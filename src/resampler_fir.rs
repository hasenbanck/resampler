#[cfg(feature = "no_std")]
use alloc::alloc::{Layout, alloc, dealloc};
use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use core::{ops::Deref, ptr, slice};
#[cfg(not(feature = "no_std"))]
use std::{
    alloc::{Layout, alloc, dealloc},
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use crate::{
    ResampleError, SampleRate,
    window::{WindowType, calculate_cutoff_kaiser, make_sincs_for_kaiser},
};

const PHASES: usize = 1024;
const INPUT_CAPACITY: usize = 4096;
const BUFFER_SIZE: usize = INPUT_CAPACITY * 2;

type ConvolveFn =
    fn(input: &[f32], coeffs1: &[f32], coeffs2: &[f32], frac: f32, taps: usize) -> f32;

/// A 64-byte aligned memory of f32 values.
pub(crate) struct AlignedMemory {
    ptr: *mut f32,
    len: usize,
    layout: Layout,
}

impl AlignedMemory {
    pub(crate) fn new(data: Vec<f32>) -> Self {
        const ALIGNMENT: usize = 64;

        let len = data.len();
        let size = len * size_of::<f32>();

        unsafe {
            let layout = Layout::from_size_align(size, ALIGNMENT).expect("invalid layout");
            let ptr = alloc(layout) as *mut f32;

            if ptr.is_null() {
                panic!("failed to allocate aligned memory for FIR coefficients");
            }

            ptr::copy_nonoverlapping(data.as_ptr(), ptr, len);

            Self { ptr, len, layout }
        }
    }
}

impl Deref for AlignedMemory {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for AlignedMemory {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

// Safety: AlignedSlice can be safely sent between threads.
unsafe impl Send for AlignedMemory {}

// Safety: AlignedSlice can be safely shared between threads (immutable access).
unsafe impl Sync for AlignedMemory {}

struct FirCacheData {
    coeffs: Arc<AlignedMemory>,
    taps: usize,
}

impl Clone for FirCacheData {
    fn clone(&self) -> Self {
        Self {
            coeffs: Arc::clone(&self.coeffs),
            taps: self.taps,
        }
    }
}

#[cfg(not(feature = "no_std"))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct FirCacheKey {
    cutoff_bits: u32,
    taps: usize,
    attenuation: Attenuation,
}

/// The desired stopband attenuation of the filter. Higher attenuation provides better stopband
/// rejection but slightly wider transition bands.
///
/// Defaults to -120 dB of stopband attenuation.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Attenuation {
    /// Stopband attenuation of around -60 dB (Inaudible threshold).
    Db60,
    /// Stopband attenuation of around -90 dB (transparent for 16-bit audio).
    Db90,
    /// Stopband attenuation of around -120 dB (transparent for 24-bit audio).
    #[default]
    Db120,
}

impl Attenuation {
    /// Returns the Kaiser window beta value for the desired attenuation level.
    ///
    /// The beta value controls the shape of the Kaiser window and directly affects
    /// the stopband attenuation of the resulting filter.
    pub(crate) fn to_kaiser_beta(self) -> f64 {
        match self {
            Attenuation::Db60 => 7.0,
            Attenuation::Db90 => 10.0,
            Attenuation::Db120 => 13.0,
        }
    }
}

/// Latency configuration for the FIR resampler.
///
/// Determines the number of filter taps, which affects both rolloff and algorithmic delay.
/// Higher tap counts provide shaper rolloff but increased latency.
///
/// The enum variants are named by their algorithmic delay in samples (taps / 2):
/// - `Sample8`: 8 samples delay (16 taps)
/// - `Sample16`: 16 samples delay (32 taps)
/// - `Sample32`: 32 samples delay (64 taps)
/// - `Sample64`: 64 samples delay (128 taps)
///
/// Defaults to 64 samples delay (128 taps).
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Latency {
    /// 8 samples algorithmic delay (16 taps).
    Sample8,
    /// 16 samples algorithmic delay (32 taps).
    Sample16,
    /// 32 samples algorithmic delay (64 taps).
    Sample32,
    /// 64 samples algorithmic delay (128 taps).
    #[default]
    Sample64,
}

impl Latency {
    /// Returns the number of filter taps for this latency setting.
    pub const fn taps(self) -> usize {
        // Taps need to be a power of two for convolve filter to run (there is no tail handling).
        match self {
            Latency::Sample8 => 16,
            Latency::Sample16 => 32,
            Latency::Sample32 => 64,
            Latency::Sample64 => 128,
        }
    }
}

#[cfg(not(feature = "no_std"))]
static FIR_CACHE: LazyLock<Mutex<HashMap<FirCacheKey, FirCacheData>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// High-quality polyphase FIR audio resampler supporting multi-channel audio with streaming API.
///
/// `ResamplerFir` uses a configurable polyphase FIR filter (32, 64, or 128 taps) decomposed
/// into 1024 branches for high-quality audio resampling with configurable latency.
/// The const generic parameter `CHANNEL` specifies the number of audio channels.
///
/// Unlike the FFT-based resampler, this implementation supports streaming with arbitrary
/// input buffer sizes, making it ideal for real-time applications. The latency can be
/// configured at construction time using the [`Latency`] enum to balance quality versus delay.
///
/// The stopband attenuation can also be configured via the [`Attenuation`] enum.
pub struct ResamplerFir {
    /// Number of audio channels.
    channels: usize,
    /// Polyphase coefficient table stored contiguously: all phases × taps in a single allocation.
    /// Layout: [phase0_tap0..N, phase1_tap0..N, ..., phase1023_tap0..N]
    coeffs: Arc<AlignedMemory>,
    /// Per-channel double-sized input buffers for efficient buffer management.
    /// Size = BUFFER_SIZE (2x INPUT_CAPACITY) to minimize copy operations.
    input_buffers: Box<[f32]>,
    /// Read position in the input buffer (where we start reading from).
    read_position: usize,
    /// Number of valid frames available for processing (from read_position).
    available_frames: usize,
    /// Current fractional position within available frames.
    position: f64,
    /// Resampling ratio (input_rate / output_rate).
    ratio: f64,
    /// Number of taps per phase.
    taps: usize,
    /// Number of polyphase branches.
    phases: usize,
    convolve_function: ConvolveFn,
}

impl ResamplerFir {
    /// Create a new [`ResamplerFir`].
    ///
    /// Parameters:
    /// - `channels`: The channel count.
    /// - `input_rate`: Input sample rate.
    /// - `output_rate`: Output sample rate.
    /// - `latency`: Latency configuration determining filter length (32, 64, or 128 taps).
    /// - `attenuation`: Desired stopband attenuation controlling filter quality.
    ///
    /// The resampler will generate polyphase filter coefficients optimized for the
    /// given sample rate pair, using a Kaiser window with beta value determined by the
    /// attenuation setting. Higher tap counts provide better frequency response at the
    /// cost of increased latency. Higher attenuation provides better stopband rejection
    /// but slightly wider transition bands.
    ///
    /// # Example
    ///
    /// ```rust
    /// use resampler::{Attenuation, Latency, ResamplerFir, SampleRate};
    ///
    /// // Create with default latency (128 taps, 64 samples delay) and 90 dB attenuation
    /// let resampler = ResamplerFir::new(
    ///     2,
    ///     SampleRate::Hz48000,
    ///     SampleRate::Hz44100,
    ///     Latency::default(),
    ///     Attenuation::default(),
    /// );
    ///
    /// // Create with low latency (32 taps, 16 samples delay) and 60 dB attenuation
    /// let resampler_low_latency = ResamplerFir::new(
    ///     2,
    ///     SampleRate::Hz48000,
    ///     SampleRate::Hz44100,
    ///     Latency::Sample16,
    ///     Attenuation::Db60,
    /// );
    /// ```
    pub fn new(
        channels: usize,
        input_rate: SampleRate,
        output_rate: SampleRate,
        latency: Latency,
        attenuation: Attenuation,
    ) -> Self {
        let input_rate_hz = u32::from(input_rate) as f64;
        let output_rate_hz = u32::from(output_rate) as f64;
        let ratio = input_rate_hz / output_rate_hz;

        let taps = latency.taps();
        let beta = attenuation.to_kaiser_beta();
        let base_cutoff = calculate_cutoff_kaiser(taps, beta);
        let cutoff = if input_rate_hz <= output_rate_hz {
            // Upsampling: preserve full input bandwidth.
            base_cutoff
        } else {
            // Downsampling: scale cutoff to output Nyquist (anti-aliasing filter).
            base_cutoff * (output_rate_hz / input_rate_hz)
        };

        let coeffs = Self::get_or_create_fir_coeffs(cutoff as f32, taps, attenuation);

        // Allocate double-sized buffers for efficient buffer management.
        let input_buffers = vec![0.0; BUFFER_SIZE * channels].into_boxed_slice();

        #[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
        let convolve_function = if std::arch::is_x86_feature_detected!("avx512f") && taps >= 16 {
            fn wrapper(
                input: &[f32],
                coeffs1: &[f32],
                coeffs2: &[f32],
                frac: f32,
                taps: usize,
            ) -> f32 {
                unsafe {
                    crate::fir::avx512::convolve_interp_avx512(input, coeffs1, coeffs2, frac, taps)
                }
            }
            wrapper
        } else if std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("fma")
        {
            fn wrapper(
                input: &[f32],
                coeffs1: &[f32],
                coeffs2: &[f32],
                frac: f32,
                taps: usize,
            ) -> f32 {
                unsafe {
                    crate::fir::avx::convolve_interp_avx_fma(input, coeffs1, coeffs2, frac, taps)
                }
            }
            wrapper
        } else if std::arch::is_x86_feature_detected!("sse4.2") {
            fn wrapper(
                input: &[f32],
                coeffs1: &[f32],
                coeffs2: &[f32],
                frac: f32,
                taps: usize,
            ) -> f32 {
                unsafe {
                    crate::fir::sse4_2::convolve_interp_sse4_2(input, coeffs1, coeffs2, frac, taps)
                }
            }
            wrapper
        } else {
            // SSE2 is always available.
            fn wrapper(
                input: &[f32],
                coeffs1: &[f32],
                coeffs2: &[f32],
                frac: f32,
                taps: usize,
            ) -> f32 {
                unsafe {
                    crate::fir::sse2::convolve_interp_sse2(input, coeffs1, coeffs2, frac, taps)
                }
            }
            wrapper
        };

        ResamplerFir {
            channels,
            coeffs,
            input_buffers,
            read_position: 0,
            available_frames: 0,
            position: 0.0,
            ratio,
            taps,
            phases: PHASES,
            #[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
            convolve_function,
            #[cfg(any(not(target_arch = "x86_64"), feature = "no_std"))]
            convolve_function: crate::fir::convolve_interp,
        }
    }

    fn create_fir_coeffs(cutoff: f32, taps: usize, beta: f64) -> FirCacheData {
        let polyphase_coeffs =
            make_sincs_for_kaiser(taps, PHASES, cutoff, beta, WindowType::Symmetric);

        // Flatten the polyphase coefficients into a single contiguous allocation.
        // Layout: [phase0_tap0..N, phase1_tap0..N, ..., phase1023_tap0..N]
        let total_size = PHASES * taps;
        let mut flattened = Vec::with_capacity(total_size);
        for phase_coeffs in polyphase_coeffs {
            flattened.extend_from_slice(&phase_coeffs);
        }

        FirCacheData {
            coeffs: Arc::new(AlignedMemory::new(flattened)),
            taps,
        }
    }

    #[cfg(not(feature = "no_std"))]
    fn get_or_create_fir_coeffs(
        cutoff: f32,
        taps: usize,
        attenuation: Attenuation,
    ) -> Arc<AlignedMemory> {
        let cache_key = FirCacheKey {
            cutoff_bits: cutoff.to_bits(),
            taps,
            attenuation,
        };
        let beta = attenuation.to_kaiser_beta();
        FIR_CACHE
            .lock()
            .unwrap()
            .entry(cache_key)
            .or_insert_with(|| Self::create_fir_coeffs(cutoff, taps, beta))
            .clone()
            .coeffs
    }

    #[cfg(feature = "no_std")]
    fn get_or_create_fir_coeffs(
        cutoff: f32,
        taps: usize,
        attenuation: Attenuation,
    ) -> Arc<AlignedMemory> {
        let beta = attenuation.to_kaiser_beta();
        Self::create_fir_coeffs(cutoff, taps, beta).coeffs
    }

    /// Calculate the maximum output buffer size that needs to be allocated.
    pub fn buffer_size_output(&self) -> usize {
        // Conservative upper bound: assume buffer could be maximally filled.
        let max_total_frames = INPUT_CAPACITY;
        let max_usable_frames = (max_total_frames - self.taps) as f64;
        #[cfg(not(feature = "no_std"))]
        let max_output_frames = (max_usable_frames / self.ratio).ceil() as usize + 2;
        #[cfg(feature = "no_std")]
        let max_output_frames = libm::ceil(max_usable_frames / self.ratio) as usize + 2;
        max_output_frames * self.channels
    }

    /// Process audio samples, resampling from input to output sample rate.
    ///
    /// This is a streaming API that accepts arbitrary input buffer sizes and produces
    /// as many output samples as possible given the available input.
    ///
    /// Input and output must be interleaved f32 slices with all channels interleaved.
    /// For stereo audio, the format is `[L0, R0, L1, R1, ...]`. For mono, it's `[S0, S1, S2, ...]`.
    ///
    /// ## Parameters
    ///
    /// - `input`: Interleaved input samples. Length must be a multiple of `CHANNEL`.
    /// - `output`: Interleaved output buffer. Length must be a multiple of `CHANNEL`.
    ///
    /// ## Returns
    ///
    /// `Ok((consumed, produced))` where:
    /// - `consumed`: Number of input samples consumed (in total f32 values, including all channels).
    /// - `produced`: Number of output samples produced (in total f32 values, including all channels).
    ///
    /// ## Example
    ///
    /// ```rust
    /// use resampler::{Attenuation, Latency, ResamplerFir, SampleRate};
    ///
    /// let mut resampler = ResamplerFir::new(
    ///     1,
    ///     SampleRate::Hz48000,
    ///     SampleRate::Hz44100,
    ///     Latency::default(),
    ///     Attenuation::default(),
    /// );
    /// let buffer_size_output = resampler.buffer_size_output();
    /// let input = vec![0.0f32; 256];
    /// let mut output = vec![0.0f32; buffer_size_output];
    ///
    /// match resampler.resample(&input, &mut output) {
    ///     Ok((consumed, produced)) => {
    ///         println!("Processed {consumed} input samples into {produced} output samples");
    ///     }
    ///     Err(error) => eprintln!("Resampling error: {error:?}"),
    /// }
    /// ```
    pub fn resample(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(usize, usize), ResampleError> {
        if !input.len().is_multiple_of(self.channels) {
            return Err(ResampleError::InvalidInputBufferSize);
        }
        if !output.len().is_multiple_of(self.channels) {
            return Err(ResampleError::InvalidOutputBufferSize);
        }

        let input_frames = input.len() / self.channels;
        let output_capacity = output.len() / self.channels;

        let write_position = self.read_position + self.available_frames;
        let remaining_capacity = BUFFER_SIZE.saturating_sub(write_position);
        let frames_to_copy = input_frames
            .min(remaining_capacity)
            .min(INPUT_CAPACITY - self.available_frames);

        // Deinterleave and copy input frames into double-sized buffers.
        for frame_idx in 0..frames_to_copy {
            for channel in 0..self.channels {
                let channel_buf = &mut self.input_buffers[BUFFER_SIZE * channel..];
                channel_buf[write_position + frame_idx] =
                    input[frame_idx * self.channels + channel];
            }
        }
        self.available_frames += frames_to_copy;

        let mut output_frame_count = 0;

        loop {
            #[cfg(not(feature = "no_std"))]
            let input_offset = self.position.floor() as usize;
            #[cfg(feature = "no_std")]
            let input_offset = libm::floor(self.position) as usize;

            // Check if we have enough input samples (need `taps` samples for convolution).
            if input_offset + self.taps > self.available_frames {
                break;
            }

            if output_frame_count >= output_capacity {
                break;
            }

            #[cfg(not(feature = "no_std"))]
            let position_fract = self.position.fract();
            #[cfg(feature = "no_std")]
            let position_fract = self.position - libm::floor(self.position);

            let phase_f = (position_fract * self.phases as f64).min((self.phases - 1) as f64);
            let phase1 = phase_f as usize;
            let phase2 = (phase1 + 1).min(self.phases - 1);
            let frac = (phase_f - phase1 as f64) as f32;

            for channel in 0..self.channels {
                // Perform N-tap convolution with linear interpolation between phases.
                let actual_pos = self.read_position + input_offset;
                let channel_buf = &self.input_buffers[BUFFER_SIZE * channel..];
                let input_slice = &channel_buf[actual_pos..actual_pos + self.taps];

                let phase1_start = phase1 * self.taps;
                let coeffs_phase1 = &self.coeffs[phase1_start..phase1_start + self.taps];
                let phase2_start = phase2 * self.taps;
                let coeffs_phase2 = &self.coeffs[phase2_start..phase2_start + self.taps];

                let sample = (self.convolve_function)(
                    input_slice,
                    coeffs_phase1,
                    coeffs_phase2,
                    frac,
                    self.taps,
                );
                output[output_frame_count * self.channels + channel] = sample;
            }

            output_frame_count += 1;
            self.position += self.ratio;
        }

        // Update buffer state: consume processed frames.
        #[cfg(not(feature = "no_std"))]
        let consumed_frames = self.position.floor() as usize;
        #[cfg(feature = "no_std")]
        let consumed_frames = libm::floor(self.position) as usize;

        self.read_position += consumed_frames;
        self.available_frames -= consumed_frames;
        self.position -= consumed_frames as f64;

        // Double-buffer optimization: only copy when read_position exceeds threshold.
        if self.read_position > INPUT_CAPACITY {
            // Copy remaining valid data to the beginning of the buffer.
            for channel in 0..self.channels {
                let channel_buf = &mut self.input_buffers[BUFFER_SIZE * channel..];
                channel_buf.copy_within(
                    self.read_position..self.read_position + self.available_frames,
                    0,
                );
            }
            self.read_position = 0;
        }

        Ok((
            frames_to_copy * self.channels,
            output_frame_count * self.channels,
        ))
    }

    /// Returns the algorithmic delay (latency) of the resampler in input samples.
    ///
    /// For the polyphase FIR resampler, this equals half the filter length due to the
    /// symmetric FIR filter design:
    /// - `Latency::_16`: 16 samples (32 taps / 2)
    /// - `Latency::_32`: 32 samples (64 taps / 2)
    /// - `Latency::_64`: 64 samples (128 taps / 2)
    pub fn delay(&self) -> usize {
        self.taps / 2
    }

    /// Resets the resampler state, clearing all internal buffers.
    ///
    /// Call this when starting to process a new audio stream to avoid
    /// discontinuities from previous audio data.
    pub fn reset(&mut self) {
        self.read_position = 0;
        self.available_frames = 0;
        self.position = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::fft::{Forward, Radix, RadixFFT};

    /// Helper function to compute frequency response magnitude in dB from impulse response.
    fn compute_frequency_response_db(impulse_response: &[f32], fft_size: usize) -> Vec<f32> {
        assert!(fft_size.is_power_of_two(), "FFT size must be power of two");

        // Create FFT object.
        let num_factors = fft_size.trailing_zeros() as usize;
        let factors = vec![Radix::Factor2; num_factors];
        let fft = RadixFFT::<Forward>::new(factors);

        // Prepare input buffer (zero-padded or truncated to fft_size).
        let mut input_buffer = vec![0.0f32; fft_size];
        let copy_len = impulse_response.len().min(fft_size);
        input_buffer[..copy_len].copy_from_slice(&impulse_response[..copy_len]);

        // Prepare output and scratchpad buffers.
        let mut output_buffer = vec![crate::fft::Complex32::zero(); fft_size / 2 + 1];
        let mut scratchpad = vec![crate::fft::Complex32::zero(); fft.scratchpad_size()];

        // Compute FFT.
        fft.process(&input_buffer, &mut output_buffer, &mut scratchpad);

        // Compute magnitudes in dB.
        output_buffer
            .iter()
            .map(|c| {
                let magnitude = (c.re * c.re + c.im * c.im).sqrt();
                if magnitude > 1e-10 {
                    20.0 * magnitude.log10()
                } else {
                    -200.0
                }
            })
            .collect()
    }

    /// Helper to get frequency bin index from frequency in Hz.
    fn freq_to_bin(freq_hz: f32, sample_rate_hz: f32, fft_size: usize) -> usize {
        ((freq_hz / sample_rate_hz) * fft_size as f32).round() as usize
    }

    /// Resample an impulse signal and extract the impulse response from output.
    fn get_resampled_impulse_response(
        input_rate: SampleRate,
        output_rate: SampleRate,
        duration_sec: f32,
    ) -> Vec<f32> {
        let input_rate_hz = u32::from(input_rate);

        let input_samples = (input_rate_hz as f32 * duration_sec) as usize;

        let impulse_pos = (input_samples as f32 * 0.5).min(input_samples as f32 - 1.0) as usize;
        let mut input = vec![0.0f32; input_samples];
        input[impulse_pos] = 1.0;

        let mut resampler = ResamplerFir::new(
            1,
            input_rate,
            output_rate,
            Latency::Sample64,
            Attenuation::Db90,
        );

        let buffer_size_output = resampler.buffer_size_output();
        let mut output_buffer = vec![0.0f32; buffer_size_output];
        let mut output = Vec::new();
        let mut input_offset = 0;

        while input_offset < input_samples {
            let remaining = input_samples - input_offset;
            let chunk_size = remaining.min(256);
            let input_chunk = &input[input_offset..input_offset + chunk_size];

            let (consumed, produced) = resampler
                .resample(input_chunk, &mut output_buffer)
                .expect("FIR resampling failed");

            output.extend_from_slice(&output_buffer[..produced]);

            input_offset += consumed;

            if consumed == 0 {
                break;
            }
        }

        output
    }

    /// Measure stopband attenuation for a given sample rate conversion.
    fn measure_stopband_attenuation(input_rate: SampleRate, output_rate: SampleRate) {
        let resampled_output = get_resampled_impulse_response(input_rate, output_rate, 5.0);

        let peak_idx = resampled_output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let output_rate_hz = u32::from(output_rate);
        let window_size = (output_rate_hz as f32 * 0.1) as usize;
        let start = peak_idx.saturating_sub(window_size / 2);
        let end = (start + window_size).min(resampled_output.len());
        let impulse_response = &resampled_output[start..end];

        let fft_size = 8192;
        let magnitude_db = compute_frequency_response_db(impulse_response, fft_size);

        let input_nyquist_hz = u32::from(input_rate) as f32 / 2.0;
        let passband_end_hz = input_nyquist_hz * 0.9; // 90% of input Nyquist
        let stopband_start_hz = input_nyquist_hz * 1.1; // 110% of input Nyquist

        let passband_start_bin = freq_to_bin(20.0, output_rate_hz as f32, fft_size);
        let passband_end_bin = freq_to_bin(passband_end_hz, output_rate_hz as f32, fft_size);
        let stopband_start_bin = freq_to_bin(stopband_start_hz, output_rate_hz as f32, fft_size);
        let stopband_end_bin = (magnitude_db.len() - 10).min(freq_to_bin(
            output_rate_hz as f32 / 2.0 * 0.95,
            output_rate_hz as f32,
            fft_size,
        ));

        let passband_values = &magnitude_db[passband_start_bin..=passband_end_bin];
        let passband_max = passband_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let stopband_values = &magnitude_db[stopband_start_bin..=stopband_end_bin];
        let stopband_max = stopband_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let _stopband_min = stopband_values
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);

        let attenuation = passband_max - stopband_max;

        #[cfg(not(feature = "no_std"))]
        {
            println!("Passband peak: {passband_max:.2} dB");
            println!("Stopband: min = {_stopband_min:.2} dB, max = {stopband_max:.2} dB");
            println!("Stopband attenuation: {attenuation:.2} dB");
        }
        assert!(
            attenuation >= 90.0,
            "FAIL: Stopband attenuation too low: {attenuation:.2} dB (required: >= 90 dB)",
        );
    }

    #[test]
    fn test_stopband_attenuation_22050_to_44100() {
        #[cfg(not(feature = "no_std"))]
        println!("=== 22050 Hz -> 44100 Hz ===");
        measure_stopband_attenuation(SampleRate::Hz22050, SampleRate::Hz44100);
    }

    #[test]
    fn test_stopband_attenuation_22050_to_48000() {
        #[cfg(not(feature = "no_std"))]
        println!("=== 22050 Hz -> 48000 Hz ===");
        measure_stopband_attenuation(SampleRate::Hz22050, SampleRate::Hz48000);
    }
}
