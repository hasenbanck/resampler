use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use core::array;
#[cfg(not(feature = "no_std"))]
use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use crate::{
    ResampleError, SampleRate,
    fir::convolve,
    window::{WindowType, calculate_cutoff_kaiser, make_sincs_for_kaiser},
};

const KAISER_BETA: f64 = 10.0;
const PHASES: usize = 1024;
const INPUT_CAPACITY: usize = 1024;

struct FirCacheData {
    coeffs: Arc<[Vec<f32>]>,
}

impl Clone for FirCacheData {
    fn clone(&self) -> Self {
        Self {
            coeffs: Arc::clone(&self.coeffs),
        }
    }
}

/// Latency configuration for the FIR resampler.
///
/// Determines the number of filter taps, which affects both rolloff and algorithmic delay.
/// Higher tap counts provide shaper rolloff but increased latency.
///
/// The enum variants are named by their algorithmic delay in samples (taps / 2):
/// - `_16`: 16 samples delay (32 taps) - lowest latency, but slow rolloff.
/// - `_32`: 32 samples delay (64 taps) - balanced latency and quality.
/// - `_64`: 64 samples delay (128 taps) - highest latency, but fasts rolloff.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Latency {
    /// 16 samples algorithmic delay (32 taps).
    _16,
    /// 32 samples algorithmic delay (64 taps).
    _32,
    /// 64 samples algorithmic delay (128 taps).
    _64,
}

impl Latency {
    /// Returns the number of filter taps for this latency setting.
    pub const fn taps(self) -> usize {
        match self {
            Latency::_16 => 32,
            Latency::_32 => 64,
            Latency::_64 => 128,
        }
    }
}

impl Default for Latency {
    /// Returns the default latency setting (`_64` = 128 taps).
    ///
    /// This provides the sharpest rolloff at the cost of higher latency (64 samples).
    fn default() -> Self {
        Latency::_64
    }
}

#[cfg(not(feature = "no_std"))]
static FIR_CACHE: LazyLock<Mutex<HashMap<u32, FirCacheData>>> =
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
pub struct ResamplerFir<const CHANNEL: usize> {
    /// Polyphase coefficient table: 1024 phases × N taps (where N is determined by latency setting).
    coeffs: Arc<[Vec<f32>]>,
    /// Per-channel fixed-size input buffers (capacity = INPUT_CAPACITY frames).
    input_buffers: [Box<[f32]>; CHANNEL],
    /// Number of valid frames currently in the input buffers.
    buffer_fill: usize,
    /// Current fractional position in input stream.
    position: f64,
    /// Resampling ratio (input_rate / output_rate).
    ratio: f64,
    /// Number of taps per phase.
    taps: usize,
    /// Number of polyphase branches.
    phases: usize,
}

impl<const CHANNEL: usize> ResamplerFir<CHANNEL> {
    /// Create a new [`ResamplerFir`].
    ///
    /// Parameters:
    /// - `input_rate`: Input sample rate.
    /// - `output_rate`: Output sample rate.
    /// - `latency`: Latency configuration determining filter length (32, 64, or 128 taps).
    ///
    /// The resampler will generate polyphase filter coefficients optimized for the
    /// given sample rate pair, using a Kaiser window with beta=10.0 for excellent
    /// stopband attenuation for 16-bit sound. Higher tap counts provide better frequency
    /// response at the cost of increased latency.
    ///
    /// # Example
    ///
    /// ```rust
    /// use resampler::{Latency, ResamplerFir, SampleRate};
    ///
    /// // Create with default latency (128 taps, 64 samples delay)
    /// let resampler =
    ///     ResamplerFir::<2>::new(SampleRate::Hz48000, SampleRate::Hz44100, Latency::default());
    ///
    /// // Create with low latency (32 taps, 16 samples delay)
    /// let resampler_low_latency =
    ///     ResamplerFir::<2>::new(SampleRate::Hz48000, SampleRate::Hz44100, Latency::_16);
    /// ```
    pub fn new(input_rate: SampleRate, output_rate: SampleRate, latency: Latency) -> Self {
        let input_rate_hz = u32::from(input_rate) as f64;
        let output_rate_hz = u32::from(output_rate) as f64;
        let ratio = input_rate_hz / output_rate_hz;

        let taps = latency.taps();
        let base_cutoff = calculate_cutoff_kaiser(taps, KAISER_BETA);
        let cutoff = if input_rate_hz <= output_rate_hz {
            // Upsampling: preserve full input bandwidth.
            base_cutoff
        } else {
            // Downsampling: scale cutoff to output Nyquist (anti-aliasing filter).
            base_cutoff * (output_rate_hz / input_rate_hz)
        };

        let coeffs = Self::get_or_create_fir_coeffs(cutoff as f32, taps);

        let input_buffers: [Box<[f32]>; CHANNEL] =
            array::from_fn(|_| vec![0.0; INPUT_CAPACITY].into_boxed_slice());

        ResamplerFir {
            coeffs,
            input_buffers,
            buffer_fill: 0,
            position: 0.0,
            ratio,
            taps,
            phases: PHASES,
        }
    }

    fn create_fir_coeffs(cutoff: f32, taps: usize) -> FirCacheData {
        let coeffs =
            make_sincs_for_kaiser(taps, PHASES, cutoff, KAISER_BETA, WindowType::Symmetric);
        FirCacheData {
            coeffs: Arc::from(coeffs.into_boxed_slice()),
        }
    }

    #[cfg(not(feature = "no_std"))]
    fn get_or_create_fir_coeffs(cutoff: f32, taps: usize) -> Arc<[Vec<f32>]> {
        let cache_key = cutoff.to_bits();
        FIR_CACHE
            .lock()
            .unwrap()
            .entry(cache_key)
            .or_insert_with(|| Self::create_fir_coeffs(cutoff, taps))
            .clone()
            .coeffs
    }

    #[cfg(feature = "no_std")]
    fn get_or_create_fir_coeffs(cutoff: f32, taps: usize) -> Arc<[Vec<f32>]> {
        Self::create_fir_coeffs(cutoff, taps).coeffs
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
        max_output_frames * CHANNEL
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
    /// use resampler::{Latency, ResamplerFir, SampleRate};
    ///
    /// let mut resampler =
    ///     ResamplerFir::<1>::new(SampleRate::Hz48000, SampleRate::Hz44100, Latency::default());
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
        if !input.len().is_multiple_of(CHANNEL) {
            return Err(ResampleError::InvalidInputBufferSize);
        }
        if !output.len().is_multiple_of(CHANNEL) {
            return Err(ResampleError::InvalidOutputBufferSize);
        }

        let input_frames = input.len() / CHANNEL;
        let output_capacity = output.len() / CHANNEL;

        let remaining_capacity = INPUT_CAPACITY - self.buffer_fill;
        let frames_to_copy = input_frames.min(remaining_capacity);

        // Deinterleave and copy input frames into fixed-size buffers.
        for frame_idx in 0..frames_to_copy {
            for channel in 0..CHANNEL {
                self.input_buffers[channel][self.buffer_fill + frame_idx] =
                    input[frame_idx * CHANNEL + channel];
            }
        }
        self.buffer_fill += frames_to_copy;

        // Generate output samples.
        let mut output_frame_count = 0;

        loop {
            #[cfg(not(feature = "no_std"))]
            let input_pos = self.position.floor() as usize;
            #[cfg(feature = "no_std")]
            let input_pos = libm::floor(self.position) as usize;

            // Check if we have enough input samples (need `taps` samples for convolution).
            if input_pos + self.taps > self.buffer_fill {
                break;
            }

            if output_frame_count >= output_capacity {
                break;
            }

            // Calculate phase index and fractional part for interpolation.
            #[cfg(not(feature = "no_std"))]
            let position_fract = self.position.fract();
            #[cfg(feature = "no_std")]
            let position_fract = self.position - libm::floor(self.position);

            let phase_f = (position_fract * self.phases as f64).min((self.phases - 1) as f64);
            let phase1 = phase_f as usize;
            let phase2 = (phase1 + 1).min(self.phases - 1);
            let frac = (phase_f - phase1 as f64) as f32;

            // Process all channels and interleave directly into output.
            for channel in 0..CHANNEL {
                // Perform N-tap convolution with linear interpolation between phases.
                let input_slice = &self.input_buffers[channel][input_pos..input_pos + self.taps];
                let sample1 = convolve(input_slice, &self.coeffs[phase1], self.taps);
                let sample2 = convolve(input_slice, &self.coeffs[phase2], self.taps);
                let sample = sample1 * (1.0 - frac) + sample2 * frac;
                output[output_frame_count * CHANNEL + channel] = sample;
            }

            output_frame_count += 1;

            self.position += self.ratio;
        }

        #[cfg(not(feature = "no_std"))]
        let consumed_frames = self.position.floor() as usize;
        #[cfg(feature = "no_std")]
        let consumed_frames = libm::floor(self.position) as usize;

        // Manually shift remaining samples to the start of buffer.
        let remaining_frames = self.buffer_fill - consumed_frames;
        for channel in 0..CHANNEL {
            for i in 0..remaining_frames {
                self.input_buffers[channel][i] = self.input_buffers[channel][consumed_frames + i];
            }
        }
        self.buffer_fill = remaining_frames;

        self.position -= consumed_frames as f64;

        Ok((frames_to_copy * CHANNEL, output_frame_count * CHANNEL))
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
        self.buffer_fill = 0;
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
        let mut scratchpad = vec![crate::fft::Complex32::zero(); fft_size];

        // Compute FFT.
        fft.process(&mut input_buffer, &mut output_buffer, &mut scratchpad);

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

        let mut resampler = ResamplerFir::<1>::new(input_rate, output_rate, Latency::_64);

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
