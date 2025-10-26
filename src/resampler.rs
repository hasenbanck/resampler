use std::{
    array,
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use crate::{
    Complex32, Forward, Inverse, LatencyMode, Radix, RadixFFT, SampleRate,
    error::ResampleError,
    planner::ConversionConfig,
    window::{calculate_cutoff_kaiser, make_kaiser_window, make_sincs_for_kaiser},
};

const KAISER_BETA: f64 = 10.0;

pub(crate) struct FftCacheData {
    filter_spectrum: Arc<[Complex32]>,
    input_window: Arc<[f32]>,
    fft: Arc<RadixFFT<Forward>>,
    ifft: Arc<RadixFFT<Inverse>>,
}

impl Clone for FftCacheData {
    fn clone(&self) -> Self {
        Self {
            filter_spectrum: Arc::clone(&self.filter_spectrum),
            input_window: Arc::clone(&self.input_window),
            fft: Arc::clone(&self.fft),
            ifft: Arc::clone(&self.ifft),
        }
    }
}

static FFT_CACHE: LazyLock<Mutex<HashMap<(usize, usize), FftCacheData>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub struct Resampler<const CHANNEL: usize> {
    fft_resampler: FftResampler,
    chunk_size_input: usize,
    chunk_size_output: usize,
    fft_size_input: usize,
    fft_size_output: usize,
    saved_frames: usize,
    overlaps: [Vec<f32>; CHANNEL],
    input_scratch: [Vec<f32>; CHANNEL],
    output_scratch: [Vec<f32>; CHANNEL],
}

impl<const CHANNEL: usize> Resampler<CHANNEL> {
    /// Create a new [`Resampler`].
    ///
    /// Parameters are:
    /// - `sample_rate_input`: Input sample rate.
    /// - `sample_rate_output`: Output sample rate.
    /// - `latency_mode`: Latency mode this resampler works in.
    pub fn new(
        sample_rate_input: SampleRate,
        sample_rate_output: SampleRate,
        latency_mode: LatencyMode,
    ) -> Self {
        // Get the optimized FFT sizes and factors directly from the conversion table.
        // These sizes are carefully chosen for efficient factorization and minimal latency.
        let config = ConversionConfig::from_sample_rates(sample_rate_input, sample_rate_output);
        let (fft_size_input, factors_in, fft_size_output, factors_out) =
            config.scale_for_latency(latency_mode);

        let overlaps: [Vec<f32>; CHANNEL] = array::from_fn(|_| vec![0.0; fft_size_output]);

        let chunk_size_input = fft_size_input;
        let chunk_size_output = fft_size_output;

        let needed_input_buffer_size = chunk_size_input + fft_size_input;
        let needed_output_buffer_size = chunk_size_output + fft_size_output;
        let input_scratch: [Vec<f32>; CHANNEL] =
            array::from_fn(|_| vec![0.0; needed_input_buffer_size]);
        let output_scratch: [Vec<f32>; CHANNEL] =
            array::from_fn(|_| vec![0.0; needed_output_buffer_size]);

        let saved_frames = 0;

        let fft_resampler =
            FftResampler::new(fft_size_input, factors_in, fft_size_output, factors_out);

        Resampler {
            chunk_size_input,
            chunk_size_output,
            fft_size_input,
            fft_size_output,
            overlaps,
            input_scratch,
            output_scratch,
            saved_frames,
            fft_resampler,
        }
    }

    pub fn chunk_size_input(&self) -> usize {
        self.chunk_size_input
    }

    pub fn chunk_size_output(&self) -> usize {
        self.chunk_size_output
    }

    /// Returns the algorithmic delay (latency) of the resampler in input samples.
    ///
    /// This delay is inherent to the FFT-based overlap-add process and equals
    /// half the FFT input size due to the windowing operation.
    pub fn delay(&self) -> usize {
        self.fft_size_input / 2
    }

    /// Input and output must be interleaved f32 slices. For example stereo
    /// would need to have the format [L0, R0, L1, R1, ...].
    pub fn process_into_buffer(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(usize, usize), ResampleError> {
        let expected_input_len = CHANNEL * self.chunk_size_input;
        let min_output_len = CHANNEL * self.chunk_size_output;

        if input.len() < expected_input_len {
            return Err(ResampleError::InputBufferSizeSize);
        }

        if output.len() < min_output_len {
            return Err(ResampleError::OutputBufferSizeSize);
        }

        // Deinterleave input into per-channel scratch buffers.
        for frame_index in 0..self.chunk_size_input {
            for channel in 0..CHANNEL {
                self.input_scratch[channel][frame_index] = input[frame_index * CHANNEL + channel];
            }
        }

        let (subchunks_to_process, output_scratch_offset) = (
            self.chunk_size_input / self.fft_size_input,
            self.saved_frames,
        );

        // Resample between input and output scratch buffers.
        for channel in 0..CHANNEL {
            for (input_chunk, output_chunk) in self.input_scratch[channel]
                .chunks(self.fft_size_input)
                .take(subchunks_to_process)
                .zip(
                    self.output_scratch[channel][output_scratch_offset..]
                        .chunks_mut(self.fft_size_output),
                )
            {
                self.fft_resampler
                    .resample(input_chunk, output_chunk, &mut self.overlaps[channel]);
            }
        }

        // Deinterleave output from per-channel scratch buffers.
        for frame_index in 0..self.chunk_size_output {
            for channel in 0..CHANNEL {
                output[frame_index * CHANNEL + channel] = self.output_scratch[channel][frame_index];
            }
        }

        Ok((self.chunk_size_input, self.chunk_size_output))
    }
}

/// FFT-based resampler using overlap-add reconstruction.
///
/// The overlap-add resampling approach is based on the Rubato crate:
/// https://github.com/HEnquist/rubato
struct FftResampler {
    fft_size_input: usize,
    fft_size_output: usize,
    input_window: Arc<[f32]>,
    fft: Arc<RadixFFT<Forward>>,
    ifft: Arc<RadixFFT<Inverse>>,
    scratchpad_forward: Vec<Complex32>,
    scratchpad_inverse: Vec<Complex32>,
    filter_spectrum: Arc<[Complex32]>,
    input_spectrum: Vec<Complex32>,
    output_spectrum: Vec<Complex32>,
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
}

impl FftResampler {
    pub(crate) fn new(
        fft_size_input: usize,
        factors_input: Vec<Radix>,
        fft_size_output: usize,
        factors_output: Vec<Radix>,
    ) -> Self {
        let cached = Self::get_or_create_cached(
            fft_size_input,
            factors_input,
            fft_size_output,
            factors_output,
        );

        let input_spectrum: Vec<Complex32> = vec![Complex32::zero(); fft_size_input + 1];
        let input_buffer: Vec<f32> = vec![0.0; 2 * fft_size_input];
        let output_spectrum: Vec<Complex32> = vec![Complex32::zero(); fft_size_output + 1];
        let output_buffer: Vec<f32> = vec![0.0; 2 * fft_size_output];

        let scratchpad_forward = vec![Complex32::zero(); cached.fft.scratchpad_size()];
        let scratchpad_inverse = vec![Complex32::zero(); cached.ifft.scratchpad_size()];

        FftResampler {
            fft_size_input,
            fft_size_output,
            input_window: cached.input_window,
            fft: cached.fft,
            ifft: cached.ifft,
            scratchpad_forward,
            scratchpad_inverse,
            filter_spectrum: cached.filter_spectrum,
            input_spectrum,
            output_spectrum,
            input_buffer,
            output_buffer,
        }
    }

    fn get_or_create_cached(
        fft_size_input: usize,
        factors_in: Vec<Radix>,
        fft_size_output: usize,
        factors_out: Vec<Radix>,
    ) -> FftCacheData {
        // Scale factors for the 2x windowing multiplier.
        let mut fft_factors_input = factors_in.clone();
        fft_factors_input.push(Radix::Factor2);
        let mut fft_factors_output = factors_out.clone();
        fft_factors_output.push(Radix::Factor2);

        let fft = RadixFFT::<Forward>::new(fft_factors_input);
        let ifft = RadixFFT::<Inverse>::new(fft_factors_output);

        FFT_CACHE
            .lock()
            .unwrap()
            .entry((fft_size_input, fft_size_output))
            .or_insert_with(|| {
                let cutoff = match fft_size_input > fft_size_output {
                    true => {
                        let scale = fft_size_output as f64 / fft_size_input as f64;
                        calculate_cutoff_kaiser(fft_size_output, KAISER_BETA) * scale
                    }
                    false => calculate_cutoff_kaiser(fft_size_input, KAISER_BETA),
                };

                // TODO: Make the Kaiser's beta configurable. For this we need to make the cutoff calculatable.
                let sincs = make_sincs_for_kaiser(fft_size_input, 1, cutoff as f32, KAISER_BETA);
                let mut filter_time = vec![0.0; 2 * fft_size_input];
                let mut filter_spectrum = vec![Complex32::zero(); fft_size_input + 1];

                for (index, filter_value) in filter_time.iter_mut().enumerate().take(fft_size_input)
                {
                    *filter_value = sincs[0][index] / (2 * fft_size_input) as f32;
                }

                let mut scratchpad = vec![Complex32::zero(); fft.scratchpad_size()];
                fft.process(&mut filter_time, &mut filter_spectrum, &mut scratchpad);

                let input_window = make_kaiser_window(fft_size_input, KAISER_BETA);

                FftCacheData {
                    filter_spectrum: filter_spectrum.into(),
                    input_window: input_window.into(),
                    fft: Arc::new(fft),
                    ifft: Arc::new(ifft),
                }
            })
            .clone()
    }

    fn resample(&mut self, wave_input: &[f32], wave_output: &mut [f32], overlap: &mut [f32]) {
        // TODO The frequency graph shows that we seem to change the volume on some ratios.

        // Apply input window for proper overlap-add reconstruction.
        for (index, (input_sample, window_value)) in
            wave_input.iter().zip(self.input_window.iter()).enumerate()
        {
            self.input_buffer[index] = input_sample * window_value;
        }

        // Zero-pad the second half of the buffer.
        for item in self
            .input_buffer
            .iter_mut()
            .skip(self.fft_size_input)
            .take(self.fft_size_input)
        {
            *item = 0.0;
        }

        self.fft.process(
            &mut self.input_buffer,
            &mut self.input_spectrum,
            &mut self.scratchpad_forward,
        );

        let new_length = match self.fft_size_input < self.fft_size_output {
            true => self.fft_size_input + 1,
            false => self.fft_size_output,
        };

        self.input_spectrum
            .iter_mut()
            .take(new_length)
            .zip(self.filter_spectrum.iter())
            .for_each(|(spec, filter)| *spec = spec.mul(filter));

        self.output_spectrum[0..new_length].copy_from_slice(&self.input_spectrum[0..new_length]);
        for value in self.output_spectrum[new_length..].iter_mut() {
            *value = Complex32::zero();
        }

        self.ifft.process(
            &mut self.output_spectrum,
            &mut self.output_buffer,
            &mut self.scratchpad_inverse,
        );

        for (index, item) in wave_output
            .iter_mut()
            .enumerate()
            .take(self.fft_size_output)
        {
            *item = self.output_buffer[index] + overlap[index];
        }
        overlap.copy_from_slice(&self.output_buffer[self.fft_size_output..]);
    }
}
