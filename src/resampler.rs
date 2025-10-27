use std::{
    array,
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use crate::{
    Complex32, Forward, Inverse, Radix, RadixFFT, SampleRate,
    error::ResampleError,
    planner::ConversionConfig,
    window::{calculate_cutoff_kaiser, make_sincs_for_kaiser},
};

const KAISER_BETA: f64 = 10.0;

pub(crate) struct FftCacheData {
    filter_spectrum: Arc<[Complex32]>,
    fft: Arc<RadixFFT<Forward>>,
    ifft: Arc<RadixFFT<Inverse>>,
}

impl Clone for FftCacheData {
    fn clone(&self) -> Self {
        Self {
            filter_spectrum: Arc::clone(&self.filter_spectrum),
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
    pub fn new(sample_rate_input: SampleRate, sample_rate_output: SampleRate) -> Self {
        // Get the optimized FFT sizes and factors directly from the conversion table.
        // These sizes are carefully chosen for efficient factorization and minimal latency.
        let config = ConversionConfig::from_sample_rates(sample_rate_input, sample_rate_output);
        let (fft_size_input, factors_in, fft_size_output, factors_out) =
            config.scale_for_throughput();

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

                let sincs = make_sincs_for_kaiser(fft_size_input, 1, cutoff as f32, KAISER_BETA);
                let mut filter_time = vec![0.0; 2 * fft_size_input];
                let mut filter_spectrum = vec![Complex32::zero(); fft_size_input + 1];

                for (index, filter_value) in filter_time.iter_mut().enumerate().take(fft_size_input)
                {
                    *filter_value = sincs[0][index] / (2 * fft_size_input) as f32;
                }

                let mut scratchpad = vec![Complex32::zero(); fft.scratchpad_size()];
                fft.process(&filter_time, &mut filter_spectrum, &mut scratchpad);

                FftCacheData {
                    filter_spectrum: filter_spectrum.into(),
                    fft: Arc::new(fft),
                    ifft: Arc::new(ifft),
                }
            })
            .clone()
    }

    fn resample(&mut self, wave_input: &[f32], wave_output: &mut [f32], overlap: &mut [f32]) {
        // Copy input and clear padding.
        self.input_buffer[..self.fft_size_input].copy_from_slice(wave_input);
        self.input_buffer
            .iter_mut()
            .skip(self.fft_size_input)
            .take(self.fft_size_input)
            .for_each(|x| *x = 0.0);

        self.fft.process(
            &self.input_buffer,
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
            .for_each(|(spectrum, filter)| *spectrum = spectrum.mul(filter));

        self.output_spectrum[0..new_length].copy_from_slice(&self.input_spectrum[0..new_length]);
        self.output_spectrum[new_length..].iter_mut().for_each(|x| {
            *x = Complex32::zero();
        });

        self.ifft.process(
            &self.output_spectrum,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 0.02;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_dc_signal_amplitude_preservation() {
        let test_cases = vec![
            (SampleRate::_48000, SampleRate::_44100, "48kHz -> 44.1kHz"),
            (SampleRate::_44100, SampleRate::_48000, "44.1kHz -> 48kHz"),
            (SampleRate::_48000, SampleRate::_32000, "48kHz -> 32kHz"),
            (SampleRate::_32000, SampleRate::_48000, "32kHz -> 48kHz"),
            (SampleRate::_96000, SampleRate::_48000, "96kHz -> 48kHz"),
            (SampleRate::_48000, SampleRate::_96000, "48kHz -> 96kHz"),
        ];

        for (input_rate, output_rate, desc) in test_cases {
            let mut resampler = Resampler::<1>::new(input_rate, output_rate);

            let dc_amplitude = 0.5f32;
            let input = vec![dc_amplitude; resampler.chunk_size_input()];
            let mut output = vec![0.0f32; resampler.chunk_size_output()];

            for _ in 0..5 {
                let _ = resampler.process_into_buffer(&input, &mut output);
            }

            let delay = resampler.delay();
            let check_start = delay.min(output.len() / 4);
            let check_end = output.len() * 3 / 4;

            for (i, &sample) in output[check_start..check_end].iter().enumerate() {
                assert!(
                    approx_eq(sample, dc_amplitude, EPSILON),
                    "{desc}: DC amplitude not preserved at sample {}: expected {dc_amplitude}, got {sample} (error: {:.2}%)",
                    i + check_start,
                    ((sample - dc_amplitude) / dc_amplitude * 100.0).abs()
                );
            }
        }
    }

    #[test]
    fn test_sine_wave_amplitude_preservation() {
        let test_cases = vec![
            (SampleRate::_48000, SampleRate::_44100, "48kHz -> 44.1kHz"),
            (SampleRate::_44100, SampleRate::_48000, "44.1kHz -> 48kHz"),
            (SampleRate::_48000, SampleRate::_32000, "48kHz -> 32kHz"),
        ];

        for (input_rate, output_rate, desc) in test_cases {
            let mut resampler = Resampler::<1>::new(input_rate, output_rate);

            let amplitude = 0.5f32;
            let frequency = 1000.0f32;
            let input_rate_hz = match input_rate {
                SampleRate::_16000 => 16000.0,
                SampleRate::_22050 => 22050.0,
                SampleRate::_32000 => 32000.0,
                SampleRate::_44100 => 44100.0,
                SampleRate::_48000 => 48000.0,
                SampleRate::_96000 => 96000.0,
                SampleRate::_192000 => 192000.0,
            };

            let chunk_size = resampler.chunk_size_input();

            let mut phase = 0.0f32;
            let phase_increment = 2.0 * PI * frequency / input_rate_hz;
            let input: Vec<f32> = (0..chunk_size)
                .map(|_| {
                    let sample = amplitude * phase.sin();
                    phase += phase_increment;
                    sample
                })
                .collect();

            let mut output = vec![0.0f32; resampler.chunk_size_output()];

            for _ in 0..5 {
                let _ = resampler.process_into_buffer(&input, &mut output);
            }

            let delay = resampler.delay();
            let check_start = delay.min(output.len() / 4);
            let check_end = output.len() * 3 / 4;

            let peak = output[check_start..check_end]
                .iter()
                .map(|&x| x.abs())
                .fold(0.0f32, f32::max);

            assert!(
                approx_eq(peak, amplitude, EPSILON),
                "{desc}: Sine wave amplitude not preserved: expected {amplitude}, got {peak} (error: {:.2}%)",
                ((peak - amplitude) / amplitude * 100.0).abs()
            );
        }
    }

    #[test]
    fn test_stereo_dc_amplitude_preservation() {
        let mut resampler = Resampler::<2>::new(SampleRate::_48000, SampleRate::_44100);

        let dc_amplitude_left = 0.3f32;
        let dc_amplitude_right = 0.6f32;
        let chunk_size = resampler.chunk_size_input();

        let mut input = vec![0.0f32; chunk_size * 2];
        for i in 0..chunk_size {
            input[i * 2] = dc_amplitude_left;
            input[i * 2 + 1] = dc_amplitude_right;
        }

        let mut output = vec![0.0f32; resampler.chunk_size_output() * 2];

        for _ in 0..5 {
            let _ = resampler.process_into_buffer(&input, &mut output);
        }

        let delay = resampler.delay();
        let check_start = delay.min(output.len() / 8) * 2;
        let check_end = output.len() * 3 / 4;

        for i in (check_start..check_end).step_by(2) {
            let left_sample = output[i];
            let right_sample = output[i + 1];

            assert!(
                approx_eq(left_sample, dc_amplitude_left, EPSILON),
                "Stereo left channel DC not preserved at frame {}: expected {dc_amplitude_left}, got {left_sample}",
                i / 2
            );

            assert!(
                approx_eq(right_sample, dc_amplitude_right, EPSILON),
                "Stereo right channel DC not preserved at frame {}: expected {dc_amplitude_right}, got {right_sample}",
                i / 2
            );
        }
    }
}
