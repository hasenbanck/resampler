use resampler::SampleRate;

/// Interpolation mode for the resampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InterpolationMode {
    /// Linear interpolation (2-point).
    Linear,
    /// Hermite interpolation (4-point, 3rd-order).
    Hermite,
}

/// A simple interpolation resampler for comparison purposes.
pub(crate) struct InterpolationResampler<const CHANNELS: usize> {
    input_rate: f64,
    output_rate: f64,
    mode: InterpolationMode,
}

impl<const CHANNELS: usize> InterpolationResampler<CHANNELS> {
    pub(crate) fn new(
        input_rate: SampleRate,
        output_rate: SampleRate,
        mode: InterpolationMode,
    ) -> Self {
        Self {
            input_rate: u32::from(input_rate) as f64,
            output_rate: u32::from(output_rate) as f64,
            mode,
        }
    }

    /// Resample the input samples using the configured interpolation mode.
    pub(crate) fn resample(&self, input: &[f32]) -> Vec<f32> {
        match self.mode {
            InterpolationMode::Linear => self.resample_linear(input),
            InterpolationMode::Hermite => self.resample_hermite(input),
        }
    }

    /// Resample using linear interpolation (2-point).
    fn resample_linear(&self, input: &[f32]) -> Vec<f32> {
        let input_frames = input.len() / CHANNELS;
        let ratio = self.output_rate / self.input_rate;
        let output_frames = (input_frames as f64 * ratio).ceil() as usize;
        let mut output = vec![0.0f32; output_frames * CHANNELS];

        for output_frame_idx in 0..output_frames {
            let input_pos = output_frame_idx as f64 / ratio;
            let input_frame_idx = input_pos.floor() as usize;
            let frac = (input_pos - input_frame_idx as f64) as f32;

            if input_frame_idx >= input_frames - 1 {
                for ch in 0..CHANNELS {
                    let input_idx = (input_frames - 1) * CHANNELS + ch;
                    let output_idx = output_frame_idx * CHANNELS + ch;
                    output[output_idx] = input[input_idx];
                }
                continue;
            }

            for ch in 0..CHANNELS {
                let input_idx0 = input_frame_idx * CHANNELS + ch;
                let input_idx1 = (input_frame_idx + 1) * CHANNELS + ch;
                let sample0 = input[input_idx0];
                let sample1 = input[input_idx1];

                // Linear interpolation: sample0 * (1 - frac) + sample1 * frac
                let interpolated = sample0 * (1.0 - frac) + sample1 * frac;

                let output_idx = output_frame_idx * CHANNELS + ch;
                output[output_idx] = interpolated;
            }
        }

        output
    }

    /// Resample using Hermite interpolation (4-point, 3rd-order).
    ///
    /// This is the 4-point, 3rd-order Hermite interpolation x-form
    /// algorithm from "Polynomial Interpolators for High-Quality
    /// Resampling of Oversampled Audio" by Olli Niemitalo, p. 43:
    /// http://yehar.com/blog/wp-content/uploads/2009/08/deip.pdf
    fn resample_hermite(&self, input: &[f32]) -> Vec<f32> {
        let input_frames = input.len() / CHANNELS;
        let ratio = self.output_rate / self.input_rate;
        let output_frames = (input_frames as f64 * ratio).ceil() as usize;
        let mut output = vec![0.0f32; output_frames * CHANNELS];

        for output_frame_idx in 0..output_frames {
            let input_pos = output_frame_idx as f64 / ratio;
            let input_frame_idx = input_pos.floor() as usize;
            let frac = (input_pos - input_frame_idx as f64) as f32;

            for ch in 0..CHANNELS {
                // Get 4 points for Hermite interpolation: previous, current, next_1, next_2
                // Clamp indices to available samples at boundaries
                let idx_prev = if input_frame_idx > 0 {
                    input_frame_idx - 1
                } else {
                    0
                };
                let idx_current = input_frame_idx.min(input_frames - 1);
                let idx_next1 = (input_frame_idx + 1).min(input_frames - 1);
                let idx_next2 = (input_frame_idx + 2).min(input_frames - 1);

                let previous = input[idx_prev * CHANNELS + ch];
                let current = input[idx_current * CHANNELS + ch];
                let next_1 = input[idx_next1 * CHANNELS + ch];
                let next_2 = input[idx_next2 * CHANNELS + ch];

                // 4-point, 3rd-order Hermite interpolation
                let c0 = current;
                let c1 = (next_1 - previous) * 0.5;
                let c2 = previous - current * 2.5 + next_1 * 2.0 - next_2 * 0.5;
                let c3 = (next_2 - previous) * 0.5 + (current - next_1) * 1.5;

                let interpolated = ((c3 * frac + c2) * frac + c1) * frac + c0;

                let output_idx = output_frame_idx * CHANNELS + ch;
                output[output_idx] = interpolated;
            }
        }

        output
    }
}
