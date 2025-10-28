use resampler::SampleRate;

/// A simple linear interpolation resampler for comparison purposes.
pub(crate) struct LinearResampler<const CHANNELS: usize> {
    input_rate: f64,
    output_rate: f64,
}

impl<const CHANNELS: usize> LinearResampler<CHANNELS> {
    pub(crate) fn new(input_rate: SampleRate, output_rate: SampleRate) -> Self {
        Self {
            input_rate: u32::from(input_rate) as f64,
            output_rate: u32::from(output_rate) as f64,
        }
    }

    /// Resample the input samples using linear interpolation.
    pub(crate) fn resample(&self, input: &[f32]) -> Vec<f32> {
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
}
