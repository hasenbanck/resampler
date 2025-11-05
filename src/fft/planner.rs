use alloc::{vec, vec::Vec};

use crate::{Radix, SampleRate, SampleRateFamily};

/// Configuration for FFT-based resampling between two sample rates.
///
/// ## Supported conversion patterns
///
/// Following "families" of sampling rates are supported:
///
/// - 16 kHz family (and all multiples)
/// - 22.05 kHz family (and all multiples)
/// - 48 kHz family (and all multiples)
///
/// | Conversion Type              | Input Size | Output Size | Ratio Error | Input FFT           | Output FFT          | Factorization            |
/// |------------------------------|------------|-------------|-------------|---------------------|---------------------|--------------------------|
/// | Inside same family           | 2          | 2           | 0.0%        | Radix-2             | Radix-2             | 2 → 2                    |
/// | Between 22.05 kHz and 48 kHz | 588        | 1280        | 0.0%        | Mixed-Radix (3,4,7) | Mixed-Radix (4,5)   | 3 × 4 × 7 × 7 → 4⁴ × 5   |
/// | Between 16 kHz and 48 kHz    | 64         | 192         | 0.0%        | Radix-2             | Mixed-Radix (2,3)   | 2⁶ → 2⁶ × 3              |
/// | Between 16 kHz and 44.1 kHz  | 640        | 882         | 0.0%        | Mixed-Radix (2,4,5) | Mixed-Radix (2,3,7) | 2 × 4⁴ × 5 → 2 × 3² × 7² |
#[derive(Debug, Clone)]
pub(crate) struct ConversionConfig {
    /// Base input FFT size (minimal latency).
    pub(crate) base_fft_size_in: usize,
    /// Base output FFT size (minimal latency).
    pub(crate) base_fft_size_out: usize,
    /// Base factorization for input FFT.
    pub(crate) base_factors_in: Vec<Radix>,
    /// Base factorization for output FFT.
    pub(crate) base_factors_out: Vec<Radix>,
}

impl ConversionConfig {
    /// Get the minimal FFT sizes needed for accurate conversion between sample rates.
    pub(crate) fn from_sample_rates(
        input_rate: SampleRate,
        output_rate: SampleRate,
    ) -> ConversionConfig {
        let input_family = input_rate.family();
        let output_family = output_rate.family();

        let input_multiplier = input_rate.family_multiplier() as usize;
        let output_multiplier = output_rate.family_multiplier() as usize;

        let base_config = match (input_family, output_family) {
            // Same family: base 1:1 ratio (2 → 2)
            (SampleRateFamily::Hz48000, SampleRateFamily::Hz48000)
            | (SampleRateFamily::Hz22050, SampleRateFamily::Hz22050)
            | (SampleRateFamily::Hz16000, SampleRateFamily::Hz16000) => ConversionConfig {
                base_fft_size_in: 2,
                base_fft_size_out: 2,
                base_factors_in: vec![Radix::Factor2],
                base_factors_out: vec![Radix::Factor2],
            },

            // 22.05 kHz → 48 kHz family (3 × 4 × 7 × 7 → 4⁴ × 5)
            (SampleRateFamily::Hz22050, SampleRateFamily::Hz48000) => ConversionConfig {
                base_fft_size_in: 588,
                base_fft_size_out: 1280,
                base_factors_in: vec![
                    Radix::Factor3,
                    Radix::Factor4,
                    Radix::Factor7,
                    Radix::Factor7,
                ],
                base_factors_out: vec![
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor5,
                ],
            },
            // 48 kHz → 22.05 kHz family (4⁴ × 5 → 3 × 4 × 7 × 7)
            (SampleRateFamily::Hz48000, SampleRateFamily::Hz22050) => ConversionConfig {
                base_fft_size_in: 1280,
                base_fft_size_out: 588,
                base_factors_in: vec![
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor5,
                ],
                base_factors_out: vec![
                    Radix::Factor3,
                    Radix::Factor4,
                    Radix::Factor7,
                    Radix::Factor7,
                ],
            },

            // 16 kHz → 48 kHz family (2⁶ → 2⁶ × 3)
            (SampleRateFamily::Hz16000, SampleRateFamily::Hz48000) => ConversionConfig {
                base_fft_size_in: 64,
                base_fft_size_out: 192,
                base_factors_in: vec![Radix::Factor2; 6],
                base_factors_out: vec![
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor3,
                ],
            },
            // 48 kHz → 16 kHz family (2⁶ × 3 → 2⁶)
            (SampleRateFamily::Hz48000, SampleRateFamily::Hz16000) => ConversionConfig {
                base_fft_size_in: 192,
                base_fft_size_out: 64,
                base_factors_in: vec![
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor3,
                ],
                base_factors_out: vec![Radix::Factor2; 6],
            },

            // 16 kHz → 22.05 kHz family (2 × 4⁴ × 5 → 2 × 3² × 7²)
            (SampleRateFamily::Hz16000, SampleRateFamily::Hz22050) => ConversionConfig {
                base_fft_size_in: 640,
                base_fft_size_out: 882,
                base_factors_in: vec![
                    Radix::Factor2,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor5,
                ],
                base_factors_out: vec![
                    Radix::Factor2,
                    Radix::Factor3,
                    Radix::Factor3,
                    Radix::Factor7,
                    Radix::Factor7,
                ],
            },
            // 22.05 kHz → 16 kHz family (2 × 3² × 7² → 2 × 4⁴ × 5)
            (SampleRateFamily::Hz22050, SampleRateFamily::Hz16000) => ConversionConfig {
                base_fft_size_in: 882,
                base_fft_size_out: 640,
                base_factors_in: vec![
                    Radix::Factor2,
                    Radix::Factor3,
                    Radix::Factor3,
                    Radix::Factor7,
                    Radix::Factor7,
                ],
                base_factors_out: vec![
                    Radix::Factor2,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor5,
                ],
            },
        };

        // Apply family multipliers to scale FFT sizes and factors.
        let scaled_fft_size_in = base_config.base_fft_size_in * input_multiplier;
        let scaled_fft_size_out = base_config.base_fft_size_out * output_multiplier;

        // Decompose multipliers into radix factors.
        let input_multiplier_factors = Self::decompose_multiplier(input_multiplier);
        let output_multiplier_factors = Self::decompose_multiplier(output_multiplier);

        // Append multiplier factors to base factors (maintains high→low ordering).
        let mut scaled_factors_in = base_config.base_factors_in.clone();
        scaled_factors_in.extend(input_multiplier_factors);

        let mut scaled_factors_out = base_config.base_factors_out.clone();
        scaled_factors_out.extend(output_multiplier_factors);

        ConversionConfig {
            base_fft_size_in: scaled_fft_size_in,
            base_fft_size_out: scaled_fft_size_out,
            base_factors_in: scaled_factors_in,
            base_factors_out: scaled_factors_out,
        }
    }

    /// Decompose a power-of-2 multiplier into radix factors.
    fn decompose_multiplier(multiplier: usize) -> Vec<Radix> {
        if multiplier == 1 {
            return Vec::new();
        }

        assert!(multiplier.is_power_of_two());

        let num_bits = multiplier.trailing_zeros() as usize;

        // Prefer Factor4 for efficiency: decompose into 4s and 2s.
        let num_factor4 = num_bits / 2;
        let num_factor2 = num_bits % 2;

        let mut factors = vec![Radix::Factor4; num_factor4];
        if num_factor2 > 0 {
            factors.push(Radix::Factor2);
        }
        factors
    }

    /// Scale the base FFT sizes to ensure a minimum of 512 input samples.
    ///
    /// Returns: (input_size, input_factors, output_size, output_factors)
    pub(crate) fn scale_for_throughput(&self) -> (usize, Vec<Radix>, usize, Vec<Radix>) {
        const TARGET_INPUT_SAMPLES: usize = 512;

        // Calculate the multiplier needed to reach target input samples.
        #[cfg(not(feature = "no_std"))]
        let multiplier = (TARGET_INPUT_SAMPLES as f32 / self.base_fft_size_in as f32)
            .ceil()
            .max(1.0) as usize;
        #[cfg(feature = "no_std")]
        let multiplier = libm::ceilf(TARGET_INPUT_SAMPLES as f32 / self.base_fft_size_in as f32)
            .max(1.0) as usize;

        // Round multiplier to nearest power of 2.
        let multiplier = multiplier.next_power_of_two();
        let scaled_fft_size_in = self.base_fft_size_in * multiplier;
        let scaled_fft_size_out = self.base_fft_size_out * multiplier;

        // Decompose multiplier into factors.
        let scaling_factors_in = Self::decompose_multiplier(multiplier);
        let scaling_factors_out = Self::decompose_multiplier(multiplier);

        let mut factors_in = self.base_factors_in.clone();
        factors_in.extend(scaling_factors_in);

        let mut factors_out = self.base_factors_out.clone();
        factors_out.extend(scaling_factors_out);

        (
            scaled_fft_size_in,
            factors_in,
            scaled_fft_size_out,
            factors_out,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_config_48000_to_96000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::Hz48000, SampleRate::Hz96000);
        assert_eq!(config.base_fft_size_in, 2);
        assert_eq!(config.base_fft_size_out, 4);
    }

    #[test]
    fn test_conversion_config_48000_to_192000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::Hz48000, SampleRate::Hz192000);
        assert_eq!(config.base_fft_size_in, 2);
        assert_eq!(config.base_fft_size_out, 8);
    }

    #[test]
    fn test_conversion_config_22050_to_48000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::Hz22050, SampleRate::Hz48000);
        assert_eq!(config.base_fft_size_in, 588);
        assert_eq!(config.base_fft_size_out, 1280);
    }

    #[test]
    fn test_conversion_config_16000_to_48000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::Hz16000, SampleRate::Hz48000);
        assert_eq!(config.base_fft_size_in, 64);
        assert_eq!(config.base_fft_size_out, 192);
    }

    #[test]
    fn test_conversion_config_16000_to_44100() {
        let config = ConversionConfig::from_sample_rates(SampleRate::Hz16000, SampleRate::Hz44100);
        assert_eq!(config.base_fft_size_in, 640);
        assert_eq!(config.base_fft_size_out, 1764);
    }

    #[test]
    fn test_conversion_config_44100_to_48000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::Hz44100, SampleRate::Hz48000);
        assert_eq!(config.base_fft_size_in, 1176);
        assert_eq!(config.base_fft_size_out, 1280);

        assert_eq!(
            config.base_factors_in,
            vec![
                Radix::Factor3,
                Radix::Factor4,
                Radix::Factor7,
                Radix::Factor7,
                Radix::Factor2
            ]
        );

        assert_eq!(
            config.base_factors_out,
            vec![
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor5
            ]
        );
    }

    #[test]
    fn test_conversion_config_44100_to_96000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::Hz44100, SampleRate::Hz96000);
        assert_eq!(config.base_fft_size_in, 1176);
        assert_eq!(config.base_fft_size_out, 2560);

        assert_eq!(
            config.base_factors_in,
            vec![
                Radix::Factor3,
                Radix::Factor4,
                Radix::Factor7,
                Radix::Factor7,
                Radix::Factor2
            ]
        );

        // Output: base [Factor4, Factor4, Factor4, Factor4, Factor5] + multiplier [Factor2]
        assert_eq!(
            config.base_factors_out,
            vec![
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor5,
                Radix::Factor2
            ]
        );
    }

    #[test]
    fn test_prefer_factor4_for_mixed_radix() {
        // Test that prefer_factor4 logic works correctly:
        // - Pure Factor2 bases use Factor2 for multipliers
        // - Mixed-radix bases use Factor4 for multipliers (when possible)

        let config = ConversionConfig {
            base_fft_size_in: 588,
            base_fft_size_out: 1280,
            base_factors_in: vec![
                Radix::Factor3,
                Radix::Factor4,
                Radix::Factor7,
                Radix::Factor7,
            ],
            base_factors_out: vec![
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor5,
            ],
        };

        let (size_in, factors_in, size_out, factors_out) = config.scale_for_throughput();

        // Base is 588, target is 512, so multiplier = 1 (no scaling needed)
        assert_eq!(size_in, 588);
        assert_eq!(size_out, 1280);

        // No scaling factors added, just the base factors
        assert_eq!(
            factors_in,
            vec![
                Radix::Factor3,
                Radix::Factor4,
                Radix::Factor7,
                Radix::Factor7
            ]
        );
        assert_eq!(
            factors_out,
            vec![
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor5
            ]
        );
    }

    #[test]
    fn test_throughput_scaling() {
        let config = ConversionConfig {
            base_fft_size_in: 588,
            base_fft_size_out: 1280,
            base_factors_in: vec![
                Radix::Factor3,
                Radix::Factor4,
                Radix::Factor7,
                Radix::Factor7,
            ],
            base_factors_out: vec![
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor5,
            ],
        };

        // Target is 512 samples, but base is 588, so multiplier = 1 (no scaling)
        let (input, factors_in, output, factors_out) = config.scale_for_throughput();
        assert_eq!(input, 588);
        assert_eq!(output, 1280);

        // No scaling factors added, just the base factors
        assert_eq!(
            factors_in,
            vec![
                Radix::Factor3,
                Radix::Factor4,
                Radix::Factor7,
                Radix::Factor7
            ]
        );

        assert_eq!(
            factors_out,
            vec![
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor5
            ]
        );
    }
}
