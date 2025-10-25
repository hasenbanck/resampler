use crate::{LatencyMode, Radix, SampleRate, SampleRateFamily};

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
/// | Conversion Type              | Input Size | Output Size | Ratio Error | Input FFT           | Output FFT        | Factorization      |
/// |------------------------------|------------|-------------|-------------|---------------------|-------------------|--------------------|
/// | Inside same family           | 2          | 2           | 0.0%        | Radix-2             | Radix-2           | 2 → 2              |
/// | Between 22.05 kHz and 48 kHz | 16         | 35          | 0.4883%     | Radix-2             | Mixed-Radix (5,7) | 2⁴ → 5 × 7         |
/// | Between 16 kHz and 48 kHz    | 64         | 192         | 0.0%        | Radix-2             | Mixed-Radix (2,3) | 2⁶ → 2⁶ × 3        |
/// | Between 16 kHz and 44.1 kHz  | 70         | 192         | 0.4859%     | Mixed-Radix (2,5,7) | Mixed-Radix (2,3) | 2 × 5 × 7 → 2⁶ × 3 |
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

        let input_multiplier = input_rate.family_multiplier();
        let output_multiplier = output_rate.family_multiplier();

        let base_config = match (input_family, output_family) {
            // Same family: base 1:1 ratio (2 → 2)
            (SampleRateFamily::_48000, SampleRateFamily::_48000)
            | (SampleRateFamily::_22050, SampleRateFamily::_22050)
            | (SampleRateFamily::_16000, SampleRateFamily::_16000) => ConversionConfig {
                base_fft_size_in: 2,
                base_fft_size_out: 2,
                base_factors_in: vec![Radix::Factor2],
                base_factors_out: vec![Radix::Factor2],
            },

            // 22.05 kHz → 48 kHz family (2⁴ → 5 × 7)
            (SampleRateFamily::_22050, SampleRateFamily::_48000) => ConversionConfig {
                base_fft_size_in: 16,
                base_fft_size_out: 35,
                base_factors_in: vec![Radix::Factor2; 4],
                base_factors_out: vec![Radix::Factor7, Radix::Factor5],
            },
            // 48 kHz → 22.05 kHz family (5 × 7 → 2⁴)
            (SampleRateFamily::_48000, SampleRateFamily::_22050) => ConversionConfig {
                base_fft_size_in: 35,
                base_fft_size_out: 16,
                base_factors_in: vec![Radix::Factor7, Radix::Factor5],
                base_factors_out: vec![Radix::Factor2; 4],
            },

            // 16 kHz → 48 kHz family (2⁶ → 2⁶ × 3)
            (SampleRateFamily::_16000, SampleRateFamily::_48000) => ConversionConfig {
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
            (SampleRateFamily::_48000, SampleRateFamily::_16000) => ConversionConfig {
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

            // 16 kHz → 22.05 kHz family (2 × 5 × 7 → 2⁶ × 3)
            (SampleRateFamily::_16000, SampleRateFamily::_22050) => ConversionConfig {
                base_fft_size_in: 70,
                base_fft_size_out: 192,
                base_factors_in: vec![Radix::Factor7, Radix::Factor5, Radix::Factor2],
                base_factors_out: vec![
                    Radix::Factor3,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor4,
                ],
            },
            // 22.05 kHz → 16 kHz family (2⁶ × 3 → 2 × 5 × 7)
            (SampleRateFamily::_22050, SampleRateFamily::_16000) => ConversionConfig {
                base_fft_size_in: 192,
                base_fft_size_out: 70,
                base_factors_in: vec![
                    Radix::Factor3,
                    Radix::Factor4,
                    Radix::Factor4,
                    Radix::Factor4,
                ],
                base_factors_out: vec![Radix::Factor7, Radix::Factor5, Radix::Factor2],
            },
        };

        // Apply family multipliers to scale FFT sizes and factors.
        let scaled_fft_size_in = base_config.base_fft_size_in * input_multiplier;
        let scaled_fft_size_out = base_config.base_fft_size_out * output_multiplier;

        // Decompose multipliers into radix factors.
        let prefer_factor4_input = base_config
            .base_factors_in
            .iter()
            .any(|factor| factor != &Radix::Factor2);
        let prefer_factor4_output = base_config
            .base_factors_out
            .iter()
            .any(|factor| factor != &Radix::Factor2);

        let input_multiplier_factors =
            Self::decompose_multiplier(input_multiplier, prefer_factor4_input);
        let output_multiplier_factors =
            Self::decompose_multiplier(output_multiplier, prefer_factor4_output);

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
    fn decompose_multiplier(multiplier: usize, prefer_factor4: bool) -> Vec<Radix> {
        if multiplier == 1 {
            return Vec::new();
        }

        assert!(multiplier.is_power_of_two());

        let num_bits = multiplier.trailing_zeros() as usize;

        if prefer_factor4 {
            // Prefer Factor4 for efficiency: decompose into 4s and 2s.
            let num_factor4 = num_bits / 2;
            let num_factor2 = num_bits % 2;

            let mut factors = vec![Radix::Factor4; num_factor4];
            if num_factor2 > 0 {
                factors.push(Radix::Factor2);
            }
            factors
        } else {
            // Pure Factor2 for power-of-2 bases (Cooley-Tukey fast path).
            vec![Radix::Factor2; num_bits]
        }
    }

    /// Scale the base FFT sizes according to the latency mode.
    ///
    /// Returns: (input_size, input_factors, output_size, output_factors)
    pub(crate) fn scale_for_latency(
        &self,
        latency_mode: LatencyMode,
    ) -> (usize, Vec<Radix>, usize, Vec<Radix>) {
        let target_input_samples = match latency_mode {
            LatencyMode::LatencyOptimized => 256,
            LatencyMode::ThroughputOptimized => 512,
        };

        // Calculate the multiplier needed to reach target input samples.
        let multiplier = (target_input_samples as f32 / self.base_fft_size_in as f32)
            .ceil()
            .max(1.0) as usize;

        // Round multiplier to nearest power of 2.
        let multiplier = multiplier.next_power_of_two();
        let scaled_fft_size_in = self.base_fft_size_in * multiplier;
        let scaled_fft_size_out = self.base_fft_size_out * multiplier;

        // Determine if base sizes are power of 2.
        let base_in_is_power_of_2 = self.base_fft_size_in.is_power_of_two();
        let base_out_is_power_of_2 = self.base_fft_size_out.is_power_of_two();

        // Decompose multiplier into factors:
        // - For power-of-2 bases: use only Factor2 to preserve Cooley-Tukey optimization.
        // - For mixed-radix bases: prefer Factor4 for efficiency.
        let scaling_factors_in = Self::decompose_multiplier(multiplier, !base_in_is_power_of_2);
        let scaling_factors_out = Self::decompose_multiplier(multiplier, !base_out_is_power_of_2);

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
        let config = ConversionConfig::from_sample_rates(SampleRate::_48000, SampleRate::_96000);
        assert_eq!(config.base_fft_size_in, 2);
        assert_eq!(config.base_fft_size_out, 4);
    }

    #[test]
    fn test_conversion_config_48000_to_192000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::_48000, SampleRate::_192000);
        assert_eq!(config.base_fft_size_in, 2);
        assert_eq!(config.base_fft_size_out, 8);
    }

    #[test]
    fn test_conversion_config_22050_to_48000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::_22050, SampleRate::_48000);
        assert_eq!(config.base_fft_size_in, 16);
        assert_eq!(config.base_fft_size_out, 35);
    }

    #[test]
    fn test_conversion_config_16000_to_48000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::_16000, SampleRate::_48000);
        assert_eq!(config.base_fft_size_in, 64);
        assert_eq!(config.base_fft_size_out, 192);
    }

    #[test]
    fn test_conversion_config_16000_to_44100() {
        let config = ConversionConfig::from_sample_rates(SampleRate::_16000, SampleRate::_44100);
        assert_eq!(config.base_fft_size_in, 70);
        assert_eq!(config.base_fft_size_out, 384);
    }

    #[test]
    fn test_conversion_config_44100_to_48000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::_44100, SampleRate::_48000);
        assert_eq!(config.base_fft_size_in, 32);
        assert_eq!(config.base_fft_size_out, 35);

        assert_eq!(config.base_factors_in, vec![Radix::Factor2; 5]);

        assert_eq!(
            config.base_factors_out,
            vec![Radix::Factor7, Radix::Factor5]
        );
    }

    #[test]
    fn test_conversion_config_44100_to_96000() {
        let config = ConversionConfig::from_sample_rates(SampleRate::_44100, SampleRate::_96000);
        assert_eq!(config.base_fft_size_in, 32);
        assert_eq!(config.base_fft_size_out, 70);

        assert_eq!(config.base_factors_in, vec![Radix::Factor2; 5]);

        // Output: base [Factor7, Factor5] + multiplier [Factor2] = [Factor7, Factor5, Factor2]
        assert_eq!(
            config.base_factors_out,
            vec![Radix::Factor7, Radix::Factor5, Radix::Factor2]
        );
    }

    #[test]
    fn test_prefer_factor4_for_mixed_radix() {
        // Test that prefer_factor4 logic works correctly:
        // - Pure Factor2 bases use Factor2 for multipliers
        // - Mixed-radix bases use Factor4 for multipliers (when possible)

        let config = ConversionConfig {
            base_fft_size_in: 16,
            base_fft_size_out: 35,
            base_factors_in: vec![Radix::Factor2; 4],
            base_factors_out: vec![Radix::Factor7, Radix::Factor5],
        };

        let (_, factors_in, _, factors_out) =
            config.scale_for_latency(LatencyMode::LatencyOptimized);

        assert_eq!(factors_in[0..2], [Radix::Factor2, Radix::Factor2]);

        // Output is mixed-radix base → should use Factor4 for multiplier
        // Base [Factor7, Factor5] + multiplier [Factor4] = [Factor7, Factor5, Factor4]
        assert_eq!(factors_out[0], Radix::Factor7);
        assert_eq!(factors_out[1], Radix::Factor5);
        assert_eq!(factors_out[2], Radix::Factor4);
    }

    #[test]
    fn test_latency_scaling() {
        let config = ConversionConfig {
            base_fft_size_in: 16,
            base_fft_size_out: 35,
            base_factors_in: vec![Radix::Factor2; 4],
            base_factors_out: vec![Radix::Factor7, Radix::Factor5],
        };

        // LatencyOptimized: target ~256 samples -> 16x multiplier
        let (input, factors_in, output, factors_out) =
            config.scale_for_latency(LatencyMode::LatencyOptimized);
        assert_eq!(input, 256);
        assert_eq!(output, 560);

        // 256 = 16 × 16, multiplier is 16 = 2^4

        // Base is power-of-2 (16), so use only Factor2: [Factor2; 4] + [Factor2; 4]
        assert_eq!(
            factors_in,
            vec![
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2
            ]
        );

        // 560 = 16 × 35, multiplier is 16, output is not power-of-2, so use Factor4: 16 = 4×4
        // Base [Factor7, Factor5] + multiplier [Factor4, Factor4] = [Factor7, Factor5, Factor4, Factor4]
        assert_eq!(
            factors_out,
            vec![
                Radix::Factor7,
                Radix::Factor5,
                Radix::Factor4,
                Radix::Factor4
            ]
        );

        // ThroughputOptimized: target ~1024 samples -> 64x multiplier
        let (input, factors_in, output, factors_out) =
            config.scale_for_latency(LatencyMode::ThroughputOptimized);
        assert_eq!(input, 1024);
        assert_eq!(output, 2240);

        // 1024 = 64 × 16, multiplier is 64 = 2^6

        // Base is power-of-2 (16), so use only Factor2: [Factor2; 6] + [Factor2; 4]
        assert_eq!(
            factors_in,
            vec![
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2
            ]
        );

        // Output is not power-of-2, so use Factor4: 64 = 4×4×4
        // Base [Factor7, Factor5] + multiplier [Factor4, Factor4, Factor4] = [Factor7, Factor5, Factor4, Factor4, Factor4]
        assert_eq!(
            factors_out,
            vec![
                Radix::Factor7,
                Radix::Factor5,
                Radix::Factor4,
                Radix::Factor4,
                Radix::Factor4
            ]
        );
    }
}
