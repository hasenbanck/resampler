#[cfg(all(
    target_arch = "x86_64",
    any(
        all(
            target_feature = "avx",
            target_feature = "fma",
            any(not(target_feature = "avx512f"), test)
        ),
        not(feature = "no_std")
    )
))]
pub(crate) mod avx;

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "avx512f", not(feature = "no_std"))
))]
pub(crate) mod avx512;

#[cfg(all(
    target_arch = "x86_64",
    any(
        all(target_feature = "sse2", any(not(target_feature = "sse4.2"), test)),
        not(feature = "no_std")
    )
))]
pub(crate) mod sse2;

#[cfg(all(
    target_arch = "x86_64",
    any(
        all(
            target_feature = "sse4.2",
            any(not(all(target_feature = "avx", target_feature = "fma")), test),
        ),
        not(feature = "no_std")
    )
))]
pub(crate) mod sse4_2;

#[cfg(target_arch = "aarch64")]
mod neon;

/// Scalar implementation of dual-phase FIR convolution with interpolation.
#[inline(always)]
#[cfg(any(test, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
pub(crate) fn convolve_interp_scalar(
    input: &[f32],
    coeffs1: &[f32],
    coeffs2: &[f32],
    frac: f32,
    taps: usize,
) -> f32 {
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    for i in 0..taps {
        let input_val = input[i];
        sum1 += coeffs1[i] * input_val;
        sum2 += coeffs2[i] * input_val;
    }
    sum1 * (1.0 - frac) + sum2 * frac
}

/// Dispatch function for dual-phase interpolated FIR convolution with compile-time SIMD selection.
///
/// Automatically selects the best available implementation based on compile-time target features.
#[cfg(any(
    all(target_arch = "x86_64", feature = "no_std"),
    not(target_arch = "x86_64")
))]
pub(crate) fn convolve_interp(
    input: &[f32],
    coeffs1: &[f32],
    coeffs2: &[f32],
    frac: f32,
    taps: usize,
) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        // Safety: We've checked that avx512f feature is enabled at compile time.
        unsafe { avx512::convolve_interp_avx512(input, coeffs1, coeffs2, frac, taps) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        target_feature = "fma",
        not(target_feature = "avx512f")
    ))]
    {
        // Safety: We've checked that avx and fma features are enabled at compile time.
        unsafe { avx::convolve_interp_avx_fma(input, coeffs1, coeffs2, frac, taps) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse4.2",
        not(target_feature = "avx"),
    ))]
    {
        // Safety: We've checked that sse4.2 feature is enabled at compile time.
        unsafe { sse4_2::convolve_interp_sse4_2(input, coeffs1, coeffs2, frac, taps) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse4.2"),
    ))]
    {
        // Safety: We've checked that sse2 feature is enabled at compile time.
        unsafe { sse2::convolve_interp_sse2(input, coeffs1, coeffs2, frac, taps) }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is mandatory on aarch64, so it's always available.
        unsafe { neon::convolve_interp_neon(input, coeffs1, coeffs2, frac, taps) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        target_arch = "aarch64"
    )))]
    convolve_interp_scalar(input, coeffs1, coeffs2, frac, taps)
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::*;
    use crate::{Latency, resampler_fir::AlignedMemory};

    const EPSILON: f32 = 1e-5;

    fn test_convolve_interp_against_scalar<F>(simd_fn: F, test_name: &str)
    where
        F: Fn(&[f32], &[f32], &[f32], f32, usize) -> f32,
    {
        let tap_counts = [
            Latency::Sample8.taps(),
            Latency::Sample16.taps(),
            Latency::Sample32.taps(),
            Latency::Sample64.taps(),
        ];

        let fractions = [0.0, 0.25, 0.5, 0.75, 1.0];

        for &taps in &tap_counts {
            for &frac in &fractions {
                let input: Vec<f32> = (0..taps).map(|i| (i as f32 * 0.1).sin()).collect();

                let coeffs1_vec: Vec<f32> = (0..taps).map(|i| (i as f32 * 0.2).cos()).collect();
                let coeffs2_vec: Vec<f32> = (0..taps).map(|i| (i as f32 * 0.15).cos()).collect();

                let coeffs1_aligned = AlignedMemory::new(coeffs1_vec.clone());
                let coeffs2_aligned = AlignedMemory::new(coeffs2_vec.clone());

                let scalar_result =
                    convolve_interp_scalar(&input, &coeffs1_aligned, &coeffs2_aligned, frac, taps);

                let simd_result = simd_fn(&input, &coeffs1_aligned, &coeffs2_aligned, frac, taps);

                let diff = (scalar_result - simd_result).abs();
                assert!(
                    diff < EPSILON,
                    "{test_name} failed for taps={taps}, frac={frac}: scalar={scalar_result}, simd={simd_result}, diff={diff}"
                );
            }
        }

        let taps = 64;
        let mut impulse = vec![0.0f32; taps];
        impulse[0] = 1.0;

        let coeffs1_vec: Vec<f32> = (0..taps).map(|i| i as f32).collect();
        let coeffs2_vec: Vec<f32> = (0..taps).map(|i| (i * 2) as f32).collect();
        let coeffs1_aligned = AlignedMemory::new(coeffs1_vec);
        let coeffs2_aligned = AlignedMemory::new(coeffs2_vec);

        let frac = 0.3;
        let scalar_result =
            convolve_interp_scalar(&impulse, &coeffs1_aligned, &coeffs2_aligned, frac, taps);
        let simd_result = simd_fn(&impulse, &coeffs1_aligned, &coeffs2_aligned, frac, taps);

        let diff = (scalar_result - simd_result).abs();
        assert!(
            diff < EPSILON,
            "{test_name} failed for impulse test: scalar={scalar_result}, simd={simd_result}, diff={diff}"
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    fn test_convolve_interp_sse2() {
        test_convolve_interp_against_scalar(
            |input, coeffs1, coeffs2, frac, taps| unsafe {
                sse2::convolve_interp_sse2(input, coeffs1, coeffs2, frac, taps)
            },
            "SSE2",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn test_convolve_interp_sse4_2() {
        test_convolve_interp_against_scalar(
            |input, coeffs1, coeffs2, frac, taps| unsafe {
                sse4_2::convolve_interp_sse4_2(input, coeffs1, coeffs2, frac, taps)
            },
            "SSE4.2",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    fn test_convolve_interp_avx_fma() {
        test_convolve_interp_against_scalar(
            |input, coeffs1, coeffs2, frac, taps| unsafe {
                avx::convolve_interp_avx_fma(input, coeffs1, coeffs2, frac, taps)
            },
            "AVX+FMA",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    fn test_convolve_interp_avx512() {
        test_convolve_interp_against_scalar(
            |input, coeffs1, coeffs2, frac, taps| unsafe {
                avx512::convolve_interp_avx512(input, coeffs1, coeffs2, frac, taps)
            },
            "AVX512",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_convolve_interp_neon() {
        test_convolve_interp_against_scalar(
            |input, coeffs1, coeffs2, frac, taps| unsafe {
                neon::convolve_interp_neon(input, coeffs1, coeffs2, frac, taps)
            },
            "NEON",
        );
    }
}
