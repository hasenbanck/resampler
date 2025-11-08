mod butterfly2;
mod butterfly3;
mod butterfly4;
mod butterfly5;
mod butterfly7;
mod butterfly8;
mod ops;

pub(crate) use butterfly2::butterfly_radix2_dispatch;
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
pub(crate) use butterfly2::{
    butterfly_radix2_generic_avx_fma_dispatch, butterfly_radix2_generic_sse2_dispatch,
    butterfly_radix2_generic_sse4_2_dispatch, butterfly_radix2_stride1_avx_fma_dispatch,
    butterfly_radix2_stride1_sse2_dispatch, butterfly_radix2_stride1_sse4_2_dispatch,
};
pub(crate) use butterfly3::butterfly_radix3_dispatch;
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
pub(crate) use butterfly3::{
    butterfly_radix3_generic_avx_fma_dispatch, butterfly_radix3_generic_sse2_dispatch,
    butterfly_radix3_generic_sse4_2_dispatch, butterfly_radix3_stride1_avx_fma_dispatch,
    butterfly_radix3_stride1_sse2_dispatch, butterfly_radix3_stride1_sse4_2_dispatch,
};
pub(crate) use butterfly4::butterfly_radix4_dispatch;
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
pub(crate) use butterfly4::{
    butterfly_radix4_generic_avx_fma_dispatch, butterfly_radix4_generic_sse2_dispatch,
    butterfly_radix4_generic_sse4_2_dispatch, butterfly_radix4_stride1_avx_fma_dispatch,
    butterfly_radix4_stride1_sse2_dispatch, butterfly_radix4_stride1_sse4_2_dispatch,
};
pub(crate) use butterfly5::butterfly_radix5_dispatch;
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
pub(crate) use butterfly5::{
    butterfly_radix5_generic_avx_fma_dispatch, butterfly_radix5_generic_sse2_dispatch,
    butterfly_radix5_generic_sse4_2_dispatch, butterfly_radix5_stride1_avx_fma_dispatch,
    butterfly_radix5_stride1_sse2_dispatch, butterfly_radix5_stride1_sse4_2_dispatch,
};
pub(crate) use butterfly7::butterfly_radix7_dispatch;
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
pub(crate) use butterfly7::{
    butterfly_radix7_generic_avx_fma_dispatch, butterfly_radix7_generic_sse2_dispatch,
    butterfly_radix7_generic_sse4_2_dispatch, butterfly_radix7_stride1_avx_fma_dispatch,
    butterfly_radix7_stride1_sse2_dispatch, butterfly_radix7_stride1_sse4_2_dispatch,
};
pub(crate) use butterfly8::butterfly_radix8_dispatch;
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
pub(crate) use butterfly8::{
    butterfly_radix8_generic_avx_fma_dispatch, butterfly_radix8_generic_sse2_dispatch,
    butterfly_radix8_generic_sse4_2_dispatch, butterfly_radix8_stride1_avx_fma_dispatch,
    butterfly_radix8_stride1_sse2_dispatch, butterfly_radix8_stride1_sse4_2_dispatch,
};

#[cfg(test)]
mod tests {
    use alloc::{format, vec, vec::Vec};

    use crate::fft::Complex32;

    /// SIMD width for twiddle packing in tests.
    #[derive(Debug, Clone, Copy)]
    #[allow(unused)]
    pub(super) enum TestSimdWidth {
        /// AVX: 4 complex numbers (256-bit)
        Width4,
        /// SSE/NEON: 2 complex numbers (128-bit)
        Width2,
    }

    fn approx_eq_complex(a: &Complex32, b: &Complex32, epsilon: f32) -> bool {
        (a.re - b.re).abs() < epsilon && (a.im - b.im).abs() < epsilon
    }

    fn assert_complex_arrays_approx_eq(
        actual: &[Complex32],
        expected: &[Complex32],
        epsilon: f32,
        context: &str,
    ) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{context}: Array lengths differ",
        );

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            // Check for NaN/Infinity mismatches first (these indicate serious bugs).
            let a_finite = a.re.is_finite() && a.im.is_finite();
            let e_finite = e.re.is_finite() && e.im.is_finite();

            if a_finite != e_finite {
                panic!(
                    "{context}: Finite/infinite mismatch at index {i}: actual = ({}, {}), expected = ({}, {})",
                    a.re, a.im, e.re, e.im
                );
            }

            // For finite values, check if the error is suspiciously large.
            // This catches cases where bit corruption produces wildly wrong values.
            if a_finite {
                let re_diff = (a.re - e.re).abs();
                let im_diff = (a.im - e.im).abs();

                // If error is more than 1000x the tolerance, it's likely a serious bug.
                let suspicious_threshold = epsilon * 1000.0;
                if re_diff > suspicious_threshold || im_diff > suspicious_threshold {
                    panic!(
                        "{context}: SUSPICIOUS LARGE ERROR at index {i}: \
                         actual = ({}, {}), expected = ({}, {}), diff = ({re_diff}, {im_diff}) \
                         (exceeds suspicious threshold {suspicious_threshold})",
                        a.re, a.im, e.re, e.im
                    );
                }

                assert!(
                    approx_eq_complex(a, e, epsilon),
                    "{context}: Mismatch at index {i}: actual = ({}, {}), expected = ({}, {}), diff = ({}, {})",
                    a.re,
                    a.im,
                    e.re,
                    e.im,
                    re_diff,
                    im_diff
                );
            }
        }
    }

    /// Generic test helper for butterfly functions.
    /// Tests a SIMD implementation against a scalar reference implementation.
    pub(super) fn test_butterfly_against_scalar<F, G>(
        scalar_fn: F,
        simd_fn: G,
        radix: usize,
        twiddles_per_stride: usize,
        simd_width: TestSimdWidth,
        test_name: &str,
    ) where
        F: Fn(&[Complex32], &mut [Complex32], &[Complex32], usize),
        G: Fn(&[Complex32], &mut [Complex32], &[Complex32], usize),
    {
        let test_configs = vec![
            // Radix-2 and radix-4 configs (powers of 2)
            (1, 16),   // stride=1, samples=16
            (1, 32),   // stride=1, samples=32
            (1, 64),   // stride=1, samples=64
            (2, 16),   // stride=2, samples=16
            (2, 32),   // stride=2, samples=32
            (4, 32),   // stride=4, samples=32
            (4, 64),   // stride=4, samples=64
            (8, 64),   // stride=8, samples=64
            (8, 128),  // stride=8, samples=128
            (16, 128), // stride=16, samples=128
            // Radix-3 configs (divisible by 3, samples/3 >= 8 for AVX SIMD)
            (1, 24), // stride=1, samples=24
            (1, 30), // stride=1, samples=30
            (1, 36), // stride=1, samples=36
            (1, 48), // stride=1, samples=48
            (1, 60), // stride=1, samples=60
            // Radix-5 configs (divisible by 5, samples/5 >= 8 for AVX SIMD)
            (1, 40), // stride=1, samples=40
            (1, 50), // stride=1, samples=50
            (1, 60), // stride=1, samples=60
            (1, 80), // stride=1, samples=80
            // Radix-7 configs (divisible by 7, samples/7 >= 8 for AVX SIMD)
            (1, 56),  // stride=1, samples=56
            (1, 70),  // stride=1, samples=70
            (1, 84),  // stride=1, samples=84
            (1, 112), // stride=1, samples=112
        ];

        for (stride, samples) in test_configs {
            if samples % radix != 0 {
                continue;
            }

            let mut src = vec![Complex32::zero(); samples];
            for i in 0..samples {
                #[cfg(not(feature = "no_std"))]
                let val = Complex32::new((i as f32 * 0.5).sin(), (i as f32 * 0.3).cos());
                #[cfg(feature = "no_std")]
                let val = Complex32::new(libm::sinf(i as f32 * 0.5), libm::cosf(i as f32 * 0.3));
                src[i] = val;
            }

            let iterations = samples / radix;

            let mut base_twiddles = Vec::with_capacity(stride * twiddles_per_stride);
            for col in 0..stride {
                for k in 0..twiddles_per_stride {
                    let angle = 2.0 * core::f32::consts::PI * (col as f32 * (k + 1) as f32)
                        / (stride as f32 * radix as f32);
                    #[cfg(not(feature = "no_std"))]
                    let tw = Complex32::new(angle.cos(), angle.sin());
                    #[cfg(feature = "no_std")]
                    let tw = Complex32::new(libm::cosf(angle), libm::sinf(angle));
                    base_twiddles.push(tw);
                }
            }

            // Pack twiddles in SIMD-friendly layout based on SIMD width.
            // SIMD iterations use packed format: [w1[i], w1[i+1], ..., w2[i], w2[i+1], ...]
            // Scalar tail uses interleaved format: [w1[i], w2[i], w3[i], ...]
            let mut twiddles = Vec::with_capacity(iterations * twiddles_per_stride);

            match simd_width {
                TestSimdWidth::Width4 => {
                    // AVX layout: pack twiddles for 4 consecutive iterations.
                    let simd_iters = (iterations / 4) * 4;
                    for i in (0..simd_iters).step_by(4) {
                        for tw_idx in 0..twiddles_per_stride {
                            for j in 0..4 {
                                let col = (i + j) % stride;
                                let base_idx = col * twiddles_per_stride + tw_idx;
                                twiddles.push(base_twiddles[base_idx]);
                            }
                        }
                    }
                    // Scalar tail: interleaved format.
                    for i in simd_iters..iterations {
                        let col = i % stride;
                        for k in 0..twiddles_per_stride {
                            twiddles.push(base_twiddles[col * twiddles_per_stride + k]);
                        }
                    }
                }
                TestSimdWidth::Width2 => {
                    // SSE/NEON layout: pack twiddles for 2 consecutive iterations.
                    let simd_iters = (iterations / 2) * 2;
                    for i in (0..simd_iters).step_by(2) {
                        for tw_idx in 0..twiddles_per_stride {
                            for j in 0..2 {
                                let col = (i + j) % stride;
                                let base_idx = col * twiddles_per_stride + tw_idx;
                                twiddles.push(base_twiddles[base_idx]);
                            }
                        }
                    }
                    // Scalar tail: interleaved format.
                    for i in simd_iters..iterations {
                        let col = i % stride;
                        for k in 0..twiddles_per_stride {
                            twiddles.push(base_twiddles[col * twiddles_per_stride + k]);
                        }
                    }
                }
            }

            let mut dst_scalar = vec![Complex32::zero(); samples];
            let mut dst_simd = vec![Complex32::zero(); samples];

            scalar_fn(&src, &mut dst_scalar, &twiddles, stride);
            simd_fn(&src, &mut dst_simd, &twiddles, stride);

            let context = format!("{test_name} with p={stride}, samples={samples}");
            assert_complex_arrays_approx_eq(&dst_simd, &dst_scalar, 1e-6, &context);

            // Additional test with specific bit patterns that would expose mask bugs.
            // These values have bit patterns that would be corrupted by XOR with 0xFFFFFFFF.
            if samples >= radix * 2 {
                let sensitive_values = [
                    1.0_f32,
                    -1.0_f32,
                    2.0_f32,
                    0.5_f32,
                    -0.5_f32,
                    core::f32::consts::PI,
                    core::f32::consts::E,
                    0.1_f32,
                ];

                for (idx, val) in src
                    .iter_mut()
                    .enumerate()
                    .take(sensitive_values.len().min(samples))
                {
                    let sv = sensitive_values[idx % sensitive_values.len()];
                    *val = Complex32::new(sv, -sv);
                }

                let mut dst_scalar2 = vec![Complex32::zero(); samples];
                let mut dst_simd2 = vec![Complex32::zero(); samples];

                scalar_fn(&src, &mut dst_scalar2, &twiddles, stride);
                simd_fn(&src, &mut dst_simd2, &twiddles, stride);

                let context2 =
                    format!("{test_name} (bit-sensitive) with p={stride}, samples={samples}");
                assert_complex_arrays_approx_eq(&dst_simd2, &dst_scalar2, 1e-6, &context2);
            }
        }
    }
}
