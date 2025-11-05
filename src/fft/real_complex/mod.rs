use crate::fft::Complex32;

#[cfg(all(
    target_arch = "x86_64",
    any(
        all(target_feature = "avx", target_feature = "fma"),
        not(feature = "no_std")
    )
))]
mod avx;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    any(
        test,
        not(feature = "no_std"),
        not(all(feature = "no_std", target_feature = "avx", target_feature = "fma"))
    ),
))]
mod sse2;

#[cfg(all(
    target_arch = "aarch64",
    any(not(feature = "no_std"), target_feature = "neon")
))]
mod neon;

/// Scalar implementation of postprocess_fft (fallback).
#[cfg(any(
    test,
    all(
        not(all(target_arch = "x86_64", target_feature = "sse2")),
        not(target_arch = "aarch64")
    )
))]
pub(crate) fn postprocess_fft_scalar(
    output_left_middle: &mut [Complex32],
    output_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    let iter_count = output_left_middle
        .len()
        .min(output_right_middle.len())
        .min(twiddles.len());

    let right_len = output_right_middle.len();
    for i in 0..iter_count {
        let out = output_left_middle[i];
        let out_rev_idx = right_len - 1 - i;
        let out_rev = output_right_middle[out_rev_idx];
        let twiddle = twiddles[i];

        let sum = out.add(&out_rev);
        let diff = out.sub(&out_rev);

        let twiddled_re_sum = Complex32::new(sum.re * twiddle.re, sum.im * twiddle.re);
        let twiddled_im_sum = Complex32::new(sum.re * twiddle.im, sum.im * twiddle.im);
        let twiddled_re_diff = Complex32::new(diff.re * twiddle.re, diff.im * twiddle.re);
        let twiddled_im_diff = Complex32::new(diff.re * twiddle.im, diff.im * twiddle.im);

        let half = 0.5;
        let half_sum_real = half * sum.re;
        let half_diff_imaginary = half * diff.im;

        let real = twiddled_re_sum.im + twiddled_im_diff.re;
        let imaginary = twiddled_im_sum.im - twiddled_re_diff.re;

        output_left_middle[i] =
            Complex32::new(half_sum_real + real, half_diff_imaginary + imaginary);
        output_right_middle[out_rev_idx] =
            Complex32::new(half_sum_real - real, imaginary - half_diff_imaginary);
    }
}

/// Scalar implementation of preprocess_ifft (fallback)
#[cfg(any(
    test,
    all(
        not(all(target_arch = "x86_64", target_feature = "sse2")),
        not(target_arch = "aarch64")
    )
))]
pub(crate) fn preprocess_ifft_scalar(
    input_left_middle: &mut [Complex32],
    input_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    let iter_count = input_left_middle
        .len()
        .min(input_right_middle.len())
        .min(twiddles.len());

    for i in 0..iter_count {
        let inp = input_left_middle[i];
        let inp_rev_idx = input_right_middle.len() - 1 - i;
        let inp_rev = input_right_middle[inp_rev_idx];
        let twiddle = twiddles[i];

        let sum = inp.add(&inp_rev);
        let diff = inp.sub(&inp_rev);

        let twiddled_re_sum = Complex32::new(sum.re * twiddle.re, sum.im * twiddle.re);
        let twiddled_im_sum = Complex32::new(sum.re * twiddle.im, sum.im * twiddle.im);
        let twiddled_re_diff = Complex32::new(diff.re * twiddle.re, diff.im * twiddle.re);
        let twiddled_im_diff = Complex32::new(diff.re * twiddle.im, diff.im * twiddle.im);

        let real = twiddled_re_sum.im + twiddled_im_diff.re;
        let imaginary = twiddled_im_sum.im - twiddled_re_diff.re;

        input_left_middle[i] = Complex32::new(sum.re - real, diff.im - imaginary);
        input_right_middle[inp_rev_idx] = Complex32::new(sum.re + real, -imaginary - diff.im);
    }
}

#[cfg(all(
    target_arch = "x86_64",
    any(
        all(target_feature = "avx", target_feature = "fma"),
        not(feature = "no_std")
    )
))]
pub(crate) fn postprocess_fft_avx_fma_wrapper(
    output_left_middle: &mut [Complex32],
    output_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    unsafe { avx::postprocess_fft_avx_fma(output_left_middle, output_right_middle, twiddles) }
}

#[cfg(all(
    target_arch = "x86_64",
    any(
        all(target_feature = "avx", target_feature = "fma"),
        not(feature = "no_std")
    )
))]
pub(crate) fn preprocess_ifft_avx_fma_wrapper(
    output_left_middle: &mut [Complex32],
    output_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    unsafe { avx::preprocess_ifft_avx_fma(output_left_middle, output_right_middle, twiddles) }
}

#[cfg(all(
    target_arch = "x86_64",
    any(
        test,
        not(feature = "no_std"),
        all(feature = "no_std", not(target_feature = "avx"))
    )
))]
pub(crate) fn postprocess_fft_sse2_wrapper(
    output_left_middle: &mut [Complex32],
    output_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    unsafe { sse2::postprocess_fft_sse2(output_left_middle, output_right_middle, twiddles) }
}

#[cfg(all(
    target_arch = "x86_64",
    any(
        test,
        not(feature = "no_std"),
        all(feature = "no_std", not(target_feature = "avx"))
    )
))]
pub(crate) fn preprocess_ifft_sse2_wrapper(
    output_left_middle: &mut [Complex32],
    output_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    unsafe { sse2::preprocess_ifft_sse2(output_left_middle, output_right_middle, twiddles) }
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn postprocess_fft_neon_wrapper(
    output_left_middle: &mut [Complex32],
    output_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    unsafe { neon::postprocess_fft_neon(output_left_middle, output_right_middle, twiddles) }
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn preprocess_ifft_neon_wrapper(
    output_left_middle: &mut [Complex32],
    output_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    unsafe { neon::preprocess_ifft_neon(output_left_middle, output_right_middle, twiddles) }
}

#[cfg(any(not(target_arch = "x86_64"), feature = "no_std"))]
pub(crate) fn select_postprocess_fn() -> fn(&mut [Complex32], &mut [Complex32], &[Complex32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        postprocess_fft_avx_fma_wrapper
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        not(target_feature = "fma")
    ))]
    {
        postprocess_fft_avx_wrapper
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx")))]
    {
        postprocess_fft_sse2_wrapper
    }

    #[cfg(target_arch = "aarch64")]
    {
        postprocess_fft_neon_wrapper
    }

    #[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
    postprocess_fft_scalar
}

#[cfg(any(not(target_arch = "x86_64"), feature = "no_std"))]
pub(crate) fn select_preprocess_fn() -> fn(&mut [Complex32], &mut [Complex32], &[Complex32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        preprocess_ifft_avx_fma_wrapper
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx")))]
    {
        preprocess_ifft_sse2_wrapper
    }

    #[cfg(target_arch = "aarch64")]
    {
        preprocess_ifft_neon_wrapper
    }

    #[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
    preprocess_ifft_scalar
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec};
    use core::f32;

    use super::*;

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
            "{context}: Array lengths differ"
        );

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_complex(a, e, epsilon),
                "{context}: Mismatch at index {i}: actual = ({}, {}), expected = ({}, {}), diff = ({}, {})",
                a.re,
                a.im,
                e.re,
                e.im,
                (a.re - e.re).abs(),
                (a.im - e.im).abs()
            );
        }
    }

    fn test_postprocess_against_scalar(
        scalar_fn: fn(&mut [Complex32], &mut [Complex32], &[Complex32]),
        simd_fn: fn(&mut [Complex32], &mut [Complex32], &[Complex32]),
        test_name: &str,
    ) {
        let test_sizes = vec![
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128,
        ];

        for middle_size in test_sizes {
            // Create test data
            let mut scalar_left = vec![Complex32::zero(); middle_size];
            let mut scalar_right = vec![Complex32::zero(); middle_size];
            let mut simd_left = vec![Complex32::zero(); middle_size];
            let mut simd_right = vec![Complex32::zero(); middle_size];

            for i in 0..middle_size {
                #[cfg(not(feature = "no_std"))]
                let val_left = Complex32::new((i as f32 * 0.5).sin(), (i as f32 * 0.3).cos());
                #[cfg(feature = "no_std")]
                let val_left =
                    Complex32::new(libm::sinf(i as f32 * 0.5), libm::cosf(i as f32 * 0.3));

                #[cfg(not(feature = "no_std"))]
                let val_right = Complex32::new((i as f32 * 0.7).cos(), (i as f32 * 0.4).sin());
                #[cfg(feature = "no_std")]
                let val_right =
                    Complex32::new(libm::cosf(i as f32 * 0.7), libm::sinf(i as f32 * 0.4));

                scalar_left[i] = val_left;
                scalar_right[i] = val_right;
                simd_left[i] = val_left;
                simd_right[i] = val_right;
            }

            // Create twiddle factors
            let mut twiddles = vec![Complex32::zero(); middle_size];
            for i in 0..middle_size {
                let angle = 2.0 * f32::consts::PI * (i as f32) / ((middle_size + 1) as f32);
                #[cfg(not(feature = "no_std"))]
                let tw = Complex32::new(angle.cos(), angle.sin());
                #[cfg(feature = "no_std")]
                let tw = Complex32::new(libm::cosf(angle), libm::sinf(angle));
                twiddles[i] = tw;
            }

            scalar_fn(&mut scalar_left, &mut scalar_right, &twiddles);
            simd_fn(&mut simd_left, &mut simd_right, &twiddles);

            let context_left = format!("{test_name} - left channel with size {middle_size}");
            let context_right = format!("{test_name} - right channel with size {middle_size}");

            assert_complex_arrays_approx_eq(&simd_left, &scalar_left, 1e-6, &context_left);
            assert_complex_arrays_approx_eq(&simd_right, &scalar_right, 1e-6, &context_right);
        }
    }

    fn test_preprocess_against_scalar(
        scalar_fn: fn(&mut [Complex32], &mut [Complex32], &[Complex32]),
        simd_fn: fn(&mut [Complex32], &mut [Complex32], &[Complex32]),
        test_name: &str,
    ) {
        let test_sizes = vec![
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128,
        ];

        for middle_size in test_sizes {
            // Create test data
            let mut scalar_left = vec![Complex32::zero(); middle_size];
            let mut scalar_right = vec![Complex32::zero(); middle_size];
            let mut simd_left = vec![Complex32::zero(); middle_size];
            let mut simd_right = vec![Complex32::zero(); middle_size];

            for i in 0..middle_size {
                #[cfg(not(feature = "no_std"))]
                let val_left = Complex32::new((i as f32 * 0.5).sin(), (i as f32 * 0.3).cos());
                #[cfg(feature = "no_std")]
                let val_left =
                    Complex32::new(libm::sinf(i as f32 * 0.5), libm::cosf(i as f32 * 0.3));

                #[cfg(not(feature = "no_std"))]
                let val_right = Complex32::new((i as f32 * 0.7).cos(), (i as f32 * 0.4).sin());
                #[cfg(feature = "no_std")]
                let val_right =
                    Complex32::new(libm::cosf(i as f32 * 0.7), libm::sinf(i as f32 * 0.4));

                scalar_left[i] = val_left;
                scalar_right[i] = val_right;
                simd_left[i] = val_left;
                simd_right[i] = val_right;
            }

            let mut twiddles = vec![Complex32::zero(); middle_size];
            for i in 0..middle_size {
                let angle = 2.0 * f32::consts::PI * (i as f32) / ((middle_size + 1) as f32);
                #[cfg(not(feature = "no_std"))]
                let tw = Complex32::new(angle.cos(), angle.sin());
                #[cfg(feature = "no_std")]
                let tw = Complex32::new(libm::cosf(angle), libm::sinf(angle));
                twiddles[i] = tw;
            }

            scalar_fn(&mut scalar_left, &mut scalar_right, &twiddles);
            simd_fn(&mut simd_left, &mut simd_right, &twiddles);

            let context_left = format!("{test_name} - left channel with size {middle_size}");
            let context_right = format!("{test_name} - right channel with size {middle_size}");

            assert_complex_arrays_approx_eq(&simd_left, &scalar_left, 1e-6, &context_left);
            assert_complex_arrays_approx_eq(&simd_right, &scalar_right, 1e-6, &context_right);
        }
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "avx")
    ))]
    fn test_preprocess_ifft_avx_fma_vs_scalar() {
        test_preprocess_against_scalar(
            preprocess_ifft_scalar,
            preprocess_ifft_avx_fma_wrapper,
            "preprocess_ifft AVX FMA",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(
            not(feature = "no_std"),
            all(target_feature = "avx", target_feature = "fma")
        )
    ))]
    fn test_postprocess_fft_avx_fma_vs_scalar() {
        test_postprocess_against_scalar(
            postprocess_fft_scalar,
            postprocess_fft_avx_fma_wrapper,
            "postprocess_fft AVX FMA",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "sse2")
    ))]
    fn test_postprocess_fft_sse2_vs_scalar() {
        test_postprocess_against_scalar(
            postprocess_fft_scalar,
            postprocess_fft_sse2_wrapper,
            "postprocess_fft SSE2",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "sse2")
    ))]
    fn test_preprocess_ifft_sse2_vs_scalar() {
        test_preprocess_against_scalar(
            preprocess_ifft_scalar,
            preprocess_ifft_sse2_wrapper,
            "preprocess_ifft SSE2",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_postprocess_fft_neon_vs_scalar() {
        test_postprocess_against_scalar(
            postprocess_fft_scalar,
            postprocess_fft_neon_wrapper,
            "postprocess_fft NEON",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_preprocess_ifft_neon_vs_scalar() {
        test_preprocess_against_scalar(
            preprocess_ifft_scalar,
            preprocess_ifft_neon_wrapper,
            "preprocess_ifft NEON",
        );
    }
}
