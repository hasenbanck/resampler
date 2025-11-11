use crate::fft::Complex32;

#[cfg(all(
    target_arch = "x86_64",
    any(
        not(feature = "no_std"),
        all(target_feature = "avx", target_feature = "fma")
    )
))]
mod avx;

#[cfg(all(
    target_arch = "x86_64",
    any(
        test,
        not(feature = "no_std"),
        all(
            feature = "no_std",
            target_feature = "sse2",
            not(target_feature = "sse4.2"),
            not(all(target_feature = "avx", target_feature = "fma"))
        )
    ),
))]
mod sse2;

#[cfg(all(
    target_arch = "x86_64",
    any(
        all(test, target_feature = "sse4.2"),
        not(feature = "no_std"),
        all(
            feature = "no_std",
            target_feature = "sse4.2",
            not(all(target_feature = "avx", target_feature = "fma"))
        )
    ),
))]
mod sse4_2;

#[cfg(all(
    target_arch = "aarch64",
    any(not(feature = "no_std"), target_feature = "neon")
))]
mod neon;

const SQRT3_2: f32 = 0.8660254; // sqrt(3)/2

/// Dispatch function for radix-3 Stockham butterfly.
#[inline(always)]
pub(crate) fn butterfly_radix3_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    #[allow(unused)]
    let samples = src.len();
    #[allow(unused)]
    let third_samples = samples / 3;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    unsafe {
        if third_samples >= 4 {
            return match stride {
                1 => avx::butterfly_radix3_stride1_avx_fma(src, dst, stage_twiddles),
                _ => avx::butterfly_radix3_generic_avx_fma(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse4.2",
        not(all(target_feature = "avx", target_feature = "fma"))
    ))]
    unsafe {
        if third_samples >= 2 {
            return match stride {
                1 => sse4_2::butterfly_radix3_stride1_sse4_2(src, dst, stage_twiddles),
                _ => sse4_2::butterfly_radix3_generic_sse4_2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse4.2")
    ))]
    unsafe {
        if third_samples >= 2 {
            return match stride {
                1 => sse2::butterfly_radix3_stride1_sse2(src, dst, stage_twiddles),
                _ => sse2::butterfly_radix3_generic_sse2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        if third_samples >= 2 {
            return match stride {
                1 => neon::butterfly_radix3_stride1_neon(src, dst, stage_twiddles),
                _ => neon::butterfly_radix3_generic_neon(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    butterfly_radix3_scalar::<4>(src, dst, stage_twiddles, stride, 0);
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
    butterfly_radix3_scalar::<2>(src, dst, stage_twiddles, stride, 0);
}

/// AVX+FMA dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix3_stride1_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let third_samples = samples / 3;

    if third_samples >= 4 {
        return unsafe { avx::butterfly_radix3_stride1_avx_fma(src, dst, stage_twiddles) };
    }

    butterfly_radix3_scalar::<4>(src, dst, stage_twiddles, 1, 0);
}

/// AVX+FMA dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix3_generic_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let third_samples = samples / 3;

    if third_samples >= 4 {
        return unsafe { avx::butterfly_radix3_generic_avx_fma(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix3_scalar::<4>(src, dst, stage_twiddles, stride, 0);
}

/// SSE2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix3_stride1_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let third_samples = samples / 3;

    if third_samples >= 2 {
        return unsafe { sse2::butterfly_radix3_stride1_sse2(src, dst, stage_twiddles) };
    }

    butterfly_radix3_scalar::<2>(src, dst, stage_twiddles, 1, 0);
}

/// SSE2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix3_generic_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let third_samples = samples / 3;

    if third_samples >= 2 {
        return unsafe { sse2::butterfly_radix3_generic_sse2(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix3_scalar::<2>(src, dst, stage_twiddles, stride, 0);
}

/// SSE4.2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix3_stride1_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let third_samples = samples / 3;

    if third_samples >= 2 {
        return unsafe { sse4_2::butterfly_radix3_stride1_sse4_2(src, dst, stage_twiddles) };
    }

    butterfly_radix3_scalar::<2>(src, dst, stage_twiddles, 1, 0);
}

/// SSE4.2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix3_generic_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let third_samples = samples / 3;

    if third_samples >= 2 {
        return unsafe {
            sse4_2::butterfly_radix3_generic_sse4_2(src, dst, stage_twiddles, stride)
        };
    }

    butterfly_radix3_scalar::<2>(src, dst, stage_twiddles, stride, 0);
}

/// Performs a single radix-3 Stockham butterfly stage (out-of-place, scalar).
#[inline(always)]
fn butterfly_radix3_scalar<const WIDTH: usize>(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
    start_index: usize,
) {
    let samples = src.len();
    let third_samples = samples / 3;
    let simd_iters = (third_samples / WIDTH) * WIDTH;

    // Stride=1 optimization: skip identity twiddle multiplications.
    if stride == 1 {
        for i in start_index..simd_iters {
            let z0 = src[i];
            let z1 = src[i + third_samples];
            let z2 = src[i + third_samples * 2];

            // Identity twiddles: t1 = (1+0i) * z1 = z1, t2 = (1+0i) * z2 = z2
            let sum_t = z1.add(&z2);
            let diff_t = z1.sub(&z2);

            let j = 3 * i;
            dst[j] = z0.add(&sum_t);

            let re_part = z0.re - 0.5 * sum_t.re;
            let im_part = z0.im - 0.5 * sum_t.im;
            let sqrt3_diff_re = SQRT3_2 * diff_t.im;
            let sqrt3_diff_im = -SQRT3_2 * diff_t.re;

            dst[j + 1] = Complex32::new(re_part + sqrt3_diff_re, im_part + sqrt3_diff_im);
            dst[j + 2] = Complex32::new(re_part - sqrt3_diff_re, im_part - sqrt3_diff_im);
        }

        // Process scalar tail.
        for i in simd_iters..third_samples {
            let z0 = src[i];
            let z1 = src[i + third_samples];
            let z2 = src[i + third_samples * 2];

            // Identity twiddles: t1 = z1, t2 = z2
            let sum_t = z1.add(&z2);
            let diff_t = z1.sub(&z2);

            let j = 3 * i;
            dst[j] = z0.add(&sum_t);

            let re_part = z0.re - 0.5 * sum_t.re;
            let im_part = z0.im - 0.5 * sum_t.im;
            let sqrt3_diff_re = SQRT3_2 * diff_t.im;
            let sqrt3_diff_im = -SQRT3_2 * diff_t.re;

            dst[j + 1] = Complex32::new(re_part + sqrt3_diff_re, im_part + sqrt3_diff_im);
            dst[j + 2] = Complex32::new(re_part - sqrt3_diff_re, im_part - sqrt3_diff_im);
        }
        return;
    }

    // Process SIMD-packed region.
    for i in start_index..simd_iters {
        let k = i % stride;
        let group_idx = i / WIDTH;
        let offset_in_group = i % WIDTH;
        let tw_base = group_idx * (2 * WIDTH) + offset_in_group;
        let w1 = stage_twiddles[tw_base];
        let w2 = stage_twiddles[tw_base + WIDTH];

        let z0 = src[i];
        let z1 = src[i + third_samples];
        let z2 = src[i + third_samples * 2];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);

        let sum_t = t1.add(&t2);
        let diff_t = t1.sub(&t2);

        let j = 3 * i - 2 * k;
        dst[j] = z0.add(&sum_t);

        let re_part = z0.re - 0.5 * sum_t.re;
        let im_part = z0.im - 0.5 * sum_t.im;
        let sqrt3_diff_re = SQRT3_2 * diff_t.im;
        let sqrt3_diff_im = -SQRT3_2 * diff_t.re;

        dst[j + stride] = Complex32::new(re_part + sqrt3_diff_re, im_part + sqrt3_diff_im);
        dst[j + stride * 2] = Complex32::new(re_part - sqrt3_diff_re, im_part - sqrt3_diff_im);
    }

    // Process scalar tail.
    let tail_offset = (simd_iters / WIDTH) * (2 * WIDTH);
    for i in simd_iters..third_samples {
        let k = i % stride;
        let tail_idx = i - simd_iters;
        let w1 = stage_twiddles[tail_offset + tail_idx * 2];
        let w2 = stage_twiddles[tail_offset + tail_idx * 2 + 1];

        let z0 = src[i];
        let z1 = src[i + third_samples];
        let z2 = src[i + third_samples * 2];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);

        let sum_t = t1.add(&t2);
        let diff_t = t1.sub(&t2);

        let j = 3 * i - 2 * k;
        dst[j] = z0.add(&sum_t);

        let re_part = z0.re - 0.5 * sum_t.re;
        let im_part = z0.im - 0.5 * sum_t.im;
        let sqrt3_diff_re = SQRT3_2 * diff_t.im;
        let sqrt3_diff_im = -SQRT3_2 * diff_t.re;

        dst[j + stride] = Complex32::new(re_part + sqrt3_diff_re, im_part + sqrt3_diff_im);
        dst[j + stride * 2] = Complex32::new(re_part - sqrt3_diff_re, im_part - sqrt3_diff_im);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(
            not(feature = "no_std"),
            all(target_feature = "avx", target_feature = "fma")
        )
    ))]
    fn test_butterfly_radix3_avx_fma_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix3_scalar::<4>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    avx::butterfly_radix3_stride1_avx_fma(src, dst, twiddles);
                } else {
                    avx::butterfly_radix3_generic_avx_fma(src, dst, twiddles, p);
                }
            },
            3,
            2,
            TestSimdWidth::Width4,
            "butterfly_radix3_avx_fma",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    fn test_butterfly_radix3_sse2_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix3_scalar::<2>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse2::butterfly_radix3_stride1_sse2(src, dst, twiddles);
                } else {
                    sse2::butterfly_radix3_generic_sse2(src, dst, twiddles, p);
                }
            },
            3,
            2,
            TestSimdWidth::Width2,
            "butterfly_radix3_sse2",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn test_butterfly_radix3_sse4_2_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix3_scalar::<2>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse4_2::butterfly_radix3_stride1_sse4_2(src, dst, twiddles);
                } else {
                    sse4_2::butterfly_radix3_generic_sse4_2(src, dst, twiddles, p);
                }
            },
            3,
            2,
            TestSimdWidth::Width2,
            "butterfly_radix3_sse4_2",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_radix3_neon_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix3_scalar::<2>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    neon::butterfly_radix3_stride1_neon(src, dst, twiddles);
                } else {
                    neon::butterfly_radix3_generic_neon(src, dst, twiddles, p);
                }
            },
            3,
            2,
            TestSimdWidth::Width2,
            "butterfly_radix3_neon",
        );
    }
}
