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

/// Dispatch function for radix-4 Stockham butterfly.
#[inline(always)]
pub(crate) fn butterfly_radix4_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    #[allow(unused)]
    let samples = src.len();
    #[allow(unused)]
    let quarter_samples = samples >> 2;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    unsafe {
        if quarter_samples >= 4 {
            return match stride {
                1 => avx::butterfly_radix4_stride1_avx_fma(src, dst, stage_twiddles),
                _ => avx::butterfly_radix4_generic_avx_fma(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse4.2",
        not(all(target_feature = "avx", target_feature = "fma"))
    ))]
    unsafe {
        if quarter_samples >= 2 {
            return match stride {
                1 => sse4_2::butterfly_radix4_stride1_sse4_2(src, dst, stage_twiddles),
                _ => sse4_2::butterfly_radix4_generic_sse4_2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse4.2")
    ))]
    unsafe {
        if quarter_samples >= 2 {
            return match stride {
                1 => sse2::butterfly_radix4_stride1_sse2(src, dst, stage_twiddles),
                _ => sse2::butterfly_radix4_generic_sse2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        if quarter_samples >= 2 {
            return match stride {
                1 => neon::butterfly_radix4_stride1_neon(src, dst, stage_twiddles),
                _ => neon::butterfly_radix4_generic_neon(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    butterfly_radix4_scalar::<4>(src, dst, stage_twiddles, stride);
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
    butterfly_radix4_scalar::<2>(src, dst, stage_twiddles, stride);
}

/// AVX+FMA dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix4_stride1_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let quarter_samples = samples >> 2;

    if quarter_samples >= 4 {
        return unsafe { avx::butterfly_radix4_stride1_avx_fma(src, dst, stage_twiddles) };
    }

    butterfly_radix4_scalar::<4>(src, dst, stage_twiddles, 1);
}

/// AVX+FMA dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix4_generic_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let quarter_samples = samples >> 2;

    if quarter_samples >= 4 {
        return unsafe { avx::butterfly_radix4_generic_avx_fma(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix4_scalar::<4>(src, dst, stage_twiddles, stride);
}

/// SSE2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix4_stride1_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let quarter_samples = samples >> 2;

    if quarter_samples >= 2 {
        return unsafe { sse2::butterfly_radix4_stride1_sse2(src, dst, stage_twiddles) };
    }

    butterfly_radix4_scalar::<2>(src, dst, stage_twiddles, 1);
}

/// SSE2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix4_generic_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let quarter_samples = samples >> 2;

    if quarter_samples >= 2 {
        return unsafe { sse2::butterfly_radix4_generic_sse2(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix4_scalar::<2>(src, dst, stage_twiddles, stride);
}

/// SSE4.2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix4_stride1_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let quarter_samples = samples >> 2;

    if quarter_samples >= 2 {
        return unsafe { sse4_2::butterfly_radix4_stride1_sse4_2(src, dst, stage_twiddles) };
    }

    butterfly_radix4_scalar::<2>(src, dst, stage_twiddles, 1);
}

/// SSE4.2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix4_generic_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let quarter_samples = samples >> 2;

    if quarter_samples >= 2 {
        return unsafe {
            sse4_2::butterfly_radix4_generic_sse4_2(src, dst, stage_twiddles, stride)
        };
    }

    butterfly_radix4_scalar::<2>(src, dst, stage_twiddles, stride);
}

/// Performs a single radix-4 Stockham butterfly stage (out-of-place, scalar).
///
/// Expects twiddles in packed format matching SIMD code:
/// - Packed portion (for pairs of iterations): [w1[i], w1[i+1], w2[i], w2[i+1], w3[i], w3[i+1], ...]
/// - Scalar tail (if any): [w1[i], w2[i], w3[i], ...] (interleaved)
#[inline(always)]
pub(super) fn butterfly_radix4_scalar<const WIDTH: usize>(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples / WIDTH) * WIDTH;

    // Process iterations using packed twiddle format
    for i in 0..simd_iters {
        let k = i % stride;

        // Calculate twiddle index in packed format (width-aware)
        let group_idx = i / WIDTH;
        let offset_in_group = i % WIDTH;
        let tw_base = group_idx * (3 * WIDTH) + offset_in_group;
        let w1 = stage_twiddles[tw_base];
        let w2 = stage_twiddles[tw_base + WIDTH];
        let w3 = stage_twiddles[tw_base + WIDTH * 2];

        let z0 = src[i];
        let z1 = src[i + quarter_samples];
        let z2 = src[i + quarter_samples * 2];
        let z3 = src[i + quarter_samples * 3];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);

        let a0 = z0.add(&t2);
        let a1 = z0.sub(&t2);
        let a2 = t1.add(&t3);
        let a3_re = t1.im - t3.im;
        let a3_im = t3.re - t1.re;

        let j = 4 * i - 3 * k;
        dst[j] = a0.add(&a2);
        dst[j + stride * 2] = a0.sub(&a2);
        dst[j + stride] = Complex32::new(a1.re + a3_re, a1.im + a3_im);
        dst[j + stride * 3] = Complex32::new(a1.re - a3_re, a1.im - a3_im);
    }

    // Process scalar tail using interleaved twiddle format
    let tail_offset = (simd_iters / WIDTH) * (3 * WIDTH);
    for i in simd_iters..quarter_samples {
        let k = i % stride;
        let tail_idx = i - simd_iters;
        let w1 = stage_twiddles[tail_offset + tail_idx * 3];
        let w2 = stage_twiddles[tail_offset + tail_idx * 3 + 1];
        let w3 = stage_twiddles[tail_offset + tail_idx * 3 + 2];

        let z0 = src[i];
        let z1 = src[i + quarter_samples];
        let z2 = src[i + quarter_samples * 2];
        let z3 = src[i + quarter_samples * 3];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);

        let a0 = z0.add(&t2);
        let a1 = z0.sub(&t2);
        let a2 = t1.add(&t3);
        let a3_re = t1.im - t3.im;
        let a3_im = t3.re - t1.re;

        let j = 4 * i - 3 * k;
        dst[j] = a0.add(&a2);
        dst[j + stride * 2] = a0.sub(&a2);
        dst[j + stride] = Complex32::new(a1.re + a3_re, a1.im + a3_im);
        dst[j + stride * 3] = Complex32::new(a1.re - a3_re, a1.im - a3_im);
    }
}

#[cfg(test)]
mod tests {
    use super::{super::tests::*, *};

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        all(target_feature = "avx", target_feature = "fma")
    ))]
    fn test_butterfly_radix4_avx_fma_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        test_butterfly_against_scalar(
            |src, dst, twiddles, p| butterfly_radix4_scalar::<4>(src, dst, twiddles, p),
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    avx::butterfly_radix4_stride1_avx_fma(src, dst, twiddles);
                } else {
                    avx::butterfly_radix4_generic_avx_fma(src, dst, twiddles, p);
                }
            },
            4,
            3,
            TestSimdWidth::Width4,
            "butterfly_radix4_avx_fma",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    fn test_butterfly_radix4_sse2_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        test_butterfly_against_scalar(
            |src, dst, twiddles, p| butterfly_radix4_scalar::<2>(src, dst, twiddles, p),
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse2::butterfly_radix4_stride1_sse2(src, dst, twiddles);
                } else {
                    sse2::butterfly_radix4_generic_sse2(src, dst, twiddles, p);
                }
            },
            4,
            3,
            TestSimdWidth::Width2,
            "butterfly_radix4_sse2",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn test_butterfly_radix4_sse4_2_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        test_butterfly_against_scalar(
            |src, dst, twiddles, p| butterfly_radix4_scalar::<2>(src, dst, twiddles, p),
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse4_2::butterfly_radix4_stride1_sse4_2(src, dst, twiddles);
                } else {
                    sse4_2::butterfly_radix4_generic_sse4_2(src, dst, twiddles, p);
                }
            },
            4,
            3,
            TestSimdWidth::Width2,
            "butterfly_radix4_sse4_2",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_radix4_neon_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        test_butterfly_against_scalar(
            |src, dst, twiddles, p| butterfly_radix4_scalar::<2>(src, dst, twiddles, p),
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    neon::butterfly_radix4_stride1_neon(src, dst, twiddles);
                } else {
                    neon::butterfly_radix4_generic_neon(src, dst, twiddles, p);
                }
            },
            4,
            3,
            TestSimdWidth::Width2,
            "butterfly_radix4_neon",
        );
    }
}
