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

/// Dispatch function for radix-2 Stockham butterfly.
#[inline(always)]
pub(crate) fn butterfly_radix2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    #[allow(unused)]
    let samples = src.len();
    #[allow(unused)]
    let half_samples = samples >> 1;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    unsafe {
        if half_samples >= 4 {
            return match stride {
                1 => avx::butterfly_radix2_stride1_avx_fma(src, dst, stage_twiddles),
                _ => avx::butterfly_radix2_generic_avx_fma(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse4.2",
        not(all(target_feature = "avx", target_feature = "fma"))
    ))]
    unsafe {
        if half_samples >= 2 {
            return match stride {
                1 => sse4_2::butterfly_radix2_stride1_sse4_2(src, dst, stage_twiddles),
                _ => sse4_2::butterfly_radix2_generic_sse4_2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse4.2")
    ))]
    unsafe {
        if half_samples >= 2 {
            return match stride {
                1 => sse2::butterfly_radix2_stride1_sse2(src, dst, stage_twiddles),
                _ => sse2::butterfly_radix2_generic_sse2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        if half_samples >= 2 {
            return match stride {
                1 => neon::butterfly_radix2_stride1_neon(src, dst, stage_twiddles),
                _ => neon::butterfly_radix2_generic_neon(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    butterfly_radix2_scalar::<4>(src, dst, stage_twiddles, stride);
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
    butterfly_radix2_scalar::<2>(src, dst, stage_twiddles, stride);
}

/// AVX+FMA dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix2_stride1_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let half_samples = samples >> 1;

    if half_samples >= 4 {
        return unsafe { avx::butterfly_radix2_stride1_avx_fma(src, dst, stage_twiddles) };
    }

    butterfly_radix2_scalar::<4>(src, dst, stage_twiddles, 1);
}

/// AVX+FMA dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix2_generic_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let half_samples = samples >> 1;

    if half_samples >= 4 {
        return unsafe { avx::butterfly_radix2_generic_avx_fma(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix2_scalar::<4>(src, dst, stage_twiddles, stride);
}

/// SSE2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix2_stride1_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let half_samples = samples >> 1;

    if half_samples >= 2 {
        return unsafe { sse2::butterfly_radix2_stride1_sse2(src, dst, stage_twiddles) };
    }

    butterfly_radix2_scalar::<2>(src, dst, stage_twiddles, 1);
}

/// SSE2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix2_generic_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let half_samples = samples >> 1;

    if half_samples >= 2 {
        return unsafe { sse2::butterfly_radix2_generic_sse2(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix2_scalar::<2>(src, dst, stage_twiddles, stride);
}

/// SSE4.2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix2_stride1_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let half_samples = samples >> 1;

    if half_samples >= 2 {
        return unsafe { sse4_2::butterfly_radix2_stride1_sse4_2(src, dst, stage_twiddles) };
    }

    butterfly_radix2_scalar::<2>(src, dst, stage_twiddles, 1);
}

/// SSE4.2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix2_generic_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let half_samples = samples >> 1;

    if half_samples >= 2 {
        return unsafe {
            sse4_2::butterfly_radix2_generic_sse4_2(src, dst, stage_twiddles, stride)
        };
    }

    butterfly_radix2_scalar::<2>(src, dst, stage_twiddles, stride);
}

/// Performs a single radix-2 Stockham butterfly stage (out-of-place, scalar).
///
/// The Stockham algorithm uses a specific indexing pattern.
/// For each element i, we compute: k = i & (p - 1), j = (i << 1) - k
/// where p is the stride parameter (number of columns in the output matrix view).
#[inline(always)]
pub(super) fn butterfly_radix2_scalar<const WIDTH: usize>(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let half_samples = samples >> 1;

    // For radix-2 with only 1 twiddle per iteration, the layout is the same
    // regardless of width, so we can use simple indexing
    for i in 0..half_samples {
        let k = i % stride;
        let twiddle = stage_twiddles[i];

        let a = src[i];
        let b = twiddle.mul(&src[i + half_samples]);

        let j = (i << 1) - k;
        dst[j] = a.add(&b);
        dst[j + stride] = a.sub(&b);
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
    fn test_butterfly_radix2_avx_fma_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        test_butterfly_against_scalar(
            |src, dst, twiddles, p| butterfly_radix2_scalar::<4>(src, dst, twiddles, p),
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    avx::butterfly_radix2_stride1_avx_fma(src, dst, twiddles);
                } else {
                    avx::butterfly_radix2_generic_avx_fma(src, dst, twiddles, p);
                }
            },
            2,
            1,
            TestSimdWidth::Width4,
            "butterfly_radix2_avx_fma",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    fn test_butterfly_radix2_sse2_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        test_butterfly_against_scalar(
            |src, dst, twiddles, p| butterfly_radix2_scalar::<2>(src, dst, twiddles, p),
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse2::butterfly_radix2_stride1_sse2(src, dst, twiddles);
                } else {
                    sse2::butterfly_radix2_generic_sse2(src, dst, twiddles, p);
                }
            },
            2,
            1,
            TestSimdWidth::Width2,
            "butterfly_radix2_sse2",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn test_butterfly_radix2_sse4_2_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        test_butterfly_against_scalar(
            |src, dst, twiddles, p| butterfly_radix2_scalar::<2>(src, dst, twiddles, p),
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse4_2::butterfly_radix2_stride1_sse4_2(src, dst, twiddles);
                } else {
                    sse4_2::butterfly_radix2_generic_sse4_2(src, dst, twiddles, p);
                }
            },
            2,
            1,
            TestSimdWidth::Width2,
            "butterfly_radix2_sse4_2",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_radix2_neon_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        test_butterfly_against_scalar(
            |src, dst, twiddles, p| butterfly_radix2_scalar::<2>(src, dst, twiddles, p),
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    neon::butterfly_radix2_stride1_neon(src, dst, twiddles);
                } else {
                    neon::butterfly_radix2_generic_neon(src, dst, twiddles, p);
                }
            },
            2,
            1,
            TestSimdWidth::Width2,
            "butterfly_radix2_neon",
        );
    }
}
