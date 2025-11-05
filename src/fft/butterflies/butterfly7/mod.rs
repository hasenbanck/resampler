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

const COS_2PI_7: f32 = 0.6234898;
const SIN_2PI_7: f32 = 0.7818315;
const COS_4PI_7: f32 = -0.22252093;
const SIN_4PI_7: f32 = 0.9749279;
const COS_6PI_7: f32 = -0.90096885;
const SIN_6PI_7: f32 = 0.43388373;

/// Dispatch function for radix-7 Stockham butterfly.
#[inline(always)]
pub(crate) fn butterfly_radix7_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    #[allow(unused)]
    let samples = src.len();
    #[allow(unused)]
    let seventh_samples = samples / 7;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    unsafe {
        if seventh_samples >= 4 {
            return match stride {
                1 => avx::butterfly_radix7_stride1_avx_fma(src, dst, stage_twiddles),
                _ => avx::butterfly_radix7_generic_avx_fma(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse4.2",
        not(all(target_feature = "avx", target_feature = "fma"))
    ))]
    unsafe {
        if seventh_samples >= 2 {
            return match stride {
                1 => sse4_2::butterfly_radix7_stride1_sse4_2(src, dst, stage_twiddles),
                _ => sse4_2::butterfly_radix7_generic_sse4_2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse4.2")
    ))]
    unsafe {
        if seventh_samples >= 2 {
            return match stride {
                1 => sse2::butterfly_radix7_stride1_sse2(src, dst, stage_twiddles),
                _ => sse2::butterfly_radix7_generic_sse2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let seventh_samples = samples / 7;
        if seventh_samples >= 2 {
            return match stride {
                1 => neon::butterfly_radix7_stride1_neon(src, dst, stage_twiddles),
                _ => neon::butterfly_radix7_generic_neon(src, dst, stage_twiddles, stride),
            };
        }
    }

    butterfly_radix7_scalar(src, dst, stage_twiddles, stride);
}

/// AVX+FMA dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix7_stride1_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let seventh_samples = samples / 7;

    if seventh_samples >= 4 {
        return unsafe { avx::butterfly_radix7_stride1_avx_fma(src, dst, stage_twiddles) };
    }

    butterfly_radix7_scalar(src, dst, stage_twiddles, 1);
}

/// AVX+FMA dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix7_generic_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let seventh_samples = samples / 7;

    if seventh_samples >= 4 {
        return unsafe { avx::butterfly_radix7_generic_avx_fma(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix7_scalar(src, dst, stage_twiddles, stride);
}

/// SSE2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix7_stride1_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let seventh_samples = samples / 7;

    if seventh_samples >= 2 {
        return unsafe { sse2::butterfly_radix7_stride1_sse2(src, dst, stage_twiddles) };
    }

    butterfly_radix7_scalar(src, dst, stage_twiddles, 1);
}

/// SSE2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix7_generic_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let seventh_samples = samples / 7;

    if seventh_samples >= 2 {
        return unsafe { sse2::butterfly_radix7_generic_sse2(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix7_scalar(src, dst, stage_twiddles, stride);
}

/// SSE4.2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix7_stride1_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let seventh_samples = samples / 7;

    if seventh_samples >= 2 {
        return unsafe { sse4_2::butterfly_radix7_stride1_sse4_2(src, dst, stage_twiddles) };
    }

    butterfly_radix7_scalar(src, dst, stage_twiddles, 1);
}

/// SSE4.2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix7_generic_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let seventh_samples = samples / 7;

    if seventh_samples >= 2 {
        return unsafe {
            sse4_2::butterfly_radix7_generic_sse4_2(src, dst, stage_twiddles, stride)
        };
    }

    butterfly_radix7_scalar(src, dst, stage_twiddles, stride);
}

/// Performs a single radix-7 Stockham butterfly stage (out-of-place, scalar).
#[inline(always)]
fn butterfly_radix7_scalar(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let seventh_samples = samples / 7;

    for i in 0..seventh_samples {
        let k = i % stride;
        let w1 = stage_twiddles[i * 6];
        let w2 = stage_twiddles[i * 6 + 1];
        let w3 = stage_twiddles[i * 6 + 2];
        let w4 = stage_twiddles[i * 6 + 3];
        let w5 = stage_twiddles[i * 6 + 4];
        let w6 = stage_twiddles[i * 6 + 5];

        let z0 = src[i];
        let z1 = src[i + seventh_samples];
        let z2 = src[i + seventh_samples * 2];
        let z3 = src[i + seventh_samples * 3];
        let z4 = src[i + seventh_samples * 4];
        let z5 = src[i + seventh_samples * 5];
        let z6 = src[i + seventh_samples * 6];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);
        let t5 = w5.mul(&z5);
        let t6 = w6.mul(&z6);

        let sum_all = t1.add(&t2).add(&t3).add(&t4).add(&t5).add(&t6);

        // Radix-7 DFT decomposition.
        let a1 = t1.add(&t6);
        let a2 = t2.add(&t5);
        let a3 = t3.add(&t4);

        let b1_re = t1.im - t6.im;
        let b1_im = t6.re - t1.re;
        let b2_re = t2.im - t5.im;
        let b2_im = t5.re - t2.re;
        let b3_re = t3.im - t4.im;
        let b3_im = t4.re - t3.re;

        let j = 7 * i - 6 * k;
        dst[j] = z0.add(&sum_all);

        for idx in 1..7 {
            let (cos1, sin1, cos2, sin2, cos3, sin3) = match idx {
                1 => (
                    COS_2PI_7, SIN_2PI_7, COS_4PI_7, SIN_4PI_7, COS_6PI_7, SIN_6PI_7,
                ),
                2 => (
                    COS_4PI_7, SIN_4PI_7, COS_6PI_7, -SIN_6PI_7, COS_2PI_7, -SIN_2PI_7,
                ),
                3 => (
                    COS_6PI_7, SIN_6PI_7, COS_2PI_7, -SIN_2PI_7, COS_4PI_7, SIN_4PI_7,
                ),
                4 => (
                    COS_6PI_7, -SIN_6PI_7, COS_2PI_7, SIN_2PI_7, COS_4PI_7, -SIN_4PI_7,
                ),
                5 => (
                    COS_4PI_7, -SIN_4PI_7, COS_6PI_7, SIN_6PI_7, COS_2PI_7, SIN_2PI_7,
                ),
                6 => (
                    COS_2PI_7, -SIN_2PI_7, COS_4PI_7, -SIN_4PI_7, COS_6PI_7, -SIN_6PI_7,
                ),
                _ => unreachable!(),
            };

            let c_re = z0.re + cos1 * a1.re + cos2 * a2.re + cos3 * a3.re;
            let c_im = z0.im + cos1 * a1.im + cos2 * a2.im + cos3 * a3.im;
            let d_re = sin1 * b1_re + sin2 * b2_re + sin3 * b3_re;
            let d_im = sin1 * b1_im + sin2 * b2_im + sin3 * b3_im;

            dst[j + stride * idx] = Complex32::new(c_re + d_re, c_im + d_im);
        }
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
    fn test_butterfly_radix7_avx_fma_vs_scalar() {
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix7_scalar,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    avx::butterfly_radix7_stride1_avx_fma(src, dst, twiddles);
                } else {
                    avx::butterfly_radix7_generic_avx_fma(src, dst, twiddles, p);
                }
            },
            7,
            6,
            "butterfly_radix7_avx_fma",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    fn test_butterfly_radix7_sse2_vs_scalar() {
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix7_scalar,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse2::butterfly_radix7_stride1_sse2(src, dst, twiddles);
                } else {
                    sse2::butterfly_radix7_generic_sse2(src, dst, twiddles, p);
                }
            },
            7,
            6,
            "butterfly_radix7_sse2",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn test_butterfly_radix7_sse4_2_vs_scalar() {
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix7_scalar,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse4_2::butterfly_radix7_stride1_sse4_2(src, dst, twiddles);
                } else {
                    sse4_2::butterfly_radix7_generic_sse4_2(src, dst, twiddles, p);
                }
            },
            7,
            6,
            "butterfly_radix7_sse4_2",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_radix7_neon_vs_scalar() {
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix7_scalar,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    neon::butterfly_radix7_stride1_neon(src, dst, twiddles);
                } else {
                    neon::butterfly_radix7_generic_neon(src, dst, twiddles, p);
                }
            },
            7,
            6,
            "butterfly_radix7_neon",
        );
    }
}
