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

const COS_2PI_5: f32 = 0.309017; // cos(2π/5)
const SIN_2PI_5: f32 = 0.95105654; // sin(2π/5)
const COS_4PI_5: f32 = -0.809017; // cos(4π/5)
const SIN_4PI_5: f32 = 0.58778524; // sin(4π/5)

/// Dispatch function for radix-5 Stockham butterfly.
#[inline(always)]
pub(crate) fn butterfly_radix5_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    #[allow(unused)]
    let samples = src.len();
    #[allow(unused)]
    let fifth_samples = samples / 5;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    unsafe {
        if fifth_samples >= 4 {
            return match stride {
                1 => avx::butterfly_radix5_stride1_avx_fma(src, dst, stage_twiddles),
                _ => avx::butterfly_radix5_generic_avx_fma(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse4.2",
        not(all(target_feature = "avx", target_feature = "fma"))
    ))]
    unsafe {
        if fifth_samples >= 2 {
            return match stride {
                1 => sse4_2::butterfly_radix5_stride1_sse4_2(src, dst, stage_twiddles),
                _ => sse4_2::butterfly_radix5_generic_sse4_2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse4.2")
    ))]
    unsafe {
        if fifth_samples >= 2 {
            return match stride {
                1 => sse2::butterfly_radix5_stride1_sse2(src, dst, stage_twiddles),
                _ => sse2::butterfly_radix5_generic_sse2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        if fifth_samples >= 2 {
            return match stride {
                1 => neon::butterfly_radix5_stride1_neon(src, dst, stage_twiddles),
                _ => neon::butterfly_radix5_generic_neon(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    butterfly_radix5_scalar::<4>(src, dst, stage_twiddles, stride, 0);
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
    butterfly_radix5_scalar::<2>(src, dst, stage_twiddles, stride, 0);
}

/// AVX+FMA dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix5_stride1_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let fifth_samples = samples / 5;

    if fifth_samples >= 4 {
        return unsafe { avx::butterfly_radix5_stride1_avx_fma(src, dst, stage_twiddles) };
    }

    butterfly_radix5_scalar::<4>(src, dst, stage_twiddles, 1, 0);
}

/// AVX+FMA dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix5_generic_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let fifth_samples = samples / 5;

    if fifth_samples >= 4 {
        return unsafe { avx::butterfly_radix5_generic_avx_fma(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix5_scalar::<4>(src, dst, stage_twiddles, stride, 0);
}

/// SSE2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix5_stride1_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let fifth_samples = samples / 5;

    if fifth_samples >= 2 {
        return unsafe { sse2::butterfly_radix5_stride1_sse2(src, dst, stage_twiddles) };
    }

    butterfly_radix5_scalar::<2>(src, dst, stage_twiddles, 1, 0);
}

/// SSE2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix5_generic_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let fifth_samples = samples / 5;

    if fifth_samples >= 2 {
        return unsafe { sse2::butterfly_radix5_generic_sse2(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix5_scalar::<2>(src, dst, stage_twiddles, stride, 0);
}

/// SSE4.2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix5_stride1_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let fifth_samples = samples / 5;

    if fifth_samples >= 2 {
        return unsafe { sse4_2::butterfly_radix5_stride1_sse4_2(src, dst, stage_twiddles) };
    }

    butterfly_radix5_scalar::<2>(src, dst, stage_twiddles, 1, 0);
}

/// SSE4.2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix5_generic_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let fifth_samples = samples / 5;

    if fifth_samples >= 2 {
        return unsafe {
            sse4_2::butterfly_radix5_generic_sse4_2(src, dst, stage_twiddles, stride)
        };
    }

    butterfly_radix5_scalar::<2>(src, dst, stage_twiddles, stride, 0);
}

/// Performs a single radix-5 Stockham butterfly stage (out-of-place, scalar).
#[inline(always)]
fn butterfly_radix5_scalar<const WIDTH: usize>(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
    start_index: usize,
) {
    let samples = src.len();
    let fifth_samples = samples / 5;
    let simd_iters = (fifth_samples / WIDTH) * WIDTH;

    // Stride=1 optimization: skip identity twiddle multiplications.
    if stride == 1 {
        for i in start_index..simd_iters {
            let z0 = src[i];
            let z1 = src[i + fifth_samples];
            let z2 = src[i + fifth_samples * 2];
            let z3 = src[i + fifth_samples * 3];
            let z4 = src[i + fifth_samples * 4];

            // Identity twiddles: t_k = (1+0i) * z_k = z_k
            let t1 = z1;
            let t2 = z2;
            let t3 = z3;
            let t4 = z4;

            // Radix-5 DFT using the standard decomposition.
            let sum_all = t1.add(&t2).add(&t3).add(&t4);

            let a1 = t1.add(&t4);
            let a2 = t2.add(&t3);
            let b1_re = t1.im - t4.im;
            let b1_im = t4.re - t1.re;
            let b2_re = t2.im - t3.im;
            let b2_im = t3.re - t2.re;

            let c1_re = z0.re + COS_2PI_5 * a1.re + COS_4PI_5 * a2.re;
            let c1_im = z0.im + COS_2PI_5 * a1.im + COS_4PI_5 * a2.im;
            let c2_re = z0.re + COS_4PI_5 * a1.re + COS_2PI_5 * a2.re;
            let c2_im = z0.im + COS_4PI_5 * a1.im + COS_2PI_5 * a2.im;

            let d1_re = SIN_2PI_5 * b1_re + SIN_4PI_5 * b2_re;
            let d1_im = SIN_2PI_5 * b1_im + SIN_4PI_5 * b2_im;
            let d2_re = SIN_4PI_5 * b1_re - SIN_2PI_5 * b2_re;
            let d2_im = SIN_4PI_5 * b1_im - SIN_2PI_5 * b2_im;

            let j = 5 * i;
            dst[j] = z0.add(&sum_all);
            dst[j + 1] = Complex32::new(c1_re + d1_re, c1_im + d1_im);
            dst[j + 2] = Complex32::new(c2_re + d2_re, c2_im + d2_im);
            dst[j + 3] = Complex32::new(c2_re - d2_re, c2_im - d2_im);
            dst[j + 4] = Complex32::new(c1_re - d1_re, c1_im - d1_im);
        }

        // Process scalar tail.
        for i in simd_iters..fifth_samples {
            let z0 = src[i];
            let z1 = src[i + fifth_samples];
            let z2 = src[i + fifth_samples * 2];
            let z3 = src[i + fifth_samples * 3];
            let z4 = src[i + fifth_samples * 4];

            // Identity twiddles: t_k = (1+0i) * z_k = z_k
            let t1 = z1;
            let t2 = z2;
            let t3 = z3;
            let t4 = z4;

            // Radix-5 DFT using the standard decomposition.
            let sum_all = t1.add(&t2).add(&t3).add(&t4);

            let a1 = t1.add(&t4);
            let a2 = t2.add(&t3);
            let b1_re = t1.im - t4.im;
            let b1_im = t4.re - t1.re;
            let b2_re = t2.im - t3.im;
            let b2_im = t3.re - t2.re;

            let c1_re = z0.re + COS_2PI_5 * a1.re + COS_4PI_5 * a2.re;
            let c1_im = z0.im + COS_2PI_5 * a1.im + COS_4PI_5 * a2.im;
            let c2_re = z0.re + COS_4PI_5 * a1.re + COS_2PI_5 * a2.re;
            let c2_im = z0.im + COS_4PI_5 * a1.im + COS_2PI_5 * a2.im;

            let d1_re = SIN_2PI_5 * b1_re + SIN_4PI_5 * b2_re;
            let d1_im = SIN_2PI_5 * b1_im + SIN_4PI_5 * b2_im;
            let d2_re = SIN_4PI_5 * b1_re - SIN_2PI_5 * b2_re;
            let d2_im = SIN_4PI_5 * b1_im - SIN_2PI_5 * b2_im;

            let j = 5 * i;
            dst[j] = z0.add(&sum_all);
            dst[j + 1] = Complex32::new(c1_re + d1_re, c1_im + d1_im);
            dst[j + 4] = Complex32::new(c1_re - d1_re, c1_im - d1_im);
            dst[j + 2] = Complex32::new(c2_re + d2_re, c2_im + d2_im);
            dst[j + 3] = Complex32::new(c2_re - d2_re, c2_im - d2_im);
        }
        return;
    }

    // Process SIMD-packed region.
    for i in start_index..simd_iters {
        let k = i % stride;
        let group_idx = i / WIDTH;
        let offset_in_group = i % WIDTH;
        let tw_base = group_idx * (4 * WIDTH) + offset_in_group;
        let w1 = stage_twiddles[tw_base];
        let w2 = stage_twiddles[tw_base + WIDTH];
        let w3 = stage_twiddles[tw_base + WIDTH * 2];
        let w4 = stage_twiddles[tw_base + WIDTH * 3];

        let z0 = src[i];
        let z1 = src[i + fifth_samples];
        let z2 = src[i + fifth_samples * 2];
        let z3 = src[i + fifth_samples * 3];
        let z4 = src[i + fifth_samples * 4];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);

        // Radix-5 DFT using the standard decomposition.
        let sum_all = t1.add(&t2).add(&t3).add(&t4);

        let a1 = t1.add(&t4);
        let a2 = t2.add(&t3);
        let b1_re = t1.im - t4.im;
        let b1_im = t4.re - t1.re;
        let b2_re = t2.im - t3.im;
        let b2_im = t3.re - t2.re;

        let c1_re = z0.re + COS_2PI_5 * a1.re + COS_4PI_5 * a2.re;
        let c1_im = z0.im + COS_2PI_5 * a1.im + COS_4PI_5 * a2.im;
        let c2_re = z0.re + COS_4PI_5 * a1.re + COS_2PI_5 * a2.re;
        let c2_im = z0.im + COS_4PI_5 * a1.im + COS_2PI_5 * a2.im;

        let d1_re = SIN_2PI_5 * b1_re + SIN_4PI_5 * b2_re;
        let d1_im = SIN_2PI_5 * b1_im + SIN_4PI_5 * b2_im;
        let d2_re = SIN_4PI_5 * b1_re - SIN_2PI_5 * b2_re;
        let d2_im = SIN_4PI_5 * b1_im - SIN_2PI_5 * b2_im;

        let j = 5 * i - 4 * k;
        dst[j] = z0.add(&sum_all);
        dst[j + stride] = Complex32::new(c1_re + d1_re, c1_im + d1_im);
        dst[j + stride * 2] = Complex32::new(c2_re + d2_re, c2_im + d2_im);
        dst[j + stride * 3] = Complex32::new(c2_re - d2_re, c2_im - d2_im);
        dst[j + stride * 4] = Complex32::new(c1_re - d1_re, c1_im - d1_im);
    }

    // Process scalar tail.
    let tail_offset = (simd_iters / WIDTH) * (4 * WIDTH);
    for i in simd_iters..fifth_samples {
        let k = i % stride;
        let tail_idx = i - simd_iters;
        let w1 = stage_twiddles[tail_offset + tail_idx * 4];
        let w2 = stage_twiddles[tail_offset + tail_idx * 4 + 1];
        let w3 = stage_twiddles[tail_offset + tail_idx * 4 + 2];
        let w4 = stage_twiddles[tail_offset + tail_idx * 4 + 3];

        let z0 = src[i];
        let z1 = src[i + fifth_samples];
        let z2 = src[i + fifth_samples * 2];
        let z3 = src[i + fifth_samples * 3];
        let z4 = src[i + fifth_samples * 4];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);

        // Radix-5 DFT using the standard decomposition.
        let sum_all = t1.add(&t2).add(&t3).add(&t4);

        let a1 = t1.add(&t4);
        let a2 = t2.add(&t3);
        let b1_re = t1.im - t4.im;
        let b1_im = t4.re - t1.re;
        let b2_re = t2.im - t3.im;
        let b2_im = t3.re - t2.re;

        let c1_re = z0.re + COS_2PI_5 * a1.re + COS_4PI_5 * a2.re;
        let c1_im = z0.im + COS_2PI_5 * a1.im + COS_4PI_5 * a2.im;
        let c2_re = z0.re + COS_4PI_5 * a1.re + COS_2PI_5 * a2.re;
        let c2_im = z0.im + COS_4PI_5 * a1.im + COS_2PI_5 * a2.im;

        let d1_re = SIN_2PI_5 * b1_re + SIN_4PI_5 * b2_re;
        let d1_im = SIN_2PI_5 * b1_im + SIN_4PI_5 * b2_im;
        let d2_re = SIN_4PI_5 * b1_re - SIN_2PI_5 * b2_re;
        let d2_im = SIN_4PI_5 * b1_im - SIN_2PI_5 * b2_im;

        let j = 5 * i - 4 * k;
        dst[j] = z0.add(&sum_all);
        dst[j + stride] = Complex32::new(c1_re + d1_re, c1_im + d1_im);
        dst[j + stride * 4] = Complex32::new(c1_re - d1_re, c1_im - d1_im);
        dst[j + stride * 2] = Complex32::new(c2_re + d2_re, c2_im + d2_im);
        dst[j + stride * 3] = Complex32::new(c2_re - d2_re, c2_im - d2_im);
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
    fn test_butterfly_radix5_avx_fma_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix5_scalar::<4>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    avx::butterfly_radix5_stride1_avx_fma(src, dst, twiddles);
                } else {
                    avx::butterfly_radix5_generic_avx_fma(src, dst, twiddles, p);
                }
            },
            5,
            4,
            TestSimdWidth::Width4,
            "butterfly_radix5_avx_fma",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    fn test_butterfly_radix5_sse2_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix5_scalar::<2>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse2::butterfly_radix5_stride1_sse2(src, dst, twiddles);
                } else {
                    sse2::butterfly_radix5_generic_sse2(src, dst, twiddles, p);
                }
            },
            5,
            4,
            TestSimdWidth::Width2,
            "butterfly_radix5_sse2",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn test_butterfly_radix5_sse4_2_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix5_scalar::<2>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse4_2::butterfly_radix5_stride1_sse4_2(src, dst, twiddles);
                } else {
                    sse4_2::butterfly_radix5_generic_sse4_2(src, dst, twiddles, p);
                }
            },
            5,
            4,
            TestSimdWidth::Width2,
            "butterfly_radix5_sse4_2",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_radix5_neon_vs_scalar() {
        use crate::fft::butterflies::tests::TestSimdWidth;
        crate::fft::butterflies::tests::test_butterfly_against_scalar(
            butterfly_radix5_scalar::<2>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    neon::butterfly_radix5_stride1_neon(src, dst, twiddles);
                } else {
                    neon::butterfly_radix5_generic_neon(src, dst, twiddles, p);
                }
            },
            5,
            4,
            TestSimdWidth::Width2,
            "butterfly_radix5_neon",
        );
    }
}
