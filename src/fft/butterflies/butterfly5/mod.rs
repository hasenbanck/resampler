use crate::Complex32;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse;

// Primitive 5th roots of unity: W_5 = exp(-2πi/5)
pub(super) const W5_1_RE: f32 = 0.309017; // cos(-2π/5)
pub(super) const W5_1_IM: f32 = -0.95105654; // sin(-2π/5)
pub(super) const W5_2_RE: f32 = -0.809017; // cos(-4π/5)
pub(super) const W5_2_IM: f32 = -0.58778524; // sin(-4π/5)
pub(super) const W5_3_RE: f32 = -0.809017; // cos(-6π/5)
pub(super) const W5_3_IM: f32 = 0.58778524; // sin(-6π/5)
pub(super) const W5_4_RE: f32 = 0.309017; // cos(-8π/5)
pub(super) const W5_4_IM: f32 = 0.95105654; // sin(-8π/5)

/// Processes a single radix-5 butterfly stage across all columns using scalar operations.
///
/// Implements the radix-5 DFT butterfly:
/// X[k] = sum_{j=0}^{4} x[j] * W_stage[j] * W_5^(k*j)
///
/// Where W_5 = exp(-2πi/5) is the primitive 5th root of unity, and
/// W_stage are the stage-specific twiddle factors.
#[inline(always)]
pub(super) fn butterfly_5_scalar(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    start_col: usize,
    num_columns: usize,
) {
    for idx in start_col..num_columns {
        let x0 = data[idx];
        let x1 = data[idx + num_columns];
        let x2 = data[idx + 2 * num_columns];
        let x3 = data[idx + 3 * num_columns];
        let x4 = data[idx + 4 * num_columns];

        // Apply stage twiddle factors.
        let w1 = stage_twiddles[idx * 4];
        let w2 = stage_twiddles[idx * 4 + 1];
        let w3 = stage_twiddles[idx * 4 + 2];
        let w4 = stage_twiddles[idx * 4 + 3];

        let t1 = x1.mul(&w1);
        let t2 = x2.mul(&w2);
        let t3 = x3.mul(&w3);
        let t4 = x4.mul(&w4);

        // X0 = x0 + t1 + t2 + t3 + t4
        data[idx] = x0.add(&t1).add(&t2).add(&t3).add(&t4);

        // X1 = x0 + t1*W_5^1 + t2*W_5^2 + t3*W_5^3 + t4*W_5^4
        let t1_w51 = Complex32::new(
            t1.re * W5_1_RE - t1.im * W5_1_IM,
            t1.re * W5_1_IM + t1.im * W5_1_RE,
        );
        let t2_w52 = Complex32::new(
            t2.re * W5_2_RE - t2.im * W5_2_IM,
            t2.re * W5_2_IM + t2.im * W5_2_RE,
        );
        let t3_w53 = Complex32::new(
            t3.re * W5_3_RE - t3.im * W5_3_IM,
            t3.re * W5_3_IM + t3.im * W5_3_RE,
        );
        let t4_w54 = Complex32::new(
            t4.re * W5_4_RE - t4.im * W5_4_IM,
            t4.re * W5_4_IM + t4.im * W5_4_RE,
        );
        data[idx + num_columns] = x0.add(&t1_w51).add(&t2_w52).add(&t3_w53).add(&t4_w54);

        // X2 = x0 + t1*W_5^2 + t2*W_5^4 + t3*W_5^1 + t4*W_5^3
        // (Note: W_5^(2*2) = W_5^4, W_5^(2*3) = W_5^6 = W_5^1, W_5^(2*4) = W_5^8 = W_5^3)
        let t1_w52 = Complex32::new(
            t1.re * W5_2_RE - t1.im * W5_2_IM,
            t1.re * W5_2_IM + t1.im * W5_2_RE,
        );
        let t2_w54 = Complex32::new(
            t2.re * W5_4_RE - t2.im * W5_4_IM,
            t2.re * W5_4_IM + t2.im * W5_4_RE,
        );
        let t3_w51 = Complex32::new(
            t3.re * W5_1_RE - t3.im * W5_1_IM,
            t3.re * W5_1_IM + t3.im * W5_1_RE,
        );
        let t4_w53 = Complex32::new(
            t4.re * W5_3_RE - t4.im * W5_3_IM,
            t4.re * W5_3_IM + t4.im * W5_3_RE,
        );
        data[idx + 2 * num_columns] = x0.add(&t1_w52).add(&t2_w54).add(&t3_w51).add(&t4_w53);

        // X3 = x0 + t1*W_5^3 + t2*W_5^1 + t3*W_5^4 + t4*W_5^2
        // (Note: W_5^(3*2) = W_5^6 = W_5^1, W_5^(3*3) = W_5^9 = W_5^4, W_5^(3*4) = W_5^12 = W_5^2)
        let t1_w53 = Complex32::new(
            t1.re * W5_3_RE - t1.im * W5_3_IM,
            t1.re * W5_3_IM + t1.im * W5_3_RE,
        );
        let t2_w51 = Complex32::new(
            t2.re * W5_1_RE - t2.im * W5_1_IM,
            t2.re * W5_1_IM + t2.im * W5_1_RE,
        );
        let t3_w54 = Complex32::new(
            t3.re * W5_4_RE - t3.im * W5_4_IM,
            t3.re * W5_4_IM + t3.im * W5_4_RE,
        );
        let t4_w52 = Complex32::new(
            t4.re * W5_2_RE - t4.im * W5_2_IM,
            t4.re * W5_2_IM + t4.im * W5_2_RE,
        );
        data[idx + 3 * num_columns] = x0.add(&t1_w53).add(&t2_w51).add(&t3_w54).add(&t4_w52);

        // X4 = x0 + t1*W_5^4 + t2*W_5^3 + t3*W_5^2 + t4*W_5^1
        // (Note: W_5^(4*2) = W_5^8 = W_5^3, W_5^(4*3) = W_5^12 = W_5^2, W_5^(4*4) = W_5^16 = W_5^1)
        let t1_w54 = Complex32::new(
            t1.re * W5_4_RE - t1.im * W5_4_IM,
            t1.re * W5_4_IM + t1.im * W5_4_RE,
        );
        let t2_w53 = Complex32::new(
            t2.re * W5_3_RE - t2.im * W5_3_IM,
            t2.re * W5_3_IM + t2.im * W5_3_RE,
        );
        let t3_w52 = Complex32::new(
            t3.re * W5_2_RE - t3.im * W5_2_IM,
            t3.re * W5_2_IM + t3.im * W5_2_RE,
        );
        let t4_w51 = Complex32::new(
            t4.re * W5_1_RE - t4.im * W5_1_IM,
            t4.re * W5_1_IM + t4.im * W5_1_RE,
        );
        data[idx + 4 * num_columns] = x0.add(&t1_w54).add(&t2_w53).add(&t3_w52).add(&t4_w51);
    }
}

/// Dispatches to the best available SIMD implementation.
#[inline(always)]
pub(crate) fn butterfly_5_dispatch(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        if num_columns >= 4 {
            return unsafe { avx::butterfly_5_avx_fma(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        not(target_feature = "fma")
    ))]
    {
        if num_columns >= 4 {
            return unsafe { avx::butterfly_5_avx(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    {
        if num_columns >= 2 {
            return unsafe { sse::butterfly_5_sse3(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse3")
    ))]
    {
        if num_columns >= 2 {
            return unsafe { sse::butterfly_5_sse2(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if num_columns >= 2 {
            return unsafe { neon::butterfly_5_neon(data, stage_twiddles, 0, num_columns) };
        }
    }

    butterfly_5_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_5_dispatch_avx_fma(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 4 {
        return unsafe { avx::butterfly_5_avx_fma(data, stage_twiddles, 0, num_columns) };
    }

    if num_columns >= 2 {
        return unsafe { sse::butterfly_5_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_5_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_5_dispatch_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 4 {
        return unsafe { avx::butterfly_5_avx(data, stage_twiddles, 0, num_columns) };
    }

    if num_columns >= 2 {
        return unsafe { sse::butterfly_5_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_5_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_5_dispatch_sse3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 2 {
        return unsafe { sse::butterfly_5_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_5_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_5_dispatch_sse2(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 2 {
        return unsafe { sse::butterfly_5_sse2(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_5_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(test)]
mod tests {
    use super::{super::test_helpers::test_butterfly_against_scalar, *};

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_5_neon_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_5_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                neon::butterfly_5_neon(data, twiddles, 0, num_columns);
            },
            5,
            4,
            "butterfly_5_neon",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "sse2")
    ))]
    fn test_butterfly_5_sse2_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_5_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                sse::butterfly_5_sse2(data, twiddles, 0, num_columns);
            },
            5,
            4,
            "butterfly_5_sse2",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "sse3")
    ))]
    fn test_butterfly_5_sse3_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_5_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                sse::butterfly_5_sse3(data, twiddles, 0, num_columns);
            },
            5,
            4,
            "butterfly_5_sse3",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "avx")
    ))]
    fn test_butterfly_5_avx_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_5_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                avx::butterfly_5_avx(data, twiddles, 0, num_columns);
            },
            5,
            4,
            "butterfly_5_avx",
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
    fn test_butterfly_5_avx_fma_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_5_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                avx::butterfly_5_avx_fma(data, twiddles, 0, num_columns);
            },
            5,
            4,
            "butterfly_5_avx_fma",
        );
    }
}
