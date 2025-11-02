use crate::Complex32;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse;

// Primitive 7th roots of unity: W_7 = exp(-2πi/7)
pub(super) const W7_1_RE: f32 = 0.6234898; // cos(-2π/7)
pub(super) const W7_1_IM: f32 = -0.7818315; // sin(-2π/7)
pub(super) const W7_2_RE: f32 = -0.2225209; // cos(-4π/7)
pub(super) const W7_2_IM: f32 = -0.9749279; // sin(-4π/7)
pub(super) const W7_3_RE: f32 = -0.9009689; // cos(-6π/7)
pub(super) const W7_3_IM: f32 = -0.4338837; // sin(-6π/7)
pub(super) const W7_4_RE: f32 = -0.9009689; // cos(-8π/7)
pub(super) const W7_4_IM: f32 = 0.4338837; // sin(-8π/7)
pub(super) const W7_5_RE: f32 = -0.2225209; // cos(-10π/7)
pub(super) const W7_5_IM: f32 = 0.9749279; // sin(-10π/7)
pub(super) const W7_6_RE: f32 = 0.6234898; // cos(-12π/7)
pub(super) const W7_6_IM: f32 = 0.7818315; // sin(-12π/7)

/// Processes a single radix-7 butterfly stage across all columns using scalar operations.
///
/// Implements the radix-7 DFT butterfly:
/// X[k] = sum_{j=0}^{6} x[j] * W_stage[j] * W_7^(k*j)
///
/// Where W_7 = exp(-2πi/7) is the primitive 7th root of unity, and
/// W_stage are the stage-specific twiddle factors.
#[inline(always)]
pub(super) fn butterfly_7_scalar(
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
        let x5 = data[idx + 5 * num_columns];
        let x6 = data[idx + 6 * num_columns];

        // Apply stage twiddle factors.
        let w1 = stage_twiddles[idx * 6];
        let w2 = stage_twiddles[idx * 6 + 1];
        let w3 = stage_twiddles[idx * 6 + 2];
        let w4 = stage_twiddles[idx * 6 + 3];
        let w5 = stage_twiddles[idx * 6 + 4];
        let w6 = stage_twiddles[idx * 6 + 5];

        let t1 = x1.mul(&w1);
        let t2 = x2.mul(&w2);
        let t3 = x3.mul(&w3);
        let t4 = x4.mul(&w4);
        let t5 = x5.mul(&w5);
        let t6 = x6.mul(&w6);

        // X0 = x0 + t1 + t2 + t3 + t4 + t5 + t6
        data[idx] = x0.add(&t1).add(&t2).add(&t3).add(&t4).add(&t5).add(&t6);

        // X1 = x0 + t1*W_7^1 + t2*W_7^2 + t3*W_7^3 + t4*W_7^4 + t5*W_7^5 + t6*W_7^6
        let t1_w71 = Complex32::new(
            t1.re * W7_1_RE - t1.im * W7_1_IM,
            t1.re * W7_1_IM + t1.im * W7_1_RE,
        );
        let t2_w72 = Complex32::new(
            t2.re * W7_2_RE - t2.im * W7_2_IM,
            t2.re * W7_2_IM + t2.im * W7_2_RE,
        );
        let t3_w73 = Complex32::new(
            t3.re * W7_3_RE - t3.im * W7_3_IM,
            t3.re * W7_3_IM + t3.im * W7_3_RE,
        );
        let t4_w74 = Complex32::new(
            t4.re * W7_4_RE - t4.im * W7_4_IM,
            t4.re * W7_4_IM + t4.im * W7_4_RE,
        );
        let t5_w75 = Complex32::new(
            t5.re * W7_5_RE - t5.im * W7_5_IM,
            t5.re * W7_5_IM + t5.im * W7_5_RE,
        );
        let t6_w76 = Complex32::new(
            t6.re * W7_6_RE - t6.im * W7_6_IM,
            t6.re * W7_6_IM + t6.im * W7_6_RE,
        );
        data[idx + num_columns] = x0
            .add(&t1_w71)
            .add(&t2_w72)
            .add(&t3_w73)
            .add(&t4_w74)
            .add(&t5_w75)
            .add(&t6_w76);

        // X2 = x0 + t1*W_7^2 + t2*W_7^4 + t3*W_7^6 + t4*W_7^1 + t5*W_7^3 + t6*W_7^5
        // (Note: W_7^(2*2) = W_7^4, W_7^(2*3) = W_7^6, W_7^(2*4) = W_7^8 = W_7^1, etc.)
        let t1_w72 = Complex32::new(
            t1.re * W7_2_RE - t1.im * W7_2_IM,
            t1.re * W7_2_IM + t1.im * W7_2_RE,
        );
        let t2_w74 = Complex32::new(
            t2.re * W7_4_RE - t2.im * W7_4_IM,
            t2.re * W7_4_IM + t2.im * W7_4_RE,
        );
        let t3_w76 = Complex32::new(
            t3.re * W7_6_RE - t3.im * W7_6_IM,
            t3.re * W7_6_IM + t3.im * W7_6_RE,
        );
        let t4_w71 = Complex32::new(
            t4.re * W7_1_RE - t4.im * W7_1_IM,
            t4.re * W7_1_IM + t4.im * W7_1_RE,
        );
        let t5_w73 = Complex32::new(
            t5.re * W7_3_RE - t5.im * W7_3_IM,
            t5.re * W7_3_IM + t5.im * W7_3_RE,
        );
        let t6_w75 = Complex32::new(
            t6.re * W7_5_RE - t6.im * W7_5_IM,
            t6.re * W7_5_IM + t6.im * W7_5_RE,
        );
        data[idx + 2 * num_columns] = x0
            .add(&t1_w72)
            .add(&t2_w74)
            .add(&t3_w76)
            .add(&t4_w71)
            .add(&t5_w73)
            .add(&t6_w75);

        // X3 = x0 + t1*W_7^3 + t2*W_7^6 + t3*W_7^2 + t4*W_7^5 + t5*W_7^1 + t6*W_7^4
        // (Note: W_7^(3*2) = W_7^6, W_7^(3*3) = W_7^9 = W_7^2, W_7^(3*4) = W_7^12 = W_7^5, etc.)
        let t1_w73 = Complex32::new(
            t1.re * W7_3_RE - t1.im * W7_3_IM,
            t1.re * W7_3_IM + t1.im * W7_3_RE,
        );
        let t2_w76 = Complex32::new(
            t2.re * W7_6_RE - t2.im * W7_6_IM,
            t2.re * W7_6_IM + t2.im * W7_6_RE,
        );
        let t3_w72 = Complex32::new(
            t3.re * W7_2_RE - t3.im * W7_2_IM,
            t3.re * W7_2_IM + t3.im * W7_2_RE,
        );
        let t4_w75 = Complex32::new(
            t4.re * W7_5_RE - t4.im * W7_5_IM,
            t4.re * W7_5_IM + t4.im * W7_5_RE,
        );
        let t5_w71 = Complex32::new(
            t5.re * W7_1_RE - t5.im * W7_1_IM,
            t5.re * W7_1_IM + t5.im * W7_1_RE,
        );
        let t6_w74 = Complex32::new(
            t6.re * W7_4_RE - t6.im * W7_4_IM,
            t6.re * W7_4_IM + t6.im * W7_4_RE,
        );
        data[idx + 3 * num_columns] = x0
            .add(&t1_w73)
            .add(&t2_w76)
            .add(&t3_w72)
            .add(&t4_w75)
            .add(&t5_w71)
            .add(&t6_w74);

        // X4 = x0 + t1*W_7^4 + t2*W_7^1 + t3*W_7^5 + t4*W_7^2 + t5*W_7^6 + t6*W_7^3
        // (Note: W_7^(4*2) = W_7^8 = W_7^1, W_7^(4*3) = W_7^12 = W_7^5, W_7^(4*4) = W_7^16 = W_7^2, etc.)
        let t1_w74 = Complex32::new(
            t1.re * W7_4_RE - t1.im * W7_4_IM,
            t1.re * W7_4_IM + t1.im * W7_4_RE,
        );
        let t2_w71 = Complex32::new(
            t2.re * W7_1_RE - t2.im * W7_1_IM,
            t2.re * W7_1_IM + t2.im * W7_1_RE,
        );
        let t3_w75 = Complex32::new(
            t3.re * W7_5_RE - t3.im * W7_5_IM,
            t3.re * W7_5_IM + t3.im * W7_5_RE,
        );
        let t4_w72 = Complex32::new(
            t4.re * W7_2_RE - t4.im * W7_2_IM,
            t4.re * W7_2_IM + t4.im * W7_2_RE,
        );
        let t5_w76 = Complex32::new(
            t5.re * W7_6_RE - t5.im * W7_6_IM,
            t5.re * W7_6_IM + t5.im * W7_6_RE,
        );
        let t6_w73 = Complex32::new(
            t6.re * W7_3_RE - t6.im * W7_3_IM,
            t6.re * W7_3_IM + t6.im * W7_3_RE,
        );
        data[idx + 4 * num_columns] = x0
            .add(&t1_w74)
            .add(&t2_w71)
            .add(&t3_w75)
            .add(&t4_w72)
            .add(&t5_w76)
            .add(&t6_w73);

        // X5 = x0 + t1*W_7^5 + t2*W_7^3 + t3*W_7^1 + t4*W_7^6 + t5*W_7^4 + t6*W_7^2
        // (Note: W_7^(5*2) = W_7^10 = W_7^3, W_7^(5*3) = W_7^15 = W_7^1, W_7^(5*4) = W_7^20 = W_7^6, etc.)
        let t1_w75 = Complex32::new(
            t1.re * W7_5_RE - t1.im * W7_5_IM,
            t1.re * W7_5_IM + t1.im * W7_5_RE,
        );
        let t2_w73 = Complex32::new(
            t2.re * W7_3_RE - t2.im * W7_3_IM,
            t2.re * W7_3_IM + t2.im * W7_3_RE,
        );
        let t3_w71 = Complex32::new(
            t3.re * W7_1_RE - t3.im * W7_1_IM,
            t3.re * W7_1_IM + t3.im * W7_1_RE,
        );
        let t4_w76 = Complex32::new(
            t4.re * W7_6_RE - t4.im * W7_6_IM,
            t4.re * W7_6_IM + t4.im * W7_6_RE,
        );
        let t5_w74 = Complex32::new(
            t5.re * W7_4_RE - t5.im * W7_4_IM,
            t5.re * W7_4_IM + t5.im * W7_4_RE,
        );
        let t6_w72 = Complex32::new(
            t6.re * W7_2_RE - t6.im * W7_2_IM,
            t6.re * W7_2_IM + t6.im * W7_2_RE,
        );
        data[idx + 5 * num_columns] = x0
            .add(&t1_w75)
            .add(&t2_w73)
            .add(&t3_w71)
            .add(&t4_w76)
            .add(&t5_w74)
            .add(&t6_w72);

        // X6 = x0 + t1*W_7^6 + t2*W_7^5 + t3*W_7^4 + t4*W_7^3 + t5*W_7^2 + t6*W_7^1
        // (Note: W_7^(6*2) = W_7^12 = W_7^5, W_7^(6*3) = W_7^18 = W_7^4, W_7^(6*4) = W_7^24 = W_7^3, etc.)
        let t1_w76 = Complex32::new(
            t1.re * W7_6_RE - t1.im * W7_6_IM,
            t1.re * W7_6_IM + t1.im * W7_6_RE,
        );
        let t2_w75 = Complex32::new(
            t2.re * W7_5_RE - t2.im * W7_5_IM,
            t2.re * W7_5_IM + t2.im * W7_5_RE,
        );
        let t3_w74 = Complex32::new(
            t3.re * W7_4_RE - t3.im * W7_4_IM,
            t3.re * W7_4_IM + t3.im * W7_4_RE,
        );
        let t4_w73 = Complex32::new(
            t4.re * W7_3_RE - t4.im * W7_3_IM,
            t4.re * W7_3_IM + t4.im * W7_3_RE,
        );
        let t5_w72 = Complex32::new(
            t5.re * W7_2_RE - t5.im * W7_2_IM,
            t5.re * W7_2_IM + t5.im * W7_2_RE,
        );
        let t6_w71 = Complex32::new(
            t6.re * W7_1_RE - t6.im * W7_1_IM,
            t6.re * W7_1_IM + t6.im * W7_1_RE,
        );
        data[idx + 6 * num_columns] = x0
            .add(&t1_w76)
            .add(&t2_w75)
            .add(&t3_w74)
            .add(&t4_w73)
            .add(&t5_w72)
            .add(&t6_w71);
    }
}

/// Dispatches to the best available SIMD implementation.
#[inline(always)]
pub(crate) fn butterfly_7_dispatch(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        if num_columns >= 4 {
            return unsafe { avx::butterfly_7_avx_fma(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        not(target_feature = "fma")
    ))]
    {
        if num_columns >= 4 {
            return unsafe { avx::butterfly_7_avx(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    {
        if num_columns >= 2 {
            return unsafe { sse::butterfly_7_sse3(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse3")
    ))]
    {
        if num_columns >= 2 {
            return unsafe { sse::butterfly_7_sse2(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if num_columns >= 2 {
            return unsafe { neon::butterfly_7_neon(data, stage_twiddles, 0, num_columns) };
        }
    }

    butterfly_7_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_7_dispatch_avx_fma(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 4 {
        return unsafe { avx::butterfly_7_avx_fma(data, stage_twiddles, 0, num_columns) };
    }

    if num_columns >= 2 {
        return unsafe { sse::butterfly_7_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_7_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_7_dispatch_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 4 {
        return unsafe { avx::butterfly_7_avx(data, stage_twiddles, 0, num_columns) };
    }

    if num_columns >= 2 {
        return unsafe { sse::butterfly_7_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_7_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_7_dispatch_sse3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 2 {
        return unsafe { sse::butterfly_7_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_7_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_7_dispatch_sse2(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 2 {
        return unsafe { sse::butterfly_7_sse2(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_7_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(test)]
mod tests {
    use super::{super::test_helpers::test_butterfly_against_scalar, *};

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_7_neon_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_7_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                neon::butterfly_7_neon(data, twiddles, 0, num_columns);
            },
            7,
            6,
            "butterfly_7_neon",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "sse2")
    ))]
    fn test_butterfly_7_sse2_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_7_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                sse::butterfly_7_sse2(data, twiddles, 0, num_columns);
            },
            7,
            6,
            "butterfly_7_sse2",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "sse3")
    ))]
    fn test_butterfly_7_sse3_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_7_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                sse::butterfly_7_sse3(data, twiddles, 0, num_columns);
            },
            7,
            6,
            "butterfly_7_sse3",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "avx")
    ))]
    fn test_butterfly_7_avx_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_7_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                avx::butterfly_7_avx(data, twiddles, 0, num_columns);
            },
            7,
            6,
            "butterfly_7_avx",
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
    fn test_butterfly_7_avx_fma_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_7_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                avx::butterfly_7_avx_fma(data, twiddles, 0, num_columns);
            },
            7,
            6,
            "butterfly_7_avx_fma",
        );
    }
}
