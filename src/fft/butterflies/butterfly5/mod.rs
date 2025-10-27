use crate::Complex32;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx",
    any(not(target_feature = "fma"), test)
))]
use avx::butterfly_5_avx;
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use avx::butterfly_5_avx_fma;
#[cfg(target_arch = "aarch64")]
use neon::butterfly_5_neon;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    any(not(target_feature = "sse3"), test)
))]
use sse::butterfly_5_sse;
#[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
use sse::butterfly_5_sse3;

// Primitive 5th roots of unity: W_5 = exp(-2πi/5)
pub(super) const W5_1_RE: f32 = 0.309017; // cos(-2π/5)
pub(super) const W5_1_IM: f32 = -0.95105654; // sin(-2π/5)
pub(super) const W5_2_RE: f32 = -0.809017; // cos(-4π/5)
pub(super) const W5_2_IM: f32 = -0.58778524; // sin(-4π/5)
pub(super) const W5_3_RE: f32 = -0.809017; // cos(-6π/5)
pub(super) const W5_3_IM: f32 = 0.58778524; // sin(-6π/5)
pub(super) const W5_4_RE: f32 = 0.309017; // cos(-8π/5)
pub(super) const W5_4_IM: f32 = 0.95105654; // sin(-8π/5)

/// Helper function to process a range of columns for radix-5 butterfly.
/// Processes columns from `start_col` to `end_col` (exclusive).
#[inline(always)]
pub(super) fn butterfly_5_columns(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
    start_col: usize,
    end_col: usize,
) {
    for idx in start_col..end_col {
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

/// Processes a single radix-5 butterfly stage across all columns using scalar operations.
///
/// Implements the radix-5 DFT butterfly:
/// X[k] = sum_{j=0}^{4} x[j] * W_stage[j] * W_5^(k*j)
///
/// Where W_5 = exp(-2πi/5) is the primitive 5th root of unity, and
/// W_stage are the stage-specific twiddle factors.
#[inline(always)]
pub fn butterfly_5_scalar(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    butterfly_5_columns(data, stage_twiddles, num_columns, 0, num_columns);
}

/// Public API that dispatches to the best available SIMD implementation.
///
/// Compile-time selection:
/// - AVX+FMA: 4 columns at once with fused multiply-add
/// - AVX: 4 columns at once
/// - SSE3: 2 columns at once with addsub instruction
/// - SSE: 2 columns at once
/// - NEON: 2 columns at once
/// - Scalar (fallback): 1 column at a time
#[inline(always)]
pub(crate) fn butterfly_5(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        if num_columns >= 4 {
            unsafe {
                return butterfly_5_avx_fma(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        not(target_feature = "fma")
    ))]
    {
        if num_columns >= 4 {
            unsafe {
                return butterfly_5_avx(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    {
        if num_columns >= 2 {
            unsafe {
                return butterfly_5_sse3(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "sse3"),
        target_feature = "sse"
    ))]
    {
        if num_columns >= 2 {
            unsafe {
                return butterfly_5_sse(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if num_columns >= 2 {
            unsafe {
                return butterfly_5_neon(data, stage_twiddles, num_columns);
            }
        }
    }

    butterfly_5_scalar(data, stage_twiddles, num_columns);
}

#[cfg(test)]
mod tests {
    use super::{super::test_helpers::test_butterfly_against_scalar, *};

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_5_neon_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_5_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_5_neon(data, twiddles, num_columns);
            },
            5,
            4,
            "butterfly_5_neon",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    fn test_butterfly_5_sse_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_5_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_5_sse(data, twiddles, num_columns);
            },
            5,
            4,
            "butterfly_5_sse",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    fn test_butterfly_5_sse3_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_5_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_5_sse3(data, twiddles, num_columns);
            },
            5,
            4,
            "butterfly_5_sse3",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    fn test_butterfly_5_avx_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_5_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_5_avx(data, twiddles, num_columns);
            },
            5,
            4,
            "butterfly_5_avx",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    fn test_butterfly_5_avx_fma_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_5_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_5_avx_fma(data, twiddles, num_columns);
            },
            5,
            4,
            "butterfly_5_avx_fma",
        );
    }
}
