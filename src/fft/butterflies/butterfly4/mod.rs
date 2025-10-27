use crate::Complex32;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse;

#[cfg(any(all(
    target_arch = "x86_64",
    target_feature = "avx",
    any(not(target_feature = "fma"), test)
)))]
use avx::butterfly_4_avx;
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use avx::butterfly_4_avx_fma;
#[cfg(target_arch = "aarch64")]
use neon::butterfly_4_neon_x2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    any(not(target_feature = "sse3"), test)
))]
use sse::butterfly_4_sse;
#[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
use sse::butterfly_4_sse3;

/// Helper function to process a range of columns for radix-4 butterfly.
/// Processes columns from `start_col` to `end_col` (exclusive).
#[inline(always)]
pub(super) fn butterfly_4_columns(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
    start_col: usize,
    end_col: usize,
) {
    for idx in start_col..end_col {
        let i0 = idx;
        let i1 = idx + num_columns;
        let i2 = idx + 2 * num_columns;
        let i3 = idx + 3 * num_columns;

        let x0 = data[i0];
        let x1 = data[i1];
        let x2 = data[i2];
        let x3 = data[i3];

        // Apply stage twiddle factors.
        let w1 = stage_twiddles[idx * 3];
        let w2 = stage_twiddles[idx * 3 + 1];
        let w3 = stage_twiddles[idx * 3 + 2];

        let t0 = x0;
        let t1 = x1.mul(&w1);
        let t2 = x2.mul(&w2);
        let t3 = x3.mul(&w3);

        // Compute radix-4 butterfly using factored form:
        // y[0] = t[0] + t[1] + t[2] + t[3]
        // y[1] = t[0] - j*t[1] - t[2] + j*t[3]
        // y[2] = t[0] - t[1] + t[2] - t[3]
        // y[3] = t[0] + j*t[1] - t[2] - j*t[3]
        let u0 = t0.add(&t2); // t0 + t2
        let u1 = t0.sub(&t2); // t0 - t2
        let u2 = t1.add(&t3); // t1 + t3
        let u3 = t1.sub(&t3); // t1 - t3

        // Multiply u3 by -j for y1: -j * (a + bi) = b - ai
        let u3_neg_j = Complex32::new(u3.im, -u3.re);
        // Multiply u3 by +j for y3: +j * (a + bi) = -b + ai
        let u3_pos_j = Complex32::new(-u3.im, u3.re);

        // Combine to produce outputs.
        let y0 = u0.add(&u2); // u0 + u2
        let y1 = u1.add(&u3_neg_j); // u1 - j*u3
        let y2 = u0.sub(&u2); // u0 - u2
        let y3 = u1.add(&u3_pos_j); // u1 + j*u3

        // Store results back.
        data[i0] = y0;
        data[i1] = y1;
        data[i2] = y2;
        data[i3] = y3;
    }
}

/// Processes a single radix-4 butterfly stage across all columns using scalar operations.
///
/// Implements the radix-4 DFT butterfly:
/// Y[0] = x[0] + x[1]*W1 + x[2]*W2 + x[3]*W3
/// Y[1] = x[0] - j*x[1]*W1 - x[2]*W2 + j*x[3]*W3
/// Y[2] = x[0] - x[1]*W1 + x[2]*W2 - x[3]*W3
/// Y[3] = x[0] + j*x[1]*W1 - x[2]*W2 - j*x[3]*W3
///
/// Where W1, W2, W3 are the stage-specific twiddle factors.
#[inline(always)]
pub fn butterfly_4_scalar(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    butterfly_4_columns(data, stage_twiddles, num_columns, 0, num_columns);
}

/// Public API that dispatches to the best available SIMD implementation.
///
/// Compile-time selection:
/// - AVX+FMA: 4 columns at once with fused multiply-add
/// - AVX: 4 columns at once
/// - SSE3: 2 columns at once with addsub instruction
/// - SSE: 2 columns at once
/// - NEON x2: 2 columns at once (vld1q/vst1q with vuzpq/vzipq)
/// - Scalar (fallback): 1 column at a time
#[inline(always)]
pub(crate) fn butterfly_4(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        if num_columns >= 4 {
            unsafe {
                return butterfly_4_avx_fma(data, stage_twiddles, num_columns);
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
                return butterfly_4_avx(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    {
        if num_columns >= 2 {
            unsafe {
                return butterfly_4_sse3(data, stage_twiddles, num_columns);
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
                return butterfly_4_sse(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if num_columns >= 2 {
            unsafe {
                return butterfly_4_neon_x2(data, stage_twiddles, num_columns);
            }
        }
    }

    butterfly_4_scalar(data, stage_twiddles, num_columns);
}

#[cfg(test)]
mod tests {
    use super::{super::test_helpers::test_butterfly_against_scalar, *};

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_4_neon_x2_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_4_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_4_neon_x2(data, twiddles, num_columns);
            },
            4,
            3,
            "butterfly_4_neon_x2",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    fn test_butterfly_4_sse_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_4_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_4_sse(data, twiddles, num_columns);
            },
            4,
            3,
            "butterfly_4_sse",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    fn test_butterfly_4_sse3_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_4_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_4_sse3(data, twiddles, num_columns);
            },
            4,
            3,
            "butterfly_4_sse3",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    fn test_butterfly_4_avx_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_4_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_4_avx(data, twiddles, num_columns);
            },
            4,
            3,
            "butterfly_4_avx",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    fn test_butterfly_4_avx_fma_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_4_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_4_avx_fma(data, twiddles, num_columns);
            },
            4,
            3,
            "butterfly_4_avx_fma",
        );
    }
}
