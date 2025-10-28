use crate::Complex32;

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
mod avx;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
mod sse;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx",
    any(not(target_feature = "fma"), test)
))]
use avx::butterfly_2_avx;
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use avx::butterfly_2_avx_fma;
#[cfg(target_arch = "aarch64")]
use neon::{butterfly_2_neon_x2, butterfly_2_neon_x4};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    any(not(target_feature = "sse3"), test)
))]
use sse::butterfly_2_sse;
#[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
use sse::butterfly_2_sse3;

/// Helper function to process a range of columns for radix-2 butterfly.
/// Processes columns from `start_col` to `end_col` (exclusive).
#[inline(always)]
pub(super) fn butterfly_2_columns(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
    start_col: usize,
    end_col: usize,
) {
    for idx in start_col..end_col {
        let d = data[idx + num_columns];
        let t = stage_twiddles[idx].mul(&d);
        let u = data[idx];
        data[idx] = u.add(&t);
        data[idx + num_columns] = u.sub(&t);
    }
}

/// Processes a single radix-2 butterfly stage across all columns using scalar operations.
#[inline(always)]
pub fn butterfly_2_scalar(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    butterfly_2_columns(data, stage_twiddles, num_columns, 0, num_columns);
}

/// Public API that dispatches to the best available SIMD implementation.
///
/// Compile-time selection:
/// - AVX+FMA: 4 columns at once with fused multiply-add
/// - AVX: 4 columns at once
/// - SSE3: 2 columns at once with addsub instruction
/// - SSE: 2 columns at once
/// - NEON x4: 4 columns at once (vld2q/vst2q)
/// - NEON x2: 2 columns at once (vld1q/vst1q)
/// - Scalar (fallback): 1 column at a time
#[inline(always)]
pub(crate) fn butterfly_2(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        if num_columns >= 4 {
            unsafe {
                return butterfly_2_avx_fma(data, stage_twiddles, num_columns);
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
                return butterfly_2_avx(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    {
        if num_columns >= 2 {
            unsafe {
                return butterfly_2_sse3(data, stage_twiddles, num_columns);
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
                return butterfly_2_sse(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if num_columns >= 4 {
            unsafe {
                return butterfly_2_neon_x4(data, stage_twiddles, num_columns);
            }
        }

        if num_columns >= 2 {
            unsafe {
                return butterfly_2_neon_x2(data, stage_twiddles, num_columns);
            }
        }
    }

    butterfly_2_scalar(data, stage_twiddles, num_columns);
}

#[cfg(test)]
mod tests {
    use super::{super::test_helpers::test_butterfly_against_scalar, *};

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_2_neon_x2_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_2_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_2_neon_x2(data, twiddles, num_columns);
            },
            2,
            1,
            "butterfly_2_neon_x2",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_2_neon_x4_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_2_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_2_neon_x4(data, twiddles, num_columns);
            },
            2,
            1,
            "butterfly_2_neon_x4",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    fn test_butterfly_2_sse_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_2_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_2_sse(data, twiddles, num_columns);
            },
            2,
            1,
            "butterfly_2_sse",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    fn test_butterfly_2_sse3_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_2_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_2_sse3(data, twiddles, num_columns);
            },
            2,
            1,
            "butterfly_2_sse3",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    fn test_butterfly_2_avx_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_2_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_2_avx(data, twiddles, num_columns);
            },
            2,
            1,
            "butterfly_2_avx",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    fn test_butterfly_2_avx_fma_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_2_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_2_avx_fma(data, twiddles, num_columns);
            },
            2,
            1,
            "butterfly_2_avx_fma",
        );
    }
}
