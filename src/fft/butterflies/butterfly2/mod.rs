use crate::Complex32;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse;

/// Processes a single radix-2 butterfly stage across all columns using scalar operations.
#[inline(always)]
pub fn butterfly_2_scalar(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    start_col: usize,
    num_columns: usize,
) {
    for idx in start_col..num_columns {
        let d = data[idx + num_columns];
        let t = stage_twiddles[idx].mul(&d);
        let u = data[idx];
        data[idx] = u.add(&t);
        data[idx + num_columns] = u.sub(&t);
    }
}

/// Dispatches to the best available SIMD implementation.
#[inline(always)]
pub(crate) fn butterfly_2_dispatch(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        if num_columns >= 4 {
            return unsafe { avx::butterfly_2_avx_fma(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        not(target_feature = "fma")
    ))]
    {
        if num_columns >= 4 {
            return unsafe { avx::butterfly_2_avx(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    {
        if num_columns >= 2 {
            return unsafe { sse::butterfly_2_sse3(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse",
        not(target_feature = "sse3")
    ))]
    {
        if num_columns >= 2 {
            return unsafe { sse::butterfly_2_sse(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if num_columns >= 4 {
            return unsafe { neon::butterfly_2_neon_x4(data, stage_twiddles, 0, num_columns) };
        }

        if num_columns >= 2 {
            return unsafe { neon::butterfly_2_neon_x2(data, stage_twiddles, 0, num_columns) };
        }
    }

    butterfly_2_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_2_dispatch_avx_fma(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 4 {
        return unsafe { avx::butterfly_2_avx_fma(data, stage_twiddles, 0, num_columns) };
    }

    if num_columns >= 2 {
        return unsafe { sse::butterfly_2_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_2_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_2_dispatch_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 4 {
        return unsafe { avx::butterfly_2_avx(data, stage_twiddles, 0, num_columns) };
    }

    if num_columns >= 2 {
        return unsafe { sse::butterfly_2_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_2_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_2_dispatch_sse3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 2 {
        return unsafe { sse::butterfly_2_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_2_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_2_dispatch_sse(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 2 {
        return unsafe { sse::butterfly_2_sse(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_2_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(test)]
mod tests {
    use super::{super::test_helpers::test_butterfly_against_scalar, *};

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_2_neon_x2_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_2_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                neon::butterfly_2_neon_x2(data, twiddles, 0, num_columns);
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
            |data, twiddles, num_columns| butterfly_2_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                neon::butterfly_2_neon_x4(data, twiddles, 0, num_columns);
            },
            2,
            1,
            "butterfly_2_neon_x4",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "sse")
    ))]
    fn test_butterfly_2_sse_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_2_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                sse::butterfly_2_sse(data, twiddles, 0, num_columns);
            },
            2,
            1,
            "butterfly_2_sse",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "sse3")
    ))]
    fn test_butterfly_2_sse3_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_2_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                sse::butterfly_2_sse3(data, twiddles, 0, num_columns);
            },
            2,
            1,
            "butterfly_2_sse3",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "avx")
    ))]
    fn test_butterfly_2_avx_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_2_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                avx::butterfly_2_avx(data, twiddles, 0, num_columns);
            },
            2,
            1,
            "butterfly_2_avx",
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
    fn test_butterfly_2_avx_fma_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_2_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                avx::butterfly_2_avx_fma(data, twiddles, 0, num_columns);
            },
            2,
            1,
            "butterfly_2_avx_fma",
        );
    }
}
