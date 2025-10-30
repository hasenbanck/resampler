use crate::Complex32;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse;

// Primitive 3rd roots of unity: W_3 = exp(-2πi/3)
pub(super) const W3_1_RE: f32 = -0.5;
pub(super) const W3_1_IM: f32 = -0.8660254; // -√3/2
pub(super) const W3_2_RE: f32 = -0.5;
pub(super) const W3_2_IM: f32 = 0.8660254; // √3/2

/// Processes a single radix-3 butterfly stage across all columns using scalar operations.
///
/// Implements the radix-3 DFT butterfly:
/// X[0] = x[0] + x[1]*W1 + x[2]*W2
/// X[1] = x[0] + x[1]*W1*W_3^1 + x[2]*W2*W_3^2
/// X[2] = x[0] + x[1]*W1*W_3^2 + x[2]*W2*W_3^1
///
/// Where W_3 = exp(-2πi/3) is the primitive 3rd root of unity, and
/// W1, W2 are the stage-specific twiddle factors.
#[inline(always)]
pub fn butterfly_3_scalar(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    start_col: usize,
    num_columns: usize,
) {
    for idx in start_col..num_columns {
        let x0 = data[idx];
        let x1 = data[idx + num_columns];
        let x2 = data[idx + 2 * num_columns];

        // Apply stage twiddle factors.
        let w1 = stage_twiddles[idx * 2];
        let w2 = stage_twiddles[idx * 2 + 1];

        let t1 = x1.mul(&w1);
        let t2 = x2.mul(&w2);

        // X0 = x0 + t1 + t2
        data[idx] = x0.add(&t1).add(&t2);

        // X1 = x0 + t1*W_3^1 + t2*W_3^2
        // Multiply t1 by W_3^1 = -0.5 - i*√3/2
        let t1_w31 = Complex32::new(
            t1.re * W3_1_RE - t1.im * W3_1_IM,
            t1.re * W3_1_IM + t1.im * W3_1_RE,
        );
        // Multiply t2 by W_3^2 = -0.5 + i*√3/2
        let t2_w32 = Complex32::new(
            t2.re * W3_2_RE - t2.im * W3_2_IM,
            t2.re * W3_2_IM + t2.im * W3_2_RE,
        );
        data[idx + num_columns] = x0.add(&t1_w31).add(&t2_w32);

        // X2 = x0 + t1*W_3^2 + t2*W_3^1
        // Multiply t1 by W_3^2
        let t1_w32 = Complex32::new(
            t1.re * W3_2_RE - t1.im * W3_2_IM,
            t1.re * W3_2_IM + t1.im * W3_2_RE,
        );
        // Multiply t2 by W_3^1
        let t2_w31 = Complex32::new(
            t2.re * W3_1_RE - t2.im * W3_1_IM,
            t2.re * W3_1_IM + t2.im * W3_1_RE,
        );
        data[idx + 2 * num_columns] = x0.add(&t1_w32).add(&t2_w31);
    }
}

/// Dispatches to the best available SIMD implementation.
#[inline(always)]
pub(crate) fn butterfly_3_dispatch(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        if num_columns >= 4 {
            return unsafe { avx::butterfly_3_avx_fma(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        not(target_feature = "fma")
    ))]
    {
        if num_columns >= 4 {
            return unsafe { avx::butterfly_3_avx(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    {
        if num_columns >= 2 {
            return unsafe { sse::butterfly_3_sse3(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse",
        not(target_feature = "sse3")
    ))]
    {
        if num_columns >= 2 {
            return unsafe { sse::butterfly_3_sse(data, stage_twiddles, 0, num_columns) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if num_columns >= 4 {
            return unsafe { neon::butterfly_3_neon_x4(data, stage_twiddles, 0, num_columns) };
        }

        if num_columns >= 2 {
            return unsafe { neon::butterfly_3_neon_x2(data, stage_twiddles, 0, num_columns) };
        }
    }

    butterfly_3_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_3_dispatch_avx_fma(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 4 {
        return unsafe { avx::butterfly_3_avx_fma(data, stage_twiddles, 0, num_columns) };
    }

    if num_columns >= 2 {
        return unsafe { sse::butterfly_3_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_3_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_3_dispatch_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 4 {
        return unsafe { avx::butterfly_3_avx(data, stage_twiddles, 0, num_columns) };
    }

    if num_columns >= 2 {
        return unsafe { sse::butterfly_3_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_3_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_3_dispatch_sse3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 2 {
        return unsafe { sse::butterfly_3_sse3(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_3_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_3_dispatch_sse(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    if num_columns >= 2 {
        return unsafe { sse::butterfly_3_sse(data, stage_twiddles, 0, num_columns) };
    }

    butterfly_3_scalar(data, stage_twiddles, 0, num_columns);
}

#[cfg(test)]
mod tests {
    use super::{super::test_helpers::test_butterfly_against_scalar, *};

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_3_neon_x2_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_3_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                neon::butterfly_3_neon_x2(data, twiddles, 0, num_columns);
            },
            3,
            2,
            "butterfly_3_neon_x2",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_butterfly_3_neon_x4_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_3_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                neon::butterfly_3_neon_x4(data, twiddles, 0, num_columns);
            },
            3,
            2,
            "butterfly_3_neon_x4",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "sse")
    ))]
    fn test_butterfly_3_sse_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_3_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                sse::butterfly_3_sse(data, twiddles, 0, num_columns);
            },
            3,
            2,
            "butterfly_3_sse",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "sse3")
    ))]
    fn test_butterfly_3_sse3_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_3_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                sse::butterfly_3_sse3(data, twiddles, 0, num_columns);
            },
            3,
            2,
            "butterfly_3_sse3",
        );
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        any(not(feature = "no_std"), target_feature = "avx")
    ))]
    fn test_butterfly_3_avx_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_3_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                avx::butterfly_3_avx(data, twiddles, 0, num_columns);
            },
            3,
            2,
            "butterfly_3_avx",
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
    fn test_butterfly_3_avx_fma_vs_scalar() {
        test_butterfly_against_scalar(
            |data, twiddles, num_columns| butterfly_3_scalar(data, twiddles, 0, num_columns),
            |data, twiddles, num_columns| unsafe {
                avx::butterfly_3_avx_fma(data, twiddles, 0, num_columns);
            },
            3,
            2,
            "butterfly_3_avx_fma",
        );
    }
}
