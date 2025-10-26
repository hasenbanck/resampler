#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::Complex32;

/// Helper function to process a range of columns for radix-2 butterfly.
/// Processes columns from `start_col` to `end_col` (exclusive).
#[inline(always)]
fn butterfly_2_columns(
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

/// Pure SSE implementation: processes 2 columns at once.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    any(not(target_feature = "sse3"), test)
))]
#[target_feature(enable = "sse")]
unsafe fn butterfly_2_sse(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 2) * 2;

    for idx in (0..simd_cols).step_by(2) {
        unsafe {
            // Load 2 complex numbers from data[idx] and data[idx+1]
            // Layout: [u0.re, u0.im, u1.re, u1.im]
            let u_ptr = data.as_ptr().add(idx) as *const f32;
            let u = _mm_loadu_ps(u_ptr);

            // Load 2 complex numbers from data[idx + num_columns]
            // Layout: [d0.re, d0.im, d1.re, d1.im]
            let d_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let d = _mm_loadu_ps(d_ptr);

            // Load 2 twiddle factors
            // Layout: [tw0.re, tw0.im, tw1.re, tw1.im]
            let tw_ptr = stage_twiddles.as_ptr().add(idx) as *const f32;
            let tw = _mm_loadu_ps(tw_ptr);

            // Complex multiply: t = tw * d
            // Result: t = (tw.re*d.re - tw.im*d.im) + i(tw.re*d.im + tw.im*d.re)

            // Duplicate real parts: [tw0.re, tw0.re, tw1.re, tw1.re]
            let tw_re = _mm_shuffle_ps(tw, tw, 0b10_10_00_00);
            // Duplicate imaginary parts: [tw0.im, tw0.im, tw1.im, tw1.im]
            let tw_im = _mm_shuffle_ps(tw, tw, 0b11_11_01_01);

            // Multiply by real parts: [tw.re*d.re, tw.re*d.im, ...]
            let prod_re = _mm_mul_ps(tw_re, d);

            // Swap d: [d.im, d.re, ...] for the imaginary multiplication
            let d_swap = _mm_shuffle_ps(d, d, 0b10_11_00_01);

            // Multiply by imaginary parts: [tw.im*d.im, tw.im*d.re, ...]
            let prod_im = _mm_mul_ps(tw_im, d_swap);

            // Emulate SSE3's addsub: [a0-b0, a1+b1, a2-b2, a3+b3]
            // We want: [prod_re[0]-prod_im[0], prod_re[1]+prod_im[1], prod_re[2]-prod_im[2], prod_re[3]+prod_im[3]]
            // Create a mask to negate elements at indices 0 and 2 (the real parts)
            // Mask layout: [sign_bit, 0, sign_bit, 0] to flip sign of real parts only
            let neg_mask = _mm_castsi128_ps(_mm_set_epi32(
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
            ));
            let prod_im_adjusted = _mm_xor_ps(prod_im, neg_mask);
            let t = _mm_add_ps(prod_re, prod_im_adjusted);

            // Butterfly: data[idx] = u + t, data[idx + num_columns] = u - t
            let out_top = _mm_add_ps(u, t);
            let out_bot = _mm_sub_ps(u, t);

            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm_storeu_ps(out_top_ptr, out_top);
            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm_storeu_ps(out_bot_ptr, out_bot);
        }
    }

    butterfly_2_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}

/// SSE3 implementation: processes 2 columns at once.
#[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
#[target_feature(enable = "sse3")]
unsafe fn butterfly_2_sse3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 2) * 2;

    for idx in (0..simd_cols).step_by(2) {
        unsafe {
            // Load 2 complex numbers from data[idx] and data[idx+1]
            // Layout: [u0.re, u0.im, u1.re, u1.im]
            let u_ptr = data.as_ptr().add(idx) as *const f32;
            let u = _mm_loadu_ps(u_ptr);

            // Load 2 complex numbers from data[idx + num_columns]
            // Layout: [d0.re, d0.im, d1.re, d1.im]
            let d_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let d = _mm_loadu_ps(d_ptr);

            // Load 2 twiddle factors.
            // Layout: [tw0.re, tw0.im, tw1.re, tw1.im]
            let tw_ptr = stage_twiddles.as_ptr().add(idx) as *const f32;
            let tw = _mm_loadu_ps(tw_ptr);

            // Complex multiply: t = tw * d
            // Result: t = (tw.re*d.re - tw.im*d.im) + i(tw.re*d.im + tw.im*d.re)

            // Duplicate real and imaginary parts.
            let tw_re = _mm_shuffle_ps(tw, tw, 0b10_10_00_00);
            let tw_im = _mm_shuffle_ps(tw, tw, 0b11_11_01_01);

            // Multiply by real parts: [tw.re*d.re, tw.re*d.im, ...]
            let prod_re = _mm_mul_ps(tw_re, d);

            // Swap d: [d.im, d.re, ...] for the imaginary multiplication.
            let d_swap = _mm_shuffle_ps(d, d, 0b10_11_00_01);

            // Multiply by imaginary parts: [tw.im*d.im, tw.im*d.re, ...]
            let prod_im = _mm_mul_ps(tw_im, d_swap);

            // Combine using SSE3 addsub: [a0-b0, a1+b1, a2-b2, a3+b3]
            // Produces: [tw.re*d.re - tw.im*d.im, tw.re*d.im + tw.im*d.re, ...]
            let t = _mm_addsub_ps(prod_re, prod_im);

            // Butterfly: data[idx] = u + t, data[idx + num_columns] = u - t
            let out_top = _mm_add_ps(u, t);
            let out_bot = _mm_sub_ps(u, t);

            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm_storeu_ps(out_top_ptr, out_top);
            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm_storeu_ps(out_bot_ptr, out_bot);
        }
    }

    butterfly_2_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}

/// AVX implementation: processes 4 columns at once.
#[cfg(any(all(
    target_arch = "x86_64",
    target_feature = "avx",
    any(not(target_feature = "fma"), test)
)))]
#[target_feature(enable = "avx")]
unsafe fn butterfly_2_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 4) * 4;

    for idx in (0..simd_cols).step_by(4) {
        unsafe {
            let u_ptr = data.as_ptr().add(idx) as *const f32;
            let u = _mm256_loadu_ps(u_ptr);

            let d_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let d = _mm256_loadu_ps(d_ptr);

            let tw_ptr = stage_twiddles.as_ptr().add(idx) as *const f32;
            let tw = _mm256_loadu_ps(tw_ptr);

            // Complex multiply using AVX.
            // Duplicate real and imaginary parts.
            let tw_re = _mm256_shuffle_ps(tw, tw, 0b10_10_00_00);
            let tw_im = _mm256_shuffle_ps(tw, tw, 0b11_11_01_01);
            let prod_re = _mm256_mul_ps(tw_re, d);

            // Swap real/imag in d.
            let d_swap = _mm256_shuffle_ps(d, d, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(tw_im, d_swap);

            // Combine using AVX addsub for efficient complex multiply.
            let t = _mm256_addsub_ps(prod_re, prod_im);

            // Butterfly operations.
            let out_top = _mm256_add_ps(u, t);
            let out_bot = _mm256_sub_ps(u, t);

            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(out_top_ptr, out_top);
            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(out_bot_ptr, out_bot);
        }
    }

    butterfly_2_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}

/// AVX+FMA implementation: processes 4 columns at once using fused multiply-add.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[target_feature(enable = "avx,fma")]
unsafe fn butterfly_2_avx_fma(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 4) * 4;

    for idx in (0..simd_cols).step_by(4) {
        unsafe {
            let u_ptr = data.as_ptr().add(idx) as *const f32;
            let u = _mm256_loadu_ps(u_ptr);
            let d_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let d = _mm256_loadu_ps(d_ptr);
            let tw_ptr = stage_twiddles.as_ptr().add(idx) as *const f32;
            let tw = _mm256_loadu_ps(tw_ptr);

            // Complex multiply using FMA: t = tw * d
            let tw_re = _mm256_shuffle_ps(tw, tw, 0b10_10_00_00);
            let tw_im = _mm256_shuffle_ps(tw, tw, 0b11_11_01_01);

            // Swap d for imaginary component: [d.im, d.re, d.im, d.re, ...]
            let d_swap = _mm256_shuffle_ps(d, d, 0b10_11_00_01);

            // Multiply imaginary parts: [tw.im*d.im, tw.im*d.re, ...]
            let prod_im = _mm256_mul_ps(tw_im, d_swap);

            // FMA: fmaddsub computes [a*b - c, a*b + c, a*b - c, a*b + c, ...]
            // Result: [tw.re*d.re - tw.im*d.im, tw.re*d.im + tw.im*d.re, ...]
            let t = _mm256_fmaddsub_ps(tw_re, d, prod_im);

            // Butterfly operations
            let out_top = _mm256_add_ps(u, t);
            let out_bot = _mm256_sub_ps(u, t);

            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(out_top_ptr, out_top);
            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(out_bot_ptr, out_bot);
        }
    }

    butterfly_2_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}

/// Public API that dispatches to the best available SIMD implementation.
///
/// Compile-time selection:
/// - AVX+FMA: 4 columns at once with fused multiply-add
/// - AVX: 4 columns at once
/// - SSE3: 2 columns at once with addsub instruction
/// - SSE: 2 columns at once
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

    butterfly_2_scalar(data, stage_twiddles, num_columns);
}

#[cfg(test)]
mod tests {
    use super::{super::test_helpers::test_butterfly_against_scalar, *};

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
