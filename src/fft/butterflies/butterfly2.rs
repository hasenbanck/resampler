#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::Complex32;

/// Processes a single radix-2 butterfly stage across all columns using scalar operations.
#[inline(always)]
pub fn butterfly_2_scalar(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    for idx in 0..num_columns {
        let d = data[idx + num_columns];
        let t = stage_twiddles[idx].mul(&d);
        let u = data[idx];
        data[idx] = u.add(&t);
        data[idx + num_columns] = u.sub(&t);
    }
}

/// SSE3 implementation: processes 2 columns at once (2 × Complex32 = 128 bits).
///
/// Complex multiplication (a+bi) * (c+di) = (ac-bd) + (ad+bc)i is implemented using:
/// - Shuffle to separate real/imaginary parts
/// - SSE3 addsub instruction for efficient combine
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse3")]
unsafe fn butterfly_2_sse2(
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

            // Duplicate real parts: [tw0.re, tw0.re, tw1.re, tw1.re]
            let tw_re = _mm_shuffle_ps(tw, tw, 0b10_10_00_00);
            // Duplicate imaginary parts: [tw0.im, tw0.im, tw1.im, tw1.im]
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

            // Store results.
            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm_storeu_ps(out_top_ptr, out_top);

            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm_storeu_ps(out_bot_ptr, out_bot);
        }
    }

    // Handle remaining columns with scalar code.
    for idx in simd_cols..num_columns {
        let d = data[idx + num_columns];
        let t = stage_twiddles[idx].mul(&d);
        let u = data[idx];
        data[idx] = u.add(&t);
        data[idx + num_columns] = u.sub(&t);
    }
}

/// AVX implementation: processes 4 columns at once (4 × Complex32 = 256 bits).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(dead_code)]
unsafe fn butterfly_2_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 4) * 4;

    for idx in (0..simd_cols).step_by(4) {
        unsafe {
            // Load 4 complex numbers (8 f32 values).
            let u_ptr = data.as_ptr().add(idx) as *const f32;
            let u = _mm256_loadu_ps(u_ptr);

            let d_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let d = _mm256_loadu_ps(d_ptr);

            let tw_ptr = stage_twiddles.as_ptr().add(idx) as *const f32;
            let tw = _mm256_loadu_ps(tw_ptr);

            // Complex multiply using AVX.
            // Duplicate real parts: [tw0.re, tw0.re, tw1.re, tw1.re, tw2.re, tw2.re, tw3.re, tw3.re]
            let tw_re = _mm256_shuffle_ps(tw, tw, 0b10_10_00_00);
            // Duplicate imaginary parts.
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

            // Store results.
            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(out_top_ptr, out_top);

            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(out_bot_ptr, out_bot);
        }
    }

    // Handle remaining columns with SSE2 or scalar.
    if num_columns - simd_cols >= 2 {
        unsafe {
            butterfly_2_sse2(
                &mut data[simd_cols..],
                &stage_twiddles[simd_cols..],
                num_columns - simd_cols,
            );
        }
    } else {
        for idx in simd_cols..num_columns {
            let d = data[idx + num_columns];
            let t = stage_twiddles[idx].mul(&d);
            let u = data[idx];
            data[idx] = u.add(&t);
            data[idx + num_columns] = u.sub(&t);
        }
    }
}

/// AVX2+FMA implementation: processes 4 columns at once using fused multiply-add.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn butterfly_2_avx2_fma(
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
            // t.re = tw.re*d.re - tw.im*d.im
            // t.im = tw.re*d.im + tw.im*d.re
            //
            // RustFFT optimization: use fmaddsub to fuse multiply and add/subtract
            // Pattern: output_im = tw.im * d_swapped, then fmaddsub(tw.re, d, output_im)

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

            // Store results
            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(out_top_ptr, out_top);

            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(out_bot_ptr, out_bot);
        }
    }

    // Handle remaining columns
    if num_columns - simd_cols >= 2 {
        unsafe {
            butterfly_2_sse2(
                &mut data[simd_cols..],
                &stage_twiddles[simd_cols..],
                num_columns - simd_cols,
            );
        }
    } else {
        for idx in simd_cols..num_columns {
            let d = data[idx + num_columns];
            let t = stage_twiddles[idx].mul(&d);
            let u = data[idx];
            data[idx] = u.add(&t);
            data[idx + num_columns] = u.sub(&t);
        }
    }
}

/// Public API that dispatches to the best available SIMD implementation.
///
/// Compile-time selection based on target features enabled in .cargo/config.toml:
/// - AVX2+FMA (best): 4 columns at once with fused multiply-add
/// - AVX: 4 columns at once
/// - SSE2 (baseline x86_64): 2 columns at once
/// - Scalar (fallback): 1 column at a time
#[inline(always)]
pub(crate) fn butterfly_2(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        if num_columns >= 4 {
            unsafe {
                return butterfly_2_avx2_fma(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        not(all(target_feature = "avx2", target_feature = "fma"))
    ))]
    {
        if num_columns >= 4 {
            unsafe {
                return butterfly_2_avx(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if num_columns >= 2 {
            unsafe {
                return butterfly_2_sse2(data, stage_twiddles, num_columns);
            }
        }
    }

    butterfly_2_scalar(data, stage_twiddles, num_columns);
}
