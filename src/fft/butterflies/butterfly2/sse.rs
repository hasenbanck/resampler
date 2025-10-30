#[cfg(any(not(feature = "no_std"), target_feature = "sse"))]
use crate::Complex32;

/// Pure SSE implementation: processes 2 columns at once.
#[cfg(any(
    test,
    not(feature = "no_std"),
    all(target_feature = "sse", not(target_feature = "sse3"))
))]
#[target_feature(enable = "sse")]
pub(super) unsafe fn butterfly_2_sse(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    start_col: usize,
    num_columns: usize,
) {
    use core::arch::x86_64::*;

    let simd_cols = ((num_columns - start_col) / 2) * 2;

    for idx in (start_col..start_col + simd_cols).step_by(2) {
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

    super::butterfly_2_scalar(data, stage_twiddles, start_col + simd_cols, num_columns);
}

/// SSE3 implementation: processes 2 columns at once.
#[cfg(any(not(feature = "no_std"), target_feature = "sse3"))]
#[target_feature(enable = "sse3")]
pub(super) unsafe fn butterfly_2_sse3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    start_col: usize,
    num_columns: usize,
) {
    use core::arch::x86_64::*;

    let simd_cols = ((num_columns - start_col) / 2) * 2;

    for idx in (start_col..start_col + simd_cols).step_by(2) {
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

    super::butterfly_2_scalar(data, stage_twiddles, start_col + simd_cols, num_columns);
}
