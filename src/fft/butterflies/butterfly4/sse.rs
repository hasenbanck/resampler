#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::Complex32;

/// Pure SSE implementation: processes 2 columns at once.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    any(not(target_feature = "sse3"), test)
))]
#[target_feature(enable = "sse")]
pub(super) unsafe fn butterfly_4_sse(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 2) * 2;

    for idx in (0..simd_cols).step_by(2) {
        unsafe {
            // Load 2 complex numbers from each row.
            // Layout: [x0[0].re, x0[0].im, x0[1].re, x0[1].im]
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = _mm_loadu_ps(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = _mm_loadu_ps(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = _mm_loadu_ps(x2_ptr);

            let x3_ptr = data.as_ptr().add(idx + 3 * num_columns) as *const f32;
            let x3 = _mm_loadu_ps(x3_ptr);

            // Load 6 twiddle factors: w1[0], w2[0], w3[0], w1[1], w2[1], w3[1]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 3) as *const f32;
            let tw_0 = _mm_loadu_ps(tw_ptr);
            // Layout: [w3[0].re, w3[0].im, w1[1].re, w1[1].im]
            let tw_1 = _mm_loadu_ps(tw_ptr.add(4));
            // Layout: [w2[1].re, w2[1].im, w3[1].re, w3[1].im]
            let tw_2 = _mm_loadu_ps(tw_ptr.add(8));

            // Extract w1, w2, w3 for both columns.
            // w1 = [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            let w1 = _mm_shuffle_ps(tw_0, tw_1, 0b11_10_01_00);
            // w2 = [w2[0].re, w2[0].im, w2[1].re, w2[1].im]
            let w2 = _mm_shuffle_ps(tw_0, tw_2, 0b01_00_11_10);
            // w3 = [w3[0].re, w3[0].im, w3[1].re, w3[1].im]
            let w3 = _mm_shuffle_ps(tw_1, tw_2, 0b11_10_01_00);

            // Complex multiply: t1 = x1 * w1
            let w1_re = _mm_shuffle_ps(w1, w1, 0b10_10_00_00);
            let w1_im = _mm_shuffle_ps(w1, w1, 0b11_11_01_01);
            let prod_re = _mm_mul_ps(w1_re, x1);
            let x1_swap = _mm_shuffle_ps(x1, x1, 0b10_11_00_01);
            let prod_im = _mm_mul_ps(w1_im, x1_swap);

            // Emulate addsub for complex multiply.
            let neg_mask = _mm_castsi128_ps(_mm_set_epi32(
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
            ));
            let prod_im_adjusted = _mm_xor_ps(prod_im, neg_mask);
            let t1 = _mm_add_ps(prod_re, prod_im_adjusted);

            // Complex multiply: t2 = x2 * w2
            let w2_re = _mm_shuffle_ps(w2, w2, 0b10_10_00_00);
            let w2_im = _mm_shuffle_ps(w2, w2, 0b11_11_01_01);
            let prod_re = _mm_mul_ps(w2_re, x2);
            let x2_swap = _mm_shuffle_ps(x2, x2, 0b10_11_00_01);
            let prod_im = _mm_mul_ps(w2_im, x2_swap);
            let prod_im_adjusted = _mm_xor_ps(prod_im, neg_mask);
            let t2 = _mm_add_ps(prod_re, prod_im_adjusted);

            // Complex multiply: t3 = x3 * w3
            let w3_re = _mm_shuffle_ps(w3, w3, 0b10_10_00_00);
            let w3_im = _mm_shuffle_ps(w3, w3, 0b11_11_01_01);
            let prod_re = _mm_mul_ps(w3_re, x3);
            let x3_swap = _mm_shuffle_ps(x3, x3, 0b10_11_00_01);
            let prod_im = _mm_mul_ps(w3_im, x3_swap);
            let prod_im_adjusted = _mm_xor_ps(prod_im, neg_mask);
            let t3 = _mm_add_ps(prod_re, prod_im_adjusted);

            // t0 = x0 (no twiddle)
            let t0 = x0;

            // Compute intermediate values for radix-4 butterfly.
            let u0 = _mm_add_ps(t0, t2); // u0 = t0 + t2
            let u1 = _mm_sub_ps(t0, t2); // u1 = t0 - t2
            let u2 = _mm_add_ps(t1, t3); // u2 = t1 + t3
            let u3 = _mm_sub_ps(t1, t3); // u3 = t1 - t3

            // Multiply u3 by -j: -j * (a + bi) = b - ai
            // Swap real/imag and negate new imaginary part: [u3.im, -u3.re]
            let u3_neg_j = _mm_shuffle_ps(u3, u3, 0b10_11_00_01);
            let sign_flip = _mm_castsi128_ps(_mm_set_epi32(
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
            ));
            let u3_neg_j = _mm_xor_ps(u3_neg_j, sign_flip);

            // Multiply u3 by +j: +j * (a + bi) = -b + ai
            // Swap real/imag and negate new real part: [-u3.im, u3.re]
            let u3_pos_j = _mm_shuffle_ps(u3, u3, 0b10_11_00_01);
            let sign_flip_re = _mm_castsi128_ps(_mm_set_epi32(
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
            ));
            let u3_pos_j = _mm_xor_ps(u3_pos_j, sign_flip_re);

            // Combine to produce outputs.
            let y0 = _mm_add_ps(u0, u2); // y0 = u0 + u2
            let y1 = _mm_add_ps(u1, u3_neg_j); // y1 = u1 - j*u3
            let y2 = _mm_sub_ps(u0, u2); // y2 = u0 - u2
            let y3 = _mm_add_ps(u1, u3_pos_j); // y3 = u1 + j*u3

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm_storeu_ps(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            _mm_storeu_ps(y3_ptr, y3);
        }
    }

    super::butterfly_4_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}

/// SSE3 implementation: processes 2 columns at once.
#[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
#[target_feature(enable = "sse3")]
pub(super) unsafe fn butterfly_4_sse3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 2) * 2;

    for idx in (0..simd_cols).step_by(2) {
        unsafe {
            // Load 2 complex numbers from each row.
            // Layout: [x0[0].re, x0[0].im, x0[1].re, x0[1].im]
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = _mm_loadu_ps(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = _mm_loadu_ps(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = _mm_loadu_ps(x2_ptr);

            let x3_ptr = data.as_ptr().add(idx + 3 * num_columns) as *const f32;
            let x3 = _mm_loadu_ps(x3_ptr);

            // Load 6 twiddle factors: w1[0], w2[0], w3[0], w1[1], w2[1], w3[1]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 3) as *const f32;
            let tw_0 = _mm_loadu_ps(tw_ptr);
            // Layout: [w3[0].re, w3[0].im, w1[1].re, w1[1].im]
            let tw_1 = _mm_loadu_ps(tw_ptr.add(4));
            // Layout: [w2[1].re, w2[1].im, w3[1].re, w3[1].im]
            let tw_2 = _mm_loadu_ps(tw_ptr.add(8));

            // Extract w1, w2, w3 for both columns.
            // w1 = [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            let w1 = _mm_shuffle_ps(tw_0, tw_1, 0b11_10_01_00);
            // w2 = [w2[0].re, w2[0].im, w2[1].re, w2[1].im]
            let w2 = _mm_shuffle_ps(tw_0, tw_2, 0b01_00_11_10);
            // w3 = [w3[0].re, w3[0].im, w3[1].re, w3[1].im]
            let w3 = _mm_shuffle_ps(tw_1, tw_2, 0b11_10_01_00);

            // Complex multiply: t1 = x1 * w1
            let w1_re = _mm_shuffle_ps(w1, w1, 0b10_10_00_00);
            let w1_im = _mm_shuffle_ps(w1, w1, 0b11_11_01_01);
            let prod_re = _mm_mul_ps(w1_re, x1);
            let x1_swap = _mm_shuffle_ps(x1, x1, 0b10_11_00_01);
            let prod_im = _mm_mul_ps(w1_im, x1_swap);

            // Use SSE3 addsub for complex multiply.
            let t1 = _mm_addsub_ps(prod_re, prod_im);

            // Complex multiply: t2 = x2 * w2
            let w2_re = _mm_shuffle_ps(w2, w2, 0b10_10_00_00);
            let w2_im = _mm_shuffle_ps(w2, w2, 0b11_11_01_01);
            let prod_re = _mm_mul_ps(w2_re, x2);
            let x2_swap = _mm_shuffle_ps(x2, x2, 0b10_11_00_01);
            let prod_im = _mm_mul_ps(w2_im, x2_swap);
            let t2 = _mm_addsub_ps(prod_re, prod_im);

            // Complex multiply: t3 = x3 * w3
            let w3_re = _mm_shuffle_ps(w3, w3, 0b10_10_00_00);
            let w3_im = _mm_shuffle_ps(w3, w3, 0b11_11_01_01);
            let prod_re = _mm_mul_ps(w3_re, x3);
            let x3_swap = _mm_shuffle_ps(x3, x3, 0b10_11_00_01);
            let prod_im = _mm_mul_ps(w3_im, x3_swap);
            let t3 = _mm_addsub_ps(prod_re, prod_im);

            // t0 = x0 (no twiddle)
            let t0 = x0;

            // Compute intermediate values for radix-4 butterfly.
            let u0 = _mm_add_ps(t0, t2); // u0 = t0 + t2
            let u1 = _mm_sub_ps(t0, t2); // u1 = t0 - t2
            let u2 = _mm_add_ps(t1, t3); // u2 = t1 + t3
            let u3 = _mm_sub_ps(t1, t3); // u3 = t1 - t3

            // Multiply u3 by -j: -j * (a + bi) = b - ai
            // Swap real/imag and negate new imaginary part: [u3.im, -u3.re]
            let u3_neg_j = _mm_shuffle_ps(u3, u3, 0b10_11_00_01);
            let sign_flip = _mm_castsi128_ps(_mm_set_epi32(
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
            ));
            let u3_neg_j = _mm_xor_ps(u3_neg_j, sign_flip);

            // Multiply u3 by +j: +j * (a + bi) = -b + ai
            // Swap real/imag and negate new real part: [-u3.im, u3.re]
            let u3_pos_j = _mm_shuffle_ps(u3, u3, 0b10_11_00_01);
            let sign_flip_re = _mm_castsi128_ps(_mm_set_epi32(
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
            ));
            let u3_pos_j = _mm_xor_ps(u3_pos_j, sign_flip_re);

            // Combine to produce outputs.
            let y0 = _mm_add_ps(u0, u2); // y0 = u0 + u2
            let y1 = _mm_add_ps(u1, u3_neg_j); // y1 = u1 - j*u3
            let y2 = _mm_sub_ps(u0, u2); // y2 = u0 - u2
            let y3 = _mm_add_ps(u1, u3_pos_j); // y3 = u1 + j*u3

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm_storeu_ps(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            _mm_storeu_ps(y3_ptr, y3);
        }
    }

    super::butterfly_4_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}
