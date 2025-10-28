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
pub(super) unsafe fn butterfly_3_sse(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 2) * 2;

    // Broadcast W3 constants for SIMD operations.
    let w3_1_re = _mm_set1_ps(super::W3_1_RE);
    let w3_1_im = _mm_set1_ps(super::W3_1_IM);
    let w3_2_re = _mm_set1_ps(super::W3_2_RE);
    let w3_2_im = _mm_set1_ps(super::W3_2_IM);

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

            // Load 4 twiddle factors: w1[0], w2[0], w1[1], w2[1]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            let tw1_ptr = stage_twiddles.as_ptr().add(idx * 2) as *const f32;
            let tw_0 = _mm_loadu_ps(tw1_ptr);
            // Layout: [w1[1].re, w1[1].im, w2[1].re, w2[1].im]
            let tw_1 = _mm_loadu_ps(tw1_ptr.add(4));

            // Extract w1 and w2 for both columns.
            // w1 = [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            let w1 = _mm_shuffle_ps(tw_0, tw_1, 0b01_00_01_00);
            // w2 = [w2[0].re, w2[0].im, w2[1].re, w2[1].im]
            let w2 = _mm_shuffle_ps(tw_0, tw_1, 0b11_10_11_10);

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

            // Y0 = x0 + t1 + t2
            let y0 = _mm_add_ps(_mm_add_ps(x0, t1), t2);

            // Y1 = x0 + t1*W_3^1 + t2*W_3^2
            // t1*W_3^1: complex multiply t1 by (super::W3_1_RE, super::W3_1_IM)
            let t1_re = _mm_shuffle_ps(t1, t1, 0b10_10_00_00);
            let t1_im = _mm_shuffle_ps(t1, t1, 0b11_11_01_01);
            let t1w31_re = _mm_sub_ps(_mm_mul_ps(t1_re, w3_1_re), _mm_mul_ps(t1_im, w3_1_im));
            let t1w31_im = _mm_add_ps(_mm_mul_ps(t1_re, w3_1_im), _mm_mul_ps(t1_im, w3_1_re));
            let t1_w31 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w31_re, t1w31_im),
                _mm_unpackhi_ps(t1w31_re, t1w31_im),
            );

            // t2*W_3^2: complex multiply t2 by (super::W3_2_RE, super::W3_2_IM)
            let t2_re = _mm_shuffle_ps(t2, t2, 0b10_10_00_00);
            let t2_im = _mm_shuffle_ps(t2, t2, 0b11_11_01_01);
            let t2w32_re = _mm_sub_ps(_mm_mul_ps(t2_re, w3_2_re), _mm_mul_ps(t2_im, w3_2_im));
            let t2w32_im = _mm_add_ps(_mm_mul_ps(t2_re, w3_2_im), _mm_mul_ps(t2_im, w3_2_re));
            let t2_w32 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w32_re, t2w32_im),
                _mm_unpackhi_ps(t2w32_re, t2w32_im),
            );

            let y1 = _mm_add_ps(_mm_add_ps(x0, t1_w31), t2_w32);

            // Y2 = x0 + t1*W_3^2 + t2*W_3^1
            // t1*W_3^2: complex multiply t1 by (super::W3_2_RE, super::W3_2_IM)
            let t1w32_re = _mm_sub_ps(_mm_mul_ps(t1_re, w3_2_re), _mm_mul_ps(t1_im, w3_2_im));
            let t1w32_im = _mm_add_ps(_mm_mul_ps(t1_re, w3_2_im), _mm_mul_ps(t1_im, w3_2_re));
            let t1_w32 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w32_re, t1w32_im),
                _mm_unpackhi_ps(t1w32_re, t1w32_im),
            );

            // t2*W_3^1: complex multiply t2 by (super::W3_1_RE, super::W3_1_IM)
            let t2w31_re = _mm_sub_ps(_mm_mul_ps(t2_re, w3_1_re), _mm_mul_ps(t2_im, w3_1_im));
            let t2w31_im = _mm_add_ps(_mm_mul_ps(t2_re, w3_1_im), _mm_mul_ps(t2_im, w3_1_re));
            let t2_w31 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w31_re, t2w31_im),
                _mm_unpackhi_ps(t2w31_re, t2w31_im),
            );

            let y2 = _mm_add_ps(_mm_add_ps(x0, t1_w32), t2_w31);

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm_storeu_ps(y2_ptr, y2);
        }
    }

    super::butterfly_3_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}

/// SSE3 implementation: processes 2 columns at once.
#[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
#[target_feature(enable = "sse3")]
pub(super) unsafe fn butterfly_3_sse3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 2) * 2;

    // Broadcast W3 constants for SIMD operations.
    let w3_1_re = _mm_set1_ps(super::W3_1_RE);
    let w3_1_im = _mm_set1_ps(super::W3_1_IM);
    let w3_2_re = _mm_set1_ps(super::W3_2_RE);
    let w3_2_im = _mm_set1_ps(super::W3_2_IM);

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

            // Load 4 twiddle factors: w1[0], w2[0], w1[1], w2[1]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            let tw1_ptr = stage_twiddles.as_ptr().add(idx * 2) as *const f32;
            let tw_0 = _mm_loadu_ps(tw1_ptr);
            // Layout: [w1[1].re, w1[1].im, w2[1].re, w2[1].im]
            let tw_1 = _mm_loadu_ps(tw1_ptr.add(4));

            // Extract w1 and w2 for both columns.
            // w1 = [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            let w1 = _mm_shuffle_ps(tw_0, tw_1, 0b01_00_01_00);
            // w2 = [w2[0].re, w2[0].im, w2[1].re, w2[1].im]
            let w2 = _mm_shuffle_ps(tw_0, tw_1, 0b11_10_11_10);

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

            // Y0 = x0 + t1 + t2
            let y0 = _mm_add_ps(_mm_add_ps(x0, t1), t2);

            // Y1 = x0 + t1*W_3^1 + t2*W_3^2
            // t1*W_3^1: complex multiply t1 by (super::W3_1_RE, super::W3_1_IM)
            let t1_re = _mm_shuffle_ps(t1, t1, 0b10_10_00_00);
            let t1_im = _mm_shuffle_ps(t1, t1, 0b11_11_01_01);
            let t1w31_re = _mm_sub_ps(_mm_mul_ps(t1_re, w3_1_re), _mm_mul_ps(t1_im, w3_1_im));
            let t1w31_im = _mm_add_ps(_mm_mul_ps(t1_re, w3_1_im), _mm_mul_ps(t1_im, w3_1_re));
            let t1_w31 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w31_re, t1w31_im),
                _mm_unpackhi_ps(t1w31_re, t1w31_im),
            );

            // t2*W_3^2: complex multiply t2 by (super::W3_2_RE, super::W3_2_IM)
            let t2_re = _mm_shuffle_ps(t2, t2, 0b10_10_00_00);
            let t2_im = _mm_shuffle_ps(t2, t2, 0b11_11_01_01);
            let t2w32_re = _mm_sub_ps(_mm_mul_ps(t2_re, w3_2_re), _mm_mul_ps(t2_im, w3_2_im));
            let t2w32_im = _mm_add_ps(_mm_mul_ps(t2_re, w3_2_im), _mm_mul_ps(t2_im, w3_2_re));
            let t2_w32 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w32_re, t2w32_im),
                _mm_unpackhi_ps(t2w32_re, t2w32_im),
            );

            let y1 = _mm_add_ps(_mm_add_ps(x0, t1_w31), t2_w32);

            // Y2 = x0 + t1*W_3^2 + t2*W_3^1
            // t1*W_3^2: complex multiply t1 by (super::W3_2_RE, super::W3_2_IM)
            let t1w32_re = _mm_sub_ps(_mm_mul_ps(t1_re, w3_2_re), _mm_mul_ps(t1_im, w3_2_im));
            let t1w32_im = _mm_add_ps(_mm_mul_ps(t1_re, w3_2_im), _mm_mul_ps(t1_im, w3_2_re));
            let t1_w32 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w32_re, t1w32_im),
                _mm_unpackhi_ps(t1w32_re, t1w32_im),
            );

            // t2*W_3^1: complex multiply t2 by (super::W3_1_RE, super::W3_1_IM)
            let t2w31_re = _mm_sub_ps(_mm_mul_ps(t2_re, w3_1_re), _mm_mul_ps(t2_im, w3_1_im));
            let t2w31_im = _mm_add_ps(_mm_mul_ps(t2_re, w3_1_im), _mm_mul_ps(t2_im, w3_1_re));
            let t2_w31 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w31_re, t2w31_im),
                _mm_unpackhi_ps(t2w31_re, t2w31_im),
            );

            let y2 = _mm_add_ps(_mm_add_ps(x0, t1_w32), t2_w31);

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm_storeu_ps(y2_ptr, y2);
        }
    }

    super::butterfly_3_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}
