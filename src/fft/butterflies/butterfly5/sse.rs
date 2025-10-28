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
pub(super) unsafe fn butterfly_5_sse(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 2) * 2;

        // Broadcast W5 constants for SIMD operations.
        let w5_1_re = _mm_set1_ps(super::W5_1_RE);
        let w5_1_im = _mm_set1_ps(super::W5_1_IM);
        let w5_2_re = _mm_set1_ps(super::W5_2_RE);
        let w5_2_im = _mm_set1_ps(super::W5_2_IM);
        let w5_3_re = _mm_set1_ps(super::W5_3_RE);
        let w5_3_im = _mm_set1_ps(super::W5_3_IM);
        let w5_4_re = _mm_set1_ps(super::W5_4_RE);
        let w5_4_im = _mm_set1_ps(super::W5_4_IM);

        // Mask for manual addsub emulation.
        let neg_mask = _mm_castsi128_ps(_mm_set_epi32(
            0,
            0x80000000u32 as i32,
            0,
            0x80000000u32 as i32,
        ));

        for idx in (0..simd_cols).step_by(2) {
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

            let x4_ptr = data.as_ptr().add(idx + 4 * num_columns) as *const f32;
            let x4 = _mm_loadu_ps(x4_ptr);

            // Load 8 twiddle factors: w1[0], w2[0], w3[0], w4[0], w1[1], w2[1], w3[1], w4[1]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 4) as *const f32;
            let tw_0 = _mm_loadu_ps(tw_ptr);
            // Layout: [w3[0].re, w3[0].im, w4[0].re, w4[0].im]
            let tw_1 = _mm_loadu_ps(tw_ptr.add(4));
            // Layout: [w1[1].re, w1[1].im, w2[1].re, w2[1].im]
            let tw_2 = _mm_loadu_ps(tw_ptr.add(8));
            // Layout: [w3[1].re, w3[1].im, w4[1].re, w4[1].im]
            let tw_3 = _mm_loadu_ps(tw_ptr.add(12));

            // Extract w1, w2, w3, w4 for both columns.
            // w1 = [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            let w1 = _mm_shuffle_ps(tw_0, tw_2, 0b01_00_01_00);
            // w2 = [w2[0].re, w2[0].im, w2[1].re, w2[1].im]
            let w2 = _mm_shuffle_ps(tw_0, tw_2, 0b11_10_11_10);
            // w3 = [w3[0].re, w3[0].im, w3[1].re, w3[1].im]
            let w3 = _mm_shuffle_ps(tw_1, tw_3, 0b01_00_01_00);
            // w4 = [w4[0].re, w4[0].im, w4[1].re, w4[1].im]
            let w4 = _mm_shuffle_ps(tw_1, tw_3, 0b11_10_11_10);

            // Complex multiply: t1 = x1 * w1
            let w1_re = _mm_shuffle_ps(w1, w1, 0b10_10_00_00);
            let w1_im = _mm_shuffle_ps(w1, w1, 0b11_11_01_01);
            let prod_re = _mm_mul_ps(w1_re, x1);
            let x1_swap = _mm_shuffle_ps(x1, x1, 0b10_11_00_01);
            let prod_im = _mm_mul_ps(w1_im, x1_swap);
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

            // Complex multiply: t4 = x4 * w4
            let w4_re = _mm_shuffle_ps(w4, w4, 0b10_10_00_00);
            let w4_im = _mm_shuffle_ps(w4, w4, 0b11_11_01_01);
            let prod_re = _mm_mul_ps(w4_re, x4);
            let x4_swap = _mm_shuffle_ps(x4, x4, 0b10_11_00_01);
            let prod_im = _mm_mul_ps(w4_im, x4_swap);
            let prod_im_adjusted = _mm_xor_ps(prod_im, neg_mask);
            let t4 = _mm_add_ps(prod_re, prod_im_adjusted);

            // Y0 = x0 + t1 + t2 + t3 + t4
            let y0 = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(x0, t1), t2), t3), t4);

            // Y1 = x0 + t1*W_5^1 + t2*W_5^2 + t3*W_5^3 + t4*W_5^4
            // t1*W_5^1: complex multiply t1 by (super::W5_1_RE, super::W5_1_IM)
            let t1_re = _mm_shuffle_ps(t1, t1, 0b10_10_00_00);
            let t1_im = _mm_shuffle_ps(t1, t1, 0b11_11_01_01);
            let t1w51_re = _mm_sub_ps(_mm_mul_ps(t1_re, w5_1_re), _mm_mul_ps(t1_im, w5_1_im));
            let t1w51_im = _mm_add_ps(_mm_mul_ps(t1_re, w5_1_im), _mm_mul_ps(t1_im, w5_1_re));
            let t1_w51 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w51_re, t1w51_im),
                _mm_unpackhi_ps(t1w51_re, t1w51_im),
            );

            // t2*W_5^2: complex multiply t2 by (super::W5_2_RE, super::W5_2_IM)
            let t2_re = _mm_shuffle_ps(t2, t2, 0b10_10_00_00);
            let t2_im = _mm_shuffle_ps(t2, t2, 0b11_11_01_01);
            let t2w52_re = _mm_sub_ps(_mm_mul_ps(t2_re, w5_2_re), _mm_mul_ps(t2_im, w5_2_im));
            let t2w52_im = _mm_add_ps(_mm_mul_ps(t2_re, w5_2_im), _mm_mul_ps(t2_im, w5_2_re));
            let t2_w52 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w52_re, t2w52_im),
                _mm_unpackhi_ps(t2w52_re, t2w52_im),
            );

            // t3*W_5^3: complex multiply t3 by (super::W5_3_RE, super::W5_3_IM)
            let t3_re = _mm_shuffle_ps(t3, t3, 0b10_10_00_00);
            let t3_im = _mm_shuffle_ps(t3, t3, 0b11_11_01_01);
            let t3w53_re = _mm_sub_ps(_mm_mul_ps(t3_re, w5_3_re), _mm_mul_ps(t3_im, w5_3_im));
            let t3w53_im = _mm_add_ps(_mm_mul_ps(t3_re, w5_3_im), _mm_mul_ps(t3_im, w5_3_re));
            let t3_w53 = _mm_movelh_ps(
                _mm_unpacklo_ps(t3w53_re, t3w53_im),
                _mm_unpackhi_ps(t3w53_re, t3w53_im),
            );

            // t4*W_5^4: complex multiply t4 by (super::W5_4_RE, super::W5_4_IM)
            let t4_re = _mm_shuffle_ps(t4, t4, 0b10_10_00_00);
            let t4_im = _mm_shuffle_ps(t4, t4, 0b11_11_01_01);
            let t4w54_re = _mm_sub_ps(_mm_mul_ps(t4_re, w5_4_re), _mm_mul_ps(t4_im, w5_4_im));
            let t4w54_im = _mm_add_ps(_mm_mul_ps(t4_re, w5_4_im), _mm_mul_ps(t4_im, w5_4_re));
            let t4_w54 = _mm_movelh_ps(
                _mm_unpacklo_ps(t4w54_re, t4w54_im),
                _mm_unpackhi_ps(t4w54_re, t4w54_im),
            );

            let y1 = _mm_add_ps(
                _mm_add_ps(_mm_add_ps(_mm_add_ps(x0, t1_w51), t2_w52), t3_w53),
                t4_w54,
            );

            // Y2 = x0 + t1*W_5^2 + t2*W_5^4 + t3*W_5^1 + t4*W_5^3
            // t1*W_5^2
            let t1w52_re = _mm_sub_ps(_mm_mul_ps(t1_re, w5_2_re), _mm_mul_ps(t1_im, w5_2_im));
            let t1w52_im = _mm_add_ps(_mm_mul_ps(t1_re, w5_2_im), _mm_mul_ps(t1_im, w5_2_re));
            let t1_w52 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w52_re, t1w52_im),
                _mm_unpackhi_ps(t1w52_re, t1w52_im),
            );

            // t2*W_5^4
            let t2w54_re = _mm_sub_ps(_mm_mul_ps(t2_re, w5_4_re), _mm_mul_ps(t2_im, w5_4_im));
            let t2w54_im = _mm_add_ps(_mm_mul_ps(t2_re, w5_4_im), _mm_mul_ps(t2_im, w5_4_re));
            let t2_w54 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w54_re, t2w54_im),
                _mm_unpackhi_ps(t2w54_re, t2w54_im),
            );

            // t3*W_5^1
            let t3w51_re = _mm_sub_ps(_mm_mul_ps(t3_re, w5_1_re), _mm_mul_ps(t3_im, w5_1_im));
            let t3w51_im = _mm_add_ps(_mm_mul_ps(t3_re, w5_1_im), _mm_mul_ps(t3_im, w5_1_re));
            let t3_w51 = _mm_movelh_ps(
                _mm_unpacklo_ps(t3w51_re, t3w51_im),
                _mm_unpackhi_ps(t3w51_re, t3w51_im),
            );

            // t4*W_5^3
            let t4w53_re = _mm_sub_ps(_mm_mul_ps(t4_re, w5_3_re), _mm_mul_ps(t4_im, w5_3_im));
            let t4w53_im = _mm_add_ps(_mm_mul_ps(t4_re, w5_3_im), _mm_mul_ps(t4_im, w5_3_re));
            let t4_w53 = _mm_movelh_ps(
                _mm_unpacklo_ps(t4w53_re, t4w53_im),
                _mm_unpackhi_ps(t4w53_re, t4w53_im),
            );

            let y2 = _mm_add_ps(
                _mm_add_ps(_mm_add_ps(_mm_add_ps(x0, t1_w52), t2_w54), t3_w51),
                t4_w53,
            );

            // Y3 = x0 + t1*W_5^3 + t2*W_5^1 + t3*W_5^4 + t4*W_5^2
            // t1*W_5^3
            let t1w53_re = _mm_sub_ps(_mm_mul_ps(t1_re, w5_3_re), _mm_mul_ps(t1_im, w5_3_im));
            let t1w53_im = _mm_add_ps(_mm_mul_ps(t1_re, w5_3_im), _mm_mul_ps(t1_im, w5_3_re));
            let t1_w53 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w53_re, t1w53_im),
                _mm_unpackhi_ps(t1w53_re, t1w53_im),
            );

            // t2*W_5^1
            let t2w51_re = _mm_sub_ps(_mm_mul_ps(t2_re, w5_1_re), _mm_mul_ps(t2_im, w5_1_im));
            let t2w51_im = _mm_add_ps(_mm_mul_ps(t2_re, w5_1_im), _mm_mul_ps(t2_im, w5_1_re));
            let t2_w51 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w51_re, t2w51_im),
                _mm_unpackhi_ps(t2w51_re, t2w51_im),
            );

            // t3*W_5^4
            let t3w54_re = _mm_sub_ps(_mm_mul_ps(t3_re, w5_4_re), _mm_mul_ps(t3_im, w5_4_im));
            let t3w54_im = _mm_add_ps(_mm_mul_ps(t3_re, w5_4_im), _mm_mul_ps(t3_im, w5_4_re));
            let t3_w54 = _mm_movelh_ps(
                _mm_unpacklo_ps(t3w54_re, t3w54_im),
                _mm_unpackhi_ps(t3w54_re, t3w54_im),
            );

            // t4*W_5^2
            let t4w52_re = _mm_sub_ps(_mm_mul_ps(t4_re, w5_2_re), _mm_mul_ps(t4_im, w5_2_im));
            let t4w52_im = _mm_add_ps(_mm_mul_ps(t4_re, w5_2_im), _mm_mul_ps(t4_im, w5_2_re));
            let t4_w52 = _mm_movelh_ps(
                _mm_unpacklo_ps(t4w52_re, t4w52_im),
                _mm_unpackhi_ps(t4w52_re, t4w52_im),
            );

            let y3 = _mm_add_ps(
                _mm_add_ps(_mm_add_ps(_mm_add_ps(x0, t1_w53), t2_w51), t3_w54),
                t4_w52,
            );

            // Y4 = x0 + t1*W_5^4 + t2*W_5^3 + t3*W_5^2 + t4*W_5^1
            // t1*W_5^4
            let t1w54_re = _mm_sub_ps(_mm_mul_ps(t1_re, w5_4_re), _mm_mul_ps(t1_im, w5_4_im));
            let t1w54_im = _mm_add_ps(_mm_mul_ps(t1_re, w5_4_im), _mm_mul_ps(t1_im, w5_4_re));
            let t1_w54 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w54_re, t1w54_im),
                _mm_unpackhi_ps(t1w54_re, t1w54_im),
            );

            // t2*W_5^3
            let t2w53_re = _mm_sub_ps(_mm_mul_ps(t2_re, w5_3_re), _mm_mul_ps(t2_im, w5_3_im));
            let t2w53_im = _mm_add_ps(_mm_mul_ps(t2_re, w5_3_im), _mm_mul_ps(t2_im, w5_3_re));
            let t2_w53 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w53_re, t2w53_im),
                _mm_unpackhi_ps(t2w53_re, t2w53_im),
            );

            // t3*W_5^2
            let t3w52_re = _mm_sub_ps(_mm_mul_ps(t3_re, w5_2_re), _mm_mul_ps(t3_im, w5_2_im));
            let t3w52_im = _mm_add_ps(_mm_mul_ps(t3_re, w5_2_im), _mm_mul_ps(t3_im, w5_2_re));
            let t3_w52 = _mm_movelh_ps(
                _mm_unpacklo_ps(t3w52_re, t3w52_im),
                _mm_unpackhi_ps(t3w52_re, t3w52_im),
            );

            // t4*W_5^1
            let t4w51_re = _mm_sub_ps(_mm_mul_ps(t4_re, w5_1_re), _mm_mul_ps(t4_im, w5_1_im));
            let t4w51_im = _mm_add_ps(_mm_mul_ps(t4_re, w5_1_im), _mm_mul_ps(t4_im, w5_1_re));
            let t4_w51 = _mm_movelh_ps(
                _mm_unpacklo_ps(t4w51_re, t4w51_im),
                _mm_unpackhi_ps(t4w51_re, t4w51_im),
            );

            let y4 = _mm_add_ps(
                _mm_add_ps(_mm_add_ps(_mm_add_ps(x0, t1_w54), t2_w53), t3_w52),
                t4_w51,
            );

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm_storeu_ps(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            _mm_storeu_ps(y3_ptr, y3);
            let y4_ptr = data.as_mut_ptr().add(idx + 4 * num_columns) as *mut f32;
            _mm_storeu_ps(y4_ptr, y4);
        }

        super::butterfly_5_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}

/// SSE3 implementation: processes 2 columns at once.
#[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
#[target_feature(enable = "sse3")]
pub(super) unsafe fn butterfly_5_sse3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 2) * 2;

        // Broadcast W5 constants for SIMD operations.
        let w5_1_re = _mm_set1_ps(super::W5_1_RE);
        let w5_1_im = _mm_set1_ps(super::W5_1_IM);
        let w5_2_re = _mm_set1_ps(super::W5_2_RE);
        let w5_2_im = _mm_set1_ps(super::W5_2_IM);
        let w5_3_re = _mm_set1_ps(super::W5_3_RE);
        let w5_3_im = _mm_set1_ps(super::W5_3_IM);
        let w5_4_re = _mm_set1_ps(super::W5_4_RE);
        let w5_4_im = _mm_set1_ps(super::W5_4_IM);

        for idx in (0..simd_cols).step_by(2) {
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

            let x4_ptr = data.as_ptr().add(idx + 4 * num_columns) as *const f32;
            let x4 = _mm_loadu_ps(x4_ptr);

            // Load 8 twiddle factors: w1[0], w2[0], w3[0], w4[0], w1[1], w2[1], w3[1], w4[1]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 4) as *const f32;
            let tw_0 = _mm_loadu_ps(tw_ptr);
            // Layout: [w3[0].re, w3[0].im, w4[0].re, w4[0].im]
            let tw_1 = _mm_loadu_ps(tw_ptr.add(4));
            // Layout: [w1[1].re, w1[1].im, w2[1].re, w2[1].im]
            let tw_2 = _mm_loadu_ps(tw_ptr.add(8));
            // Layout: [w3[1].re, w3[1].im, w4[1].re, w4[1].im]
            let tw_3 = _mm_loadu_ps(tw_ptr.add(12));

            // Extract w1, w2, w3, w4 for both columns.
            // w1 = [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            let w1 = _mm_shuffle_ps(tw_0, tw_2, 0b01_00_01_00);
            // w2 = [w2[0].re, w2[0].im, w2[1].re, w2[1].im]
            let w2 = _mm_shuffle_ps(tw_0, tw_2, 0b11_10_11_10);
            // w3 = [w3[0].re, w3[0].im, w3[1].re, w3[1].im]
            let w3 = _mm_shuffle_ps(tw_1, tw_3, 0b01_00_01_00);
            // w4 = [w4[0].re, w4[0].im, w4[1].re, w4[1].im]
            let w4 = _mm_shuffle_ps(tw_1, tw_3, 0b11_10_11_10);

            // Complex multiply: t1 = x1 * w1
            let w1_re = _mm_shuffle_ps(w1, w1, 0b10_10_00_00);
            let w1_im = _mm_shuffle_ps(w1, w1, 0b11_11_01_01);
            let prod_re = _mm_mul_ps(w1_re, x1);
            let x1_swap = _mm_shuffle_ps(x1, x1, 0b10_11_00_01);
            let prod_im = _mm_mul_ps(w1_im, x1_swap);
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

            // Complex multiply: t4 = x4 * w4
            let w4_re = _mm_shuffle_ps(w4, w4, 0b10_10_00_00);
            let w4_im = _mm_shuffle_ps(w4, w4, 0b11_11_01_01);
            let prod_re = _mm_mul_ps(w4_re, x4);
            let x4_swap = _mm_shuffle_ps(x4, x4, 0b10_11_00_01);
            let prod_im = _mm_mul_ps(w4_im, x4_swap);
            let t4 = _mm_addsub_ps(prod_re, prod_im);

            // Y0 = x0 + t1 + t2 + t3 + t4
            let y0 = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(x0, t1), t2), t3), t4);

            // Y1 = x0 + t1*W_5^1 + t2*W_5^2 + t3*W_5^3 + t4*W_5^4
            // t1*W_5^1: complex multiply t1 by (super::W5_1_RE, super::W5_1_IM)
            let t1_re = _mm_shuffle_ps(t1, t1, 0b10_10_00_00);
            let t1_im = _mm_shuffle_ps(t1, t1, 0b11_11_01_01);
            let t1w51_re = _mm_sub_ps(_mm_mul_ps(t1_re, w5_1_re), _mm_mul_ps(t1_im, w5_1_im));
            let t1w51_im = _mm_add_ps(_mm_mul_ps(t1_re, w5_1_im), _mm_mul_ps(t1_im, w5_1_re));
            let t1_w51 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w51_re, t1w51_im),
                _mm_unpackhi_ps(t1w51_re, t1w51_im),
            );

            // t2*W_5^2: complex multiply t2 by (super::W5_2_RE, super::W5_2_IM)
            let t2_re = _mm_shuffle_ps(t2, t2, 0b10_10_00_00);
            let t2_im = _mm_shuffle_ps(t2, t2, 0b11_11_01_01);
            let t2w52_re = _mm_sub_ps(_mm_mul_ps(t2_re, w5_2_re), _mm_mul_ps(t2_im, w5_2_im));
            let t2w52_im = _mm_add_ps(_mm_mul_ps(t2_re, w5_2_im), _mm_mul_ps(t2_im, w5_2_re));
            let t2_w52 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w52_re, t2w52_im),
                _mm_unpackhi_ps(t2w52_re, t2w52_im),
            );

            // t3*W_5^3: complex multiply t3 by (super::W5_3_RE, super::W5_3_IM)
            let t3_re = _mm_shuffle_ps(t3, t3, 0b10_10_00_00);
            let t3_im = _mm_shuffle_ps(t3, t3, 0b11_11_01_01);
            let t3w53_re = _mm_sub_ps(_mm_mul_ps(t3_re, w5_3_re), _mm_mul_ps(t3_im, w5_3_im));
            let t3w53_im = _mm_add_ps(_mm_mul_ps(t3_re, w5_3_im), _mm_mul_ps(t3_im, w5_3_re));
            let t3_w53 = _mm_movelh_ps(
                _mm_unpacklo_ps(t3w53_re, t3w53_im),
                _mm_unpackhi_ps(t3w53_re, t3w53_im),
            );

            // t4*W_5^4: complex multiply t4 by (super::W5_4_RE, super::W5_4_IM)
            let t4_re = _mm_shuffle_ps(t4, t4, 0b10_10_00_00);
            let t4_im = _mm_shuffle_ps(t4, t4, 0b11_11_01_01);
            let t4w54_re = _mm_sub_ps(_mm_mul_ps(t4_re, w5_4_re), _mm_mul_ps(t4_im, w5_4_im));
            let t4w54_im = _mm_add_ps(_mm_mul_ps(t4_re, w5_4_im), _mm_mul_ps(t4_im, w5_4_re));
            let t4_w54 = _mm_movelh_ps(
                _mm_unpacklo_ps(t4w54_re, t4w54_im),
                _mm_unpackhi_ps(t4w54_re, t4w54_im),
            );

            let y1 = _mm_add_ps(
                _mm_add_ps(_mm_add_ps(_mm_add_ps(x0, t1_w51), t2_w52), t3_w53),
                t4_w54,
            );

            // Y2 = x0 + t1*W_5^2 + t2*W_5^4 + t3*W_5^1 + t4*W_5^3
            // t1*W_5^2
            let t1w52_re = _mm_sub_ps(_mm_mul_ps(t1_re, w5_2_re), _mm_mul_ps(t1_im, w5_2_im));
            let t1w52_im = _mm_add_ps(_mm_mul_ps(t1_re, w5_2_im), _mm_mul_ps(t1_im, w5_2_re));
            let t1_w52 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w52_re, t1w52_im),
                _mm_unpackhi_ps(t1w52_re, t1w52_im),
            );

            // t2*W_5^4
            let t2w54_re = _mm_sub_ps(_mm_mul_ps(t2_re, w5_4_re), _mm_mul_ps(t2_im, w5_4_im));
            let t2w54_im = _mm_add_ps(_mm_mul_ps(t2_re, w5_4_im), _mm_mul_ps(t2_im, w5_4_re));
            let t2_w54 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w54_re, t2w54_im),
                _mm_unpackhi_ps(t2w54_re, t2w54_im),
            );

            // t3*W_5^1
            let t3w51_re = _mm_sub_ps(_mm_mul_ps(t3_re, w5_1_re), _mm_mul_ps(t3_im, w5_1_im));
            let t3w51_im = _mm_add_ps(_mm_mul_ps(t3_re, w5_1_im), _mm_mul_ps(t3_im, w5_1_re));
            let t3_w51 = _mm_movelh_ps(
                _mm_unpacklo_ps(t3w51_re, t3w51_im),
                _mm_unpackhi_ps(t3w51_re, t3w51_im),
            );

            // t4*W_5^3
            let t4w53_re = _mm_sub_ps(_mm_mul_ps(t4_re, w5_3_re), _mm_mul_ps(t4_im, w5_3_im));
            let t4w53_im = _mm_add_ps(_mm_mul_ps(t4_re, w5_3_im), _mm_mul_ps(t4_im, w5_3_re));
            let t4_w53 = _mm_movelh_ps(
                _mm_unpacklo_ps(t4w53_re, t4w53_im),
                _mm_unpackhi_ps(t4w53_re, t4w53_im),
            );

            let y2 = _mm_add_ps(
                _mm_add_ps(_mm_add_ps(_mm_add_ps(x0, t1_w52), t2_w54), t3_w51),
                t4_w53,
            );

            // Y3 = x0 + t1*W_5^3 + t2*W_5^1 + t3*W_5^4 + t4*W_5^2
            // t1*W_5^3
            let t1w53_re = _mm_sub_ps(_mm_mul_ps(t1_re, w5_3_re), _mm_mul_ps(t1_im, w5_3_im));
            let t1w53_im = _mm_add_ps(_mm_mul_ps(t1_re, w5_3_im), _mm_mul_ps(t1_im, w5_3_re));
            let t1_w53 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w53_re, t1w53_im),
                _mm_unpackhi_ps(t1w53_re, t1w53_im),
            );

            // t2*W_5^1
            let t2w51_re = _mm_sub_ps(_mm_mul_ps(t2_re, w5_1_re), _mm_mul_ps(t2_im, w5_1_im));
            let t2w51_im = _mm_add_ps(_mm_mul_ps(t2_re, w5_1_im), _mm_mul_ps(t2_im, w5_1_re));
            let t2_w51 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w51_re, t2w51_im),
                _mm_unpackhi_ps(t2w51_re, t2w51_im),
            );

            // t3*W_5^4
            let t3w54_re = _mm_sub_ps(_mm_mul_ps(t3_re, w5_4_re), _mm_mul_ps(t3_im, w5_4_im));
            let t3w54_im = _mm_add_ps(_mm_mul_ps(t3_re, w5_4_im), _mm_mul_ps(t3_im, w5_4_re));
            let t3_w54 = _mm_movelh_ps(
                _mm_unpacklo_ps(t3w54_re, t3w54_im),
                _mm_unpackhi_ps(t3w54_re, t3w54_im),
            );

            // t4*W_5^2
            let t4w52_re = _mm_sub_ps(_mm_mul_ps(t4_re, w5_2_re), _mm_mul_ps(t4_im, w5_2_im));
            let t4w52_im = _mm_add_ps(_mm_mul_ps(t4_re, w5_2_im), _mm_mul_ps(t4_im, w5_2_re));
            let t4_w52 = _mm_movelh_ps(
                _mm_unpacklo_ps(t4w52_re, t4w52_im),
                _mm_unpackhi_ps(t4w52_re, t4w52_im),
            );

            let y3 = _mm_add_ps(
                _mm_add_ps(_mm_add_ps(_mm_add_ps(x0, t1_w53), t2_w51), t3_w54),
                t4_w52,
            );

            // Y4 = x0 + t1*W_5^4 + t2*W_5^3 + t3*W_5^2 + t4*W_5^1
            // t1*W_5^4
            let t1w54_re = _mm_sub_ps(_mm_mul_ps(t1_re, w5_4_re), _mm_mul_ps(t1_im, w5_4_im));
            let t1w54_im = _mm_add_ps(_mm_mul_ps(t1_re, w5_4_im), _mm_mul_ps(t1_im, w5_4_re));
            let t1_w54 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w54_re, t1w54_im),
                _mm_unpackhi_ps(t1w54_re, t1w54_im),
            );

            // t2*W_5^3
            let t2w53_re = _mm_sub_ps(_mm_mul_ps(t2_re, w5_3_re), _mm_mul_ps(t2_im, w5_3_im));
            let t2w53_im = _mm_add_ps(_mm_mul_ps(t2_re, w5_3_im), _mm_mul_ps(t2_im, w5_3_re));
            let t2_w53 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w53_re, t2w53_im),
                _mm_unpackhi_ps(t2w53_re, t2w53_im),
            );

            // t3*W_5^2
            let t3w52_re = _mm_sub_ps(_mm_mul_ps(t3_re, w5_2_re), _mm_mul_ps(t3_im, w5_2_im));
            let t3w52_im = _mm_add_ps(_mm_mul_ps(t3_re, w5_2_im), _mm_mul_ps(t3_im, w5_2_re));
            let t3_w52 = _mm_movelh_ps(
                _mm_unpacklo_ps(t3w52_re, t3w52_im),
                _mm_unpackhi_ps(t3w52_re, t3w52_im),
            );

            // t4*W_5^1
            let t4w51_re = _mm_sub_ps(_mm_mul_ps(t4_re, w5_1_re), _mm_mul_ps(t4_im, w5_1_im));
            let t4w51_im = _mm_add_ps(_mm_mul_ps(t4_re, w5_1_im), _mm_mul_ps(t4_im, w5_1_re));
            let t4_w51 = _mm_movelh_ps(
                _mm_unpacklo_ps(t4w51_re, t4w51_im),
                _mm_unpackhi_ps(t4w51_re, t4w51_im),
            );

            let y4 = _mm_add_ps(
                _mm_add_ps(_mm_add_ps(_mm_add_ps(x0, t1_w54), t2_w53), t3_w52),
                t4_w51,
            );

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm_storeu_ps(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            _mm_storeu_ps(y3_ptr, y3);
            let y4_ptr = data.as_mut_ptr().add(idx + 4 * num_columns) as *mut f32;
            _mm_storeu_ps(y4_ptr, y4);
        }

        super::butterfly_5_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}
