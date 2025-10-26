#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::Complex32;

// Primitive 5th roots of unity: W_5 = exp(-2πi/5)
const W5_1_RE: f32 = 0.309017; // cos(-2π/5)
const W5_1_IM: f32 = -0.95105654; // sin(-2π/5)
const W5_2_RE: f32 = -0.809017; // cos(-4π/5)
const W5_2_IM: f32 = -0.58778524; // sin(-4π/5)
const W5_3_RE: f32 = -0.809017; // cos(-6π/5)
const W5_3_IM: f32 = 0.58778524; // sin(-6π/5)
const W5_4_RE: f32 = 0.309017; // cos(-8π/5)
const W5_4_IM: f32 = 0.95105654; // sin(-8π/5)

/// Helper function to process a range of columns for radix-5 butterfly.
/// Processes columns from `start_col` to `end_col` (exclusive).
#[inline(always)]
fn butterfly_5_columns(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
    start_col: usize,
    end_col: usize,
) {
    for idx in start_col..end_col {
        let x0 = data[idx];
        let x1 = data[idx + num_columns];
        let x2 = data[idx + 2 * num_columns];
        let x3 = data[idx + 3 * num_columns];
        let x4 = data[idx + 4 * num_columns];

        // Apply stage twiddle factors.
        let w1 = stage_twiddles[idx * 4];
        let w2 = stage_twiddles[idx * 4 + 1];
        let w3 = stage_twiddles[idx * 4 + 2];
        let w4 = stage_twiddles[idx * 4 + 3];

        let t1 = x1.mul(&w1);
        let t2 = x2.mul(&w2);
        let t3 = x3.mul(&w3);
        let t4 = x4.mul(&w4);

        // X0 = x0 + t1 + t2 + t3 + t4
        data[idx] = x0.add(&t1).add(&t2).add(&t3).add(&t4);

        // X1 = x0 + t1*W_5^1 + t2*W_5^2 + t3*W_5^3 + t4*W_5^4
        let t1_w51 = Complex32::new(
            t1.re * W5_1_RE - t1.im * W5_1_IM,
            t1.re * W5_1_IM + t1.im * W5_1_RE,
        );
        let t2_w52 = Complex32::new(
            t2.re * W5_2_RE - t2.im * W5_2_IM,
            t2.re * W5_2_IM + t2.im * W5_2_RE,
        );
        let t3_w53 = Complex32::new(
            t3.re * W5_3_RE - t3.im * W5_3_IM,
            t3.re * W5_3_IM + t3.im * W5_3_RE,
        );
        let t4_w54 = Complex32::new(
            t4.re * W5_4_RE - t4.im * W5_4_IM,
            t4.re * W5_4_IM + t4.im * W5_4_RE,
        );
        data[idx + num_columns] = x0.add(&t1_w51).add(&t2_w52).add(&t3_w53).add(&t4_w54);

        // X2 = x0 + t1*W_5^2 + t2*W_5^4 + t3*W_5^1 + t4*W_5^3
        // (Note: W_5^(2*2) = W_5^4, W_5^(2*3) = W_5^6 = W_5^1, W_5^(2*4) = W_5^8 = W_5^3)
        let t1_w52 = Complex32::new(
            t1.re * W5_2_RE - t1.im * W5_2_IM,
            t1.re * W5_2_IM + t1.im * W5_2_RE,
        );
        let t2_w54 = Complex32::new(
            t2.re * W5_4_RE - t2.im * W5_4_IM,
            t2.re * W5_4_IM + t2.im * W5_4_RE,
        );
        let t3_w51 = Complex32::new(
            t3.re * W5_1_RE - t3.im * W5_1_IM,
            t3.re * W5_1_IM + t3.im * W5_1_RE,
        );
        let t4_w53 = Complex32::new(
            t4.re * W5_3_RE - t4.im * W5_3_IM,
            t4.re * W5_3_IM + t4.im * W5_3_RE,
        );
        data[idx + 2 * num_columns] = x0.add(&t1_w52).add(&t2_w54).add(&t3_w51).add(&t4_w53);

        // X3 = x0 + t1*W_5^3 + t2*W_5^1 + t3*W_5^4 + t4*W_5^2
        // (Note: W_5^(3*2) = W_5^6 = W_5^1, W_5^(3*3) = W_5^9 = W_5^4, W_5^(3*4) = W_5^12 = W_5^2)
        let t1_w53 = Complex32::new(
            t1.re * W5_3_RE - t1.im * W5_3_IM,
            t1.re * W5_3_IM + t1.im * W5_3_RE,
        );
        let t2_w51 = Complex32::new(
            t2.re * W5_1_RE - t2.im * W5_1_IM,
            t2.re * W5_1_IM + t2.im * W5_1_RE,
        );
        let t3_w54 = Complex32::new(
            t3.re * W5_4_RE - t3.im * W5_4_IM,
            t3.re * W5_4_IM + t3.im * W5_4_RE,
        );
        let t4_w52 = Complex32::new(
            t4.re * W5_2_RE - t4.im * W5_2_IM,
            t4.re * W5_2_IM + t4.im * W5_2_RE,
        );
        data[idx + 3 * num_columns] = x0.add(&t1_w53).add(&t2_w51).add(&t3_w54).add(&t4_w52);

        // X4 = x0 + t1*W_5^4 + t2*W_5^3 + t3*W_5^2 + t4*W_5^1
        // (Note: W_5^(4*2) = W_5^8 = W_5^3, W_5^(4*3) = W_5^12 = W_5^2, W_5^(4*4) = W_5^16 = W_5^1)
        let t1_w54 = Complex32::new(
            t1.re * W5_4_RE - t1.im * W5_4_IM,
            t1.re * W5_4_IM + t1.im * W5_4_RE,
        );
        let t2_w53 = Complex32::new(
            t2.re * W5_3_RE - t2.im * W5_3_IM,
            t2.re * W5_3_IM + t2.im * W5_3_RE,
        );
        let t3_w52 = Complex32::new(
            t3.re * W5_2_RE - t3.im * W5_2_IM,
            t3.re * W5_2_IM + t3.im * W5_2_RE,
        );
        let t4_w51 = Complex32::new(
            t4.re * W5_1_RE - t4.im * W5_1_IM,
            t4.re * W5_1_IM + t4.im * W5_1_RE,
        );
        data[idx + 4 * num_columns] = x0.add(&t1_w54).add(&t2_w53).add(&t3_w52).add(&t4_w51);
    }
}

/// Processes a single radix-5 butterfly stage across all columns using scalar operations.
///
/// Implements the radix-5 DFT butterfly:
/// X[k] = sum_{j=0}^{4} x[j] * W_stage[j] * W_5^(k*j)
///
/// Where W_5 = exp(-2πi/5) is the primitive 5th root of unity, and
/// W_stage are the stage-specific twiddle factors.
#[inline(always)]
pub fn butterfly_5_scalar(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    butterfly_5_columns(data, stage_twiddles, num_columns, 0, num_columns);
}

/// Pure SSE implementation: processes 2 columns at once.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    any(not(target_feature = "sse3"), test)
))]
#[target_feature(enable = "sse")]
unsafe fn butterfly_5_sse(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 2) * 2;

        // Broadcast W5 constants for SIMD operations.
        let w5_1_re = _mm_set1_ps(W5_1_RE);
        let w5_1_im = _mm_set1_ps(W5_1_IM);
        let w5_2_re = _mm_set1_ps(W5_2_RE);
        let w5_2_im = _mm_set1_ps(W5_2_IM);
        let w5_3_re = _mm_set1_ps(W5_3_RE);
        let w5_3_im = _mm_set1_ps(W5_3_IM);
        let w5_4_re = _mm_set1_ps(W5_4_RE);
        let w5_4_im = _mm_set1_ps(W5_4_IM);

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
            // t1*W_5^1: complex multiply t1 by (W5_1_RE, W5_1_IM)
            let t1_re = _mm_shuffle_ps(t1, t1, 0b10_10_00_00);
            let t1_im = _mm_shuffle_ps(t1, t1, 0b11_11_01_01);
            let t1w51_re = _mm_sub_ps(_mm_mul_ps(t1_re, w5_1_re), _mm_mul_ps(t1_im, w5_1_im));
            let t1w51_im = _mm_add_ps(_mm_mul_ps(t1_re, w5_1_im), _mm_mul_ps(t1_im, w5_1_re));
            let t1_w51 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w51_re, t1w51_im),
                _mm_unpackhi_ps(t1w51_re, t1w51_im),
            );

            // t2*W_5^2: complex multiply t2 by (W5_2_RE, W5_2_IM)
            let t2_re = _mm_shuffle_ps(t2, t2, 0b10_10_00_00);
            let t2_im = _mm_shuffle_ps(t2, t2, 0b11_11_01_01);
            let t2w52_re = _mm_sub_ps(_mm_mul_ps(t2_re, w5_2_re), _mm_mul_ps(t2_im, w5_2_im));
            let t2w52_im = _mm_add_ps(_mm_mul_ps(t2_re, w5_2_im), _mm_mul_ps(t2_im, w5_2_re));
            let t2_w52 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w52_re, t2w52_im),
                _mm_unpackhi_ps(t2w52_re, t2w52_im),
            );

            // t3*W_5^3: complex multiply t3 by (W5_3_RE, W5_3_IM)
            let t3_re = _mm_shuffle_ps(t3, t3, 0b10_10_00_00);
            let t3_im = _mm_shuffle_ps(t3, t3, 0b11_11_01_01);
            let t3w53_re = _mm_sub_ps(_mm_mul_ps(t3_re, w5_3_re), _mm_mul_ps(t3_im, w5_3_im));
            let t3w53_im = _mm_add_ps(_mm_mul_ps(t3_re, w5_3_im), _mm_mul_ps(t3_im, w5_3_re));
            let t3_w53 = _mm_movelh_ps(
                _mm_unpacklo_ps(t3w53_re, t3w53_im),
                _mm_unpackhi_ps(t3w53_re, t3w53_im),
            );

            // t4*W_5^4: complex multiply t4 by (W5_4_RE, W5_4_IM)
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

        butterfly_5_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}

/// SSE3 implementation: processes 2 columns at once.
#[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
#[target_feature(enable = "sse3")]
unsafe fn butterfly_5_sse3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 2) * 2;

        // Broadcast W5 constants for SIMD operations.
        let w5_1_re = _mm_set1_ps(W5_1_RE);
        let w5_1_im = _mm_set1_ps(W5_1_IM);
        let w5_2_re = _mm_set1_ps(W5_2_RE);
        let w5_2_im = _mm_set1_ps(W5_2_IM);
        let w5_3_re = _mm_set1_ps(W5_3_RE);
        let w5_3_im = _mm_set1_ps(W5_3_IM);
        let w5_4_re = _mm_set1_ps(W5_4_RE);
        let w5_4_im = _mm_set1_ps(W5_4_IM);

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
            // t1*W_5^1: complex multiply t1 by (W5_1_RE, W5_1_IM)
            let t1_re = _mm_shuffle_ps(t1, t1, 0b10_10_00_00);
            let t1_im = _mm_shuffle_ps(t1, t1, 0b11_11_01_01);
            let t1w51_re = _mm_sub_ps(_mm_mul_ps(t1_re, w5_1_re), _mm_mul_ps(t1_im, w5_1_im));
            let t1w51_im = _mm_add_ps(_mm_mul_ps(t1_re, w5_1_im), _mm_mul_ps(t1_im, w5_1_re));
            let t1_w51 = _mm_movelh_ps(
                _mm_unpacklo_ps(t1w51_re, t1w51_im),
                _mm_unpackhi_ps(t1w51_re, t1w51_im),
            );

            // t2*W_5^2: complex multiply t2 by (W5_2_RE, W5_2_IM)
            let t2_re = _mm_shuffle_ps(t2, t2, 0b10_10_00_00);
            let t2_im = _mm_shuffle_ps(t2, t2, 0b11_11_01_01);
            let t2w52_re = _mm_sub_ps(_mm_mul_ps(t2_re, w5_2_re), _mm_mul_ps(t2_im, w5_2_im));
            let t2w52_im = _mm_add_ps(_mm_mul_ps(t2_re, w5_2_im), _mm_mul_ps(t2_im, w5_2_re));
            let t2_w52 = _mm_movelh_ps(
                _mm_unpacklo_ps(t2w52_re, t2w52_im),
                _mm_unpackhi_ps(t2w52_re, t2w52_im),
            );

            // t3*W_5^3: complex multiply t3 by (W5_3_RE, W5_3_IM)
            let t3_re = _mm_shuffle_ps(t3, t3, 0b10_10_00_00);
            let t3_im = _mm_shuffle_ps(t3, t3, 0b11_11_01_01);
            let t3w53_re = _mm_sub_ps(_mm_mul_ps(t3_re, w5_3_re), _mm_mul_ps(t3_im, w5_3_im));
            let t3w53_im = _mm_add_ps(_mm_mul_ps(t3_re, w5_3_im), _mm_mul_ps(t3_im, w5_3_re));
            let t3_w53 = _mm_movelh_ps(
                _mm_unpacklo_ps(t3w53_re, t3w53_im),
                _mm_unpackhi_ps(t3w53_re, t3w53_im),
            );

            // t4*W_5^4: complex multiply t4 by (W5_4_RE, W5_4_IM)
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

        butterfly_5_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}

/// AVX implementation: processes 4 columns at once.
#[cfg(any(all(
    target_arch = "x86_64",
    target_feature = "avx",
    any(not(target_feature = "fma"), test)
)))]
#[target_feature(enable = "avx")]
unsafe fn butterfly_5_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 4) * 4;

        // Broadcast W5 constants for SIMD operations.
        let w5_1_re = _mm256_set1_ps(W5_1_RE);
        let w5_1_im = _mm256_set1_ps(W5_1_IM);
        let w5_2_re = _mm256_set1_ps(W5_2_RE);
        let w5_2_im = _mm256_set1_ps(W5_2_IM);
        let w5_3_re = _mm256_set1_ps(W5_3_RE);
        let w5_3_im = _mm256_set1_ps(W5_3_IM);
        let w5_4_re = _mm256_set1_ps(W5_4_RE);
        let w5_4_im = _mm256_set1_ps(W5_4_IM);

        for idx in (0..simd_cols).step_by(4) {
            // Load 4 complex numbers from each row.
            // Layout: [x0[0].re, x0[0].im, x0[1].re, x0[1].im, x0[2].re, x0[2].im, x0[3].re, x0[3].im]
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = _mm256_loadu_ps(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = _mm256_loadu_ps(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = _mm256_loadu_ps(x2_ptr);

            let x3_ptr = data.as_ptr().add(idx + 3 * num_columns) as *const f32;
            let x3 = _mm256_loadu_ps(x3_ptr);

            let x4_ptr = data.as_ptr().add(idx + 4 * num_columns) as *const f32;
            let x4 = _mm256_loadu_ps(x4_ptr);

            // Load 16 twiddle factors: w1[0], w2[0], w3[0], w4[0], w1[1], w2[1], w3[1], w4[1], ...
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im, w3[0].re, w3[0].im, w4[0].re, w4[0].im]
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 4) as *const f32;
            let tw_0 = _mm256_loadu_ps(tw_ptr);
            // Layout: [w1[1].re, w1[1].im, w2[1].re, w2[1].im, w3[1].re, w3[1].im, w4[1].re, w4[1].im]
            let tw_1 = _mm256_loadu_ps(tw_ptr.add(8));
            // Layout: [w1[2].re, w1[2].im, w2[2].re, w2[2].im, w3[2].re, w3[2].im, w4[2].re, w4[2].im]
            let tw_2 = _mm256_loadu_ps(tw_ptr.add(16));
            // Layout: [w1[3].re, w1[3].im, w2[3].re, w2[3].im, w3[3].re, w3[3].im, w4[3].re, w4[3].im]
            let tw_3 = _mm256_loadu_ps(tw_ptr.add(24));

            // Extract w1, w2, w3, w4 for all 4 columns.
            // tw_0 = [w1[0], w2[0] | w3[0], w4[0]]
            // tw_1 = [w1[1], w2[1] | w3[1], w4[1]]
            // tw_2 = [w1[2], w2[2] | w3[2], w4[2]]
            // tw_3 = [w1[3], w2[3] | w3[3], w4[3]]

            // Rearrange to group w1/w2 and w3/w4 separately
            // temp_01 = [w1[0], w2[0] | w1[2], w2[2]]
            let temp_01 = _mm256_permute2f128_ps(tw_0, tw_2, 0x20);
            // temp_23 = [w1[1], w2[1] | w1[3], w2[3]]
            let temp_23 = _mm256_permute2f128_ps(tw_1, tw_3, 0x20);
            // temp_45 = [w3[0], w4[0] | w3[2], w4[2]]
            let temp_45 = _mm256_permute2f128_ps(tw_0, tw_2, 0x31);
            // temp_67 = [w3[1], w4[1] | w3[3], w4[3]]
            let temp_67 = _mm256_permute2f128_ps(tw_1, tw_3, 0x31);

            // Extract w1, w2, w3, w4
            let w1 = _mm256_shuffle_ps(temp_01, temp_23, 0b01_00_01_00);
            let w2 = _mm256_shuffle_ps(temp_01, temp_23, 0b11_10_11_10);
            let w3 = _mm256_shuffle_ps(temp_45, temp_67, 0b01_00_01_00);
            let w4 = _mm256_shuffle_ps(temp_45, temp_67, 0b11_10_11_10);

            // Complex multiply: t1 = x1 * w1
            let w1_re = _mm256_shuffle_ps(w1, w1, 0b10_10_00_00);
            let w1_im = _mm256_shuffle_ps(w1, w1, 0b11_11_01_01);
            let prod_re = _mm256_mul_ps(w1_re, x1);
            let x1_swap = _mm256_shuffle_ps(x1, x1, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w1_im, x1_swap);
            let t1 = _mm256_addsub_ps(prod_re, prod_im);

            // Complex multiply: t2 = x2 * w2
            let w2_re = _mm256_shuffle_ps(w2, w2, 0b10_10_00_00);
            let w2_im = _mm256_shuffle_ps(w2, w2, 0b11_11_01_01);
            let prod_re = _mm256_mul_ps(w2_re, x2);
            let x2_swap = _mm256_shuffle_ps(x2, x2, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w2_im, x2_swap);
            let t2 = _mm256_addsub_ps(prod_re, prod_im);

            // Complex multiply: t3 = x3 * w3
            let w3_re = _mm256_shuffle_ps(w3, w3, 0b10_10_00_00);
            let w3_im = _mm256_shuffle_ps(w3, w3, 0b11_11_01_01);
            let prod_re = _mm256_mul_ps(w3_re, x3);
            let x3_swap = _mm256_shuffle_ps(x3, x3, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w3_im, x3_swap);
            let t3 = _mm256_addsub_ps(prod_re, prod_im);

            // Complex multiply: t4 = x4 * w4
            let w4_re = _mm256_shuffle_ps(w4, w4, 0b10_10_00_00);
            let w4_im = _mm256_shuffle_ps(w4, w4, 0b11_11_01_01);
            let prod_re = _mm256_mul_ps(w4_re, x4);
            let x4_swap = _mm256_shuffle_ps(x4, x4, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w4_im, x4_swap);
            let t4 = _mm256_addsub_ps(prod_re, prod_im);

            // Y0 = x0 + t1 + t2 + t3 + t4
            let y0 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, t1), t2), t3),
                t4,
            );

            // Y1 = x0 + t1*W_5^1 + t2*W_5^2 + t3*W_5^3 + t4*W_5^4
            // t1*W_5^1: complex multiply t1 by (W5_1_RE, W5_1_IM)
            let t1_re = _mm256_shuffle_ps(t1, t1, 0b10_10_00_00);
            let t1_im = _mm256_shuffle_ps(t1, t1, 0b11_11_01_01);
            let t1w51_re =
                _mm256_sub_ps(_mm256_mul_ps(t1_re, w5_1_re), _mm256_mul_ps(t1_im, w5_1_im));
            let t1w51_im =
                _mm256_add_ps(_mm256_mul_ps(t1_re, w5_1_im), _mm256_mul_ps(t1_im, w5_1_re));

            let t1_w51_lo = _mm256_unpacklo_ps(t1w51_re, t1w51_im);
            let t1_w51_hi = _mm256_unpackhi_ps(t1w51_re, t1w51_im);
            let t1_w51 = _mm256_shuffle_ps(t1_w51_lo, t1_w51_hi, 0b01_00_01_00);

            // t2*W_5^2: complex multiply t2 by (W5_2_RE, W5_2_IM)
            let t2_re = _mm256_shuffle_ps(t2, t2, 0b10_10_00_00);
            let t2_im = _mm256_shuffle_ps(t2, t2, 0b11_11_01_01);
            let t2w52_re =
                _mm256_sub_ps(_mm256_mul_ps(t2_re, w5_2_re), _mm256_mul_ps(t2_im, w5_2_im));
            let t2w52_im =
                _mm256_add_ps(_mm256_mul_ps(t2_re, w5_2_im), _mm256_mul_ps(t2_im, w5_2_re));

            let t2_w52_lo = _mm256_unpacklo_ps(t2w52_re, t2w52_im);
            let t2_w52_hi = _mm256_unpackhi_ps(t2w52_re, t2w52_im);
            let t2_w52 = _mm256_shuffle_ps(t2_w52_lo, t2_w52_hi, 0b01_00_01_00);

            // t3*W_5^3: complex multiply t3 by (W5_3_RE, W5_3_IM)
            let t3_re = _mm256_shuffle_ps(t3, t3, 0b10_10_00_00);
            let t3_im = _mm256_shuffle_ps(t3, t3, 0b11_11_01_01);
            let t3w53_re =
                _mm256_sub_ps(_mm256_mul_ps(t3_re, w5_3_re), _mm256_mul_ps(t3_im, w5_3_im));
            let t3w53_im =
                _mm256_add_ps(_mm256_mul_ps(t3_re, w5_3_im), _mm256_mul_ps(t3_im, w5_3_re));

            let t3_w53_lo = _mm256_unpacklo_ps(t3w53_re, t3w53_im);
            let t3_w53_hi = _mm256_unpackhi_ps(t3w53_re, t3w53_im);
            let t3_w53 = _mm256_shuffle_ps(t3_w53_lo, t3_w53_hi, 0b01_00_01_00);

            // t4*W_5^4: complex multiply t4 by (W5_4_RE, W5_4_IM)
            let t4_re = _mm256_shuffle_ps(t4, t4, 0b10_10_00_00);
            let t4_im = _mm256_shuffle_ps(t4, t4, 0b11_11_01_01);
            let t4w54_re =
                _mm256_sub_ps(_mm256_mul_ps(t4_re, w5_4_re), _mm256_mul_ps(t4_im, w5_4_im));
            let t4w54_im =
                _mm256_add_ps(_mm256_mul_ps(t4_re, w5_4_im), _mm256_mul_ps(t4_im, w5_4_re));

            let t4_w54_lo = _mm256_unpacklo_ps(t4w54_re, t4w54_im);
            let t4_w54_hi = _mm256_unpackhi_ps(t4w54_re, t4w54_im);
            let t4_w54 = _mm256_shuffle_ps(t4_w54_lo, t4_w54_hi, 0b01_00_01_00);

            let y1 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, t1_w51), t2_w52), t3_w53),
                t4_w54,
            );

            // Y2 = x0 + t1*W_5^2 + t2*W_5^4 + t3*W_5^1 + t4*W_5^3
            // t1*W_5^2
            let t1w52_re =
                _mm256_sub_ps(_mm256_mul_ps(t1_re, w5_2_re), _mm256_mul_ps(t1_im, w5_2_im));
            let t1w52_im =
                _mm256_add_ps(_mm256_mul_ps(t1_re, w5_2_im), _mm256_mul_ps(t1_im, w5_2_re));

            let t1_w52_lo = _mm256_unpacklo_ps(t1w52_re, t1w52_im);
            let t1_w52_hi = _mm256_unpackhi_ps(t1w52_re, t1w52_im);
            let t1_w52 = _mm256_shuffle_ps(t1_w52_lo, t1_w52_hi, 0b01_00_01_00);

            // t2*W_5^4
            let t2w54_re =
                _mm256_sub_ps(_mm256_mul_ps(t2_re, w5_4_re), _mm256_mul_ps(t2_im, w5_4_im));
            let t2w54_im =
                _mm256_add_ps(_mm256_mul_ps(t2_re, w5_4_im), _mm256_mul_ps(t2_im, w5_4_re));

            let t2_w54_lo = _mm256_unpacklo_ps(t2w54_re, t2w54_im);
            let t2_w54_hi = _mm256_unpackhi_ps(t2w54_re, t2w54_im);
            let t2_w54 = _mm256_shuffle_ps(t2_w54_lo, t2_w54_hi, 0b01_00_01_00);

            // t3*W_5^1
            let t3w51_re =
                _mm256_sub_ps(_mm256_mul_ps(t3_re, w5_1_re), _mm256_mul_ps(t3_im, w5_1_im));
            let t3w51_im =
                _mm256_add_ps(_mm256_mul_ps(t3_re, w5_1_im), _mm256_mul_ps(t3_im, w5_1_re));

            let t3_w51_lo = _mm256_unpacklo_ps(t3w51_re, t3w51_im);
            let t3_w51_hi = _mm256_unpackhi_ps(t3w51_re, t3w51_im);
            let t3_w51 = _mm256_shuffle_ps(t3_w51_lo, t3_w51_hi, 0b01_00_01_00);

            // t4*W_5^3
            let t4w53_re =
                _mm256_sub_ps(_mm256_mul_ps(t4_re, w5_3_re), _mm256_mul_ps(t4_im, w5_3_im));
            let t4w53_im =
                _mm256_add_ps(_mm256_mul_ps(t4_re, w5_3_im), _mm256_mul_ps(t4_im, w5_3_re));

            let t4_w53_lo = _mm256_unpacklo_ps(t4w53_re, t4w53_im);
            let t4_w53_hi = _mm256_unpackhi_ps(t4w53_re, t4w53_im);
            let t4_w53 = _mm256_shuffle_ps(t4_w53_lo, t4_w53_hi, 0b01_00_01_00);

            let y2 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, t1_w52), t2_w54), t3_w51),
                t4_w53,
            );

            // Y3 = x0 + t1*W_5^3 + t2*W_5^1 + t3*W_5^4 + t4*W_5^2
            // t1*W_5^3
            let t1w53_re =
                _mm256_sub_ps(_mm256_mul_ps(t1_re, w5_3_re), _mm256_mul_ps(t1_im, w5_3_im));
            let t1w53_im =
                _mm256_add_ps(_mm256_mul_ps(t1_re, w5_3_im), _mm256_mul_ps(t1_im, w5_3_re));

            let t1_w53_lo = _mm256_unpacklo_ps(t1w53_re, t1w53_im);
            let t1_w53_hi = _mm256_unpackhi_ps(t1w53_re, t1w53_im);
            let t1_w53 = _mm256_shuffle_ps(t1_w53_lo, t1_w53_hi, 0b01_00_01_00);

            // t2*W_5^1
            let t2w51_re =
                _mm256_sub_ps(_mm256_mul_ps(t2_re, w5_1_re), _mm256_mul_ps(t2_im, w5_1_im));
            let t2w51_im =
                _mm256_add_ps(_mm256_mul_ps(t2_re, w5_1_im), _mm256_mul_ps(t2_im, w5_1_re));

            let t2_w51_lo = _mm256_unpacklo_ps(t2w51_re, t2w51_im);
            let t2_w51_hi = _mm256_unpackhi_ps(t2w51_re, t2w51_im);
            let t2_w51 = _mm256_shuffle_ps(t2_w51_lo, t2_w51_hi, 0b01_00_01_00);

            // t3*W_5^4
            let t3w54_re =
                _mm256_sub_ps(_mm256_mul_ps(t3_re, w5_4_re), _mm256_mul_ps(t3_im, w5_4_im));
            let t3w54_im =
                _mm256_add_ps(_mm256_mul_ps(t3_re, w5_4_im), _mm256_mul_ps(t3_im, w5_4_re));

            let t3_w54_lo = _mm256_unpacklo_ps(t3w54_re, t3w54_im);
            let t3_w54_hi = _mm256_unpackhi_ps(t3w54_re, t3w54_im);
            let t3_w54 = _mm256_shuffle_ps(t3_w54_lo, t3_w54_hi, 0b01_00_01_00);

            // t4*W_5^2
            let t4w52_re =
                _mm256_sub_ps(_mm256_mul_ps(t4_re, w5_2_re), _mm256_mul_ps(t4_im, w5_2_im));
            let t4w52_im =
                _mm256_add_ps(_mm256_mul_ps(t4_re, w5_2_im), _mm256_mul_ps(t4_im, w5_2_re));

            let t4_w52_lo = _mm256_unpacklo_ps(t4w52_re, t4w52_im);
            let t4_w52_hi = _mm256_unpackhi_ps(t4w52_re, t4w52_im);
            let t4_w52 = _mm256_shuffle_ps(t4_w52_lo, t4_w52_hi, 0b01_00_01_00);

            let y3 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, t1_w53), t2_w51), t3_w54),
                t4_w52,
            );

            // Y4 = x0 + t1*W_5^4 + t2*W_5^3 + t3*W_5^2 + t4*W_5^1
            // t1*W_5^4
            let t1w54_re =
                _mm256_sub_ps(_mm256_mul_ps(t1_re, w5_4_re), _mm256_mul_ps(t1_im, w5_4_im));
            let t1w54_im =
                _mm256_add_ps(_mm256_mul_ps(t1_re, w5_4_im), _mm256_mul_ps(t1_im, w5_4_re));

            let t1_w54_lo = _mm256_unpacklo_ps(t1w54_re, t1w54_im);
            let t1_w54_hi = _mm256_unpackhi_ps(t1w54_re, t1w54_im);
            let t1_w54 = _mm256_shuffle_ps(t1_w54_lo, t1_w54_hi, 0b01_00_01_00);

            // t2*W_5^3
            let t2w53_re =
                _mm256_sub_ps(_mm256_mul_ps(t2_re, w5_3_re), _mm256_mul_ps(t2_im, w5_3_im));
            let t2w53_im =
                _mm256_add_ps(_mm256_mul_ps(t2_re, w5_3_im), _mm256_mul_ps(t2_im, w5_3_re));

            let t2_w53_lo = _mm256_unpacklo_ps(t2w53_re, t2w53_im);
            let t2_w53_hi = _mm256_unpackhi_ps(t2w53_re, t2w53_im);
            let t2_w53 = _mm256_shuffle_ps(t2_w53_lo, t2_w53_hi, 0b01_00_01_00);

            // t3*W_5^2
            let t3w52_re =
                _mm256_sub_ps(_mm256_mul_ps(t3_re, w5_2_re), _mm256_mul_ps(t3_im, w5_2_im));
            let t3w52_im =
                _mm256_add_ps(_mm256_mul_ps(t3_re, w5_2_im), _mm256_mul_ps(t3_im, w5_2_re));

            let t3_w52_lo = _mm256_unpacklo_ps(t3w52_re, t3w52_im);
            let t3_w52_hi = _mm256_unpackhi_ps(t3w52_re, t3w52_im);
            let t3_w52 = _mm256_shuffle_ps(t3_w52_lo, t3_w52_hi, 0b01_00_01_00);

            // t4*W_5^1
            let t4w51_re =
                _mm256_sub_ps(_mm256_mul_ps(t4_re, w5_1_re), _mm256_mul_ps(t4_im, w5_1_im));
            let t4w51_im =
                _mm256_add_ps(_mm256_mul_ps(t4_re, w5_1_im), _mm256_mul_ps(t4_im, w5_1_re));

            let t4_w51_lo = _mm256_unpacklo_ps(t4w51_re, t4w51_im);
            let t4_w51_hi = _mm256_unpackhi_ps(t4w51_re, t4w51_im);
            let t4_w51 = _mm256_shuffle_ps(t4_w51_lo, t4_w51_hi, 0b01_00_01_00);

            let y4 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, t1_w54), t2_w53), t3_w52),
                t4_w51,
            );

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm256_storeu_ps(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            _mm256_storeu_ps(y3_ptr, y3);
            let y4_ptr = data.as_mut_ptr().add(idx + 4 * num_columns) as *mut f32;
            _mm256_storeu_ps(y4_ptr, y4);
        }

        butterfly_5_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}

/// AVX+FMA implementation: processes 4 columns at once using fused multiply-add.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[target_feature(enable = "avx,fma")]
unsafe fn butterfly_5_avx_fma(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 4) * 4;

        // Broadcast W5 constants for SIMD operations.
        let w5_1_re = _mm256_set1_ps(W5_1_RE);
        let w5_1_im = _mm256_set1_ps(W5_1_IM);
        let w5_2_re = _mm256_set1_ps(W5_2_RE);
        let w5_2_im = _mm256_set1_ps(W5_2_IM);
        let w5_3_re = _mm256_set1_ps(W5_3_RE);
        let w5_3_im = _mm256_set1_ps(W5_3_IM);
        let w5_4_re = _mm256_set1_ps(W5_4_RE);
        let w5_4_im = _mm256_set1_ps(W5_4_IM);

        for idx in (0..simd_cols).step_by(4) {
            // Load 4 complex numbers from each row.
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = _mm256_loadu_ps(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = _mm256_loadu_ps(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = _mm256_loadu_ps(x2_ptr);

            let x3_ptr = data.as_ptr().add(idx + 3 * num_columns) as *const f32;
            let x3 = _mm256_loadu_ps(x3_ptr);

            let x4_ptr = data.as_ptr().add(idx + 4 * num_columns) as *const f32;
            let x4 = _mm256_loadu_ps(x4_ptr);

            // Load 16 twiddle factors.
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 4) as *const f32;
            let tw_0 = _mm256_loadu_ps(tw_ptr);
            let tw_1 = _mm256_loadu_ps(tw_ptr.add(8));
            let tw_2 = _mm256_loadu_ps(tw_ptr.add(16));
            let tw_3 = _mm256_loadu_ps(tw_ptr.add(24));

            // Extract w1, w2, w3, w4 for all 4 columns.
            let temp_01 = _mm256_permute2f128_ps(tw_0, tw_2, 0x20);
            let temp_23 = _mm256_permute2f128_ps(tw_1, tw_3, 0x20);
            let temp_45 = _mm256_permute2f128_ps(tw_0, tw_2, 0x31);
            let temp_67 = _mm256_permute2f128_ps(tw_1, tw_3, 0x31);

            let w1 = _mm256_shuffle_ps(temp_01, temp_23, 0b01_00_01_00);
            let w2 = _mm256_shuffle_ps(temp_01, temp_23, 0b11_10_11_10);
            let w3 = _mm256_shuffle_ps(temp_45, temp_67, 0b01_00_01_00);
            let w4 = _mm256_shuffle_ps(temp_45, temp_67, 0b11_10_11_10);

            // Complex multiply using FMA: t1 = x1 * w1
            let w1_re = _mm256_shuffle_ps(w1, w1, 0b10_10_00_00);
            let w1_im = _mm256_shuffle_ps(w1, w1, 0b11_11_01_01);
            let x1_swap = _mm256_shuffle_ps(x1, x1, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w1_im, x1_swap);
            let t1 = _mm256_fmaddsub_ps(w1_re, x1, prod_im);

            // Complex multiply using FMA: t2 = x2 * w2
            let w2_re = _mm256_shuffle_ps(w2, w2, 0b10_10_00_00);
            let w2_im = _mm256_shuffle_ps(w2, w2, 0b11_11_01_01);
            let x2_swap = _mm256_shuffle_ps(x2, x2, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w2_im, x2_swap);
            let t2 = _mm256_fmaddsub_ps(w2_re, x2, prod_im);

            // Complex multiply using FMA: t3 = x3 * w3
            let w3_re = _mm256_shuffle_ps(w3, w3, 0b10_10_00_00);
            let w3_im = _mm256_shuffle_ps(w3, w3, 0b11_11_01_01);
            let x3_swap = _mm256_shuffle_ps(x3, x3, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w3_im, x3_swap);
            let t3 = _mm256_fmaddsub_ps(w3_re, x3, prod_im);

            // Complex multiply using FMA: t4 = x4 * w4
            let w4_re = _mm256_shuffle_ps(w4, w4, 0b10_10_00_00);
            let w4_im = _mm256_shuffle_ps(w4, w4, 0b11_11_01_01);
            let x4_swap = _mm256_shuffle_ps(x4, x4, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w4_im, x4_swap);
            let t4 = _mm256_fmaddsub_ps(w4_re, x4, prod_im);

            // Y0 = x0 + t1 + t2 + t3 + t4
            let y0 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, t1), t2), t3),
                t4,
            );

            // Y1 = x0 + t1*W_5^1 + t2*W_5^2 + t3*W_5^3 + t4*W_5^4
            // t1*W_5^1: complex multiply using FMA
            let t1_re = _mm256_shuffle_ps(t1, t1, 0b10_10_00_00);
            let t1_im = _mm256_shuffle_ps(t1, t1, 0b11_11_01_01);
            let t1w51_re = _mm256_fmsub_ps(t1_re, w5_1_re, _mm256_mul_ps(t1_im, w5_1_im));
            let t1w51_im = _mm256_fmadd_ps(t1_re, w5_1_im, _mm256_mul_ps(t1_im, w5_1_re));

            let t1_w51_lo = _mm256_unpacklo_ps(t1w51_re, t1w51_im);
            let t1_w51_hi = _mm256_unpackhi_ps(t1w51_re, t1w51_im);
            let t1_w51 = _mm256_shuffle_ps(t1_w51_lo, t1_w51_hi, 0b01_00_01_00);

            // t2*W_5^2: complex multiply using FMA
            let t2_re = _mm256_shuffle_ps(t2, t2, 0b10_10_00_00);
            let t2_im = _mm256_shuffle_ps(t2, t2, 0b11_11_01_01);
            let t2w52_re = _mm256_fmsub_ps(t2_re, w5_2_re, _mm256_mul_ps(t2_im, w5_2_im));
            let t2w52_im = _mm256_fmadd_ps(t2_re, w5_2_im, _mm256_mul_ps(t2_im, w5_2_re));

            let t2_w52_lo = _mm256_unpacklo_ps(t2w52_re, t2w52_im);
            let t2_w52_hi = _mm256_unpackhi_ps(t2w52_re, t2w52_im);
            let t2_w52 = _mm256_shuffle_ps(t2_w52_lo, t2_w52_hi, 0b01_00_01_00);

            // t3*W_5^3: complex multiply using FMA
            let t3_re = _mm256_shuffle_ps(t3, t3, 0b10_10_00_00);
            let t3_im = _mm256_shuffle_ps(t3, t3, 0b11_11_01_01);
            let t3w53_re = _mm256_fmsub_ps(t3_re, w5_3_re, _mm256_mul_ps(t3_im, w5_3_im));
            let t3w53_im = _mm256_fmadd_ps(t3_re, w5_3_im, _mm256_mul_ps(t3_im, w5_3_re));

            let t3_w53_lo = _mm256_unpacklo_ps(t3w53_re, t3w53_im);
            let t3_w53_hi = _mm256_unpackhi_ps(t3w53_re, t3w53_im);
            let t3_w53 = _mm256_shuffle_ps(t3_w53_lo, t3_w53_hi, 0b01_00_01_00);

            // t4*W_5^4: complex multiply using FMA
            let t4_re = _mm256_shuffle_ps(t4, t4, 0b10_10_00_00);
            let t4_im = _mm256_shuffle_ps(t4, t4, 0b11_11_01_01);
            let t4w54_re = _mm256_fmsub_ps(t4_re, w5_4_re, _mm256_mul_ps(t4_im, w5_4_im));
            let t4w54_im = _mm256_fmadd_ps(t4_re, w5_4_im, _mm256_mul_ps(t4_im, w5_4_re));

            let t4_w54_lo = _mm256_unpacklo_ps(t4w54_re, t4w54_im);
            let t4_w54_hi = _mm256_unpackhi_ps(t4w54_re, t4w54_im);
            let t4_w54 = _mm256_shuffle_ps(t4_w54_lo, t4_w54_hi, 0b01_00_01_00);

            let y1 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, t1_w51), t2_w52), t3_w53),
                t4_w54,
            );

            // Y2 = x0 + t1*W_5^2 + t2*W_5^4 + t3*W_5^1 + t4*W_5^3
            // t1*W_5^2
            let t1w52_re = _mm256_fmsub_ps(t1_re, w5_2_re, _mm256_mul_ps(t1_im, w5_2_im));
            let t1w52_im = _mm256_fmadd_ps(t1_re, w5_2_im, _mm256_mul_ps(t1_im, w5_2_re));

            let t1_w52_lo = _mm256_unpacklo_ps(t1w52_re, t1w52_im);
            let t1_w52_hi = _mm256_unpackhi_ps(t1w52_re, t1w52_im);
            let t1_w52 = _mm256_shuffle_ps(t1_w52_lo, t1_w52_hi, 0b01_00_01_00);

            // t2*W_5^4
            let t2w54_re = _mm256_fmsub_ps(t2_re, w5_4_re, _mm256_mul_ps(t2_im, w5_4_im));
            let t2w54_im = _mm256_fmadd_ps(t2_re, w5_4_im, _mm256_mul_ps(t2_im, w5_4_re));

            let t2_w54_lo = _mm256_unpacklo_ps(t2w54_re, t2w54_im);
            let t2_w54_hi = _mm256_unpackhi_ps(t2w54_re, t2w54_im);
            let t2_w54 = _mm256_shuffle_ps(t2_w54_lo, t2_w54_hi, 0b01_00_01_00);

            // t3*W_5^1
            let t3w51_re = _mm256_fmsub_ps(t3_re, w5_1_re, _mm256_mul_ps(t3_im, w5_1_im));
            let t3w51_im = _mm256_fmadd_ps(t3_re, w5_1_im, _mm256_mul_ps(t3_im, w5_1_re));

            let t3_w51_lo = _mm256_unpacklo_ps(t3w51_re, t3w51_im);
            let t3_w51_hi = _mm256_unpackhi_ps(t3w51_re, t3w51_im);
            let t3_w51 = _mm256_shuffle_ps(t3_w51_lo, t3_w51_hi, 0b01_00_01_00);

            // t4*W_5^3
            let t4w53_re = _mm256_fmsub_ps(t4_re, w5_3_re, _mm256_mul_ps(t4_im, w5_3_im));
            let t4w53_im = _mm256_fmadd_ps(t4_re, w5_3_im, _mm256_mul_ps(t4_im, w5_3_re));

            let t4_w53_lo = _mm256_unpacklo_ps(t4w53_re, t4w53_im);
            let t4_w53_hi = _mm256_unpackhi_ps(t4w53_re, t4w53_im);
            let t4_w53 = _mm256_shuffle_ps(t4_w53_lo, t4_w53_hi, 0b01_00_01_00);

            let y2 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, t1_w52), t2_w54), t3_w51),
                t4_w53,
            );

            // Y3 = x0 + t1*W_5^3 + t2*W_5^1 + t3*W_5^4 + t4*W_5^2
            // t1*W_5^3
            let t1w53_re = _mm256_fmsub_ps(t1_re, w5_3_re, _mm256_mul_ps(t1_im, w5_3_im));
            let t1w53_im = _mm256_fmadd_ps(t1_re, w5_3_im, _mm256_mul_ps(t1_im, w5_3_re));

            let t1_w53_lo = _mm256_unpacklo_ps(t1w53_re, t1w53_im);
            let t1_w53_hi = _mm256_unpackhi_ps(t1w53_re, t1w53_im);
            let t1_w53 = _mm256_shuffle_ps(t1_w53_lo, t1_w53_hi, 0b01_00_01_00);

            // t2*W_5^1
            let t2w51_re = _mm256_fmsub_ps(t2_re, w5_1_re, _mm256_mul_ps(t2_im, w5_1_im));
            let t2w51_im = _mm256_fmadd_ps(t2_re, w5_1_im, _mm256_mul_ps(t2_im, w5_1_re));

            let t2_w51_lo = _mm256_unpacklo_ps(t2w51_re, t2w51_im);
            let t2_w51_hi = _mm256_unpackhi_ps(t2w51_re, t2w51_im);
            let t2_w51 = _mm256_shuffle_ps(t2_w51_lo, t2_w51_hi, 0b01_00_01_00);

            // t3*W_5^4
            let t3w54_re = _mm256_fmsub_ps(t3_re, w5_4_re, _mm256_mul_ps(t3_im, w5_4_im));
            let t3w54_im = _mm256_fmadd_ps(t3_re, w5_4_im, _mm256_mul_ps(t3_im, w5_4_re));

            let t3_w54_lo = _mm256_unpacklo_ps(t3w54_re, t3w54_im);
            let t3_w54_hi = _mm256_unpackhi_ps(t3w54_re, t3w54_im);
            let t3_w54 = _mm256_shuffle_ps(t3_w54_lo, t3_w54_hi, 0b01_00_01_00);

            // t4*W_5^2
            let t4w52_re = _mm256_fmsub_ps(t4_re, w5_2_re, _mm256_mul_ps(t4_im, w5_2_im));
            let t4w52_im = _mm256_fmadd_ps(t4_re, w5_2_im, _mm256_mul_ps(t4_im, w5_2_re));

            let t4_w52_lo = _mm256_unpacklo_ps(t4w52_re, t4w52_im);
            let t4_w52_hi = _mm256_unpackhi_ps(t4w52_re, t4w52_im);
            let t4_w52 = _mm256_shuffle_ps(t4_w52_lo, t4_w52_hi, 0b01_00_01_00);

            let y3 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, t1_w53), t2_w51), t3_w54),
                t4_w52,
            );

            // Y4 = x0 + t1*W_5^4 + t2*W_5^3 + t3*W_5^2 + t4*W_5^1
            // t1*W_5^4
            let t1w54_re = _mm256_fmsub_ps(t1_re, w5_4_re, _mm256_mul_ps(t1_im, w5_4_im));
            let t1w54_im = _mm256_fmadd_ps(t1_re, w5_4_im, _mm256_mul_ps(t1_im, w5_4_re));

            let t1_w54_lo = _mm256_unpacklo_ps(t1w54_re, t1w54_im);
            let t1_w54_hi = _mm256_unpackhi_ps(t1w54_re, t1w54_im);
            let t1_w54 = _mm256_shuffle_ps(t1_w54_lo, t1_w54_hi, 0b01_00_01_00);

            // t2*W_5^3
            let t2w53_re = _mm256_fmsub_ps(t2_re, w5_3_re, _mm256_mul_ps(t2_im, w5_3_im));
            let t2w53_im = _mm256_fmadd_ps(t2_re, w5_3_im, _mm256_mul_ps(t2_im, w5_3_re));

            let t2_w53_lo = _mm256_unpacklo_ps(t2w53_re, t2w53_im);
            let t2_w53_hi = _mm256_unpackhi_ps(t2w53_re, t2w53_im);
            let t2_w53 = _mm256_shuffle_ps(t2_w53_lo, t2_w53_hi, 0b01_00_01_00);

            // t3*W_5^2
            let t3w52_re = _mm256_fmsub_ps(t3_re, w5_2_re, _mm256_mul_ps(t3_im, w5_2_im));
            let t3w52_im = _mm256_fmadd_ps(t3_re, w5_2_im, _mm256_mul_ps(t3_im, w5_2_re));

            let t3_w52_lo = _mm256_unpacklo_ps(t3w52_re, t3w52_im);
            let t3_w52_hi = _mm256_unpackhi_ps(t3w52_re, t3w52_im);
            let t3_w52 = _mm256_shuffle_ps(t3_w52_lo, t3_w52_hi, 0b01_00_01_00);

            // t4*W_5^1
            let t4w51_re = _mm256_fmsub_ps(t4_re, w5_1_re, _mm256_mul_ps(t4_im, w5_1_im));
            let t4w51_im = _mm256_fmadd_ps(t4_re, w5_1_im, _mm256_mul_ps(t4_im, w5_1_re));

            let t4_w51_lo = _mm256_unpacklo_ps(t4w51_re, t4w51_im);
            let t4_w51_hi = _mm256_unpackhi_ps(t4w51_re, t4w51_im);
            let t4_w51 = _mm256_shuffle_ps(t4_w51_lo, t4_w51_hi, 0b01_00_01_00);

            let y4 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, t1_w54), t2_w53), t3_w52),
                t4_w51,
            );

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm256_storeu_ps(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            _mm256_storeu_ps(y3_ptr, y3);
            let y4_ptr = data.as_mut_ptr().add(idx + 4 * num_columns) as *mut f32;
            _mm256_storeu_ps(y4_ptr, y4);
        }

        butterfly_5_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
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
pub(crate) fn butterfly_5(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        if num_columns >= 4 {
            unsafe {
                return butterfly_5_avx_fma(data, stage_twiddles, num_columns);
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
                return butterfly_5_avx(data, stage_twiddles, num_columns);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    {
        if num_columns >= 2 {
            unsafe {
                return butterfly_5_sse3(data, stage_twiddles, num_columns);
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
                return butterfly_5_sse(data, stage_twiddles, num_columns);
            }
        }
    }

    butterfly_5_scalar(data, stage_twiddles, num_columns);
}

#[cfg(test)]
mod tests {
    use super::{super::test_helpers::test_butterfly_against_scalar, *};

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    fn test_butterfly_5_sse_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_5_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_5_sse(data, twiddles, num_columns);
            },
            5,
            4,
            "butterfly_5_sse",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
    fn test_butterfly_5_sse3_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_5_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_5_sse3(data, twiddles, num_columns);
            },
            5,
            4,
            "butterfly_5_sse3",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    fn test_butterfly_5_avx_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_5_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_5_avx(data, twiddles, num_columns);
            },
            5,
            4,
            "butterfly_5_avx",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    fn test_butterfly_5_avx_fma_vs_scalar() {
        test_butterfly_against_scalar(
            butterfly_5_scalar,
            |data, twiddles, num_columns| unsafe {
                butterfly_5_avx_fma(data, twiddles, num_columns);
            },
            5,
            4,
            "butterfly_5_avx_fma",
        );
    }
}
