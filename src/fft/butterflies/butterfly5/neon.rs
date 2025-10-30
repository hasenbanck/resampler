use crate::Complex32;

/// NEON implementation: processes 2 columns at once.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_5_neon(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    start_col: usize,
    num_columns: usize,
) {
    use core::arch::aarch64::*;

    unsafe {
        let simd_cols = ((num_columns - start_col) / 2) * 2;

        // Broadcast W5 constants for SIMD operations.
        let w5_1_re = vdupq_n_f32(super::W5_1_RE);
        let w5_1_im = vdupq_n_f32(super::W5_1_IM);
        let w5_2_re = vdupq_n_f32(super::W5_2_RE);
        let w5_2_im = vdupq_n_f32(super::W5_2_IM);
        let w5_3_re = vdupq_n_f32(super::W5_3_RE);
        let w5_3_im = vdupq_n_f32(super::W5_3_IM);
        let w5_4_re = vdupq_n_f32(super::W5_4_RE);
        let w5_4_im = vdupq_n_f32(super::W5_4_IM);

        for idx in (start_col..start_col + simd_cols).step_by(2) {
            // Load 2 complex numbers from each row.
            // Layout: [x0[0].re, x0[0].im, x0[1].re, x0[1].im]
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = vld1q_f32(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = vld1q_f32(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = vld1q_f32(x2_ptr);

            let x3_ptr = data.as_ptr().add(idx + 3 * num_columns) as *const f32;
            let x3 = vld1q_f32(x3_ptr);

            let x4_ptr = data.as_ptr().add(idx + 4 * num_columns) as *const f32;
            let x4 = vld1q_f32(x4_ptr);

            // Load 8 twiddle factors: w1[0], w2[0], w3[0], w4[0], w1[1], w2[1], w3[1], w4[1]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 4) as *const f32;
            let tw_0 = vld1q_f32(tw_ptr);
            // Layout: [w3[0].re, w3[0].im, w4[0].re, w4[0].im]
            let tw_1 = vld1q_f32(tw_ptr.add(4));
            // Layout: [w1[1].re, w1[1].im, w2[1].re, w2[1].im]
            let tw_2 = vld1q_f32(tw_ptr.add(8));
            // Layout: [w3[1].re, w3[1].im, w4[1].re, w4[1].im]
            let tw_3 = vld1q_f32(tw_ptr.add(12));

            // Extract w1, w2, w3, w4 for both columns.
            // w1 = [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            // w2 = [w2[0].re, w2[0].im, w2[1].re, w2[1].im]
            // Use vtrn (transpose) and vzip to rearrange
            let trn_0_1 = vtrn1q_f32(tw_0, tw_2); // [w1[0].re, w1[1].re, w2[0].re, w2[1].re]
            let trn_0_2 = vtrn2q_f32(tw_0, tw_2); // [w1[0].im, w1[1].im, w2[0].im, w2[1].im]
            let w1 = vzip1q_f32(trn_0_1, trn_0_2); // [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            let w2 = vzip2q_f32(trn_0_1, trn_0_2); // [w2[0].re, w2[0].im, w2[1].re, w2[1].im]

            let trn_1_1 = vtrn1q_f32(tw_1, tw_3); // [w3[0].re, w3[1].re, w4[0].re, w4[1].re]
            let trn_1_2 = vtrn2q_f32(tw_1, tw_3); // [w3[0].im, w3[1].im, w4[0].im, w4[1].im]
            let w3 = vzip1q_f32(trn_1_1, trn_1_2); // [w3[0].re, w3[0].im, w3[1].re, w3[1].im]
            let w4 = vzip2q_f32(trn_1_1, trn_1_2); // [w4[0].re, w4[0].im, w4[1].re, w4[1].im]

            // Mask for manual addsub emulation.
            let neg_mask =
                vreinterpretq_f32_u32(vld1q_u32([0x80000000u32, 0, 0x80000000u32, 0].as_ptr()));

            // Complex multiply: t1 = x1 * w1
            let w1_re = vtrn1q_f32(w1, w1);
            let w1_im = vtrn2q_f32(w1, w1);
            let prod_re = vmulq_f32(w1_re, x1);
            let x1_swap = vrev64q_f32(x1);
            let prod_im = vmulq_f32(w1_im, x1_swap);
            let prod_im_adjusted = veorq_u32(
                vreinterpretq_u32_f32(prod_im),
                vreinterpretq_u32_f32(neg_mask),
            );
            let t1 = vaddq_f32(prod_re, vreinterpretq_f32_u32(prod_im_adjusted));

            // Complex multiply: t2 = x2 * w2
            let w2_re = vtrn1q_f32(w2, w2);
            let w2_im = vtrn2q_f32(w2, w2);
            let prod_re = vmulq_f32(w2_re, x2);
            let x2_swap = vrev64q_f32(x2);
            let prod_im = vmulq_f32(w2_im, x2_swap);
            let prod_im_adjusted = veorq_u32(
                vreinterpretq_u32_f32(prod_im),
                vreinterpretq_u32_f32(neg_mask),
            );
            let t2 = vaddq_f32(prod_re, vreinterpretq_f32_u32(prod_im_adjusted));

            // Complex multiply: t3 = x3 * w3
            let w3_re = vtrn1q_f32(w3, w3);
            let w3_im = vtrn2q_f32(w3, w3);
            let prod_re = vmulq_f32(w3_re, x3);
            let x3_swap = vrev64q_f32(x3);
            let prod_im = vmulq_f32(w3_im, x3_swap);
            let prod_im_adjusted = veorq_u32(
                vreinterpretq_u32_f32(prod_im),
                vreinterpretq_u32_f32(neg_mask),
            );
            let t3 = vaddq_f32(prod_re, vreinterpretq_f32_u32(prod_im_adjusted));

            // Complex multiply: t4 = x4 * w4
            let w4_re = vtrn1q_f32(w4, w4);
            let w4_im = vtrn2q_f32(w4, w4);
            let prod_re = vmulq_f32(w4_re, x4);
            let x4_swap = vrev64q_f32(x4);
            let prod_im = vmulq_f32(w4_im, x4_swap);
            let prod_im_adjusted = veorq_u32(
                vreinterpretq_u32_f32(prod_im),
                vreinterpretq_u32_f32(neg_mask),
            );
            let t4 = vaddq_f32(prod_re, vreinterpretq_f32_u32(prod_im_adjusted));

            // Y0 = x0 + t1 + t2 + t3 + t4
            let y0 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(x0, t1), t2), t3), t4);

            // Y1 = x0 + t1*W_5^1 + t2*W_5^2 + t3*W_5^3 + t4*W_5^4
            // t1*W_5^1: complex multiply t1 by (super::W5_1_RE, super::W5_1_IM)
            let t1_re = vtrn1q_f32(t1, t1);
            let t1_im = vtrn2q_f32(t1, t1);
            let t1w51_re = vsubq_f32(vmulq_f32(t1_re, w5_1_re), vmulq_f32(t1_im, w5_1_im));
            let t1w51_im = vaddq_f32(vmulq_f32(t1_re, w5_1_im), vmulq_f32(t1_im, w5_1_re));
            let t1_w51 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t1w51_re, t1w51_im)),
                vget_low_f32(vzip2q_f32(t1w51_re, t1w51_im)),
            );

            // t2*W_5^2: complex multiply t2 by (super::W5_2_RE, super::W5_2_IM)
            let t2_re = vtrn1q_f32(t2, t2);
            let t2_im = vtrn2q_f32(t2, t2);
            let t2w52_re = vsubq_f32(vmulq_f32(t2_re, w5_2_re), vmulq_f32(t2_im, w5_2_im));
            let t2w52_im = vaddq_f32(vmulq_f32(t2_re, w5_2_im), vmulq_f32(t2_im, w5_2_re));
            let t2_w52 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t2w52_re, t2w52_im)),
                vget_low_f32(vzip2q_f32(t2w52_re, t2w52_im)),
            );

            // t3*W_5^3: complex multiply t3 by (super::W5_3_RE, super::W5_3_IM)
            let t3_re = vtrn1q_f32(t3, t3);
            let t3_im = vtrn2q_f32(t3, t3);
            let t3w53_re = vsubq_f32(vmulq_f32(t3_re, w5_3_re), vmulq_f32(t3_im, w5_3_im));
            let t3w53_im = vaddq_f32(vmulq_f32(t3_re, w5_3_im), vmulq_f32(t3_im, w5_3_re));
            let t3_w53 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t3w53_re, t3w53_im)),
                vget_low_f32(vzip2q_f32(t3w53_re, t3w53_im)),
            );

            // t4*W_5^4: complex multiply t4 by (super::W5_4_RE, super::W5_4_IM)
            let t4_re = vtrn1q_f32(t4, t4);
            let t4_im = vtrn2q_f32(t4, t4);
            let t4w54_re = vsubq_f32(vmulq_f32(t4_re, w5_4_re), vmulq_f32(t4_im, w5_4_im));
            let t4w54_im = vaddq_f32(vmulq_f32(t4_re, w5_4_im), vmulq_f32(t4_im, w5_4_re));
            let t4_w54 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t4w54_re, t4w54_im)),
                vget_low_f32(vzip2q_f32(t4w54_re, t4w54_im)),
            );

            let y1 = vaddq_f32(
                vaddq_f32(vaddq_f32(vaddq_f32(x0, t1_w51), t2_w52), t3_w53),
                t4_w54,
            );

            // Y2 = x0 + t1*W_5^2 + t2*W_5^4 + t3*W_5^1 + t4*W_5^3
            // t1*W_5^2
            let t1w52_re = vsubq_f32(vmulq_f32(t1_re, w5_2_re), vmulq_f32(t1_im, w5_2_im));
            let t1w52_im = vaddq_f32(vmulq_f32(t1_re, w5_2_im), vmulq_f32(t1_im, w5_2_re));
            let t1_w52 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t1w52_re, t1w52_im)),
                vget_low_f32(vzip2q_f32(t1w52_re, t1w52_im)),
            );

            // t2*W_5^4
            let t2w54_re = vsubq_f32(vmulq_f32(t2_re, w5_4_re), vmulq_f32(t2_im, w5_4_im));
            let t2w54_im = vaddq_f32(vmulq_f32(t2_re, w5_4_im), vmulq_f32(t2_im, w5_4_re));
            let t2_w54 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t2w54_re, t2w54_im)),
                vget_low_f32(vzip2q_f32(t2w54_re, t2w54_im)),
            );

            // t3*W_5^1
            let t3w51_re = vsubq_f32(vmulq_f32(t3_re, w5_1_re), vmulq_f32(t3_im, w5_1_im));
            let t3w51_im = vaddq_f32(vmulq_f32(t3_re, w5_1_im), vmulq_f32(t3_im, w5_1_re));
            let t3_w51 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t3w51_re, t3w51_im)),
                vget_low_f32(vzip2q_f32(t3w51_re, t3w51_im)),
            );

            // t4*W_5^3
            let t4w53_re = vsubq_f32(vmulq_f32(t4_re, w5_3_re), vmulq_f32(t4_im, w5_3_im));
            let t4w53_im = vaddq_f32(vmulq_f32(t4_re, w5_3_im), vmulq_f32(t4_im, w5_3_re));
            let t4_w53 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t4w53_re, t4w53_im)),
                vget_low_f32(vzip2q_f32(t4w53_re, t4w53_im)),
            );

            let y2 = vaddq_f32(
                vaddq_f32(vaddq_f32(vaddq_f32(x0, t1_w52), t2_w54), t3_w51),
                t4_w53,
            );

            // Y3 = x0 + t1*W_5^3 + t2*W_5^1 + t3*W_5^4 + t4*W_5^2
            // t1*W_5^3
            let t1w53_re = vsubq_f32(vmulq_f32(t1_re, w5_3_re), vmulq_f32(t1_im, w5_3_im));
            let t1w53_im = vaddq_f32(vmulq_f32(t1_re, w5_3_im), vmulq_f32(t1_im, w5_3_re));
            let t1_w53 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t1w53_re, t1w53_im)),
                vget_low_f32(vzip2q_f32(t1w53_re, t1w53_im)),
            );

            // t2*W_5^1
            let t2w51_re = vsubq_f32(vmulq_f32(t2_re, w5_1_re), vmulq_f32(t2_im, w5_1_im));
            let t2w51_im = vaddq_f32(vmulq_f32(t2_re, w5_1_im), vmulq_f32(t2_im, w5_1_re));
            let t2_w51 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t2w51_re, t2w51_im)),
                vget_low_f32(vzip2q_f32(t2w51_re, t2w51_im)),
            );

            // t3*W_5^4
            let t3w54_re = vsubq_f32(vmulq_f32(t3_re, w5_4_re), vmulq_f32(t3_im, w5_4_im));
            let t3w54_im = vaddq_f32(vmulq_f32(t3_re, w5_4_im), vmulq_f32(t3_im, w5_4_re));
            let t3_w54 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t3w54_re, t3w54_im)),
                vget_low_f32(vzip2q_f32(t3w54_re, t3w54_im)),
            );

            // t4*W_5^2
            let t4w52_re = vsubq_f32(vmulq_f32(t4_re, w5_2_re), vmulq_f32(t4_im, w5_2_im));
            let t4w52_im = vaddq_f32(vmulq_f32(t4_re, w5_2_im), vmulq_f32(t4_im, w5_2_re));
            let t4_w52 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t4w52_re, t4w52_im)),
                vget_low_f32(vzip2q_f32(t4w52_re, t4w52_im)),
            );

            let y3 = vaddq_f32(
                vaddq_f32(vaddq_f32(vaddq_f32(x0, t1_w53), t2_w51), t3_w54),
                t4_w52,
            );

            // Y4 = x0 + t1*W_5^4 + t2*W_5^3 + t3*W_5^2 + t4*W_5^1
            // t1*W_5^4
            let t1w54_re = vsubq_f32(vmulq_f32(t1_re, w5_4_re), vmulq_f32(t1_im, w5_4_im));
            let t1w54_im = vaddq_f32(vmulq_f32(t1_re, w5_4_im), vmulq_f32(t1_im, w5_4_re));
            let t1_w54 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t1w54_re, t1w54_im)),
                vget_low_f32(vzip2q_f32(t1w54_re, t1w54_im)),
            );

            // t2*W_5^3
            let t2w53_re = vsubq_f32(vmulq_f32(t2_re, w5_3_re), vmulq_f32(t2_im, w5_3_im));
            let t2w53_im = vaddq_f32(vmulq_f32(t2_re, w5_3_im), vmulq_f32(t2_im, w5_3_re));
            let t2_w53 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t2w53_re, t2w53_im)),
                vget_low_f32(vzip2q_f32(t2w53_re, t2w53_im)),
            );

            // t3*W_5^2
            let t3w52_re = vsubq_f32(vmulq_f32(t3_re, w5_2_re), vmulq_f32(t3_im, w5_2_im));
            let t3w52_im = vaddq_f32(vmulq_f32(t3_re, w5_2_im), vmulq_f32(t3_im, w5_2_re));
            let t3_w52 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t3w52_re, t3w52_im)),
                vget_low_f32(vzip2q_f32(t3w52_re, t3w52_im)),
            );

            // t4*W_5^1
            let t4w51_re = vsubq_f32(vmulq_f32(t4_re, w5_1_re), vmulq_f32(t4_im, w5_1_im));
            let t4w51_im = vaddq_f32(vmulq_f32(t4_re, w5_1_im), vmulq_f32(t4_im, w5_1_re));
            let t4_w51 = vcombine_f32(
                vget_low_f32(vzip1q_f32(t4w51_re, t4w51_im)),
                vget_low_f32(vzip2q_f32(t4w51_re, t4w51_im)),
            );

            let y4 = vaddq_f32(
                vaddq_f32(vaddq_f32(vaddq_f32(x0, t1_w54), t2_w53), t3_w52),
                t4_w51,
            );

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            vst1q_f32(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            vst1q_f32(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            vst1q_f32(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            vst1q_f32(y3_ptr, y3);
            let y4_ptr = data.as_mut_ptr().add(idx + 4 * num_columns) as *mut f32;
            vst1q_f32(y4_ptr, y4);
        }

        super::butterfly_5_scalar(data, stage_twiddles, start_col + simd_cols, num_columns)
    }
}
