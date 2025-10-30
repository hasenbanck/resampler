use crate::Complex32;

/// NEON implementation: processes 2 columns at once.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_7_neon(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    start_col: usize,
    num_columns: usize,
) {
    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::*;

    unsafe {
        let simd_cols = ((num_columns - start_col) / 2) * 2;

        // Broadcast W7 constants for SIMD operations.
        let w7_1_re = vdupq_n_f32(super::W7_1_RE);
        let w7_1_im = vdupq_n_f32(super::W7_1_IM);
        let w7_2_re = vdupq_n_f32(super::W7_2_RE);
        let w7_2_im = vdupq_n_f32(super::W7_2_IM);
        let w7_3_re = vdupq_n_f32(super::W7_3_RE);
        let w7_3_im = vdupq_n_f32(super::W7_3_IM);
        let w7_4_re = vdupq_n_f32(super::W7_4_RE);
        let w7_4_im = vdupq_n_f32(super::W7_4_IM);
        let w7_5_re = vdupq_n_f32(super::W7_5_RE);
        let w7_5_im = vdupq_n_f32(super::W7_5_IM);
        let w7_6_re = vdupq_n_f32(super::W7_6_RE);
        let w7_6_im = vdupq_n_f32(super::W7_6_IM);

        for idx in (start_col..start_col + simd_cols).step_by(2) {
            // Load 2 complex numbers from each row.
            // Layout: [x[0].re, x[0].im, x[1].re, x[1].im]
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

            let x5_ptr = data.as_ptr().add(idx + 5 * num_columns) as *const f32;
            let x5 = vld1q_f32(x5_ptr);

            let x6_ptr = data.as_ptr().add(idx + 6 * num_columns) as *const f32;
            let x6 = vld1q_f32(x6_ptr);

            // Load 12 twiddle factors.
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 6) as *const f32;
            let tw_0 = vld1q_f32(tw_ptr);
            let tw_1 = vld1q_f32(tw_ptr.add(4));
            let tw_2 = vld1q_f32(tw_ptr.add(8));
            let tw_3 = vld1q_f32(tw_ptr.add(12));
            let tw_4 = vld1q_f32(tw_ptr.add(16));
            let tw_5 = vld1q_f32(tw_ptr.add(20));

            // Extract w1, w2, w3, w4, w5, w6 for both columns.
            // First transpose to group by w1/w2/w3/w4/w5/w6
            let trn_0_1 = vtrn1q_f32(tw_0, tw_3); // [w1[0].re, w1[1].re, w2[0].re, w2[1].re]
            let trn_0_2 = vtrn2q_f32(tw_0, tw_3); // [w1[0].im, w1[1].im, w2[0].im, w2[1].im]
            let trn_1_1 = vtrn1q_f32(tw_1, tw_4); // [w3[0].re, w3[1].re, w4[0].re, w4[1].re]
            let trn_1_2 = vtrn2q_f32(tw_1, tw_4); // [w3[0].im, w3[1].im, w4[0].im, w4[1].im]
            let trn_2_1 = vtrn1q_f32(tw_2, tw_5); // [w5[0].re, w5[1].re, w6[0].re, w6[1].re]
            let trn_2_2 = vtrn2q_f32(tw_2, tw_5); // [w5[0].im, w5[1].im, w6[0].im, w6[1].im]

            // Then zip to interleave re/im
            let w1 = vzip1q_f32(trn_0_1, trn_0_2); // [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            let w2 = vzip2q_f32(trn_0_1, trn_0_2); // [w2[0].re, w2[0].im, w2[1].re, w2[1].im]
            let w3 = vzip1q_f32(trn_1_1, trn_1_2); // [w3[0].re, w3[0].im, w3[1].re, w3[1].im]
            let w4 = vzip2q_f32(trn_1_1, trn_1_2); // [w4[0].re, w4[0].im, w4[1].re, w4[1].im]
            let w5 = vzip1q_f32(trn_2_1, trn_2_2); // [w5[0].re, w5[0].im, w5[1].re, w5[1].im]
            let w6 = vzip2q_f32(trn_2_1, trn_2_2); // [w6[0].re, w6[0].im, w6[1].re, w6[1].im]

            // Helper macro for complex multiply using NEON.
            macro_rules! cmul_neon {
                ($x:expr, $w:expr) => {{
                    let w_re = vtrn1q_f32($w, $w);
                    let w_im = vtrn2q_f32($w, $w);
                    let prod_re = vmulq_f32(w_re, $x);
                    let x_swap = vrev64q_f32($x);
                    let prod_im = vmulq_f32(w_im, x_swap);

                    // Emulate addsub for complex multiply.
                    let neg_mask = vreinterpretq_f32_u32(vld1q_u32(
                        [0x80000000u32, 0, 0x80000000u32, 0].as_ptr(),
                    ));
                    let prod_im_adjusted = veorq_u32(
                        vreinterpretq_u32_f32(prod_im),
                        vreinterpretq_u32_f32(neg_mask),
                    );
                    vaddq_f32(prod_re, vreinterpretq_f32_u32(prod_im_adjusted))
                }};
            }

            // Complex multiply: t1 = x1 * w1, t2 = x2 * w2, etc.
            let t1 = cmul_neon!(x1, w1);
            let t2 = cmul_neon!(x2, w2);
            let t3 = cmul_neon!(x3, w3);
            let t4 = cmul_neon!(x4, w4);
            let t5 = cmul_neon!(x5, w5);
            let t6 = cmul_neon!(x6, w6);

            // Y0 = x0 + t1 + t2 + t3 + t4 + t5 + t6
            let y0 = vaddq_f32(
                vaddq_f32(vaddq_f32(x0, t1), vaddq_f32(t2, t3)),
                vaddq_f32(vaddq_f32(t4, t5), t6),
            );

            // Helper macro for complex multiply by W7 constant.
            macro_rules! cmul_w7 {
                ($t:expr, $w_re:expr, $w_im:expr) => {{
                    let t_re = vtrn1q_f32($t, $t);
                    let t_im = vtrn2q_f32($t, $t);
                    let re = vsubq_f32(vmulq_f32(t_re, $w_re), vmulq_f32(t_im, $w_im));
                    let im = vaddq_f32(vmulq_f32(t_re, $w_im), vmulq_f32(t_im, $w_re));
                    vcombine_f32(
                        vget_low_f32(vzip1q_f32(re, im)),
                        vget_low_f32(vzip2q_f32(re, im)),
                    )
                }};
            }

            // Y1 = x0 + t1*W_7^1 + t2*W_7^2 + t3*W_7^3 + t4*W_7^4 + t5*W_7^5 + t6*W_7^6
            let t1_w71 = cmul_w7!(t1, w7_1_re, w7_1_im);
            let t2_w72 = cmul_w7!(t2, w7_2_re, w7_2_im);
            let t3_w73 = cmul_w7!(t3, w7_3_re, w7_3_im);
            let t4_w74 = cmul_w7!(t4, w7_4_re, w7_4_im);
            let t5_w75 = cmul_w7!(t5, w7_5_re, w7_5_im);
            let t6_w76 = cmul_w7!(t6, w7_6_re, w7_6_im);

            let y1 = vaddq_f32(
                vaddq_f32(vaddq_f32(x0, t1_w71), vaddq_f32(t2_w72, t3_w73)),
                vaddq_f32(vaddq_f32(t4_w74, t5_w75), t6_w76),
            );

            // Y2 = x0 + t1*W_7^2 + t2*W_7^4 + t3*W_7^6 + t4*W_7^1 + t5*W_7^3 + t6*W_7^5
            let t1_w72 = cmul_w7!(t1, w7_2_re, w7_2_im);
            let t2_w74 = cmul_w7!(t2, w7_4_re, w7_4_im);
            let t3_w76 = cmul_w7!(t3, w7_6_re, w7_6_im);
            let t4_w71 = cmul_w7!(t4, w7_1_re, w7_1_im);
            let t5_w73 = cmul_w7!(t5, w7_3_re, w7_3_im);
            let t6_w75 = cmul_w7!(t6, w7_5_re, w7_5_im);

            let y2 = vaddq_f32(
                vaddq_f32(vaddq_f32(x0, t1_w72), vaddq_f32(t2_w74, t3_w76)),
                vaddq_f32(vaddq_f32(t4_w71, t5_w73), t6_w75),
            );

            // Y3 = x0 + t1*W_7^3 + t2*W_7^6 + t3*W_7^2 + t4*W_7^5 + t5*W_7^1 + t6*W_7^4
            let t1_w73 = cmul_w7!(t1, w7_3_re, w7_3_im);
            let t2_w76 = cmul_w7!(t2, w7_6_re, w7_6_im);
            let t3_w72 = cmul_w7!(t3, w7_2_re, w7_2_im);
            let t4_w75 = cmul_w7!(t4, w7_5_re, w7_5_im);
            let t5_w71 = cmul_w7!(t5, w7_1_re, w7_1_im);
            let t6_w74 = cmul_w7!(t6, w7_4_re, w7_4_im);

            let y3 = vaddq_f32(
                vaddq_f32(vaddq_f32(x0, t1_w73), vaddq_f32(t2_w76, t3_w72)),
                vaddq_f32(vaddq_f32(t4_w75, t5_w71), t6_w74),
            );

            // Y4 = x0 + t1*W_7^4 + t2*W_7^1 + t3*W_7^5 + t4*W_7^2 + t5*W_7^6 + t6*W_7^3
            let t1_w74 = cmul_w7!(t1, w7_4_re, w7_4_im);
            let t2_w71 = cmul_w7!(t2, w7_1_re, w7_1_im);
            let t3_w75 = cmul_w7!(t3, w7_5_re, w7_5_im);
            let t4_w72 = cmul_w7!(t4, w7_2_re, w7_2_im);
            let t5_w76 = cmul_w7!(t5, w7_6_re, w7_6_im);
            let t6_w73 = cmul_w7!(t6, w7_3_re, w7_3_im);

            let y4 = vaddq_f32(
                vaddq_f32(vaddq_f32(x0, t1_w74), vaddq_f32(t2_w71, t3_w75)),
                vaddq_f32(vaddq_f32(t4_w72, t5_w76), t6_w73),
            );

            // Y5 = x0 + t1*W_7^5 + t2*W_7^3 + t3*W_7^1 + t4*W_7^6 + t5*W_7^4 + t6*W_7^2
            let t1_w75 = cmul_w7!(t1, w7_5_re, w7_5_im);
            let t2_w73 = cmul_w7!(t2, w7_3_re, w7_3_im);
            let t3_w71 = cmul_w7!(t3, w7_1_re, w7_1_im);
            let t4_w76 = cmul_w7!(t4, w7_6_re, w7_6_im);
            let t5_w74 = cmul_w7!(t5, w7_4_re, w7_4_im);
            let t6_w72 = cmul_w7!(t6, w7_2_re, w7_2_im);

            let y5 = vaddq_f32(
                vaddq_f32(vaddq_f32(x0, t1_w75), vaddq_f32(t2_w73, t3_w71)),
                vaddq_f32(vaddq_f32(t4_w76, t5_w74), t6_w72),
            );

            // Y6 = x0 + t1*W_7^6 + t2*W_7^5 + t3*W_7^4 + t4*W_7^3 + t5*W_7^2 + t6*W_7^1
            let t1_w76 = cmul_w7!(t1, w7_6_re, w7_6_im);
            let t2_w75 = cmul_w7!(t2, w7_5_re, w7_5_im);
            let t3_w74 = cmul_w7!(t3, w7_4_re, w7_4_im);
            let t4_w73 = cmul_w7!(t4, w7_3_re, w7_3_im);
            let t5_w72 = cmul_w7!(t5, w7_2_re, w7_2_im);
            let t6_w71 = cmul_w7!(t6, w7_1_re, w7_1_im);

            let y6 = vaddq_f32(
                vaddq_f32(vaddq_f32(x0, t1_w76), vaddq_f32(t2_w75, t3_w74)),
                vaddq_f32(vaddq_f32(t4_w73, t5_w72), t6_w71),
            );

            // Store results.
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
            let y5_ptr = data.as_mut_ptr().add(idx + 5 * num_columns) as *mut f32;
            vst1q_f32(y5_ptr, y5);
            let y6_ptr = data.as_mut_ptr().add(idx + 6 * num_columns) as *mut f32;
            vst1q_f32(y6_ptr, y6);
        }

        super::butterfly_7_scalar(data, stage_twiddles, start_col + simd_cols, num_columns);
    }
}
