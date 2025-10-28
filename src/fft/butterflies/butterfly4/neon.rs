#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::Complex32;
/// NEON implementation: processes 2 columns at once using vld1q/vst1q.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_4_neon_x2(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 2) * 2;

        for idx in (0..simd_cols).step_by(2) {
            // Load 2 complex numbers (4 f32) from each row as interleaved
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = vld1q_f32(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = vld1q_f32(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = vld1q_f32(x2_ptr);

            let x3_ptr = data.as_ptr().add(idx + 3 * num_columns) as *const f32;
            let x3 = vld1q_f32(x3_ptr);

            // Load 6 twiddle factors: w1[0], w2[0], w3[0], w1[1], w2[1], w3[1]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 3) as *const f32;
            let tw_0 = vld1q_f32(tw_ptr);
            // Layout: [w3[0].re, w3[0].im, w1[1].re, w1[1].im]
            let tw_1 = vld1q_f32(tw_ptr.add(4));
            // Layout: [w2[1].re, w2[1].im, w3[1].re, w3[1].im]
            let tw_2 = vld1q_f32(tw_ptr.add(8));

            // Extract w1, w2, w3 using vcombine_f32 with vget_low/high
            // w1 = [w1[0].re, w1[0].im, w1[1].re, w1[1].im] = low(tw_0) + high(tw_1)
            let w1 = vcombine_f32(vget_low_f32(tw_0), vget_high_f32(tw_1));
            // w2 = [w2[0].re, w2[0].im, w2[1].re, w2[1].im] = high(tw_0) + low(tw_2)
            let w2 = vcombine_f32(vget_high_f32(tw_0), vget_low_f32(tw_2));
            // w3 = [w3[0].re, w3[0].im, w3[1].re, w3[1].im] = low(tw_1) + high(tw_2)
            let w3 = vcombine_f32(vget_low_f32(tw_1), vget_high_f32(tw_2));

            // Deinterleave x0, x1, x2, x3, w1, w2, w3 using vuzpq
            let x0_parts = vuzpq_f32(x0, x0); // .0 = real, .1 = imag
            let x1_parts = vuzpq_f32(x1, x1);
            let x2_parts = vuzpq_f32(x2, x2);
            let x3_parts = vuzpq_f32(x3, x3);
            let w1_parts = vuzpq_f32(w1, w1);
            let w2_parts = vuzpq_f32(w2, w2);
            let w3_parts = vuzpq_f32(w3, w3);

            // t0 = x0 (no twiddle)
            let t0_re = x0_parts.0;
            let t0_im = x0_parts.1;

            // Complex multiply: t1 = x1 * w1
            // t1.re = w1.re * x1.re - w1.im * x1.im
            // t1.im = w1.re * x1.im + w1.im * x1.re
            let t1_re = vmulq_f32(w1_parts.0, x1_parts.0);
            let t1_re = vmlsq_f32(t1_re, w1_parts.1, x1_parts.1);

            let t1_im = vmulq_f32(w1_parts.0, x1_parts.1);
            let t1_im = vmlaq_f32(t1_im, w1_parts.1, x1_parts.0);

            // Complex multiply: t2 = x2 * w2
            let t2_re = vmulq_f32(w2_parts.0, x2_parts.0);
            let t2_re = vmlsq_f32(t2_re, w2_parts.1, x2_parts.1);

            let t2_im = vmulq_f32(w2_parts.0, x2_parts.1);
            let t2_im = vmlaq_f32(t2_im, w2_parts.1, x2_parts.0);

            // Complex multiply: t3 = x3 * w3
            let t3_re = vmulq_f32(w3_parts.0, x3_parts.0);
            let t3_re = vmlsq_f32(t3_re, w3_parts.1, x3_parts.1);

            let t3_im = vmulq_f32(w3_parts.0, x3_parts.1);
            let t3_im = vmlaq_f32(t3_im, w3_parts.1, x3_parts.0);

            // Compute intermediate values for radix-4 butterfly.
            let u0_re = vaddq_f32(t0_re, t2_re); // u0 = t0 + t2
            let u0_im = vaddq_f32(t0_im, t2_im);

            let u1_re = vsubq_f32(t0_re, t2_re); // u1 = t0 - t2
            let u1_im = vsubq_f32(t0_im, t2_im);

            let u2_re = vaddq_f32(t1_re, t3_re); // u2 = t1 + t3
            let u2_im = vaddq_f32(t1_im, t3_im);

            let u3_re = vsubq_f32(t1_re, t3_re); // u3 = t1 - t3
            let u3_im = vsubq_f32(t1_im, t3_im);

            // Multiply u3 by -j: -j * (a + bi) = b - ai
            // u3_neg_j.re = u3.im, u3_neg_j.im = -u3.re
            let u3_neg_j_re = u3_im;
            let u3_neg_j_im = vnegq_f32(u3_re);

            // Multiply u3 by +j: +j * (a + bi) = -b + ai
            // u3_pos_j.re = -u3.im, u3_pos_j.im = u3.re
            let u3_pos_j_re = vnegq_f32(u3_im);
            let u3_pos_j_im = u3_re;

            // Combine to produce outputs.
            let y0_re = vaddq_f32(u0_re, u2_re); // y0 = u0 + u2
            let y0_im = vaddq_f32(u0_im, u2_im);

            let y1_re = vaddq_f32(u1_re, u3_neg_j_re); // y1 = u1 - j*u3
            let y1_im = vaddq_f32(u1_im, u3_neg_j_im);

            let y2_re = vsubq_f32(u0_re, u2_re); // y2 = u0 - u2
            let y2_im = vsubq_f32(u0_im, u2_im);

            let y3_re = vaddq_f32(u1_re, u3_pos_j_re); // y3 = u1 + j*u3
            let y3_im = vaddq_f32(u1_im, u3_pos_j_im);

            // Reinterleave using vzipq and store
            let y0_interleaved = vzipq_f32(y0_re, y0_im);
            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            vst1q_f32(y0_ptr, y0_interleaved.0);

            let y1_interleaved = vzipq_f32(y1_re, y1_im);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            vst1q_f32(y1_ptr, y1_interleaved.0);

            let y2_interleaved = vzipq_f32(y2_re, y2_im);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            vst1q_f32(y2_ptr, y2_interleaved.0);

            let y3_interleaved = vzipq_f32(y3_re, y3_im);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            vst1q_f32(y3_ptr, y3_interleaved.0);
        }

        super::butterfly_4_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}
