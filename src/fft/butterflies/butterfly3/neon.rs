#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::Complex32;
/// NEON implementation: processes 2 columns at once using vld1q/vst1q.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_3_neon_x2(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 2) * 2;

        // Broadcast W3 constants for SIMD operations.
        let w3_1_re = vdupq_n_f32(super::W3_1_RE);
        let w3_1_im = vdupq_n_f32(super::W3_1_IM);
        let w3_2_re = vdupq_n_f32(super::W3_2_RE);
        let w3_2_im = vdupq_n_f32(super::W3_2_IM);

        for idx in (0..simd_cols).step_by(2) {
            // Load 2 complex numbers (4 f32) from each row as interleaved
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = vld1q_f32(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = vld1q_f32(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = vld1q_f32(x2_ptr);

            // Load 4 twiddle factors: w1[0], w2[0], w1[1], w2[1]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            let tw1_ptr = stage_twiddles.as_ptr().add(idx * 2) as *const f32;
            let tw_0 = vld1q_f32(tw1_ptr);
            // Layout: [w1[1].re, w1[1].im, w2[1].re, w2[1].im]
            let tw_1 = vld1q_f32(tw1_ptr.add(4));

            // Extract w1 and w2 using vtrn1q_f64/vtrn2q_f64
            // Treat as pairs of f64 to transpose: [w1[0], w2[0]] and [w1[1], w2[1]]
            // vtrn1q_f64 selects first element from each pair: [w1[0], w1[1]]
            // vtrn2q_f64 selects second element from each pair: [w2[0], w2[1]]
            let tw_0_f64 = vreinterpretq_f64_f32(tw_0);
            let tw_1_f64 = vreinterpretq_f64_f32(tw_1);
            let w1 = vreinterpretq_f32_f64(vtrn1q_f64(tw_0_f64, tw_1_f64)); // [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            let w2 = vreinterpretq_f32_f64(vtrn2q_f64(tw_0_f64, tw_1_f64)); // [w2[0].re, w2[0].im, w2[1].re, w2[1].im]

            // Deinterleave x0, x1, x2, w1, w2 using vuzpq
            let x0_parts = vuzpq_f32(x0, x0); // .0 = real, .1 = imag
            let x1_parts = vuzpq_f32(x1, x1);
            let x2_parts = vuzpq_f32(x2, x2);
            let w1_parts = vuzpq_f32(w1, w1);
            let w2_parts = vuzpq_f32(w2, w2);

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

            // Y0 = x0 + t1 + t2
            let y0_re = vaddq_f32(vaddq_f32(x0_parts.0, t1_re), t2_re);
            let y0_im = vaddq_f32(vaddq_f32(x0_parts.1, t1_im), t2_im);

            // Y1 = x0 + t1*W_3^1 + t2*W_3^2
            // t1*W_3^1: complex multiply t1 by (super::W3_1_RE, super::W3_1_IM)
            let t1w31_re = vmulq_f32(t1_re, w3_1_re);
            let t1w31_re = vmlsq_f32(t1w31_re, t1_im, w3_1_im);

            let t1w31_im = vmulq_f32(t1_re, w3_1_im);
            let t1w31_im = vmlaq_f32(t1w31_im, t1_im, w3_1_re);

            // t2*W_3^2: complex multiply t2 by (super::W3_2_RE, super::W3_2_IM)
            let t2w32_re = vmulq_f32(t2_re, w3_2_re);
            let t2w32_re = vmlsq_f32(t2w32_re, t2_im, w3_2_im);

            let t2w32_im = vmulq_f32(t2_re, w3_2_im);
            let t2w32_im = vmlaq_f32(t2w32_im, t2_im, w3_2_re);

            let y1_re = vaddq_f32(vaddq_f32(x0_parts.0, t1w31_re), t2w32_re);
            let y1_im = vaddq_f32(vaddq_f32(x0_parts.1, t1w31_im), t2w32_im);

            // Y2 = x0 + t1*W_3^2 + t2*W_3^1
            // t1*W_3^2: complex multiply t1 by (super::W3_2_RE, super::W3_2_IM)
            let t1w32_re = vmulq_f32(t1_re, w3_2_re);
            let t1w32_re = vmlsq_f32(t1w32_re, t1_im, w3_2_im);

            let t1w32_im = vmulq_f32(t1_re, w3_2_im);
            let t1w32_im = vmlaq_f32(t1w32_im, t1_im, w3_2_re);

            // t2*W_3^1: complex multiply t2 by (super::W3_1_RE, super::W3_1_IM)
            let t2w31_re = vmulq_f32(t2_re, w3_1_re);
            let t2w31_re = vmlsq_f32(t2w31_re, t2_im, w3_1_im);

            let t2w31_im = vmulq_f32(t2_re, w3_1_im);
            let t2w31_im = vmlaq_f32(t2w31_im, t2_im, w3_1_re);

            let y2_re = vaddq_f32(vaddq_f32(x0_parts.0, t1w32_re), t2w31_re);
            let y2_im = vaddq_f32(vaddq_f32(x0_parts.1, t1w32_im), t2w31_im);

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
        }

        super::butterfly_3_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}
/// NEON implementation: processes 4 columns at once using vld2q/vst2q.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_3_neon_x4(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 4) * 4;

        // Broadcast W3 constants for SIMD operations.
        let w3_1_re = vdupq_n_f32(super::W3_1_RE);
        let w3_1_im = vdupq_n_f32(super::W3_1_IM);
        let w3_2_re = vdupq_n_f32(super::W3_2_RE);
        let w3_2_im = vdupq_n_f32(super::W3_2_IM);

        for idx in (0..simd_cols).step_by(4) {
            // Load 4 complex numbers using vld2q (automatically deinterleaves into real/imag)
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = vld2q_f32(x0_ptr); // x0.0 = real parts, x0.1 = imag parts

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = vld2q_f32(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = vld2q_f32(x2_ptr);

            // Load twiddle factors: 8 complex numbers total (w1[0..3], w2[0..3])
            // Memory layout: [w1[0], w2[0], w1[1], w2[1], w1[2], w2[2], w1[3], w2[3]]
            // We need to extract: w1 = [w1[0], w1[1], w1[2], w1[3]], w2 = [w2[0], w2[1], w2[2], w2[3]]
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 2) as *const f32;

            // Load first 4 twiddle pairs: [w1[0], w2[0], w1[1], w2[1]]
            let tw_01 = vld2q_f32(tw_ptr); // tw_01.0 = [w1[0].re, w2[0].re, w1[1].re, w2[1].re], tw_01.1 = imag

            // Load second 4 twiddle pairs: [w1[2], w2[2], w1[3], w2[3]]
            let tw_23 = vld2q_f32(tw_ptr.add(8)); // tw_23.0 = [w1[2].re, w2[2].re, w1[3].re, w2[3].re], tw_23.1 = imag

            // Extract w1 and w2 by deinterleaving the pairs
            // vuzpq extracts even/odd indexed elements
            let tw_01_parts = vuzpq_f32(tw_01.0, tw_01.1); // .0 = [w1[0].re, w1[1].re, w1[0].im, w1[1].im], .1 = [w2[0].re, w2[1].re, w2[0].im, w2[1].im]
            let tw_23_parts = vuzpq_f32(tw_23.0, tw_23.1);

            // Combine to get w1 and w2
            let w1_re = vcombine_f32(vget_low_f32(tw_01_parts.0), vget_low_f32(tw_23_parts.0)); // [w1[0].re, w1[1].re, w1[2].re, w1[3].re]
            let w1_im = vcombine_f32(vget_high_f32(tw_01_parts.0), vget_high_f32(tw_23_parts.0)); // [w1[0].im, w1[1].im, w1[2].im, w1[3].im]
            let w2_re = vcombine_f32(vget_low_f32(tw_01_parts.1), vget_low_f32(tw_23_parts.1));
            let w2_im = vcombine_f32(vget_high_f32(tw_01_parts.1), vget_high_f32(tw_23_parts.1));

            // Complex multiply: t1 = x1 * w1
            // t1.re = w1.re * x1.re - w1.im * x1.im
            // t1.im = w1.re * x1.im + w1.im * x1.re
            let t1_re = vmulq_f32(w1_re, x1.0);
            let t1_re = vmlsq_f32(t1_re, w1_im, x1.1);

            let t1_im = vmulq_f32(w1_re, x1.1);
            let t1_im = vmlaq_f32(t1_im, w1_im, x1.0);

            // Complex multiply: t2 = x2 * w2
            let t2_re = vmulq_f32(w2_re, x2.0);
            let t2_re = vmlsq_f32(t2_re, w2_im, x2.1);

            let t2_im = vmulq_f32(w2_re, x2.1);
            let t2_im = vmlaq_f32(t2_im, w2_im, x2.0);

            // Y0 = x0 + t1 + t2
            let y0_re = vaddq_f32(vaddq_f32(x0.0, t1_re), t2_re);
            let y0_im = vaddq_f32(vaddq_f32(x0.1, t1_im), t2_im);

            // Y1 = x0 + t1*W_3^1 + t2*W_3^2
            // t1*W_3^1: complex multiply t1 by (super::W3_1_RE, super::W3_1_IM)
            let t1w31_re = vmulq_f32(t1_re, w3_1_re);
            let t1w31_re = vmlsq_f32(t1w31_re, t1_im, w3_1_im);

            let t1w31_im = vmulq_f32(t1_re, w3_1_im);
            let t1w31_im = vmlaq_f32(t1w31_im, t1_im, w3_1_re);

            // t2*W_3^2: complex multiply t2 by (super::W3_2_RE, super::W3_2_IM)
            let t2w32_re = vmulq_f32(t2_re, w3_2_re);
            let t2w32_re = vmlsq_f32(t2w32_re, t2_im, w3_2_im);

            let t2w32_im = vmulq_f32(t2_re, w3_2_im);
            let t2w32_im = vmlaq_f32(t2w32_im, t2_im, w3_2_re);

            let y1_re = vaddq_f32(vaddq_f32(x0.0, t1w31_re), t2w32_re);
            let y1_im = vaddq_f32(vaddq_f32(x0.1, t1w31_im), t2w32_im);

            // Y2 = x0 + t1*W_3^2 + t2*W_3^1
            // t1*W_3^2: complex multiply t1 by (super::W3_2_RE, super::W3_2_IM)
            let t1w32_re = vmulq_f32(t1_re, w3_2_re);
            let t1w32_re = vmlsq_f32(t1w32_re, t1_im, w3_2_im);

            let t1w32_im = vmulq_f32(t1_re, w3_2_im);
            let t1w32_im = vmlaq_f32(t1w32_im, t1_im, w3_2_re);

            // t2*W_3^1: complex multiply t2 by (super::W3_1_RE, super::W3_1_IM)
            let t2w31_re = vmulq_f32(t2_re, w3_1_re);
            let t2w31_re = vmlsq_f32(t2w31_re, t2_im, w3_1_im);

            let t2w31_im = vmulq_f32(t2_re, w3_1_im);
            let t2w31_im = vmlaq_f32(t2w31_im, t2_im, w3_1_re);

            let y2_re = vaddq_f32(vaddq_f32(x0.0, t1w32_re), t2w31_re);
            let y2_im = vaddq_f32(vaddq_f32(x0.1, t1w32_im), t2w31_im);

            // Store using vst2q (automatically reinterleaves)
            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            vst2q_f32(y0_ptr, float32x4x2_t(y0_re, y0_im));

            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            vst2q_f32(y1_ptr, float32x4x2_t(y1_re, y1_im));

            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            vst2q_f32(y2_ptr, float32x4x2_t(y2_re, y2_im));
        }

        super::butterfly_3_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}
