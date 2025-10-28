#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::Complex32;

/// NEON implementation: processes 2 columns at once using vld1q/vst1q.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_2_neon_x2(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 2) * 2;

        for idx in (0..simd_cols).step_by(2) {
            // Load 2 complex numbers (4 f32) as interleaved: [u0.re, u0.im, u1.re, u1.im]
            let u_ptr = data.as_ptr().add(idx) as *const f32;
            let u = vld1q_f32(u_ptr);

            let d_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let d = vld1q_f32(d_ptr);

            let tw_ptr = stage_twiddles.as_ptr().add(idx) as *const f32;
            let tw = vld1q_f32(tw_ptr);

            // Deinterleave real and imaginary parts using vuzpq
            // vuzpq extracts even-indexed elements (.0) and odd-indexed elements (.1)
            let u_parts = vuzpq_f32(u, u); // .0 = [u0.re, u1.re, ...], .1 = [u0.im, u1.im, ...]
            let d_parts = vuzpq_f32(d, d); // .0 = [d0.re, d1.re, ...], .1 = [d0.im, d1.im, ...]
            let tw_parts = vuzpq_f32(tw, tw); // .0 = [tw0.re, tw1.re, ...], .1 = [tw0.im, tw1.im, ...]

            // Complex multiply: t = tw * d
            // t.re = tw.re * d.re - tw.im * d.im
            // t.im = tw.re * d.im + tw.im * d.re
            // Using vmlsq/vmlaq for fused multiply-add/subtract
            let t_re = vmulq_f32(tw_parts.0, d_parts.0);
            let t_re = vmlsq_f32(t_re, tw_parts.1, d_parts.1);

            let t_im = vmulq_f32(tw_parts.0, d_parts.1);
            let t_im = vmlaq_f32(t_im, tw_parts.1, d_parts.0);

            // Butterfly: out_top = u + t, out_bot = u - t
            let out_top_re = vaddq_f32(u_parts.0, t_re);
            let out_top_im = vaddq_f32(u_parts.1, t_im);
            let out_bot_re = vsubq_f32(u_parts.0, t_re);
            let out_bot_im = vsubq_f32(u_parts.1, t_im);

            // Reinterleave real and imaginary parts using vzipq
            // vzipq.0 interleaves low halves: [a0, b0, a1, b1]
            let out_top_interleaved = vzipq_f32(out_top_re, out_top_im);
            let out_bot_interleaved = vzipq_f32(out_bot_re, out_bot_im);

            // Store results (use .0 which contains the interleaved result)
            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            vst1q_f32(out_top_ptr, out_top_interleaved.0);
            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            vst1q_f32(out_bot_ptr, out_bot_interleaved.0);
        }

        super::butterfly_2_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}

/// NEON implementation: processes 4 columns at once using vld2q/vst2q.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_2_neon_x4(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 4) * 4;

        for idx in (0..simd_cols).step_by(4) {
            // Load 4 complex numbers using vld2q (automatically deinterleaves into real/imag)
            let u_ptr = data.as_ptr().add(idx) as *const f32;
            let u = vld2q_f32(u_ptr); // u.0 = real parts, u.1 = imag parts

            let d_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let d = vld2q_f32(d_ptr);

            let tw_ptr = stage_twiddles.as_ptr().add(idx) as *const f32;
            let tw = vld2q_f32(tw_ptr);

            // Complex multiply: t = tw * d
            // t.re = tw.re * d.re - tw.im * d.im
            // t.im = tw.re * d.im + tw.im * d.re
            // Using vmlsq/vmlaq for fused multiply-add/subtract
            let t_re = vmulq_f32(tw.0, d.0);
            let t_re = vmlsq_f32(t_re, tw.1, d.1);

            let t_im = vmulq_f32(tw.0, d.1);
            let t_im = vmlaq_f32(t_im, tw.1, d.0);

            // Butterfly: out_top = u + t, out_bot = u - t
            let out_top_re = vaddq_f32(u.0, t_re);
            let out_top_im = vaddq_f32(u.1, t_im);
            let out_bot_re = vsubq_f32(u.0, t_re);
            let out_bot_im = vsubq_f32(u.1, t_im);

            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            vst2q_f32(out_top_ptr, float32x4x2_t(out_top_re, out_top_im));
            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            vst2q_f32(out_bot_ptr, float32x4x2_t(out_bot_re, out_bot_im));
        }

        super::butterfly_2_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}
