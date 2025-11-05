use crate::Complex32;

/// Postprocess FFT output for real-valued FFT using NEON SIMD.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn postprocess_fft_neon(
    output_left_middle: &mut [Complex32],
    output_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    use core::arch::aarch64::*;

    unsafe {
        let len = output_left_middle.len();
        let right_len = output_right_middle.len();

        let simd_len = len / 2 * 2;

        for i in (0..simd_len).step_by(2) {
            let rev_idx0 = right_len - 1 - i;
            let rev_idx1 = right_len - 2 - i;

            let out_left_ptr = output_left_middle.as_ptr().add(i) as *const f32;
            let out_left = vld1q_f32(out_left_ptr);

            let low = vld1_f32(&output_right_middle[rev_idx0] as *const Complex32 as *const f32);
            let high = vld1_f32(&output_right_middle[rev_idx1] as *const Complex32 as *const f32);
            let out_rev = vcombine_f32(low, high);

            let tw_ptr = twiddles.as_ptr().add(i) as *const f32;
            let tw = vld1q_f32(tw_ptr);

            let out_left_parts = vuzpq_f32(out_left, out_left);
            let out_left_re = out_left_parts.0;
            let out_left_im = out_left_parts.1;

            let out_rev_parts = vuzpq_f32(out_rev, out_rev);
            let out_rev_re = out_rev_parts.0;
            let out_rev_im = out_rev_parts.1;

            let tw_parts = vuzpq_f32(tw, tw);
            let tw_re = tw_parts.0;
            let tw_im = tw_parts.1;

            let sum_re = vaddq_f32(out_left_re, out_rev_re);
            let sum_im = vaddq_f32(out_left_im, out_rev_im);

            let diff_re = vsubq_f32(out_left_re, out_rev_re);
            let diff_im = vsubq_f32(out_left_im, out_rev_im);

            let real_part = vmulq_f32(sum_im, tw_re);
            let real_part = vmlaq_f32(real_part, diff_re, tw_im);

            let imaginary_part = vmulq_f32(sum_im, tw_im);
            let imaginary_part = vmlsq_f32(imaginary_part, diff_re, tw_re);

            let half = vdupq_n_f32(0.5);
            let sum_re_half = vmulq_f32(sum_re, half);
            let diff_im_half = vmulq_f32(diff_im, half);

            let left_re = vaddq_f32(sum_re_half, real_part);
            let left_im = vaddq_f32(diff_im_half, imaginary_part);

            let right_re = vsubq_f32(sum_re_half, real_part);
            let right_im = vsubq_f32(imaginary_part, diff_im_half);

            let left_output = vzipq_f32(left_re, left_im);
            let right_output = vzipq_f32(right_re, right_im);

            let out_left_ptr_mut = output_left_middle.as_mut_ptr().add(i) as *mut f32;
            vst1q_f32(out_left_ptr_mut, left_output.0);

            let right_low = vget_low_f32(right_output.0);
            let right_high = vget_high_f32(right_output.0);
            let out_rev0_ptr_mut = output_right_middle.as_mut_ptr().add(rev_idx0) as *mut f32;
            let out_rev1_ptr_mut = output_right_middle.as_mut_ptr().add(rev_idx1) as *mut f32;
            vst1_f32(out_rev0_ptr_mut, right_low);
            vst1_f32(out_rev1_ptr_mut, right_high);
        }

        for i in simd_len..len {
            let rev_idx = right_len - 1 - i;
            let out_left = output_left_middle[i];
            let out_rev = output_right_middle[rev_idx];
            let tw = twiddles[i];

            let sum = out_left.add(&out_rev);
            let diff = out_left.sub(&out_rev);

            let real_part = sum.im * tw.re + diff.re * tw.im;
            let imaginary_part = sum.im * tw.im - diff.re * tw.re;

            output_left_middle[i] = Complex32 {
                re: sum.re * 0.5 + real_part,
                im: diff.im * 0.5 + imaginary_part,
            };

            output_right_middle[rev_idx] = Complex32 {
                re: sum.re * 0.5 - real_part,
                im: imaginary_part - diff.im * 0.5,
            };
        }
    }
}

/// Preprocess IFFT input for real-valued IFFT using NEON SIMD.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn preprocess_ifft_neon(
    input_left_middle: &mut [Complex32],
    input_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    use core::arch::aarch64::*;

    unsafe {
        let len = input_left_middle.len();
        let right_len = input_right_middle.len();

        let simd_len = len / 2 * 2;

        for i in (0..simd_len).step_by(2) {
            let rev_idx0 = right_len - 1 - i;
            let rev_idx1 = right_len - 2 - i;

            let inp_left_ptr = input_left_middle.as_ptr().add(i) as *const f32;
            let inp_left = vld1q_f32(inp_left_ptr);

            let low = vld1_f32(input_right_middle.as_ptr().add(rev_idx0) as *const f32);
            let high = vld1_f32(input_right_middle.as_ptr().add(rev_idx1) as *const f32);
            let inp_rev = vcombine_f32(low, high);

            let tw_ptr = twiddles.as_ptr().add(i) as *const f32;
            let tw = vld1q_f32(tw_ptr);

            let inp_left_parts = vuzpq_f32(inp_left, inp_left);
            let inp_left_re = inp_left_parts.0;
            let inp_left_im = inp_left_parts.1;

            let inp_rev_parts = vuzpq_f32(inp_rev, inp_rev);
            let inp_rev_re = inp_rev_parts.0;
            let inp_rev_im = inp_rev_parts.1;

            let tw_parts = vuzpq_f32(tw, tw);
            let tw_re = tw_parts.0;
            let tw_im = tw_parts.1;

            let sum_re = vaddq_f32(inp_left_re, inp_rev_re);
            let sum_im = vaddq_f32(inp_left_im, inp_rev_im);

            let diff_re = vsubq_f32(inp_left_re, inp_rev_re);
            let diff_im = vsubq_f32(inp_left_im, inp_rev_im);

            let real_part = vmulq_f32(sum_im, tw_re);
            let real_part = vmlaq_f32(real_part, diff_re, tw_im);

            let imaginary_part = vmulq_f32(sum_im, tw_im);
            let imaginary_part = vmlsq_f32(imaginary_part, diff_re, tw_re);

            let left_re = vsubq_f32(sum_re, real_part);
            let left_im = vsubq_f32(diff_im, imaginary_part);

            let right_re = vaddq_f32(sum_re, real_part);
            let right_im = vnegq_f32(imaginary_part);
            let right_im = vsubq_f32(right_im, diff_im);

            let left_output = vzipq_f32(left_re, left_im);
            let right_output = vzipq_f32(right_re, right_im);

            let inp_left_ptr_mut = input_left_middle.as_mut_ptr().add(i) as *mut f32;
            vst1q_f32(inp_left_ptr_mut, left_output.0);

            let right_low = vget_low_f32(right_output.0);
            let right_high = vget_high_f32(right_output.0);
            let inp_rev0_ptr_mut = input_right_middle.as_mut_ptr().add(rev_idx0) as *mut f32;
            let inp_rev1_ptr_mut = input_right_middle.as_mut_ptr().add(rev_idx1) as *mut f32;
            vst1_f32(inp_rev0_ptr_mut, right_low);
            vst1_f32(inp_rev1_ptr_mut, right_high);
        }

        for i in simd_len..len {
            let rev_idx = right_len - 1 - i;
            let inp_left = input_left_middle[i];
            let inp_rev = input_right_middle[rev_idx];
            let tw = twiddles[i];

            let sum = inp_left.add(&inp_rev);
            let diff = inp_left.sub(&inp_rev);

            let real_part = sum.im * tw.re + diff.re * tw.im;
            let imaginary_part = sum.im * tw.im - diff.re * tw.re;

            input_left_middle[i] = Complex32 {
                re: sum.re - real_part,
                im: diff.im - imaginary_part,
            };

            input_right_middle[rev_idx] = Complex32 {
                re: sum.re + real_part,
                im: -imaginary_part - diff.im,
            };
        }
    }
}
