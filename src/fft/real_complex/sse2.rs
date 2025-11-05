use crate::fft::Complex32;

/// SSE2 implementation of postprocess_fft.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn postprocess_fft_sse2(
    output_left_middle: &mut [Complex32],
    output_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    use core::arch::x86_64::*;

    unsafe {
        let iter_count = output_left_middle
            .len()
            .min(output_right_middle.len())
            .min(twiddles.len());

        let simd_count = iter_count / 2;
        let right_len = output_right_middle.len();

        let half = _mm_set1_ps(0.5);

        for chunk in 0..simd_count {
            let i = chunk * 2;

            let out_left_ptr = output_left_middle.as_ptr().add(i) as *const f32;
            let out_left = _mm_loadu_ps(out_left_ptr);

            let idx0 = right_len - 1 - i;
            let idx1 = right_len - 2 - i;

            let out_rev = _mm_set_ps(
                output_right_middle[idx1].im,
                output_right_middle[idx1].re,
                output_right_middle[idx0].im,
                output_right_middle[idx0].re,
            );

            let tw_ptr = twiddles.as_ptr().add(i) as *const f32;
            let tw = _mm_loadu_ps(tw_ptr);

            let sum = _mm_add_ps(out_left, out_rev);
            let diff = _mm_sub_ps(out_left, out_rev);

            let tw_re = _mm_shuffle_ps(tw, tw, 0b10_10_00_00);
            let tw_im = _mm_shuffle_ps(tw, tw, 0b11_11_01_01);

            let twiddled_re_sum = _mm_mul_ps(sum, tw_re);
            let twiddled_im_sum = _mm_mul_ps(sum, tw_im);
            let twiddled_re_diff = _mm_mul_ps(diff, tw_re);
            let twiddled_im_diff = _mm_mul_ps(diff, tw_im);

            let twiddled_re_sum_im =
                _mm_shuffle_ps(twiddled_re_sum, twiddled_re_sum, 0b11_11_01_01);
            let twiddled_im_diff_re =
                _mm_shuffle_ps(twiddled_im_diff, twiddled_im_diff, 0b10_10_00_00);

            let real = _mm_add_ps(twiddled_re_sum_im, twiddled_im_diff_re);

            let twiddled_im_sum_im =
                _mm_shuffle_ps(twiddled_im_sum, twiddled_im_sum, 0b11_11_01_01);
            let twiddled_re_diff_re =
                _mm_shuffle_ps(twiddled_re_diff, twiddled_re_diff, 0b10_10_00_00);

            let imaginary = _mm_sub_ps(twiddled_im_sum_im, twiddled_re_diff_re);

            let half_sum = _mm_mul_ps(half, sum);
            let half_diff = _mm_mul_ps(half, diff);

            let half_sum_plus_real = _mm_add_ps(half_sum, real);
            let half_diff_plus_imag = _mm_add_ps(half_diff, imaginary);

            let mask_pattern = _mm_castsi128_ps(_mm_setr_epi32(0, -1, 0, -1));
            let left_output = _mm_or_ps(
                _mm_andnot_ps(mask_pattern, half_sum_plus_real),
                _mm_and_ps(mask_pattern, half_diff_plus_imag),
            );

            let out_left_ptr = output_left_middle.as_mut_ptr().add(i) as *mut f32;
            _mm_storeu_ps(out_left_ptr, left_output);

            let half_sum_minus_real = _mm_sub_ps(half_sum, real);
            let imag_minus_half_diff = _mm_sub_ps(imaginary, half_diff);
            let right_output = _mm_or_ps(
                _mm_andnot_ps(mask_pattern, half_sum_minus_real),
                _mm_and_ps(mask_pattern, imag_minus_half_diff),
            );

            let mut right_vals = [0.0f32; 4];
            _mm_storeu_ps(right_vals.as_mut_ptr(), right_output);

            output_right_middle[right_len - 1 - i] = Complex32::new(right_vals[0], right_vals[1]);
            output_right_middle[right_len - 1 - (i + 1)] =
                Complex32::new(right_vals[2], right_vals[3]);
        }

        let remaining_start = simd_count * 2;
        for i in remaining_start..iter_count {
            let out = output_left_middle[i];
            let out_rev_idx = right_len - 1 - i;
            let out_rev = output_right_middle[out_rev_idx];
            let twiddle = twiddles[i];

            let sum = out.add(&out_rev);
            let diff = out.sub(&out_rev);

            let twiddled_re_sum = Complex32::new(sum.re * twiddle.re, sum.im * twiddle.re);
            let twiddled_im_sum = Complex32::new(sum.re * twiddle.im, sum.im * twiddle.im);
            let twiddled_re_diff = Complex32::new(diff.re * twiddle.re, diff.im * twiddle.re);
            let twiddled_im_diff = Complex32::new(diff.re * twiddle.im, diff.im * twiddle.im);

            let half = 0.5;
            let half_sum_real = half * sum.re;
            let half_diff_imaginary = half * diff.im;

            let real = twiddled_re_sum.im + twiddled_im_diff.re;
            let imaginary = twiddled_im_sum.im - twiddled_re_diff.re;

            output_left_middle[i] =
                Complex32::new(half_sum_real + real, half_diff_imaginary + imaginary);
            output_right_middle[out_rev_idx] =
                Complex32::new(half_sum_real - real, imaginary - half_diff_imaginary);
        }
    }
}

/// SSE2 implementation of preprocess_ifft.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn preprocess_ifft_sse2(
    input_left_middle: &mut [Complex32],
    input_right_middle: &mut [Complex32],
    twiddles: &[Complex32],
) {
    use core::arch::x86_64::*;

    unsafe {
        let iter_count = input_left_middle
            .len()
            .min(input_right_middle.len())
            .min(twiddles.len());

        let simd_count = iter_count / 2;

        let mask_pattern = _mm_castsi128_ps(_mm_setr_epi32(0, -1, 0, -1));

        for chunk in 0..simd_count {
            let i = chunk * 2;

            let inp_left_ptr = input_left_middle.as_ptr().add(i) as *const f32;
            let inp = _mm_loadu_ps(inp_left_ptr);

            let idx0 = input_right_middle.len() - 1 - i;
            let idx1 = input_right_middle.len() - 2 - i;

            let inp_rev = _mm_set_ps(
                input_right_middle[idx1].im,
                input_right_middle[idx1].re,
                input_right_middle[idx0].im,
                input_right_middle[idx0].re,
            );

            let tw_ptr = twiddles.as_ptr().add(i) as *const f32;
            let tw = _mm_loadu_ps(tw_ptr);

            let sum = _mm_add_ps(inp, inp_rev);
            let diff = _mm_sub_ps(inp, inp_rev);

            let tw_re = _mm_shuffle_ps(tw, tw, 0b10_10_00_00);
            let tw_im = _mm_shuffle_ps(tw, tw, 0b11_11_01_01);

            let twiddled_re_sum = _mm_mul_ps(sum, tw_re);
            let twiddled_im_sum = _mm_mul_ps(sum, tw_im);
            let twiddled_re_diff = _mm_mul_ps(diff, tw_re);
            let twiddled_im_diff = _mm_mul_ps(diff, tw_im);

            let twiddled_re_sum_im =
                _mm_shuffle_ps(twiddled_re_sum, twiddled_re_sum, 0b11_11_01_01);
            let twiddled_im_diff_re =
                _mm_shuffle_ps(twiddled_im_diff, twiddled_im_diff, 0b10_10_00_00);
            let real = _mm_add_ps(twiddled_re_sum_im, twiddled_im_diff_re);

            let twiddled_im_sum_im =
                _mm_shuffle_ps(twiddled_im_sum, twiddled_im_sum, 0b11_11_01_01);
            let twiddled_re_diff_re =
                _mm_shuffle_ps(twiddled_re_diff, twiddled_re_diff, 0b10_10_00_00);
            let imaginary = _mm_sub_ps(twiddled_im_sum_im, twiddled_re_diff_re);

            let sum_minus_real = _mm_sub_ps(sum, real);
            let diff_minus_imag = _mm_sub_ps(diff, imaginary);
            let left_input = _mm_or_ps(
                _mm_andnot_ps(mask_pattern, sum_minus_real),
                _mm_and_ps(mask_pattern, diff_minus_imag),
            );

            let inp_left_ptr = input_left_middle.as_mut_ptr().add(i) as *mut f32;
            _mm_storeu_ps(inp_left_ptr, left_input);

            let sum_plus_real = _mm_add_ps(sum, real);
            let imag_plus_diff = _mm_add_ps(imaginary, diff);
            let neg_imag_minus_diff = _mm_sub_ps(_mm_setzero_ps(), imag_plus_diff);
            let right_input = _mm_or_ps(
                _mm_andnot_ps(mask_pattern, sum_plus_real),
                _mm_and_ps(mask_pattern, neg_imag_minus_diff),
            );

            let mut right_vals = [0.0f32; 4];
            _mm_storeu_ps(right_vals.as_mut_ptr(), right_input);

            input_right_middle[input_right_middle.len() - 1 - i] =
                Complex32::new(right_vals[0], right_vals[1]);
            input_right_middle[input_right_middle.len() - 1 - (i + 1)] =
                Complex32::new(right_vals[2], right_vals[3]);
        }

        let remaining_start = simd_count * 2;
        for i in remaining_start..iter_count {
            let inp = input_left_middle[i];
            let inp_rev_idx = input_right_middle.len() - 1 - i;
            let inp_rev = input_right_middle[inp_rev_idx];
            let twiddle = twiddles[i];

            let sum = inp.add(&inp_rev);
            let diff = inp.sub(&inp_rev);

            let twiddled_re_sum = Complex32::new(sum.re * twiddle.re, sum.im * twiddle.re);
            let twiddled_im_sum = Complex32::new(sum.re * twiddle.im, sum.im * twiddle.im);
            let twiddled_re_diff = Complex32::new(diff.re * twiddle.re, diff.im * twiddle.re);
            let twiddled_im_diff = Complex32::new(diff.re * twiddle.im, diff.im * twiddle.im);

            let real = twiddled_re_sum.im + twiddled_im_diff.re;
            let imaginary = twiddled_im_sum.im - twiddled_re_diff.re;

            input_left_middle[i] = Complex32::new(sum.re - real, diff.im - imaginary);
            input_right_middle[inp_rev_idx] = Complex32::new(sum.re + real, -imaginary - diff.im);
        }
    }
}
