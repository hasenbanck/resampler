use core::arch::x86_64::*;

use crate::Complex32;

/// AVX implementation: processes 4 columns at once.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx",
    any(not(target_feature = "fma"), test)
))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn butterfly_2_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 4) * 4;

    for idx in (0..simd_cols).step_by(4) {
        unsafe {
            let u_ptr = data.as_ptr().add(idx) as *const f32;
            let u = _mm256_loadu_ps(u_ptr);

            let d_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let d = _mm256_loadu_ps(d_ptr);

            let tw_ptr = stage_twiddles.as_ptr().add(idx) as *const f32;
            let tw = _mm256_loadu_ps(tw_ptr);

            // Complex multiply using AVX.
            // Duplicate real and imaginary parts.
            let tw_re = _mm256_shuffle_ps(tw, tw, 0b10_10_00_00);
            let tw_im = _mm256_shuffle_ps(tw, tw, 0b11_11_01_01);
            let prod_re = _mm256_mul_ps(tw_re, d);

            // Swap real/imag in d.
            let d_swap = _mm256_shuffle_ps(d, d, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(tw_im, d_swap);

            // Combine using AVX addsub for efficient complex multiply.
            let t = _mm256_addsub_ps(prod_re, prod_im);

            // Butterfly operations.
            let out_top = _mm256_add_ps(u, t);
            let out_bot = _mm256_sub_ps(u, t);

            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(out_top_ptr, out_top);
            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(out_bot_ptr, out_bot);
        }
    }

    super::butterfly_2_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}

/// AVX+FMA implementation: processes 4 columns at once using fused multiply-add.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_2_avx_fma(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 4) * 4;

    for idx in (0..simd_cols).step_by(4) {
        unsafe {
            let u_ptr = data.as_ptr().add(idx) as *const f32;
            let u = _mm256_loadu_ps(u_ptr);
            let d_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let d = _mm256_loadu_ps(d_ptr);
            let tw_ptr = stage_twiddles.as_ptr().add(idx) as *const f32;
            let tw = _mm256_loadu_ps(tw_ptr);

            // Complex multiply using FMA: t = tw * d
            let tw_re = _mm256_shuffle_ps(tw, tw, 0b10_10_00_00);
            let tw_im = _mm256_shuffle_ps(tw, tw, 0b11_11_01_01);

            // Swap d for imaginary component: [d.im, d.re, d.im, d.re, ...]
            let d_swap = _mm256_shuffle_ps(d, d, 0b10_11_00_01);

            // Multiply imaginary parts: [tw.im*d.im, tw.im*d.re, ...]
            let prod_im = _mm256_mul_ps(tw_im, d_swap);

            // FMA: fmaddsub computes [a*b - c, a*b + c, a*b - c, a*b + c, ...]
            // Result: [tw.re*d.re - tw.im*d.im, tw.re*d.im + tw.im*d.re, ...]
            let t = _mm256_fmaddsub_ps(tw_re, d, prod_im);

            // Butterfly operations
            let out_top = _mm256_add_ps(u, t);
            let out_bot = _mm256_sub_ps(u, t);

            let out_top_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(out_top_ptr, out_top);
            let out_bot_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(out_bot_ptr, out_bot);
        }
    }

    super::butterfly_2_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}
