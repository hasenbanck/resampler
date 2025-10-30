#[cfg(any(test, not(feature = "no_std"), target_feature = "avx"))]
use crate::Complex32;

/// AVX implementation: processes 4 columns at once.
#[cfg(any(
    test,
    not(feature = "no_std"),
    all(target_feature = "avx", not(target_feature = "fma"))
))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn butterfly_3_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    start_col: usize,
    num_columns: usize,
) {
    use core::arch::x86_64::*;

    unsafe {
        let simd_cols = ((num_columns - start_col) / 4) * 4;

        // Broadcast W3 constants for SIMD operations.
        let w3_1_re = _mm256_set1_ps(super::W3_1_RE);
        let w3_1_im = _mm256_set1_ps(super::W3_1_IM);
        let w3_2_re = _mm256_set1_ps(super::W3_2_RE);
        let w3_2_im = _mm256_set1_ps(super::W3_2_IM);

        for idx in (start_col..start_col + simd_cols).step_by(4) {
            // Load 4 complex numbers from each row.
            // Layout: [x0[0].re, x0[0].im, x0[1].re, x0[1].im, x0[2].re, x0[2].im, x0[3].re, x0[3].im]
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = _mm256_loadu_ps(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = _mm256_loadu_ps(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = _mm256_loadu_ps(x2_ptr);

            // Load 8 twiddle factors: w1[0], w2[0], w1[1], w2[1], w1[2], w2[2], w1[3], w2[3]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im, w1[1].re, w1[1].im, w2[1].re, w2[1].im]
            let tw1_ptr = stage_twiddles.as_ptr().add(idx * 2) as *const f32;
            let tw_0 = _mm256_loadu_ps(tw1_ptr);
            // Layout: [w1[2].re, w1[2].im, w2[2].re, w2[2].im, w1[3].re, w1[3].im, w2[3].re, w2[3].im]
            let tw_1 = _mm256_loadu_ps(tw1_ptr.add(8));

            // Extract w1 and w2 for all 4 columns.
            // tw_0 = [w1[0], w2[0] | w1[1], w2[1]]
            // tw_1 = [w1[2], w2[2] | w1[3], w2[3]]
            // Step 1: Use permute2f128 to group alternating columns
            // temp_0_low = [w1[0], w2[0] | w1[2], w2[2]]
            let temp_0_low = _mm256_permute2f128_ps(tw_0, tw_1, 0x20);
            // temp_0_high = [w1[1], w2[1] | w1[3], w2[3]]
            let temp_0_high = _mm256_permute2f128_ps(tw_0, tw_1, 0x31);

            // Step 2: Shuffle to extract w1 and w2
            // w1 = [w1[0].re, w1[0].im, w1[1].re, w1[1].im, w1[2].re, w1[2].im, w1[3].re, w1[3].im]
            let w1 = _mm256_shuffle_ps(temp_0_low, temp_0_high, 0b01_00_01_00);
            // w2 = [w2[0].re, w2[0].im, w2[1].re, w2[1].im, w2[2].re, w2[2].im, w2[3].re, w2[3].im]
            let w2 = _mm256_shuffle_ps(temp_0_low, temp_0_high, 0b11_10_11_10);

            // Complex multiply: t1 = x1 * w1
            let w1_re = _mm256_shuffle_ps(w1, w1, 0b10_10_00_00);
            let w1_im = _mm256_shuffle_ps(w1, w1, 0b11_11_01_01);
            let prod_re = _mm256_mul_ps(w1_re, x1);
            let x1_swap = _mm256_shuffle_ps(x1, x1, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w1_im, x1_swap);

            // Use AVX addsub for complex multiply.
            let t1 = _mm256_addsub_ps(prod_re, prod_im);

            // Complex multiply: t2 = x2 * w2
            let w2_re = _mm256_shuffle_ps(w2, w2, 0b10_10_00_00);
            let w2_im = _mm256_shuffle_ps(w2, w2, 0b11_11_01_01);
            let prod_re = _mm256_mul_ps(w2_re, x2);
            let x2_swap = _mm256_shuffle_ps(x2, x2, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w2_im, x2_swap);
            let t2 = _mm256_addsub_ps(prod_re, prod_im);

            // Y0 = x0 + t1 + t2
            let y0 = _mm256_add_ps(_mm256_add_ps(x0, t1), t2);

            // Y1 = x0 + t1*W_3^1 + t2*W_3^2
            // t1*W_3^1: complex multiply t1 by (super::W3_1_RE, super::W3_1_IM)
            let t1_re = _mm256_shuffle_ps(t1, t1, 0b10_10_00_00);
            let t1_im = _mm256_shuffle_ps(t1, t1, 0b11_11_01_01);
            let t1w31_re =
                _mm256_sub_ps(_mm256_mul_ps(t1_re, w3_1_re), _mm256_mul_ps(t1_im, w3_1_im));
            let t1w31_im =
                _mm256_add_ps(_mm256_mul_ps(t1_re, w3_1_im), _mm256_mul_ps(t1_im, w3_1_re));

            // Interleave real and imaginary parts
            let t1_w31_lo = _mm256_unpacklo_ps(t1w31_re, t1w31_im);
            let t1_w31_hi = _mm256_unpackhi_ps(t1w31_re, t1w31_im);
            let t1_w31 = _mm256_shuffle_ps(t1_w31_lo, t1_w31_hi, 0b01_00_01_00);

            // t2*W_3^2: complex multiply t2 by (super::W3_2_RE, super::W3_2_IM)
            let t2_re = _mm256_shuffle_ps(t2, t2, 0b10_10_00_00);
            let t2_im = _mm256_shuffle_ps(t2, t2, 0b11_11_01_01);
            let t2w32_re =
                _mm256_sub_ps(_mm256_mul_ps(t2_re, w3_2_re), _mm256_mul_ps(t2_im, w3_2_im));
            let t2w32_im =
                _mm256_add_ps(_mm256_mul_ps(t2_re, w3_2_im), _mm256_mul_ps(t2_im, w3_2_re));

            let t2_w32_lo = _mm256_unpacklo_ps(t2w32_re, t2w32_im);
            let t2_w32_hi = _mm256_unpackhi_ps(t2w32_re, t2w32_im);
            let t2_w32 = _mm256_shuffle_ps(t2_w32_lo, t2_w32_hi, 0b01_00_01_00);

            let y1 = _mm256_add_ps(_mm256_add_ps(x0, t1_w31), t2_w32);

            // Y2 = x0 + t1*W_3^2 + t2*W_3^1
            // t1*W_3^2: complex multiply t1 by (super::W3_2_RE, super::W3_2_IM)
            let t1w32_re =
                _mm256_sub_ps(_mm256_mul_ps(t1_re, w3_2_re), _mm256_mul_ps(t1_im, w3_2_im));
            let t1w32_im =
                _mm256_add_ps(_mm256_mul_ps(t1_re, w3_2_im), _mm256_mul_ps(t1_im, w3_2_re));

            let t1_w32_lo = _mm256_unpacklo_ps(t1w32_re, t1w32_im);
            let t1_w32_hi = _mm256_unpackhi_ps(t1w32_re, t1w32_im);
            let t1_w32 = _mm256_shuffle_ps(t1_w32_lo, t1_w32_hi, 0b01_00_01_00);

            // t2*W_3^1: complex multiply t2 by (super::W3_1_RE, super::W3_1_IM)
            let t2w31_re =
                _mm256_sub_ps(_mm256_mul_ps(t2_re, w3_1_re), _mm256_mul_ps(t2_im, w3_1_im));
            let t2w31_im =
                _mm256_add_ps(_mm256_mul_ps(t2_re, w3_1_im), _mm256_mul_ps(t2_im, w3_1_re));

            let t2_w31_lo = _mm256_unpacklo_ps(t2w31_re, t2w31_im);
            let t2_w31_hi = _mm256_unpackhi_ps(t2w31_re, t2w31_im);
            let t2_w31 = _mm256_shuffle_ps(t2_w31_lo, t2_w31_hi, 0b01_00_01_00);

            let y2 = _mm256_add_ps(_mm256_add_ps(x0, t1_w32), t2_w31);

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm256_storeu_ps(y2_ptr, y2);
        }

        super::butterfly_3_scalar(data, stage_twiddles, start_col + simd_cols, num_columns)
    }
}

/// AVX+FMA implementation: processes 4 columns at once using fused multiply-add.
#[cfg(any(
    not(feature = "no_std"),
    all(target_feature = "avx", target_feature = "fma")
))]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_3_avx_fma(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    start_col: usize,
    num_columns: usize,
) {
    use core::arch::x86_64::*;

    unsafe {
        let simd_cols = ((num_columns - start_col) / 4) * 4;

        // Broadcast W3 constants for SIMD operations.
        let w3_1_re = _mm256_set1_ps(super::W3_1_RE);
        let w3_1_im = _mm256_set1_ps(super::W3_1_IM);
        let w3_2_re = _mm256_set1_ps(super::W3_2_RE);
        let w3_2_im = _mm256_set1_ps(super::W3_2_IM);

        for idx in (start_col..start_col + simd_cols).step_by(4) {
            // Load 4 complex numbers from each row.
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = _mm256_loadu_ps(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = _mm256_loadu_ps(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = _mm256_loadu_ps(x2_ptr);

            // Load 8 twiddle factors.
            let tw1_ptr = stage_twiddles.as_ptr().add(idx * 2) as *const f32;
            let tw_0 = _mm256_loadu_ps(tw1_ptr);
            let tw_1 = _mm256_loadu_ps(tw1_ptr.add(8));

            // Extract w1 and w2 for all 4 columns.
            let temp_0_low = _mm256_permute2f128_ps(tw_0, tw_1, 0x20);
            let temp_0_high = _mm256_permute2f128_ps(tw_0, tw_1, 0x31);

            let w1 = _mm256_shuffle_ps(temp_0_low, temp_0_high, 0b01_00_01_00);
            let w2 = _mm256_shuffle_ps(temp_0_low, temp_0_high, 0b11_10_11_10);

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

            // Y0 = x0 + t1 + t2
            let y0 = _mm256_add_ps(_mm256_add_ps(x0, t1), t2);

            // Y1 = x0 + t1*W_3^1 + t2*W_3^2
            // t1*W_3^1: complex multiply using FMA
            let t1_re = _mm256_shuffle_ps(t1, t1, 0b10_10_00_00);
            let t1_im = _mm256_shuffle_ps(t1, t1, 0b11_11_01_01);
            let t1w31_re = _mm256_fmsub_ps(t1_re, w3_1_re, _mm256_mul_ps(t1_im, w3_1_im));
            let t1w31_im = _mm256_fmadd_ps(t1_re, w3_1_im, _mm256_mul_ps(t1_im, w3_1_re));

            let t1_w31_lo = _mm256_unpacklo_ps(t1w31_re, t1w31_im);
            let t1_w31_hi = _mm256_unpackhi_ps(t1w31_re, t1w31_im);
            let t1_w31 = _mm256_shuffle_ps(t1_w31_lo, t1_w31_hi, 0b01_00_01_00);

            // t2*W_3^2: complex multiply using FMA
            let t2_re = _mm256_shuffle_ps(t2, t2, 0b10_10_00_00);
            let t2_im = _mm256_shuffle_ps(t2, t2, 0b11_11_01_01);
            let t2w32_re = _mm256_fmsub_ps(t2_re, w3_2_re, _mm256_mul_ps(t2_im, w3_2_im));
            let t2w32_im = _mm256_fmadd_ps(t2_re, w3_2_im, _mm256_mul_ps(t2_im, w3_2_re));

            let t2_w32_lo = _mm256_unpacklo_ps(t2w32_re, t2w32_im);
            let t2_w32_hi = _mm256_unpackhi_ps(t2w32_re, t2w32_im);
            let t2_w32 = _mm256_shuffle_ps(t2_w32_lo, t2_w32_hi, 0b01_00_01_00);

            let y1 = _mm256_add_ps(_mm256_add_ps(x0, t1_w31), t2_w32);

            // Y2 = x0 + t1*W_3^2 + t2*W_3^1
            // t1*W_3^2: complex multiply using FMA
            let t1w32_re = _mm256_fmsub_ps(t1_re, w3_2_re, _mm256_mul_ps(t1_im, w3_2_im));
            let t1w32_im = _mm256_fmadd_ps(t1_re, w3_2_im, _mm256_mul_ps(t1_im, w3_2_re));

            let t1_w32_lo = _mm256_unpacklo_ps(t1w32_re, t1w32_im);
            let t1_w32_hi = _mm256_unpackhi_ps(t1w32_re, t1w32_im);
            let t1_w32 = _mm256_shuffle_ps(t1_w32_lo, t1_w32_hi, 0b01_00_01_00);

            // t2*W_3^1: complex multiply using FMA
            let t2w31_re = _mm256_fmsub_ps(t2_re, w3_1_re, _mm256_mul_ps(t2_im, w3_1_im));
            let t2w31_im = _mm256_fmadd_ps(t2_re, w3_1_im, _mm256_mul_ps(t2_im, w3_1_re));

            let t2_w31_lo = _mm256_unpacklo_ps(t2w31_re, t2w31_im);
            let t2_w31_hi = _mm256_unpackhi_ps(t2w31_re, t2w31_im);
            let t2_w31 = _mm256_shuffle_ps(t2_w31_lo, t2_w31_hi, 0b01_00_01_00);

            let y2 = _mm256_add_ps(_mm256_add_ps(x0, t1_w32), t2_w31);

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm256_storeu_ps(y2_ptr, y2);
        }

        super::butterfly_3_scalar(data, stage_twiddles, start_col + simd_cols, num_columns)
    }
}
