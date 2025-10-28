#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::Complex32;

/// AVX implementation: processes 4 columns at once.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx",
    any(not(target_feature = "fma"), test)
))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn butterfly_4_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 4) * 4;

    for idx in (0..simd_cols).step_by(4) {
        unsafe {
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

            // Load 12 twiddle factors: w1[0-3], w2[0-3], w3[0-3]
            // Memory layout: [w1[0], w2[0], w3[0], w1[1], w2[1], w3[1], w1[2], w2[2], w3[2], w1[3], w2[3], w3[3]]
            // Layout: [w1[0].re, w1[0].im, w2[0].re, w2[0].im, w3[0].re, w3[0].im, w1[1].re, w1[1].im]
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 3) as *const f32;
            let tw_0 = _mm256_loadu_ps(tw_ptr);
            // Layout: [w2[1].re, w2[1].im, w3[1].re, w3[1].im, w1[2].re, w1[2].im, w2[2].re, w2[2].im]
            let tw_1 = _mm256_loadu_ps(tw_ptr.add(8));
            // Layout: [w3[2].re, w3[2].im, w1[3].re, w1[3].im, w2[3].re, w2[3].im, w3[3].re, w3[3].im]
            let tw_2 = _mm256_loadu_ps(tw_ptr.add(16));

            // Extract w1, w2, w3 by carefully shuffling the twiddles.
            // tw_0 has: [w1[0], w2[0] | w3[0], w1[1]]
            // tw_1 has: [w2[1], w3[1] | w1[2], w2[2]]
            // tw_2 has: [w3[2], w1[3] | w2[3], w3[3]]

            // Get intermediate permutations.
            let temp_02 = _mm256_permute2f128_ps(tw_0, tw_1, 0x20); // [w1[0], w2[0] | w2[1], w3[1]]
            let temp_13 = _mm256_permute2f128_ps(tw_0, tw_2, 0x31); // [w3[0], w1[1] | w2[3], w3[3]]
            let temp_mid = _mm256_permute2f128_ps(tw_1, tw_2, 0x21); // [w1[2], w2[2] | w3[2], w1[3]]

            // Extract w1 = [w1[0], w1[1], w1[2], w1[3]]
            let w1_low_lane = _mm256_shuffle_ps(temp_02, temp_13, 0b11_10_01_00);
            let temp_mid_swapped = _mm256_permute2f128_ps(temp_mid, temp_mid, 0x01);
            let w1_high_lane = _mm256_shuffle_ps(temp_mid, temp_mid_swapped, 0b11_10_01_00);
            let w1 = _mm256_permute2f128_ps(w1_low_lane, w1_high_lane, 0x20);

            // Extract w2 = [w2[0], w2[1], w2[2], w2[3]]
            let temp_02_swapped = _mm256_permute2f128_ps(temp_02, temp_02, 0x01);
            let w2_low_lane = _mm256_shuffle_ps(temp_02, temp_02_swapped, 0b01_00_11_10);
            let temp_13_swapped = _mm256_permute2f128_ps(temp_13, temp_13, 0x01);
            let w2_high_lane = _mm256_shuffle_ps(temp_mid, temp_13_swapped, 0b01_00_11_10);
            let w2 = _mm256_permute2f128_ps(w2_low_lane, w2_high_lane, 0x20);

            // Extract w3 = [w3[0], w3[1], w3[2], w3[3]]
            let w3_low_lane = _mm256_shuffle_ps(temp_13, temp_02_swapped, 0b11_10_01_00);
            let w3_high_lane = _mm256_shuffle_ps(temp_mid_swapped, temp_13_swapped, 0b11_10_01_00);
            let w3 = _mm256_permute2f128_ps(w3_low_lane, w3_high_lane, 0x20);

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

            // Complex multiply: t3 = x3 * w3
            let w3_re = _mm256_shuffle_ps(w3, w3, 0b10_10_00_00);
            let w3_im = _mm256_shuffle_ps(w3, w3, 0b11_11_01_01);
            let prod_re = _mm256_mul_ps(w3_re, x3);
            let x3_swap = _mm256_shuffle_ps(x3, x3, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w3_im, x3_swap);
            let t3 = _mm256_addsub_ps(prod_re, prod_im);

            // t0 = x0 (no twiddle)
            let t0 = x0;

            // Compute intermediate values for radix-4 butterfly.
            let u0 = _mm256_add_ps(t0, t2); // u0 = t0 + t2
            let u1 = _mm256_sub_ps(t0, t2); // u1 = t0 - t2
            let u2 = _mm256_add_ps(t1, t3); // u2 = t1 + t3
            let u3 = _mm256_sub_ps(t1, t3); // u3 = t1 - t3

            // Multiply u3 by -j: -j * (a + bi) = b - ai
            // Swap real/imag and negate new imaginary part: [u3.im, -u3.re]
            let u3_neg_j = _mm256_shuffle_ps(u3, u3, 0b10_11_00_01);
            let sign_flip = _mm256_castsi256_ps(_mm256_set_epi32(
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
            ));
            let u3_neg_j = _mm256_xor_ps(u3_neg_j, sign_flip);

            // Multiply u3 by +j: +j * (a + bi) = -b + ai
            // Swap real/imag and negate new real part: [-u3.im, u3.re]
            let u3_pos_j = _mm256_shuffle_ps(u3, u3, 0b10_11_00_01);
            let sign_flip_re = _mm256_castsi256_ps(_mm256_set_epi32(
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
            ));
            let u3_pos_j = _mm256_xor_ps(u3_pos_j, sign_flip_re);

            // Combine to produce outputs.
            let y0 = _mm256_add_ps(u0, u2); // y0 = u0 + u2
            let y1 = _mm256_add_ps(u1, u3_neg_j); // y1 = u1 - j*u3
            let y2 = _mm256_sub_ps(u0, u2); // y2 = u0 - u2
            let y3 = _mm256_add_ps(u1, u3_pos_j); // y3 = u1 + j*u3

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm256_storeu_ps(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            _mm256_storeu_ps(y3_ptr, y3);
        }
    }

    super::butterfly_4_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}

/// AVX+FMA implementation: processes 4 columns at once using fused multiply-add.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_4_avx_fma(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    let simd_cols = (num_columns / 4) * 4;

    for idx in (0..simd_cols).step_by(4) {
        unsafe {
            // Load 4 complex numbers from each row.
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = _mm256_loadu_ps(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = _mm256_loadu_ps(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = _mm256_loadu_ps(x2_ptr);

            let x3_ptr = data.as_ptr().add(idx + 3 * num_columns) as *const f32;
            let x3 = _mm256_loadu_ps(x3_ptr);

            // Load 12 twiddle factors.
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 3) as *const f32;
            let tw_0 = _mm256_loadu_ps(tw_ptr);
            let tw_1 = _mm256_loadu_ps(tw_ptr.add(8));
            let tw_2 = _mm256_loadu_ps(tw_ptr.add(16));

            // Extract w1, w2, w3 by shuffling the twiddles.
            let temp_02 = _mm256_permute2f128_ps(tw_0, tw_1, 0x20); // [w1[0], w2[0] | w2[1], w3[1]]
            let temp_13 = _mm256_permute2f128_ps(tw_0, tw_2, 0x31); // [w3[0], w1[1] | w2[3], w3[3]]
            let temp_mid = _mm256_permute2f128_ps(tw_1, tw_2, 0x21); // [w1[2], w2[2] | w3[2], w1[3]]

            let w1_low_lane = _mm256_shuffle_ps(temp_02, temp_13, 0b11_10_01_00);
            let temp_mid_swapped = _mm256_permute2f128_ps(temp_mid, temp_mid, 0x01);
            let w1_high_lane = _mm256_shuffle_ps(temp_mid, temp_mid_swapped, 0b11_10_01_00);
            let w1 = _mm256_permute2f128_ps(w1_low_lane, w1_high_lane, 0x20);

            let temp_02_swapped = _mm256_permute2f128_ps(temp_02, temp_02, 0x01);
            let w2_low_lane = _mm256_shuffle_ps(temp_02, temp_02_swapped, 0b01_00_11_10);
            let temp_13_swapped = _mm256_permute2f128_ps(temp_13, temp_13, 0x01);
            let w2_high_lane = _mm256_shuffle_ps(temp_mid, temp_13_swapped, 0b01_00_11_10);
            let w2 = _mm256_permute2f128_ps(w2_low_lane, w2_high_lane, 0x20);

            let w3_low_lane = _mm256_shuffle_ps(temp_13, temp_02_swapped, 0b11_10_01_00);
            let w3_high_lane = _mm256_shuffle_ps(temp_mid_swapped, temp_13_swapped, 0b11_10_01_00);
            let w3 = _mm256_permute2f128_ps(w3_low_lane, w3_high_lane, 0x20);

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

            // t0 = x0 (no twiddle)
            let t0 = x0;

            // Compute intermediate values for radix-4 butterfly.
            let u0 = _mm256_add_ps(t0, t2); // u0 = t0 + t2
            let u1 = _mm256_sub_ps(t0, t2); // u1 = t0 - t2
            let u2 = _mm256_add_ps(t1, t3); // u2 = t1 + t3
            let u3 = _mm256_sub_ps(t1, t3); // u3 = t1 - t3

            // Multiply u3 by -j: -j * (a + bi) = b - ai
            // Swap real/imag and negate new imaginary part: [u3.im, -u3.re]
            let u3_neg_j = _mm256_shuffle_ps(u3, u3, 0b10_11_00_01);
            let sign_flip = _mm256_castsi256_ps(_mm256_set_epi32(
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
            ));
            let u3_neg_j = _mm256_xor_ps(u3_neg_j, sign_flip);

            // Multiply u3 by +j: +j * (a + bi) = -b + ai
            // Swap real/imag and negate new real part: [-u3.im, u3.re]
            let u3_pos_j = _mm256_shuffle_ps(u3, u3, 0b10_11_00_01);
            let sign_flip_re = _mm256_castsi256_ps(_mm256_set_epi32(
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
                0,
                0x80000000u32 as i32,
            ));
            let u3_pos_j = _mm256_xor_ps(u3_pos_j, sign_flip_re);

            // Combine to produce outputs.
            let y0 = _mm256_add_ps(u0, u2); // y0 = u0 + u2
            let y1 = _mm256_add_ps(u1, u3_neg_j); // y1 = u1 - j*u3
            let y2 = _mm256_sub_ps(u0, u2); // y2 = u0 - u2
            let y3 = _mm256_add_ps(u1, u3_pos_j); // y3 = u1 + j*u3

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm256_storeu_ps(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            _mm256_storeu_ps(y3_ptr, y3);
        }
    }

    super::butterfly_4_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
}
