#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::Complex32;

/// AVX implementation: processes 4 columns at once.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx",
    any(not(target_feature = "fma"), test)
))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn butterfly_7_avx(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 4) * 4;

        // Broadcast W7 constants for SIMD operations.
        let w7_1_re = _mm256_set1_ps(super::W7_1_RE);
        let w7_1_im = _mm256_set1_ps(super::W7_1_IM);
        let w7_2_re = _mm256_set1_ps(super::W7_2_RE);
        let w7_2_im = _mm256_set1_ps(super::W7_2_IM);
        let w7_3_re = _mm256_set1_ps(super::W7_3_RE);
        let w7_3_im = _mm256_set1_ps(super::W7_3_IM);
        let w7_4_re = _mm256_set1_ps(super::W7_4_RE);
        let w7_4_im = _mm256_set1_ps(super::W7_4_IM);
        let w7_5_re = _mm256_set1_ps(super::W7_5_RE);
        let w7_5_im = _mm256_set1_ps(super::W7_5_IM);
        let w7_6_re = _mm256_set1_ps(super::W7_6_RE);
        let w7_6_im = _mm256_set1_ps(super::W7_6_IM);

        for idx in (0..simd_cols).step_by(4) {
            // Load 4 complex numbers from each row.
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = _mm256_loadu_ps(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = _mm256_loadu_ps(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = _mm256_loadu_ps(x2_ptr);

            let x3_ptr = data.as_ptr().add(idx + 3 * num_columns) as *const f32;
            let x3 = _mm256_loadu_ps(x3_ptr);

            let x4_ptr = data.as_ptr().add(idx + 4 * num_columns) as *const f32;
            let x4 = _mm256_loadu_ps(x4_ptr);

            let x5_ptr = data.as_ptr().add(idx + 5 * num_columns) as *const f32;
            let x5 = _mm256_loadu_ps(x5_ptr);

            let x6_ptr = data.as_ptr().add(idx + 6 * num_columns) as *const f32;
            let x6 = _mm256_loadu_ps(x6_ptr);

            // Load 24 twiddle factors for 4 columns.
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 6) as *const f32;
            let tw_0 = _mm256_loadu_ps(tw_ptr); // w1[0], w2[0], w3[0], w4[0]
            let tw_1 = _mm256_loadu_ps(tw_ptr.add(8)); // w5[0], w6[0], w1[1], w2[1]
            let tw_2 = _mm256_loadu_ps(tw_ptr.add(16)); // w3[1], w4[1], w5[1], w6[1]
            let tw_3 = _mm256_loadu_ps(tw_ptr.add(24)); // w1[2], w2[2], w3[2], w4[2]
            let tw_4 = _mm256_loadu_ps(tw_ptr.add(32)); // w5[2], w6[2], w1[3], w2[3]
            let tw_5 = _mm256_loadu_ps(tw_ptr.add(40)); // w3[3], w4[3], w5[3], w6[3]

            // Extract w1, w2, w3, w4, w5, w6 for all 4 columns using permute and shuffle.
            let temp_0 = _mm256_permute2f128_ps(tw_0, tw_3, 0x20);
            let temp_1 = _mm256_permute2f128_ps(tw_1, tw_4, 0x20);
            let temp_2 = _mm256_permute2f128_ps(tw_0, tw_3, 0x31);
            let temp_3 = _mm256_permute2f128_ps(tw_1, tw_4, 0x31);
            let temp_4 = _mm256_permute2f128_ps(tw_2, tw_5, 0x20);
            let temp_5 = _mm256_permute2f128_ps(tw_2, tw_5, 0x31);

            let w1 = _mm256_shuffle_ps(temp_0, temp_3, 0b01_00_01_00);
            let w2 = _mm256_shuffle_ps(temp_0, temp_3, 0b11_10_11_10);
            let w3 = _mm256_shuffle_ps(temp_2, temp_4, 0b01_00_01_00);
            let w4 = _mm256_shuffle_ps(temp_2, temp_4, 0b11_10_11_10);
            let w5 = _mm256_shuffle_ps(temp_1, temp_5, 0b01_00_01_00);
            let w6 = _mm256_shuffle_ps(temp_1, temp_5, 0b11_10_11_10);

            // Helper macro for complex multiply using AVX addsub.
            macro_rules! cmul_avx {
                ($x:expr, $w:expr) => {{
                    let w_re = _mm256_shuffle_ps($w, $w, 0b10_10_00_00);
                    let w_im = _mm256_shuffle_ps($w, $w, 0b11_11_01_01);
                    let prod_re = _mm256_mul_ps(w_re, $x);
                    let x_swap = _mm256_shuffle_ps($x, $x, 0b10_11_00_01);
                    let prod_im = _mm256_mul_ps(w_im, x_swap);
                    _mm256_addsub_ps(prod_re, prod_im)
                }};
            }

            // Complex multiply: t1 = x1 * w1, t2 = x2 * w2, etc.
            let t1 = cmul_avx!(x1, w1);
            let t2 = cmul_avx!(x2, w2);
            let t3 = cmul_avx!(x3, w3);
            let t4 = cmul_avx!(x4, w4);
            let t5 = cmul_avx!(x5, w5);
            let t6 = cmul_avx!(x6, w6);

            // Y0 = x0 + t1 + t2 + t3 + t4 + t5 + t6
            let y0 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1), _mm256_add_ps(t2, t3)),
                _mm256_add_ps(_mm256_add_ps(t4, t5), t6),
            );

            // Helper macro for complex multiply by W7 constant.
            macro_rules! cmul_w7 {
                ($t:expr, $w_re:expr, $w_im:expr) => {{
                    let t_re = _mm256_shuffle_ps($t, $t, 0b10_10_00_00);
                    let t_im = _mm256_shuffle_ps($t, $t, 0b11_11_01_01);
                    let re = _mm256_sub_ps(_mm256_mul_ps(t_re, $w_re), _mm256_mul_ps(t_im, $w_im));
                    let im = _mm256_add_ps(_mm256_mul_ps(t_re, $w_im), _mm256_mul_ps(t_im, $w_re));
                    let lo = _mm256_unpacklo_ps(re, im);
                    let hi = _mm256_unpackhi_ps(re, im);
                    _mm256_shuffle_ps(lo, hi, 0b01_00_01_00)
                }};
            }

            // Y1 = x0 + t1*W_7^1 + t2*W_7^2 + t3*W_7^3 + t4*W_7^4 + t5*W_7^5 + t6*W_7^6
            let t1_w71 = cmul_w7!(t1, w7_1_re, w7_1_im);
            let t2_w72 = cmul_w7!(t2, w7_2_re, w7_2_im);
            let t3_w73 = cmul_w7!(t3, w7_3_re, w7_3_im);
            let t4_w74 = cmul_w7!(t4, w7_4_re, w7_4_im);
            let t5_w75 = cmul_w7!(t5, w7_5_re, w7_5_im);
            let t6_w76 = cmul_w7!(t6, w7_6_re, w7_6_im);

            let y1 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w71), _mm256_add_ps(t2_w72, t3_w73)),
                _mm256_add_ps(_mm256_add_ps(t4_w74, t5_w75), t6_w76),
            );

            // Y2 = x0 + t1*W_7^2 + t2*W_7^4 + t3*W_7^6 + t4*W_7^1 + t5*W_7^3 + t6*W_7^5
            let t1_w72 = cmul_w7!(t1, w7_2_re, w7_2_im);
            let t2_w74 = cmul_w7!(t2, w7_4_re, w7_4_im);
            let t3_w76 = cmul_w7!(t3, w7_6_re, w7_6_im);
            let t4_w71 = cmul_w7!(t4, w7_1_re, w7_1_im);
            let t5_w73 = cmul_w7!(t5, w7_3_re, w7_3_im);
            let t6_w75 = cmul_w7!(t6, w7_5_re, w7_5_im);

            let y2 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w72), _mm256_add_ps(t2_w74, t3_w76)),
                _mm256_add_ps(_mm256_add_ps(t4_w71, t5_w73), t6_w75),
            );

            // Y3 = x0 + t1*W_7^3 + t2*W_7^6 + t3*W_7^2 + t4*W_7^5 + t5*W_7^1 + t6*W_7^4
            let t1_w73 = cmul_w7!(t1, w7_3_re, w7_3_im);
            let t2_w76 = cmul_w7!(t2, w7_6_re, w7_6_im);
            let t3_w72 = cmul_w7!(t3, w7_2_re, w7_2_im);
            let t4_w75 = cmul_w7!(t4, w7_5_re, w7_5_im);
            let t5_w71 = cmul_w7!(t5, w7_1_re, w7_1_im);
            let t6_w74 = cmul_w7!(t6, w7_4_re, w7_4_im);

            let y3 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w73), _mm256_add_ps(t2_w76, t3_w72)),
                _mm256_add_ps(_mm256_add_ps(t4_w75, t5_w71), t6_w74),
            );

            // Y4 = x0 + t1*W_7^4 + t2*W_7^1 + t3*W_7^5 + t4*W_7^2 + t5*W_7^6 + t6*W_7^3
            let t1_w74 = cmul_w7!(t1, w7_4_re, w7_4_im);
            let t2_w71 = cmul_w7!(t2, w7_1_re, w7_1_im);
            let t3_w75 = cmul_w7!(t3, w7_5_re, w7_5_im);
            let t4_w72 = cmul_w7!(t4, w7_2_re, w7_2_im);
            let t5_w76 = cmul_w7!(t5, w7_6_re, w7_6_im);
            let t6_w73 = cmul_w7!(t6, w7_3_re, w7_3_im);

            let y4 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w74), _mm256_add_ps(t2_w71, t3_w75)),
                _mm256_add_ps(_mm256_add_ps(t4_w72, t5_w76), t6_w73),
            );

            // Y5 = x0 + t1*W_7^5 + t2*W_7^3 + t3*W_7^1 + t4*W_7^6 + t5*W_7^4 + t6*W_7^2
            let t1_w75 = cmul_w7!(t1, w7_5_re, w7_5_im);
            let t2_w73 = cmul_w7!(t2, w7_3_re, w7_3_im);
            let t3_w71 = cmul_w7!(t3, w7_1_re, w7_1_im);
            let t4_w76 = cmul_w7!(t4, w7_6_re, w7_6_im);
            let t5_w74 = cmul_w7!(t5, w7_4_re, w7_4_im);
            let t6_w72 = cmul_w7!(t6, w7_2_re, w7_2_im);

            let y5 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w75), _mm256_add_ps(t2_w73, t3_w71)),
                _mm256_add_ps(_mm256_add_ps(t4_w76, t5_w74), t6_w72),
            );

            // Y6 = x0 + t1*W_7^6 + t2*W_7^5 + t3*W_7^4 + t4*W_7^3 + t5*W_7^2 + t6*W_7^1
            let t1_w76 = cmul_w7!(t1, w7_6_re, w7_6_im);
            let t2_w75 = cmul_w7!(t2, w7_5_re, w7_5_im);
            let t3_w74 = cmul_w7!(t3, w7_4_re, w7_4_im);
            let t4_w73 = cmul_w7!(t4, w7_3_re, w7_3_im);
            let t5_w72 = cmul_w7!(t5, w7_2_re, w7_2_im);
            let t6_w71 = cmul_w7!(t6, w7_1_re, w7_1_im);

            let y6 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w76), _mm256_add_ps(t2_w75, t3_w74)),
                _mm256_add_ps(_mm256_add_ps(t4_w73, t5_w72), t6_w71),
            );

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm256_storeu_ps(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            _mm256_storeu_ps(y3_ptr, y3);
            let y4_ptr = data.as_mut_ptr().add(idx + 4 * num_columns) as *mut f32;
            _mm256_storeu_ps(y4_ptr, y4);
            let y5_ptr = data.as_mut_ptr().add(idx + 5 * num_columns) as *mut f32;
            _mm256_storeu_ps(y5_ptr, y5);
            let y6_ptr = data.as_mut_ptr().add(idx + 6 * num_columns) as *mut f32;
            _mm256_storeu_ps(y6_ptr, y6);
        }

        super::butterfly_7_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}

/// AVX+FMA implementation: processes 4 columns at once using fused multiply-add.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_7_avx_fma(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    unsafe {
        let simd_cols = (num_columns / 4) * 4;

        // Broadcast W7 constants for SIMD operations.
        let w7_1_re = _mm256_set1_ps(super::W7_1_RE);
        let w7_1_im = _mm256_set1_ps(super::W7_1_IM);
        let w7_2_re = _mm256_set1_ps(super::W7_2_RE);
        let w7_2_im = _mm256_set1_ps(super::W7_2_IM);
        let w7_3_re = _mm256_set1_ps(super::W7_3_RE);
        let w7_3_im = _mm256_set1_ps(super::W7_3_IM);
        let w7_4_re = _mm256_set1_ps(super::W7_4_RE);
        let w7_4_im = _mm256_set1_ps(super::W7_4_IM);
        let w7_5_re = _mm256_set1_ps(super::W7_5_RE);
        let w7_5_im = _mm256_set1_ps(super::W7_5_IM);
        let w7_6_re = _mm256_set1_ps(super::W7_6_RE);
        let w7_6_im = _mm256_set1_ps(super::W7_6_IM);

        for idx in (0..simd_cols).step_by(4) {
            // Load 4 complex numbers from each row.
            let x0_ptr = data.as_ptr().add(idx) as *const f32;
            let x0 = _mm256_loadu_ps(x0_ptr);

            let x1_ptr = data.as_ptr().add(idx + num_columns) as *const f32;
            let x1 = _mm256_loadu_ps(x1_ptr);

            let x2_ptr = data.as_ptr().add(idx + 2 * num_columns) as *const f32;
            let x2 = _mm256_loadu_ps(x2_ptr);

            let x3_ptr = data.as_ptr().add(idx + 3 * num_columns) as *const f32;
            let x3 = _mm256_loadu_ps(x3_ptr);

            let x4_ptr = data.as_ptr().add(idx + 4 * num_columns) as *const f32;
            let x4 = _mm256_loadu_ps(x4_ptr);

            let x5_ptr = data.as_ptr().add(idx + 5 * num_columns) as *const f32;
            let x5 = _mm256_loadu_ps(x5_ptr);

            let x6_ptr = data.as_ptr().add(idx + 6 * num_columns) as *const f32;
            let x6 = _mm256_loadu_ps(x6_ptr);

            // Load 24 twiddle factors for 4 columns.
            let tw_ptr = stage_twiddles.as_ptr().add(idx * 6) as *const f32;
            let tw_0 = _mm256_loadu_ps(tw_ptr);
            let tw_1 = _mm256_loadu_ps(tw_ptr.add(8));
            let tw_2 = _mm256_loadu_ps(tw_ptr.add(16));
            let tw_3 = _mm256_loadu_ps(tw_ptr.add(24));
            let tw_4 = _mm256_loadu_ps(tw_ptr.add(32));
            let tw_5 = _mm256_loadu_ps(tw_ptr.add(40));

            // Extract w1, w2, w3, w4, w5, w6 for all 4 columns.
            let temp_0 = _mm256_permute2f128_ps(tw_0, tw_3, 0x20);
            let temp_1 = _mm256_permute2f128_ps(tw_1, tw_4, 0x20);
            let temp_2 = _mm256_permute2f128_ps(tw_0, tw_3, 0x31);
            let temp_3 = _mm256_permute2f128_ps(tw_1, tw_4, 0x31);
            let temp_4 = _mm256_permute2f128_ps(tw_2, tw_5, 0x20);
            let temp_5 = _mm256_permute2f128_ps(tw_2, tw_5, 0x31);

            let w1 = _mm256_shuffle_ps(temp_0, temp_3, 0b01_00_01_00);
            let w2 = _mm256_shuffle_ps(temp_0, temp_3, 0b11_10_11_10);
            let w3 = _mm256_shuffle_ps(temp_2, temp_4, 0b01_00_01_00);
            let w4 = _mm256_shuffle_ps(temp_2, temp_4, 0b11_10_11_10);
            let w5 = _mm256_shuffle_ps(temp_1, temp_5, 0b01_00_01_00);
            let w6 = _mm256_shuffle_ps(temp_1, temp_5, 0b11_10_11_10);

            // Helper macro for complex multiply using FMA.
            macro_rules! cmul_fma {
                ($x:expr, $w:expr) => {{
                    let w_re = _mm256_shuffle_ps($w, $w, 0b10_10_00_00);
                    let w_im = _mm256_shuffle_ps($w, $w, 0b11_11_01_01);
                    let x_swap = _mm256_shuffle_ps($x, $x, 0b10_11_00_01);
                    let prod_im = _mm256_mul_ps(w_im, x_swap);
                    _mm256_fmaddsub_ps(w_re, $x, prod_im)
                }};
            }

            // Complex multiply: t1 = x1 * w1, t2 = x2 * w2, etc.
            let t1 = cmul_fma!(x1, w1);
            let t2 = cmul_fma!(x2, w2);
            let t3 = cmul_fma!(x3, w3);
            let t4 = cmul_fma!(x4, w4);
            let t5 = cmul_fma!(x5, w5);
            let t6 = cmul_fma!(x6, w6);

            // Y0 = x0 + t1 + t2 + t3 + t4 + t5 + t6
            let y0 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1), _mm256_add_ps(t2, t3)),
                _mm256_add_ps(_mm256_add_ps(t4, t5), t6),
            );

            // Helper macro for complex multiply by W7 constant using FMA.
            macro_rules! cmul_w7 {
                ($t:expr, $w_re:expr, $w_im:expr) => {{
                    let t_re = _mm256_shuffle_ps($t, $t, 0b10_10_00_00);
                    let t_im = _mm256_shuffle_ps($t, $t, 0b11_11_01_01);
                    let re = _mm256_fmsub_ps(t_re, $w_re, _mm256_mul_ps(t_im, $w_im));
                    let im = _mm256_fmadd_ps(t_re, $w_im, _mm256_mul_ps(t_im, $w_re));
                    let lo = _mm256_unpacklo_ps(re, im);
                    let hi = _mm256_unpackhi_ps(re, im);
                    _mm256_shuffle_ps(lo, hi, 0b01_00_01_00)
                }};
            }

            // Y1 = x0 + t1*W_7^1 + t2*W_7^2 + t3*W_7^3 + t4*W_7^4 + t5*W_7^5 + t6*W_7^6
            let t1_w71 = cmul_w7!(t1, w7_1_re, w7_1_im);
            let t2_w72 = cmul_w7!(t2, w7_2_re, w7_2_im);
            let t3_w73 = cmul_w7!(t3, w7_3_re, w7_3_im);
            let t4_w74 = cmul_w7!(t4, w7_4_re, w7_4_im);
            let t5_w75 = cmul_w7!(t5, w7_5_re, w7_5_im);
            let t6_w76 = cmul_w7!(t6, w7_6_re, w7_6_im);

            let y1 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w71), _mm256_add_ps(t2_w72, t3_w73)),
                _mm256_add_ps(_mm256_add_ps(t4_w74, t5_w75), t6_w76),
            );

            // Y2 = x0 + t1*W_7^2 + t2*W_7^4 + t3*W_7^6 + t4*W_7^1 + t5*W_7^3 + t6*W_7^5
            let t1_w72 = cmul_w7!(t1, w7_2_re, w7_2_im);
            let t2_w74 = cmul_w7!(t2, w7_4_re, w7_4_im);
            let t3_w76 = cmul_w7!(t3, w7_6_re, w7_6_im);
            let t4_w71 = cmul_w7!(t4, w7_1_re, w7_1_im);
            let t5_w73 = cmul_w7!(t5, w7_3_re, w7_3_im);
            let t6_w75 = cmul_w7!(t6, w7_5_re, w7_5_im);

            let y2 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w72), _mm256_add_ps(t2_w74, t3_w76)),
                _mm256_add_ps(_mm256_add_ps(t4_w71, t5_w73), t6_w75),
            );

            // Y3 = x0 + t1*W_7^3 + t2*W_7^6 + t3*W_7^2 + t4*W_7^5 + t5*W_7^1 + t6*W_7^4
            let t1_w73 = cmul_w7!(t1, w7_3_re, w7_3_im);
            let t2_w76 = cmul_w7!(t2, w7_6_re, w7_6_im);
            let t3_w72 = cmul_w7!(t3, w7_2_re, w7_2_im);
            let t4_w75 = cmul_w7!(t4, w7_5_re, w7_5_im);
            let t5_w71 = cmul_w7!(t5, w7_1_re, w7_1_im);
            let t6_w74 = cmul_w7!(t6, w7_4_re, w7_4_im);

            let y3 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w73), _mm256_add_ps(t2_w76, t3_w72)),
                _mm256_add_ps(_mm256_add_ps(t4_w75, t5_w71), t6_w74),
            );

            // Y4 = x0 + t1*W_7^4 + t2*W_7^1 + t3*W_7^5 + t4*W_7^2 + t5*W_7^6 + t6*W_7^3
            let t1_w74 = cmul_w7!(t1, w7_4_re, w7_4_im);
            let t2_w71 = cmul_w7!(t2, w7_1_re, w7_1_im);
            let t3_w75 = cmul_w7!(t3, w7_5_re, w7_5_im);
            let t4_w72 = cmul_w7!(t4, w7_2_re, w7_2_im);
            let t5_w76 = cmul_w7!(t5, w7_6_re, w7_6_im);
            let t6_w73 = cmul_w7!(t6, w7_3_re, w7_3_im);

            let y4 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w74), _mm256_add_ps(t2_w71, t3_w75)),
                _mm256_add_ps(_mm256_add_ps(t4_w72, t5_w76), t6_w73),
            );

            // Y5 = x0 + t1*W_7^5 + t2*W_7^3 + t3*W_7^1 + t4*W_7^6 + t5*W_7^4 + t6*W_7^2
            let t1_w75 = cmul_w7!(t1, w7_5_re, w7_5_im);
            let t2_w73 = cmul_w7!(t2, w7_3_re, w7_3_im);
            let t3_w71 = cmul_w7!(t3, w7_1_re, w7_1_im);
            let t4_w76 = cmul_w7!(t4, w7_6_re, w7_6_im);
            let t5_w74 = cmul_w7!(t5, w7_4_re, w7_4_im);
            let t6_w72 = cmul_w7!(t6, w7_2_re, w7_2_im);

            let y5 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w75), _mm256_add_ps(t2_w73, t3_w71)),
                _mm256_add_ps(_mm256_add_ps(t4_w76, t5_w74), t6_w72),
            );

            // Y6 = x0 + t1*W_7^6 + t2*W_7^5 + t3*W_7^4 + t4*W_7^3 + t5*W_7^2 + t6*W_7^1
            let t1_w76 = cmul_w7!(t1, w7_6_re, w7_6_im);
            let t2_w75 = cmul_w7!(t2, w7_5_re, w7_5_im);
            let t3_w74 = cmul_w7!(t3, w7_4_re, w7_4_im);
            let t4_w73 = cmul_w7!(t4, w7_3_re, w7_3_im);
            let t5_w72 = cmul_w7!(t5, w7_2_re, w7_2_im);
            let t6_w71 = cmul_w7!(t6, w7_1_re, w7_1_im);

            let y6 = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(x0, t1_w76), _mm256_add_ps(t2_w75, t3_w74)),
                _mm256_add_ps(_mm256_add_ps(t4_w73, t5_w72), t6_w71),
            );

            let y0_ptr = data.as_mut_ptr().add(idx) as *mut f32;
            _mm256_storeu_ps(y0_ptr, y0);
            let y1_ptr = data.as_mut_ptr().add(idx + num_columns) as *mut f32;
            _mm256_storeu_ps(y1_ptr, y1);
            let y2_ptr = data.as_mut_ptr().add(idx + 2 * num_columns) as *mut f32;
            _mm256_storeu_ps(y2_ptr, y2);
            let y3_ptr = data.as_mut_ptr().add(idx + 3 * num_columns) as *mut f32;
            _mm256_storeu_ps(y3_ptr, y3);
            let y4_ptr = data.as_mut_ptr().add(idx + 4 * num_columns) as *mut f32;
            _mm256_storeu_ps(y4_ptr, y4);
            let y5_ptr = data.as_mut_ptr().add(idx + 5 * num_columns) as *mut f32;
            _mm256_storeu_ps(y5_ptr, y5);
            let y6_ptr = data.as_mut_ptr().add(idx + 6 * num_columns) as *mut f32;
            _mm256_storeu_ps(y6_ptr, y6);
        }

        super::butterfly_7_columns(data, stage_twiddles, num_columns, simd_cols, num_columns);
    }
}
