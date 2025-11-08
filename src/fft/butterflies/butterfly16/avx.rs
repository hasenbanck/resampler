use core::{arch::x86_64::*, f32::consts::FRAC_1_SQRT_2};

use super::{W16_1_IM, W16_1_RE, W16_3_IM, W16_3_RE};
use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_avx, complex_mul_i_avx, load_neg_imag_mask_avx},
};

/// Performs a single radix-16 Stockham butterfly stage for stride=1 (out-of-place, AVX+FMA).
///
/// This is a specialized version for the stride=1 case (first stage) that uses the unpack-permute
/// pattern to enable sequential SIMD stores. For stride=1, output indices are sequential: j=16*i.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix16_stride1_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let sixteenth_samples = samples >> 4;
    let simd_iters = (sixteenth_samples >> 2) << 2;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_avx();

        for i in (0..simd_iters).step_by(4) {
            // Load 16 input values from the 16 "sixteenths" of the input array.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + sixteenth_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + sixteenth_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + sixteenth_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + sixteenth_samples * 4) as *const f32;
            let z4 = _mm256_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + sixteenth_samples * 5) as *const f32;
            let z5 = _mm256_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + sixteenth_samples * 6) as *const f32;
            let z6 = _mm256_loadu_ps(z6_ptr);

            let z7_ptr = src.as_ptr().add(i + sixteenth_samples * 7) as *const f32;
            let z7 = _mm256_loadu_ps(z7_ptr);

            let z8_ptr = src.as_ptr().add(i + sixteenth_samples * 8) as *const f32;
            let z8 = _mm256_loadu_ps(z8_ptr);

            let z9_ptr = src.as_ptr().add(i + sixteenth_samples * 9) as *const f32;
            let z9 = _mm256_loadu_ps(z9_ptr);

            let z10_ptr = src.as_ptr().add(i + sixteenth_samples * 10) as *const f32;
            let z10 = _mm256_loadu_ps(z10_ptr);

            let z11_ptr = src.as_ptr().add(i + sixteenth_samples * 11) as *const f32;
            let z11 = _mm256_loadu_ps(z11_ptr);

            let z12_ptr = src.as_ptr().add(i + sixteenth_samples * 12) as *const f32;
            let z12 = _mm256_loadu_ps(z12_ptr);

            let z13_ptr = src.as_ptr().add(i + sixteenth_samples * 13) as *const f32;
            let z13 = _mm256_loadu_ps(z13_ptr);

            let z14_ptr = src.as_ptr().add(i + sixteenth_samples * 14) as *const f32;
            let z14 = _mm256_loadu_ps(z14_ptr);

            let z15_ptr = src.as_ptr().add(i + sixteenth_samples * 15) as *const f32;
            let z15 = _mm256_loadu_ps(z15_ptr);

            // Load prepackaged twiddles in packed format (15 twiddles per group of 4 lanes).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 15) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]
            let w3 = _mm256_loadu_ps(tw_ptr.add(16)); // w3[i..i+4]
            let w4 = _mm256_loadu_ps(tw_ptr.add(24)); // w4[i..i+4]
            let w5 = _mm256_loadu_ps(tw_ptr.add(32)); // w5[i..i+4]
            let w6 = _mm256_loadu_ps(tw_ptr.add(40)); // w6[i..i+4]
            let w7 = _mm256_loadu_ps(tw_ptr.add(48)); // w7[i..i+4]
            let w8 = _mm256_loadu_ps(tw_ptr.add(56)); // w8[i..i+4]
            let w9 = _mm256_loadu_ps(tw_ptr.add(64)); // w9[i..i+4]
            let w10 = _mm256_loadu_ps(tw_ptr.add(72)); // w10[i..i+4]
            let w11 = _mm256_loadu_ps(tw_ptr.add(80)); // w11[i..i+4]
            let w12 = _mm256_loadu_ps(tw_ptr.add(88)); // w12[i..i+4]
            let w13 = _mm256_loadu_ps(tw_ptr.add(96)); // w13[i..i+4]
            let w14 = _mm256_loadu_ps(tw_ptr.add(104)); // w14[i..i+4]
            let w15 = _mm256_loadu_ps(tw_ptr.add(112)); // w15[i..i+4]

            // Apply twiddle factors: t1-t15 = w1-w15 * z1-z15
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);
            let t3 = complex_mul_avx(w3, z3);
            let t4 = complex_mul_avx(w4, z4);
            let t5 = complex_mul_avx(w5, z5);
            let t6 = complex_mul_avx(w6, z6);
            let t7 = complex_mul_avx(w7, z7);
            let t8 = complex_mul_avx(w8, z8);
            let t9 = complex_mul_avx(w9, z9);
            let t10 = complex_mul_avx(w10, z10);
            let t11 = complex_mul_avx(w11, z11);
            let t12 = complex_mul_avx(w12, z12);
            let t13 = complex_mul_avx(w13, z13);
            let t14 = complex_mul_avx(w14, z14);
            let t15 = complex_mul_avx(w15, z15);

            // Split-radix decomposition:
            // EVEN indices: (0, 2, 4, 6, 8, 10, 12, 14) -> (z0, t2, t4, t6, t8, t10, t12, t14)
            // ODD indices: (1, 3, 5, 7, 9, 11, 13, 15) -> (t1, t3, t5, t7, t9, t11, t13, t15)

            // Process EVEN group using radix-8 split-radix.
            // Even further split into even-even and even-odd:
            // Even-even: (z0, t4, t8, t12)
            // Even-odd: (t2, t6, t10, t14)

            let even_ee_a0 = _mm256_add_ps(z0, t8);
            let even_ee_a1 = _mm256_sub_ps(z0, t8);
            let even_ee_a2 = _mm256_add_ps(t4, t12);
            let t4_sub_t12 = _mm256_sub_ps(t4, t12);
            let even_ee_a3 = complex_mul_i_avx(t4_sub_t12, neg_imag_mask);

            let x_ee_0 = _mm256_add_ps(even_ee_a0, even_ee_a2);
            let x_ee_2 = _mm256_sub_ps(even_ee_a0, even_ee_a2);
            let x_ee_1 = _mm256_add_ps(even_ee_a1, even_ee_a3);
            let x_ee_3 = _mm256_sub_ps(even_ee_a1, even_ee_a3);

            let even_eo_a0 = _mm256_add_ps(t2, t10);
            let even_eo_a1 = _mm256_sub_ps(t2, t10);
            let even_eo_a2 = _mm256_add_ps(t6, t14);
            let t6_sub_t14 = _mm256_sub_ps(t6, t14);
            let even_eo_a3 = complex_mul_i_avx(t6_sub_t14, neg_imag_mask);

            let x_eo_0 = _mm256_add_ps(even_eo_a0, even_eo_a2);
            let x_eo_2 = _mm256_sub_ps(even_eo_a0, even_eo_a2);
            let x_eo_1 = _mm256_add_ps(even_eo_a1, even_eo_a3);
            let x_eo_3 = _mm256_sub_ps(even_eo_a1, even_eo_a3);

            // Combine with W_8 twiddles to get x_even[0..7]
            let x_even_0 = _mm256_add_ps(x_ee_0, x_eo_0);
            let x_even_4 = _mm256_sub_ps(x_ee_0, x_eo_0);

            let w8_1 = _mm256_setr_ps(
                FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
            );
            let w8_1_x_eo_1 = complex_mul_avx(w8_1, x_eo_1);
            let x_even_1 = _mm256_add_ps(x_ee_1, w8_1_x_eo_1);
            let x_even_5 = _mm256_sub_ps(x_ee_1, w8_1_x_eo_1);

            let w8_2_x_eo_2 = complex_mul_i_avx(x_eo_2, neg_imag_mask);
            let x_even_2 = _mm256_add_ps(x_ee_2, w8_2_x_eo_2);
            let x_even_6 = _mm256_sub_ps(x_ee_2, w8_2_x_eo_2);

            let w8_3 = _mm256_setr_ps(
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
            );
            let w8_3_x_eo_3 = complex_mul_avx(w8_3, x_eo_3);
            let x_even_3 = _mm256_add_ps(x_ee_3, w8_3_x_eo_3);
            let x_even_7 = _mm256_sub_ps(x_ee_3, w8_3_x_eo_3);

            // Process ODD group using radix-8 split-radix.
            // Odd further split into odd-even and odd-odd:
            // Odd-even: (t1, t5, t9, t13)
            // Odd-odd: (t3, t7, t11, t15)

            let odd_ee_a0 = _mm256_add_ps(t1, t9);
            let odd_ee_a1 = _mm256_sub_ps(t1, t9);
            let odd_ee_a2 = _mm256_add_ps(t5, t13);
            let t5_sub_t13 = _mm256_sub_ps(t5, t13);
            let odd_ee_a3 = complex_mul_i_avx(t5_sub_t13, neg_imag_mask);

            let x_oe_0 = _mm256_add_ps(odd_ee_a0, odd_ee_a2);
            let x_oe_2 = _mm256_sub_ps(odd_ee_a0, odd_ee_a2);
            let x_oe_1 = _mm256_add_ps(odd_ee_a1, odd_ee_a3);
            let x_oe_3 = _mm256_sub_ps(odd_ee_a1, odd_ee_a3);

            let odd_eo_a0 = _mm256_add_ps(t3, t11);
            let odd_eo_a1 = _mm256_sub_ps(t3, t11);
            let odd_eo_a2 = _mm256_add_ps(t7, t15);
            let t7_sub_t15 = _mm256_sub_ps(t7, t15);
            let odd_eo_a3 = complex_mul_i_avx(t7_sub_t15, neg_imag_mask);

            let x_oo_0 = _mm256_add_ps(odd_eo_a0, odd_eo_a2);
            let x_oo_2 = _mm256_sub_ps(odd_eo_a0, odd_eo_a2);
            let x_oo_1 = _mm256_add_ps(odd_eo_a1, odd_eo_a3);
            let x_oo_3 = _mm256_sub_ps(odd_eo_a1, odd_eo_a3);

            // Combine with W_8 twiddles to get x_odd[0..7]
            let x_odd_0 = _mm256_add_ps(x_oe_0, x_oo_0);
            let x_odd_4 = _mm256_sub_ps(x_oe_0, x_oo_0);

            let w8_1_x_oo_1 = complex_mul_avx(w8_1, x_oo_1);
            let x_odd_1 = _mm256_add_ps(x_oe_1, w8_1_x_oo_1);
            let x_odd_5 = _mm256_sub_ps(x_oe_1, w8_1_x_oo_1);

            let w8_2_x_oo_2 = complex_mul_i_avx(x_oo_2, neg_imag_mask);
            let x_odd_2 = _mm256_add_ps(x_oe_2, w8_2_x_oo_2);
            let x_odd_6 = _mm256_sub_ps(x_oe_2, w8_2_x_oo_2);

            let w8_3_x_oo_3 = complex_mul_avx(w8_3, x_oo_3);
            let x_odd_3 = _mm256_add_ps(x_oe_3, w8_3_x_oo_3);
            let x_odd_7 = _mm256_sub_ps(x_oe_3, w8_3_x_oo_3);

            // Apply W_16 twiddles and combine even/odd results.
            // out[k] = x_even[k] + W_16^k * x_odd[k]
            // out[k+8] = x_even[k] - W_16^k * x_odd[k]

            // k=0: W_16^0 = 1
            let out0 = _mm256_add_ps(x_even_0, x_odd_0);
            let out8 = _mm256_sub_ps(x_even_0, x_odd_0);

            // k=1: W_16^1 = (0.92387953, -0.38268343)
            let w16_1_const = _mm256_setr_ps(
                W16_1_RE, W16_1_IM, W16_1_RE, W16_1_IM, W16_1_RE, W16_1_IM, W16_1_RE, W16_1_IM,
            );
            let w16_1_x_odd_1 = complex_mul_avx(w16_1_const, x_odd_1);
            let out1 = _mm256_add_ps(x_even_1, w16_1_x_odd_1);
            let out9 = _mm256_sub_ps(x_even_1, w16_1_x_odd_1);

            // k=2: W_16^2 = W_8^1 = (1-i)/√2
            let w16_2_x_odd_2 = complex_mul_avx(w8_1, x_odd_2);
            let out2 = _mm256_add_ps(x_even_2, w16_2_x_odd_2);
            let out10 = _mm256_sub_ps(x_even_2, w16_2_x_odd_2);

            // k=3: W_16^3 = (0.38268343, -0.92387953)
            let w16_3_const = _mm256_setr_ps(
                W16_3_RE, W16_3_IM, W16_3_RE, W16_3_IM, W16_3_RE, W16_3_IM, W16_3_RE, W16_3_IM,
            );
            let w16_3_x_odd_3 = complex_mul_avx(w16_3_const, x_odd_3);
            let out3 = _mm256_add_ps(x_even_3, w16_3_x_odd_3);
            let out11 = _mm256_sub_ps(x_even_3, w16_3_x_odd_3);

            // k=4: W_16^4 = W_8^2 = -i
            let w16_4_x_odd_4 = complex_mul_i_avx(x_odd_4, neg_imag_mask);
            let out4 = _mm256_add_ps(x_even_4, w16_4_x_odd_4);
            let out12 = _mm256_sub_ps(x_even_4, w16_4_x_odd_4);

            // k=5: W_16^5 = -W_16^3 = (-0.38268343, -0.92387953)
            let w16_5_const = _mm256_setr_ps(
                -W16_3_RE, W16_3_IM, -W16_3_RE, W16_3_IM, -W16_3_RE, W16_3_IM, -W16_3_RE, W16_3_IM,
            );
            let w16_5_x_odd_5 = complex_mul_avx(w16_5_const, x_odd_5);
            let out5 = _mm256_add_ps(x_even_5, w16_5_x_odd_5);
            let out13 = _mm256_sub_ps(x_even_5, w16_5_x_odd_5);

            // k=6: W_16^6 = W_8^3 = (-1-i)/√2
            let w16_6_x_odd_6 = complex_mul_avx(w8_3, x_odd_6);
            let out6 = _mm256_add_ps(x_even_6, w16_6_x_odd_6);
            let out14 = _mm256_sub_ps(x_even_6, w16_6_x_odd_6);

            // k=7: W_16^7 = -W_16^1 = (-0.92387953, -0.38268343)
            let w16_7_const = _mm256_setr_ps(
                -W16_1_RE, W16_1_IM, -W16_1_RE, W16_1_IM, -W16_1_RE, W16_1_IM, -W16_1_RE, W16_1_IM,
            );
            let w16_7_x_odd_7 = complex_mul_avx(w16_7_const, x_odd_7);
            let out7 = _mm256_add_ps(x_even_7, w16_7_x_odd_7);
            let out15 = _mm256_sub_ps(x_even_7, w16_7_x_odd_7);

            // Interleave 16 outputs for sequential storage (stride-1 optimization).
            // Cast to double for easier interleaving (treats each complex pair as one 64-bit value).
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);
            let out4_pd = _mm256_castps_pd(out4);
            let out5_pd = _mm256_castps_pd(out5);
            let out6_pd = _mm256_castps_pd(out6);
            let out7_pd = _mm256_castps_pd(out7);
            let out8_pd = _mm256_castps_pd(out8);
            let out9_pd = _mm256_castps_pd(out9);
            let out10_pd = _mm256_castps_pd(out10);
            let out11_pd = _mm256_castps_pd(out11);
            let out12_pd = _mm256_castps_pd(out12);
            let out13_pd = _mm256_castps_pd(out13);
            let out14_pd = _mm256_castps_pd(out14);
            let out15_pd = _mm256_castps_pd(out15);

            // Extract 128-bit lanes (each lane has 2 doubles = 1 complex in float representation)
            let out0_lo = _mm256_castpd256_pd128(out0_pd);
            let out0_hi = _mm256_extractf128_pd(out0_pd, 1);
            let out1_lo = _mm256_castpd256_pd128(out1_pd);
            let out1_hi = _mm256_extractf128_pd(out1_pd, 1);
            let out2_lo = _mm256_castpd256_pd128(out2_pd);
            let out2_hi = _mm256_extractf128_pd(out2_pd, 1);
            let out3_lo = _mm256_castpd256_pd128(out3_pd);
            let out3_hi = _mm256_extractf128_pd(out3_pd, 1);
            let out4_lo = _mm256_castpd256_pd128(out4_pd);
            let out4_hi = _mm256_extractf128_pd(out4_pd, 1);
            let out5_lo = _mm256_castpd256_pd128(out5_pd);
            let out5_hi = _mm256_extractf128_pd(out5_pd, 1);
            let out6_lo = _mm256_castpd256_pd128(out6_pd);
            let out6_hi = _mm256_extractf128_pd(out6_pd, 1);
            let out7_lo = _mm256_castpd256_pd128(out7_pd);
            let out7_hi = _mm256_extractf128_pd(out7_pd, 1);
            let out8_lo = _mm256_castpd256_pd128(out8_pd);
            let out8_hi = _mm256_extractf128_pd(out8_pd, 1);
            let out9_lo = _mm256_castpd256_pd128(out9_pd);
            let out9_hi = _mm256_extractf128_pd(out9_pd, 1);
            let out10_lo = _mm256_castpd256_pd128(out10_pd);
            let out10_hi = _mm256_extractf128_pd(out10_pd, 1);
            let out11_lo = _mm256_castpd256_pd128(out11_pd);
            let out11_hi = _mm256_extractf128_pd(out11_pd, 1);
            let out12_lo = _mm256_castpd256_pd128(out12_pd);
            let out12_hi = _mm256_extractf128_pd(out12_pd, 1);
            let out13_lo = _mm256_castpd256_pd128(out13_pd);
            let out13_hi = _mm256_extractf128_pd(out13_pd, 1);
            let out14_lo = _mm256_castpd256_pd128(out14_pd);
            let out14_hi = _mm256_extractf128_pd(out14_pd, 1);
            let out15_lo = _mm256_castpd256_pd128(out15_pd);
            let out15_hi = _mm256_extractf128_pd(out15_pd, 1);

            // Build 16 result vectors by interleaving pairs from low lane (iteration i and i+1)
            // result0: [out0[0], out1[0], out2[0], out3[0]]
            let temp0_lo = _mm_unpacklo_pd(out0_lo, out1_lo);
            let temp0_hi = _mm_unpacklo_pd(out2_lo, out3_lo);
            let result0 = _mm256_castpd_ps(_mm256_set_m128d(temp0_hi, temp0_lo));

            // result1: [out4[0], out5[0], out6[0], out7[0]]
            let temp1_lo = _mm_unpacklo_pd(out4_lo, out5_lo);
            let temp1_hi = _mm_unpacklo_pd(out6_lo, out7_lo);
            let result1 = _mm256_castpd_ps(_mm256_set_m128d(temp1_hi, temp1_lo));

            // result2: [out8[0], out9[0], out10[0], out11[0]]
            let temp2_lo = _mm_unpacklo_pd(out8_lo, out9_lo);
            let temp2_hi = _mm_unpacklo_pd(out10_lo, out11_lo);
            let result2 = _mm256_castpd_ps(_mm256_set_m128d(temp2_hi, temp2_lo));

            // result3: [out12[0], out13[0], out14[0], out15[0]]
            let temp3_lo = _mm_unpacklo_pd(out12_lo, out13_lo);
            let temp3_hi = _mm_unpacklo_pd(out14_lo, out15_lo);
            let result3 = _mm256_castpd_ps(_mm256_set_m128d(temp3_hi, temp3_lo));

            // High parts (iterations i+2 and i+3)
            // result4: [out0[1], out1[1], out2[1], out3[1]]
            let temp4_lo = _mm_unpackhi_pd(out0_lo, out1_lo);
            let temp4_hi = _mm_unpackhi_pd(out2_lo, out3_lo);
            let result4 = _mm256_castpd_ps(_mm256_set_m128d(temp4_hi, temp4_lo));

            // result5: [out4[1], out5[1], out6[1], out7[1]]
            let temp5_lo = _mm_unpackhi_pd(out4_lo, out5_lo);
            let temp5_hi = _mm_unpackhi_pd(out6_lo, out7_lo);
            let result5 = _mm256_castpd_ps(_mm256_set_m128d(temp5_hi, temp5_lo));

            // result6: [out8[1], out9[1], out10[1], out11[1]]
            let temp6_lo = _mm_unpackhi_pd(out8_lo, out9_lo);
            let temp6_hi = _mm_unpackhi_pd(out10_lo, out11_lo);
            let result6 = _mm256_castpd_ps(_mm256_set_m128d(temp6_hi, temp6_lo));

            // result7: [out12[1], out13[1], out14[1], out15[1]]
            let temp7_lo = _mm_unpackhi_pd(out12_lo, out13_lo);
            let temp7_hi = _mm_unpackhi_pd(out14_lo, out15_lo);
            let result7 = _mm256_castpd_ps(_mm256_set_m128d(temp7_hi, temp7_lo));

            // High lane (iterations i+2, i+3)
            // result8: [out0[2], out1[2], out2[2], out3[2]]
            let temp8_lo = _mm_unpacklo_pd(out0_hi, out1_hi);
            let temp8_hi = _mm_unpacklo_pd(out2_hi, out3_hi);
            let result8 = _mm256_castpd_ps(_mm256_set_m128d(temp8_hi, temp8_lo));

            // result9: [out4[2], out5[2], out6[2], out7[2]]
            let temp9_lo = _mm_unpacklo_pd(out4_hi, out5_hi);
            let temp9_hi = _mm_unpacklo_pd(out6_hi, out7_hi);
            let result9 = _mm256_castpd_ps(_mm256_set_m128d(temp9_hi, temp9_lo));

            // result10: [out8[2], out9[2], out10[2], out11[2]]
            let temp10_lo = _mm_unpacklo_pd(out8_hi, out9_hi);
            let temp10_hi = _mm_unpacklo_pd(out10_hi, out11_hi);
            let result10 = _mm256_castpd_ps(_mm256_set_m128d(temp10_hi, temp10_lo));

            // result11: [out12[2], out13[2], out14[2], out15[2]]
            let temp11_lo = _mm_unpacklo_pd(out12_hi, out13_hi);
            let temp11_hi = _mm_unpacklo_pd(out14_hi, out15_hi);
            let result11 = _mm256_castpd_ps(_mm256_set_m128d(temp11_hi, temp11_lo));

            // result12: [out0[3], out1[3], out2[3], out3[3]]
            let temp12_lo = _mm_unpackhi_pd(out0_hi, out1_hi);
            let temp12_hi = _mm_unpackhi_pd(out2_hi, out3_hi);
            let result12 = _mm256_castpd_ps(_mm256_set_m128d(temp12_hi, temp12_lo));

            // result13: [out4[3], out5[3], out6[3], out7[3]]
            let temp13_lo = _mm_unpackhi_pd(out4_hi, out5_hi);
            let temp13_hi = _mm_unpackhi_pd(out6_hi, out7_hi);
            let result13 = _mm256_castpd_ps(_mm256_set_m128d(temp13_hi, temp13_lo));

            // result14: [out8[3], out9[3], out10[3], out11[3]]
            let temp14_lo = _mm_unpackhi_pd(out8_hi, out9_hi);
            let temp14_hi = _mm_unpackhi_pd(out10_hi, out11_hi);
            let result14 = _mm256_castpd_ps(_mm256_set_m128d(temp14_hi, temp14_lo));

            // result15: [out12[3], out13[3], out14[3], out15[3]]
            let temp15_lo = _mm_unpackhi_pd(out12_hi, out13_hi);
            let temp15_hi = _mm_unpackhi_pd(out14_hi, out15_hi);
            let result15 = _mm256_castpd_ps(_mm256_set_m128d(temp15_hi, temp15_lo));

            // Sequential stores - 16 stores for 16 output vectors
            let j = 16 * i;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            _mm256_storeu_ps(dst_ptr, result0);
            _mm256_storeu_ps(dst_ptr.add(8), result1);
            _mm256_storeu_ps(dst_ptr.add(16), result2);
            _mm256_storeu_ps(dst_ptr.add(24), result3);
            _mm256_storeu_ps(dst_ptr.add(32), result4);
            _mm256_storeu_ps(dst_ptr.add(40), result5);
            _mm256_storeu_ps(dst_ptr.add(48), result6);
            _mm256_storeu_ps(dst_ptr.add(56), result7);
            _mm256_storeu_ps(dst_ptr.add(64), result8);
            _mm256_storeu_ps(dst_ptr.add(72), result9);
            _mm256_storeu_ps(dst_ptr.add(80), result10);
            _mm256_storeu_ps(dst_ptr.add(88), result11);
            _mm256_storeu_ps(dst_ptr.add(96), result12);
            _mm256_storeu_ps(dst_ptr.add(104), result13);
            _mm256_storeu_ps(dst_ptr.add(112), result14);
            _mm256_storeu_ps(dst_ptr.add(120), result15);
        }
    }

    // Scalar tail - process remaining iterations
    let tail_offset = (simd_iters / 4) * (15 * 4);
    for i in simd_iters..sixteenth_samples {
        let tail_idx = i - simd_iters;

        // Load twiddles for this iteration (interleaved format)
        let w1 = stage_twiddles[tail_offset + tail_idx * 15];
        let w2 = stage_twiddles[tail_offset + tail_idx * 15 + 1];
        let w3 = stage_twiddles[tail_offset + tail_idx * 15 + 2];
        let w4 = stage_twiddles[tail_offset + tail_idx * 15 + 3];
        let w5 = stage_twiddles[tail_offset + tail_idx * 15 + 4];
        let w6 = stage_twiddles[tail_offset + tail_idx * 15 + 5];
        let w7 = stage_twiddles[tail_offset + tail_idx * 15 + 6];
        let w8 = stage_twiddles[tail_offset + tail_idx * 15 + 7];
        let w9 = stage_twiddles[tail_offset + tail_idx * 15 + 8];
        let w10 = stage_twiddles[tail_offset + tail_idx * 15 + 9];
        let w11 = stage_twiddles[tail_offset + tail_idx * 15 + 10];
        let w12 = stage_twiddles[tail_offset + tail_idx * 15 + 11];
        let w13 = stage_twiddles[tail_offset + tail_idx * 15 + 12];
        let w14 = stage_twiddles[tail_offset + tail_idx * 15 + 13];
        let w15 = stage_twiddles[tail_offset + tail_idx * 15 + 14];

        // Load 16 input values
        let z0 = src[i];
        let z1 = src[i + sixteenth_samples];
        let z2 = src[i + sixteenth_samples * 2];
        let z3 = src[i + sixteenth_samples * 3];
        let z4 = src[i + sixteenth_samples * 4];
        let z5 = src[i + sixteenth_samples * 5];
        let z6 = src[i + sixteenth_samples * 6];
        let z7 = src[i + sixteenth_samples * 7];
        let z8 = src[i + sixteenth_samples * 8];
        let z9 = src[i + sixteenth_samples * 9];
        let z10 = src[i + sixteenth_samples * 10];
        let z11 = src[i + sixteenth_samples * 11];
        let z12 = src[i + sixteenth_samples * 12];
        let z13 = src[i + sixteenth_samples * 13];
        let z14 = src[i + sixteenth_samples * 14];
        let z15 = src[i + sixteenth_samples * 15];

        // Apply twiddles
        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);
        let t5 = w5.mul(&z5);
        let t6 = w6.mul(&z6);
        let t7 = w7.mul(&z7);
        let t8 = w8.mul(&z8);
        let t9 = w9.mul(&z9);
        let t10 = w10.mul(&z10);
        let t11 = w11.mul(&z11);
        let t12 = w12.mul(&z12);
        let t13 = w13.mul(&z13);
        let t14 = w14.mul(&z14);
        let t15 = w15.mul(&z15);

        // Process EVEN group using scalar split-radix
        let z0_val = z0;
        let z2_val = t2;
        let z4_val = t4;
        let z6_val = t6;
        let z8_val = t8;
        let z10_val = t10;
        let z12_val = t12;
        let z14_val = t14;

        // Radix-4 on even-even
        let ee_a0 = z0_val.add(&z8_val);
        let ee_a1 = z0_val.sub(&z8_val);
        let ee_a2 = z4_val.add(&z12_val);
        let ee_a3_re = z4_val.im - z12_val.im;
        let ee_a3_im = z12_val.re - z4_val.re;

        let x_ee_0 = ee_a0.add(&ee_a2);
        let x_ee_2 = ee_a0.sub(&ee_a2);
        let x_ee_1 = Complex32::new(ee_a1.re + ee_a3_re, ee_a1.im + ee_a3_im);
        let x_ee_3 = Complex32::new(ee_a1.re - ee_a3_re, ee_a1.im - ee_a3_im);

        // Radix-4 on even-odd
        let eo_a0 = z2_val.add(&z10_val);
        let eo_a1 = z2_val.sub(&z10_val);
        let eo_a2 = z6_val.add(&z14_val);
        let eo_a3_re = z6_val.im - z14_val.im;
        let eo_a3_im = z14_val.re - z6_val.re;

        let x_eo_0 = eo_a0.add(&eo_a2);
        let x_eo_2 = eo_a0.sub(&eo_a2);
        let x_eo_1 = Complex32::new(eo_a1.re + eo_a3_re, eo_a1.im + eo_a3_im);
        let x_eo_3 = Complex32::new(eo_a1.re - eo_a3_re, eo_a1.im - eo_a3_im);

        // Combine with W_8 twiddles to get X_even[0..7]
        let x_even = [
            x_ee_0.add(&x_eo_0),
            Complex32::new(
                x_ee_1.re + FRAC_1_SQRT_2 * (x_eo_1.re + x_eo_1.im),
                x_ee_1.im + FRAC_1_SQRT_2 * (x_eo_1.im - x_eo_1.re),
            ),
            x_ee_2.add(&Complex32::new(x_eo_2.im, -x_eo_2.re)),
            Complex32::new(
                x_ee_3.re + FRAC_1_SQRT_2 * (x_eo_3.im - x_eo_3.re),
                x_ee_3.im - FRAC_1_SQRT_2 * (x_eo_3.re + x_eo_3.im),
            ),
            x_ee_0.sub(&x_eo_0),
            Complex32::new(
                x_ee_1.re - FRAC_1_SQRT_2 * (x_eo_1.re + x_eo_1.im),
                x_ee_1.im - FRAC_1_SQRT_2 * (x_eo_1.im - x_eo_1.re),
            ),
            x_ee_2.sub(&Complex32::new(x_eo_2.im, -x_eo_2.re)),
            Complex32::new(
                x_ee_3.re - FRAC_1_SQRT_2 * (x_eo_3.im - x_eo_3.re),
                x_ee_3.im + FRAC_1_SQRT_2 * (x_eo_3.re + x_eo_3.im),
            ),
        ];

        // Process ODD group
        let z1_val = t1;
        let z3_val = t3;
        let z5_val = t5;
        let z7_val = t7;
        let z9_val = t9;
        let z11_val = t11;
        let z13_val = t13;
        let z15_val = t15;

        // Radix-4 on odd-even
        let oe_a0 = z1_val.add(&z9_val);
        let oe_a1 = z1_val.sub(&z9_val);
        let oe_a2 = z5_val.add(&z13_val);
        let oe_a3_re = z5_val.im - z13_val.im;
        let oe_a3_im = z13_val.re - z5_val.re;

        let x_oe_0 = oe_a0.add(&oe_a2);
        let x_oe_2 = oe_a0.sub(&oe_a2);
        let x_oe_1 = Complex32::new(oe_a1.re + oe_a3_re, oe_a1.im + oe_a3_im);
        let x_oe_3 = Complex32::new(oe_a1.re - oe_a3_re, oe_a1.im - oe_a3_im);

        // Radix-4 on odd-odd
        let oo_a0 = z3_val.add(&z11_val);
        let oo_a1 = z3_val.sub(&z11_val);
        let oo_a2 = z7_val.add(&z15_val);
        let oo_a3_re = z7_val.im - z15_val.im;
        let oo_a3_im = z15_val.re - z7_val.re;

        let x_oo_0 = oo_a0.add(&oo_a2);
        let x_oo_2 = oo_a0.sub(&oo_a2);
        let x_oo_1 = Complex32::new(oo_a1.re + oo_a3_re, oo_a1.im + oo_a3_im);
        let x_oo_3 = Complex32::new(oo_a1.re - oo_a3_re, oo_a1.im - oo_a3_im);

        // Combine with W_8 twiddles to get X_odd[0..7]
        let x_odd = [
            x_oe_0.add(&x_oo_0),
            Complex32::new(
                x_oe_1.re + FRAC_1_SQRT_2 * (x_oo_1.re + x_oo_1.im),
                x_oe_1.im + FRAC_1_SQRT_2 * (x_oo_1.im - x_oo_1.re),
            ),
            x_oe_2.add(&Complex32::new(x_oo_2.im, -x_oo_2.re)),
            Complex32::new(
                x_oe_3.re + FRAC_1_SQRT_2 * (x_oo_3.im - x_oo_3.re),
                x_oe_3.im - FRAC_1_SQRT_2 * (x_oo_3.re + x_oo_3.im),
            ),
            x_oe_0.sub(&x_oo_0),
            Complex32::new(
                x_oe_1.re - FRAC_1_SQRT_2 * (x_oo_1.re + x_oo_1.im),
                x_oe_1.im - FRAC_1_SQRT_2 * (x_oo_1.im - x_oo_1.re),
            ),
            x_oe_2.sub(&Complex32::new(x_oo_2.im, -x_oo_2.re)),
            Complex32::new(
                x_oe_3.re - FRAC_1_SQRT_2 * (x_oo_3.im - x_oo_3.re),
                x_oe_3.im + FRAC_1_SQRT_2 * (x_oo_3.re + x_oo_3.im),
            ),
        ];

        // Final combining with W_16 twiddles
        let j = 16 * i;

        let w16_0_odd = x_odd[0];
        dst[j] = x_even[0].add(&w16_0_odd);
        dst[j + 8] = x_even[0].sub(&w16_0_odd);

        let w16_1_odd = Complex32::new(
            W16_1_RE * x_odd[1].re - W16_1_IM * x_odd[1].im,
            W16_1_RE * x_odd[1].im + W16_1_IM * x_odd[1].re,
        );
        dst[j + 1] = x_even[1].add(&w16_1_odd);
        dst[j + 9] = x_even[1].sub(&w16_1_odd);

        let w16_2_odd = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd[2].re + x_odd[2].im),
            FRAC_1_SQRT_2 * (x_odd[2].im - x_odd[2].re),
        );
        dst[j + 2] = x_even[2].add(&w16_2_odd);
        dst[j + 10] = x_even[2].sub(&w16_2_odd);

        let w16_3_odd = Complex32::new(
            W16_3_RE * x_odd[3].re - W16_3_IM * x_odd[3].im,
            W16_3_RE * x_odd[3].im + W16_3_IM * x_odd[3].re,
        );
        dst[j + 3] = x_even[3].add(&w16_3_odd);
        dst[j + 11] = x_even[3].sub(&w16_3_odd);

        let w16_4_odd = Complex32::new(x_odd[4].im, -x_odd[4].re);
        dst[j + 4] = x_even[4].add(&w16_4_odd);
        dst[j + 12] = x_even[4].sub(&w16_4_odd);

        let w16_5_odd = Complex32::new(
            -W16_3_RE * x_odd[5].re - W16_3_IM * x_odd[5].im,
            -W16_3_RE * x_odd[5].im + W16_3_IM * x_odd[5].re,
        );
        dst[j + 5] = x_even[5].add(&w16_5_odd);
        dst[j + 13] = x_even[5].sub(&w16_5_odd);

        let w16_6_odd = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd[6].im - x_odd[6].re),
            -FRAC_1_SQRT_2 * (x_odd[6].re + x_odd[6].im),
        );
        dst[j + 6] = x_even[6].add(&w16_6_odd);
        dst[j + 14] = x_even[6].sub(&w16_6_odd);

        let w16_7_odd = Complex32::new(
            -W16_1_RE * x_odd[7].re - W16_1_IM * x_odd[7].im,
            -W16_1_RE * x_odd[7].im + W16_1_IM * x_odd[7].re,
        );
        dst[j + 7] = x_even[7].add(&w16_7_odd);
        dst[j + 15] = x_even[7].sub(&w16_7_odd);
    }
}

/// Performs a single radix-16 Stockham butterfly stage for p>1 (out-of-place, AVX+FMA).
///
/// Generic version for p>1 cases. Uses direct SIMD stores, accepting scattered writes
/// as the shuffle overhead isn't justified for larger strides.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix16_generic_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    if stride == 0 {
        return;
    }

    let samples = src.len();
    let sixteenth_samples = samples >> 4;
    let simd_iters = (sixteenth_samples >> 2) << 2;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_avx();

        for i in (0..simd_iters).step_by(4) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;
            let k2 = k + 2 - ((k + 2 >= stride) as usize) * stride;
            let k3 = k + 3 - ((k + 3 >= stride) as usize) * stride;

            // Load 16 input values
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + sixteenth_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + sixteenth_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + sixteenth_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + sixteenth_samples * 4) as *const f32;
            let z4 = _mm256_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + sixteenth_samples * 5) as *const f32;
            let z5 = _mm256_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + sixteenth_samples * 6) as *const f32;
            let z6 = _mm256_loadu_ps(z6_ptr);

            let z7_ptr = src.as_ptr().add(i + sixteenth_samples * 7) as *const f32;
            let z7 = _mm256_loadu_ps(z7_ptr);

            let z8_ptr = src.as_ptr().add(i + sixteenth_samples * 8) as *const f32;
            let z8 = _mm256_loadu_ps(z8_ptr);

            let z9_ptr = src.as_ptr().add(i + sixteenth_samples * 9) as *const f32;
            let z9 = _mm256_loadu_ps(z9_ptr);

            let z10_ptr = src.as_ptr().add(i + sixteenth_samples * 10) as *const f32;
            let z10 = _mm256_loadu_ps(z10_ptr);

            let z11_ptr = src.as_ptr().add(i + sixteenth_samples * 11) as *const f32;
            let z11 = _mm256_loadu_ps(z11_ptr);

            let z12_ptr = src.as_ptr().add(i + sixteenth_samples * 12) as *const f32;
            let z12 = _mm256_loadu_ps(z12_ptr);

            let z13_ptr = src.as_ptr().add(i + sixteenth_samples * 13) as *const f32;
            let z13 = _mm256_loadu_ps(z13_ptr);

            let z14_ptr = src.as_ptr().add(i + sixteenth_samples * 14) as *const f32;
            let z14 = _mm256_loadu_ps(z14_ptr);

            let z15_ptr = src.as_ptr().add(i + sixteenth_samples * 15) as *const f32;
            let z15 = _mm256_loadu_ps(z15_ptr);

            // Load twiddles
            let tw_ptr = stage_twiddles.as_ptr().add(i * 15) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr);
            let w2 = _mm256_loadu_ps(tw_ptr.add(8));
            let w3 = _mm256_loadu_ps(tw_ptr.add(16));
            let w4 = _mm256_loadu_ps(tw_ptr.add(24));
            let w5 = _mm256_loadu_ps(tw_ptr.add(32));
            let w6 = _mm256_loadu_ps(tw_ptr.add(40));
            let w7 = _mm256_loadu_ps(tw_ptr.add(48));
            let w8 = _mm256_loadu_ps(tw_ptr.add(56));
            let w9 = _mm256_loadu_ps(tw_ptr.add(64));
            let w10 = _mm256_loadu_ps(tw_ptr.add(72));
            let w11 = _mm256_loadu_ps(tw_ptr.add(80));
            let w12 = _mm256_loadu_ps(tw_ptr.add(88));
            let w13 = _mm256_loadu_ps(tw_ptr.add(96));
            let w14 = _mm256_loadu_ps(tw_ptr.add(104));
            let w15 = _mm256_loadu_ps(tw_ptr.add(112));

            // Apply twiddles
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);
            let t3 = complex_mul_avx(w3, z3);
            let t4 = complex_mul_avx(w4, z4);
            let t5 = complex_mul_avx(w5, z5);
            let t6 = complex_mul_avx(w6, z6);
            let t7 = complex_mul_avx(w7, z7);
            let t8 = complex_mul_avx(w8, z8);
            let t9 = complex_mul_avx(w9, z9);
            let t10 = complex_mul_avx(w10, z10);
            let t11 = complex_mul_avx(w11, z11);
            let t12 = complex_mul_avx(w12, z12);
            let t13 = complex_mul_avx(w13, z13);
            let t14 = complex_mul_avx(w14, z14);
            let t15 = complex_mul_avx(w15, z15);

            // Process EVEN group (same as stride-1)
            let even_ee_a0 = _mm256_add_ps(z0, t8);
            let even_ee_a1 = _mm256_sub_ps(z0, t8);
            let even_ee_a2 = _mm256_add_ps(t4, t12);
            let t4_sub_t12 = _mm256_sub_ps(t4, t12);
            let even_ee_a3 = complex_mul_i_avx(t4_sub_t12, neg_imag_mask);

            let x_ee_0 = _mm256_add_ps(even_ee_a0, even_ee_a2);
            let x_ee_2 = _mm256_sub_ps(even_ee_a0, even_ee_a2);
            let x_ee_1 = _mm256_add_ps(even_ee_a1, even_ee_a3);
            let x_ee_3 = _mm256_sub_ps(even_ee_a1, even_ee_a3);

            let even_eo_a0 = _mm256_add_ps(t2, t10);
            let even_eo_a1 = _mm256_sub_ps(t2, t10);
            let even_eo_a2 = _mm256_add_ps(t6, t14);
            let t6_sub_t14 = _mm256_sub_ps(t6, t14);
            let even_eo_a3 = complex_mul_i_avx(t6_sub_t14, neg_imag_mask);

            let x_eo_0 = _mm256_add_ps(even_eo_a0, even_eo_a2);
            let x_eo_2 = _mm256_sub_ps(even_eo_a0, even_eo_a2);
            let x_eo_1 = _mm256_add_ps(even_eo_a1, even_eo_a3);
            let x_eo_3 = _mm256_sub_ps(even_eo_a1, even_eo_a3);

            let x_even_0 = _mm256_add_ps(x_ee_0, x_eo_0);
            let x_even_4 = _mm256_sub_ps(x_ee_0, x_eo_0);

            let w8_1 = _mm256_setr_ps(
                FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
            );
            let w8_1_x_eo_1 = complex_mul_avx(w8_1, x_eo_1);
            let x_even_1 = _mm256_add_ps(x_ee_1, w8_1_x_eo_1);
            let x_even_5 = _mm256_sub_ps(x_ee_1, w8_1_x_eo_1);

            let w8_2_x_eo_2 = complex_mul_i_avx(x_eo_2, neg_imag_mask);
            let x_even_2 = _mm256_add_ps(x_ee_2, w8_2_x_eo_2);
            let x_even_6 = _mm256_sub_ps(x_ee_2, w8_2_x_eo_2);

            let w8_3 = _mm256_setr_ps(
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
            );
            let w8_3_x_eo_3 = complex_mul_avx(w8_3, x_eo_3);
            let x_even_3 = _mm256_add_ps(x_ee_3, w8_3_x_eo_3);
            let x_even_7 = _mm256_sub_ps(x_ee_3, w8_3_x_eo_3);

            // Process ODD group
            let odd_ee_a0 = _mm256_add_ps(t1, t9);
            let odd_ee_a1 = _mm256_sub_ps(t1, t9);
            let odd_ee_a2 = _mm256_add_ps(t5, t13);
            let t5_sub_t13 = _mm256_sub_ps(t5, t13);
            let odd_ee_a3 = complex_mul_i_avx(t5_sub_t13, neg_imag_mask);

            let x_oe_0 = _mm256_add_ps(odd_ee_a0, odd_ee_a2);
            let x_oe_2 = _mm256_sub_ps(odd_ee_a0, odd_ee_a2);
            let x_oe_1 = _mm256_add_ps(odd_ee_a1, odd_ee_a3);
            let x_oe_3 = _mm256_sub_ps(odd_ee_a1, odd_ee_a3);

            let odd_eo_a0 = _mm256_add_ps(t3, t11);
            let odd_eo_a1 = _mm256_sub_ps(t3, t11);
            let odd_eo_a2 = _mm256_add_ps(t7, t15);
            let t7_sub_t15 = _mm256_sub_ps(t7, t15);
            let odd_eo_a3 = complex_mul_i_avx(t7_sub_t15, neg_imag_mask);

            let x_oo_0 = _mm256_add_ps(odd_eo_a0, odd_eo_a2);
            let x_oo_2 = _mm256_sub_ps(odd_eo_a0, odd_eo_a2);
            let x_oo_1 = _mm256_add_ps(odd_eo_a1, odd_eo_a3);
            let x_oo_3 = _mm256_sub_ps(odd_eo_a1, odd_eo_a3);

            let x_odd_0 = _mm256_add_ps(x_oe_0, x_oo_0);
            let x_odd_4 = _mm256_sub_ps(x_oe_0, x_oo_0);

            let w8_1_x_oo_1 = complex_mul_avx(w8_1, x_oo_1);
            let x_odd_1 = _mm256_add_ps(x_oe_1, w8_1_x_oo_1);
            let x_odd_5 = _mm256_sub_ps(x_oe_1, w8_1_x_oo_1);

            let w8_2_x_oo_2 = complex_mul_i_avx(x_oo_2, neg_imag_mask);
            let x_odd_2 = _mm256_add_ps(x_oe_2, w8_2_x_oo_2);
            let x_odd_6 = _mm256_sub_ps(x_oe_2, w8_2_x_oo_2);

            let w8_3_x_oo_3 = complex_mul_avx(w8_3, x_oo_3);
            let x_odd_3 = _mm256_add_ps(x_oe_3, w8_3_x_oo_3);
            let x_odd_7 = _mm256_sub_ps(x_oe_3, w8_3_x_oo_3);

            // Apply W_16 twiddles
            let out0 = _mm256_add_ps(x_even_0, x_odd_0);
            let out8 = _mm256_sub_ps(x_even_0, x_odd_0);

            let w16_1_const = _mm256_setr_ps(
                W16_1_RE, W16_1_IM, W16_1_RE, W16_1_IM, W16_1_RE, W16_1_IM, W16_1_RE, W16_1_IM,
            );
            let w16_1_x_odd_1 = complex_mul_avx(w16_1_const, x_odd_1);
            let out1 = _mm256_add_ps(x_even_1, w16_1_x_odd_1);
            let out9 = _mm256_sub_ps(x_even_1, w16_1_x_odd_1);

            let w16_2_x_odd_2 = complex_mul_avx(w8_1, x_odd_2);
            let out2 = _mm256_add_ps(x_even_2, w16_2_x_odd_2);
            let out10 = _mm256_sub_ps(x_even_2, w16_2_x_odd_2);

            let w16_3_const = _mm256_setr_ps(
                W16_3_RE, W16_3_IM, W16_3_RE, W16_3_IM, W16_3_RE, W16_3_IM, W16_3_RE, W16_3_IM,
            );
            let w16_3_x_odd_3 = complex_mul_avx(w16_3_const, x_odd_3);
            let out3 = _mm256_add_ps(x_even_3, w16_3_x_odd_3);
            let out11 = _mm256_sub_ps(x_even_3, w16_3_x_odd_3);

            let w16_4_x_odd_4 = complex_mul_i_avx(x_odd_4, neg_imag_mask);
            let out4 = _mm256_add_ps(x_even_4, w16_4_x_odd_4);
            let out12 = _mm256_sub_ps(x_even_4, w16_4_x_odd_4);

            let w16_5_const = _mm256_setr_ps(
                -W16_3_RE, W16_3_IM, -W16_3_RE, W16_3_IM, -W16_3_RE, W16_3_IM, -W16_3_RE, W16_3_IM,
            );
            let w16_5_x_odd_5 = complex_mul_avx(w16_5_const, x_odd_5);
            let out5 = _mm256_add_ps(x_even_5, w16_5_x_odd_5);
            let out13 = _mm256_sub_ps(x_even_5, w16_5_x_odd_5);

            let w16_6_x_odd_6 = complex_mul_avx(w8_3, x_odd_6);
            let out6 = _mm256_add_ps(x_even_6, w16_6_x_odd_6);
            let out14 = _mm256_sub_ps(x_even_6, w16_6_x_odd_6);

            let w16_7_const = _mm256_setr_ps(
                -W16_1_RE, W16_1_IM, -W16_1_RE, W16_1_IM, -W16_1_RE, W16_1_IM, -W16_1_RE, W16_1_IM,
            );
            let w16_7_x_odd_7 = complex_mul_avx(w16_7_const, x_odd_7);
            let out7 = _mm256_add_ps(x_even_7, w16_7_x_odd_7);
            let out15 = _mm256_sub_ps(x_even_7, w16_7_x_odd_7);

            // Calculate output indices: j = 16*i - 15*k
            let j0 = 16 * i - 15 * k0;
            let j1 = 16 * (i + 1) - 15 * k1;
            let j2 = 16 * (i + 2) - 15 * k2;
            let j3 = 16 * (i + 3) - 15 * k3;

            // Direct SIMD stores using 64-bit operations.
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);
            let out4_pd = _mm256_castps_pd(out4);
            let out5_pd = _mm256_castps_pd(out5);
            let out6_pd = _mm256_castps_pd(out6);
            let out7_pd = _mm256_castps_pd(out7);
            let out8_pd = _mm256_castps_pd(out8);
            let out9_pd = _mm256_castps_pd(out9);
            let out10_pd = _mm256_castps_pd(out10);
            let out11_pd = _mm256_castps_pd(out11);
            let out12_pd = _mm256_castps_pd(out12);
            let out13_pd = _mm256_castps_pd(out13);
            let out14_pd = _mm256_castps_pd(out14);
            let out15_pd = _mm256_castps_pd(out15);

            // Extract lanes
            let out0_lo = _mm256_castpd256_pd128(out0_pd);
            let out0_hi = _mm256_extractf128_pd(out0_pd, 1);
            let out1_lo = _mm256_castpd256_pd128(out1_pd);
            let out1_hi = _mm256_extractf128_pd(out1_pd, 1);
            let out2_lo = _mm256_castpd256_pd128(out2_pd);
            let out2_hi = _mm256_extractf128_pd(out2_pd, 1);
            let out3_lo = _mm256_castpd256_pd128(out3_pd);
            let out3_hi = _mm256_extractf128_pd(out3_pd, 1);
            let out4_lo = _mm256_castpd256_pd128(out4_pd);
            let out4_hi = _mm256_extractf128_pd(out4_pd, 1);
            let out5_lo = _mm256_castpd256_pd128(out5_pd);
            let out5_hi = _mm256_extractf128_pd(out5_pd, 1);
            let out6_lo = _mm256_castpd256_pd128(out6_pd);
            let out6_hi = _mm256_extractf128_pd(out6_pd, 1);
            let out7_lo = _mm256_castpd256_pd128(out7_pd);
            let out7_hi = _mm256_extractf128_pd(out7_pd, 1);
            let out8_lo = _mm256_castpd256_pd128(out8_pd);
            let out8_hi = _mm256_extractf128_pd(out8_pd, 1);
            let out9_lo = _mm256_castpd256_pd128(out9_pd);
            let out9_hi = _mm256_extractf128_pd(out9_pd, 1);
            let out10_lo = _mm256_castpd256_pd128(out10_pd);
            let out10_hi = _mm256_extractf128_pd(out10_pd, 1);
            let out11_lo = _mm256_castpd256_pd128(out11_pd);
            let out11_hi = _mm256_extractf128_pd(out11_pd, 1);
            let out12_lo = _mm256_castpd256_pd128(out12_pd);
            let out12_hi = _mm256_extractf128_pd(out12_pd, 1);
            let out13_lo = _mm256_castpd256_pd128(out13_pd);
            let out13_hi = _mm256_extractf128_pd(out13_pd, 1);
            let out14_lo = _mm256_castpd256_pd128(out14_pd);
            let out14_hi = _mm256_extractf128_pd(out14_pd, 1);
            let out15_lo = _mm256_castpd256_pd128(out15_pd);
            let out15_hi = _mm256_extractf128_pd(out15_pd, 1);

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            // Store results for all 16 outputs across 4 lanes
            _mm_storel_pd(dst_ptr.add(j0), out0_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 3), out3_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 4), out4_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 5), out5_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 6), out6_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 7), out7_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 8), out8_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 9), out9_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 10), out10_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 11), out11_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 12), out12_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 13), out13_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 14), out14_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 15), out15_lo);

            _mm_storeh_pd(dst_ptr.add(j1), out0_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 3), out3_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 4), out4_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 5), out5_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 6), out6_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 7), out7_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 8), out8_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 9), out9_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 10), out10_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 11), out11_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 12), out12_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 13), out13_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 14), out14_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 15), out15_lo);

            _mm_storel_pd(dst_ptr.add(j2), out0_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride), out1_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 2), out2_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 3), out3_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 4), out4_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 5), out5_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 6), out6_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 7), out7_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 8), out8_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 9), out9_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 10), out10_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 11), out11_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 12), out12_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 13), out13_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 14), out14_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 15), out15_hi);

            _mm_storeh_pd(dst_ptr.add(j3), out0_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride), out1_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 2), out2_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 3), out3_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 4), out4_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 5), out5_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 6), out6_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 7), out7_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 8), out8_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 9), out9_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 10), out10_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 11), out11_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 12), out12_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 13), out13_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 14), out14_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 15), out15_hi);
        }
    }

    let tail_offset = (simd_iters / 4) * (15 * 4);
    for i in simd_iters..sixteenth_samples {
        let k = i % stride;
        let tail_idx = i - simd_iters;

        // Load twiddles for this iteration (interleaved format).
        let w1 = stage_twiddles[tail_offset + tail_idx * 15];
        let w2 = stage_twiddles[tail_offset + tail_idx * 15 + 1];
        let w3 = stage_twiddles[tail_offset + tail_idx * 15 + 2];
        let w4 = stage_twiddles[tail_offset + tail_idx * 15 + 3];
        let w5 = stage_twiddles[tail_offset + tail_idx * 15 + 4];
        let w6 = stage_twiddles[tail_offset + tail_idx * 15 + 5];
        let w7 = stage_twiddles[tail_offset + tail_idx * 15 + 6];
        let w8 = stage_twiddles[tail_offset + tail_idx * 15 + 7];
        let w9 = stage_twiddles[tail_offset + tail_idx * 15 + 8];
        let w10 = stage_twiddles[tail_offset + tail_idx * 15 + 9];
        let w11 = stage_twiddles[tail_offset + tail_idx * 15 + 10];
        let w12 = stage_twiddles[tail_offset + tail_idx * 15 + 11];
        let w13 = stage_twiddles[tail_offset + tail_idx * 15 + 12];
        let w14 = stage_twiddles[tail_offset + tail_idx * 15 + 13];
        let w15 = stage_twiddles[tail_offset + tail_idx * 15 + 14];

        // Load 16 input values.
        let z0 = src[i];
        let z1 = src[i + sixteenth_samples];
        let z2 = src[i + sixteenth_samples * 2];
        let z3 = src[i + sixteenth_samples * 3];
        let z4 = src[i + sixteenth_samples * 4];
        let z5 = src[i + sixteenth_samples * 5];
        let z6 = src[i + sixteenth_samples * 6];
        let z7 = src[i + sixteenth_samples * 7];
        let z8 = src[i + sixteenth_samples * 8];
        let z9 = src[i + sixteenth_samples * 9];
        let z10 = src[i + sixteenth_samples * 10];
        let z11 = src[i + sixteenth_samples * 11];
        let z12 = src[i + sixteenth_samples * 12];
        let z13 = src[i + sixteenth_samples * 13];
        let z14 = src[i + sixteenth_samples * 14];
        let z15 = src[i + sixteenth_samples * 15];

        // Apply twiddles.
        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);
        let t5 = w5.mul(&z5);
        let t6 = w6.mul(&z6);
        let t7 = w7.mul(&z7);
        let t8 = w8.mul(&z8);
        let t9 = w9.mul(&z9);
        let t10 = w10.mul(&z10);
        let t11 = w11.mul(&z11);
        let t12 = w12.mul(&z12);
        let t13 = w13.mul(&z13);
        let t14 = w14.mul(&z14);
        let t15 = w15.mul(&z15);

        // Process EVEN group using scalar split-radix.
        let z0_val = z0;
        let z2_val = t2;
        let z4_val = t4;
        let z6_val = t6;
        let z8_val = t8;
        let z10_val = t10;
        let z12_val = t12;
        let z14_val = t14;

        // Radix-4 on even-even.
        let ee_a0 = z0_val.add(&z8_val);
        let ee_a1 = z0_val.sub(&z8_val);
        let ee_a2 = z4_val.add(&z12_val);
        let ee_a3_re = z4_val.im - z12_val.im;
        let ee_a3_im = z12_val.re - z4_val.re;

        let x_ee_0 = ee_a0.add(&ee_a2);
        let x_ee_2 = ee_a0.sub(&ee_a2);
        let x_ee_1 = Complex32::new(ee_a1.re + ee_a3_re, ee_a1.im + ee_a3_im);
        let x_ee_3 = Complex32::new(ee_a1.re - ee_a3_re, ee_a1.im - ee_a3_im);

        // Radix-4 on even-odd.
        let eo_a0 = z2_val.add(&z10_val);
        let eo_a1 = z2_val.sub(&z10_val);
        let eo_a2 = z6_val.add(&z14_val);
        let eo_a3_re = z6_val.im - z14_val.im;
        let eo_a3_im = z14_val.re - z6_val.re;

        let x_eo_0 = eo_a0.add(&eo_a2);
        let x_eo_2 = eo_a0.sub(&eo_a2);
        let x_eo_1 = Complex32::new(eo_a1.re + eo_a3_re, eo_a1.im + eo_a3_im);
        let x_eo_3 = Complex32::new(eo_a1.re - eo_a3_re, eo_a1.im - eo_a3_im);

        // Combine with W_8 twiddles to get X_even[0..7]
        let x_even = [
            x_ee_0.add(&x_eo_0),
            Complex32::new(
                x_ee_1.re + FRAC_1_SQRT_2 * (x_eo_1.re + x_eo_1.im),
                x_ee_1.im + FRAC_1_SQRT_2 * (x_eo_1.im - x_eo_1.re),
            ),
            x_ee_2.add(&Complex32::new(x_eo_2.im, -x_eo_2.re)),
            Complex32::new(
                x_ee_3.re + FRAC_1_SQRT_2 * (x_eo_3.im - x_eo_3.re),
                x_ee_3.im - FRAC_1_SQRT_2 * (x_eo_3.re + x_eo_3.im),
            ),
            x_ee_0.sub(&x_eo_0),
            Complex32::new(
                x_ee_1.re - FRAC_1_SQRT_2 * (x_eo_1.re + x_eo_1.im),
                x_ee_1.im - FRAC_1_SQRT_2 * (x_eo_1.im - x_eo_1.re),
            ),
            x_ee_2.sub(&Complex32::new(x_eo_2.im, -x_eo_2.re)),
            Complex32::new(
                x_ee_3.re - FRAC_1_SQRT_2 * (x_eo_3.im - x_eo_3.re),
                x_ee_3.im + FRAC_1_SQRT_2 * (x_eo_3.re + x_eo_3.im),
            ),
        ];

        // Process ODD group.
        let z1_val = t1;
        let z3_val = t3;
        let z5_val = t5;
        let z7_val = t7;
        let z9_val = t9;
        let z11_val = t11;
        let z13_val = t13;
        let z15_val = t15;

        // Radix-4 on odd-even.
        let oe_a0 = z1_val.add(&z9_val);
        let oe_a1 = z1_val.sub(&z9_val);
        let oe_a2 = z5_val.add(&z13_val);
        let oe_a3_re = z5_val.im - z13_val.im;
        let oe_a3_im = z13_val.re - z5_val.re;

        let x_oe_0 = oe_a0.add(&oe_a2);
        let x_oe_2 = oe_a0.sub(&oe_a2);
        let x_oe_1 = Complex32::new(oe_a1.re + oe_a3_re, oe_a1.im + oe_a3_im);
        let x_oe_3 = Complex32::new(oe_a1.re - oe_a3_re, oe_a1.im - oe_a3_im);

        // Radix-4 on odd-odd.
        let oo_a0 = z3_val.add(&z11_val);
        let oo_a1 = z3_val.sub(&z11_val);
        let oo_a2 = z7_val.add(&z15_val);
        let oo_a3_re = z7_val.im - z15_val.im;
        let oo_a3_im = z15_val.re - z7_val.re;

        let x_oo_0 = oo_a0.add(&oo_a2);
        let x_oo_2 = oo_a0.sub(&oo_a2);
        let x_oo_1 = Complex32::new(oo_a1.re + oo_a3_re, oo_a1.im + oo_a3_im);
        let x_oo_3 = Complex32::new(oo_a1.re - oo_a3_re, oo_a1.im - oo_a3_im);

        // Combine with W_8 twiddles to get X_odd[0..7]
        let x_odd = [
            x_oe_0.add(&x_oo_0),
            Complex32::new(
                x_oe_1.re + FRAC_1_SQRT_2 * (x_oo_1.re + x_oo_1.im),
                x_oe_1.im + FRAC_1_SQRT_2 * (x_oo_1.im - x_oo_1.re),
            ),
            x_oe_2.add(&Complex32::new(x_oo_2.im, -x_oo_2.re)),
            Complex32::new(
                x_oe_3.re + FRAC_1_SQRT_2 * (x_oo_3.im - x_oo_3.re),
                x_oe_3.im - FRAC_1_SQRT_2 * (x_oo_3.re + x_oo_3.im),
            ),
            x_oe_0.sub(&x_oo_0),
            Complex32::new(
                x_oe_1.re - FRAC_1_SQRT_2 * (x_oo_1.re + x_oo_1.im),
                x_oe_1.im - FRAC_1_SQRT_2 * (x_oo_1.im - x_oo_1.re),
            ),
            x_oe_2.sub(&Complex32::new(x_oo_2.im, -x_oo_2.re)),
            Complex32::new(
                x_oe_3.re - FRAC_1_SQRT_2 * (x_oo_3.im - x_oo_3.re),
                x_oe_3.im + FRAC_1_SQRT_2 * (x_oo_3.re + x_oo_3.im),
            ),
        ];

        // Final combining with W_16 twiddles.
        let j = 16 * i - 15 * k;

        let w16_0_odd = x_odd[0];
        dst[j] = x_even[0].add(&w16_0_odd);
        dst[j + stride * 8] = x_even[0].sub(&w16_0_odd);

        let w16_1_odd = Complex32::new(
            W16_1_RE * x_odd[1].re - W16_1_IM * x_odd[1].im,
            W16_1_RE * x_odd[1].im + W16_1_IM * x_odd[1].re,
        );
        dst[j + stride] = x_even[1].add(&w16_1_odd);
        dst[j + stride * 9] = x_even[1].sub(&w16_1_odd);

        let w16_2_odd = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd[2].re + x_odd[2].im),
            FRAC_1_SQRT_2 * (x_odd[2].im - x_odd[2].re),
        );
        dst[j + stride * 2] = x_even[2].add(&w16_2_odd);
        dst[j + stride * 10] = x_even[2].sub(&w16_2_odd);

        let w16_3_odd = Complex32::new(
            W16_3_RE * x_odd[3].re - W16_3_IM * x_odd[3].im,
            W16_3_RE * x_odd[3].im + W16_3_IM * x_odd[3].re,
        );
        dst[j + stride * 3] = x_even[3].add(&w16_3_odd);
        dst[j + stride * 11] = x_even[3].sub(&w16_3_odd);

        let w16_4_odd = Complex32::new(x_odd[4].im, -x_odd[4].re);
        dst[j + stride * 4] = x_even[4].add(&w16_4_odd);
        dst[j + stride * 12] = x_even[4].sub(&w16_4_odd);

        let w16_5_odd = Complex32::new(
            -W16_3_RE * x_odd[5].re - W16_3_IM * x_odd[5].im,
            -W16_3_RE * x_odd[5].im + W16_3_IM * x_odd[5].re,
        );
        dst[j + stride * 5] = x_even[5].add(&w16_5_odd);
        dst[j + stride * 13] = x_even[5].sub(&w16_5_odd);

        let w16_6_odd = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd[6].im - x_odd[6].re),
            -FRAC_1_SQRT_2 * (x_odd[6].re + x_odd[6].im),
        );
        dst[j + stride * 6] = x_even[6].add(&w16_6_odd);
        dst[j + stride * 14] = x_even[6].sub(&w16_6_odd);

        let w16_7_odd = Complex32::new(
            -W16_1_RE * x_odd[7].re - W16_1_IM * x_odd[7].im,
            -W16_1_RE * x_odd[7].im + W16_1_IM * x_odd[7].re,
        );
        dst[j + stride * 7] = x_even[7].add(&w16_7_odd);
        dst[j + stride * 15] = x_even[7].sub(&w16_7_odd);
    }
}
