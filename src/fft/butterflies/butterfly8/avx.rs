use core::{arch::x86_64::*, f32::consts::FRAC_1_SQRT_2};

use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_avx, complex_mul_i_avx, load_neg_imag_mask_avx},
};

/// Performs a single radix-8 Stockham butterfly stage for stride=1 (out-of-place, AVX+FMA).
///
/// This is a specialized version for the stride=1 case (first stage) that uses the unpack-permute
/// pattern to enable sequential SIMD stores. For stride=1, output indices are sequential: j=8*i.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix8_stride1_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;
    let simd_iters = (eighth_samples >> 2) << 2;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_avx();
        let frac_1_sqrt2_scalar = FRAC_1_SQRT_2;

        for i in (0..simd_iters).step_by(4) {
            // Load 8 input values from the 8 "eighths" of the input array.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + eighth_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + eighth_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + eighth_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + eighth_samples * 4) as *const f32;
            let z4 = _mm256_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + eighth_samples * 5) as *const f32;
            let z5 = _mm256_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + eighth_samples * 6) as *const f32;
            let z6 = _mm256_loadu_ps(z6_ptr);

            let z7_ptr = src.as_ptr().add(i + eighth_samples * 7) as *const f32;
            let z7 = _mm256_loadu_ps(z7_ptr);

            // Load prepackaged twiddles in packed format (7 twiddles per group of 4 lanes).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 7) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]
            let w3 = _mm256_loadu_ps(tw_ptr.add(16)); // w3[i..i+4]
            let w4 = _mm256_loadu_ps(tw_ptr.add(24)); // w4[i..i+4]
            let w5 = _mm256_loadu_ps(tw_ptr.add(32)); // w5[i..i+4]
            let w6 = _mm256_loadu_ps(tw_ptr.add(40)); // w6[i..i+4]
            let w7 = _mm256_loadu_ps(tw_ptr.add(48)); // w7[i..i+4]

            // Apply twiddle factors: t1-t7 = w1-w7 * z1-z7
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);
            let t3 = complex_mul_avx(w3, z3);
            let t4 = complex_mul_avx(w4, z4);
            let t5 = complex_mul_avx(w5, z5);
            let t6 = complex_mul_avx(w6, z6);
            let t7 = complex_mul_avx(w7, z7);

            // Split-radix decomposition: compute even radix-4 on (z0, t2, t4, t6)
            let even_a0 = _mm256_add_ps(z0, t4);
            let even_a1 = _mm256_sub_ps(z0, t4);
            let even_a2 = _mm256_add_ps(t2, t6);
            // even_a3 = i * (t2 - t6): (real, imag) -> (imag, -real)
            let t2_sub_t6 = _mm256_sub_ps(t2, t6);
            let even_a3 = complex_mul_i_avx(t2_sub_t6, neg_imag_mask);

            let x_even_0 = _mm256_add_ps(even_a0, even_a2);
            let x_even_2 = _mm256_sub_ps(even_a0, even_a2);
            let x_even_1 = _mm256_add_ps(even_a1, even_a3);
            let x_even_3 = _mm256_sub_ps(even_a1, even_a3);

            // Compute odd radix-4 on (t1, t3, t5, t7).
            let odd_a0 = _mm256_add_ps(t1, t5);
            let odd_a1 = _mm256_sub_ps(t1, t5);
            let odd_a2 = _mm256_add_ps(t3, t7);
            let t3_sub_t7 = _mm256_sub_ps(t3, t7);
            let odd_a3 = complex_mul_i_avx(t3_sub_t7, neg_imag_mask);

            let x_odd_0 = _mm256_add_ps(odd_a0, odd_a2);
            let x_odd_2 = _mm256_sub_ps(odd_a0, odd_a2);
            let x_odd_1 = _mm256_add_ps(odd_a1, odd_a3);
            let x_odd_3 = _mm256_sub_ps(odd_a1, odd_a3);

            // Apply W_8 twiddles and combine even/odd results.
            // W_8^0 = 1, W_8^1 = (1-i)/√2, W_8^2 = -i, W_8^3 = (-1-i)/√2

            // out0 = x_even_0 + x_odd_0
            let out0 = _mm256_add_ps(x_even_0, x_odd_0);
            // out4 = x_even_0 - x_odd_0
            let out4 = _mm256_sub_ps(x_even_0, x_odd_0);

            // out1 = x_even_1 + W_8^1 * x_odd_1
            // W_8^1 = (1-i)/√2 as interleaved complex: [re, im, re, im, ...] = [1/√2, -1/√2, ...]
            let w8_1 = _mm256_setr_ps(
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
            );
            let w8_1_x_odd_1 = complex_mul_avx(w8_1, x_odd_1);
            let out1 = _mm256_add_ps(x_even_1, w8_1_x_odd_1);
            // out5 = x_even_1 - W_8^1 * x_odd_1
            let out5 = _mm256_sub_ps(x_even_1, w8_1_x_odd_1);

            // out2 = x_even_2 + W_8^2 * x_odd_2
            // W_8^2 = -i = (0, -1), so multiply by -i: (a+bi)*(-i) = (b, -a)
            let w8_2_x_odd_2 = complex_mul_i_avx(x_odd_2, neg_imag_mask);
            let out2 = _mm256_add_ps(x_even_2, w8_2_x_odd_2);
            // out6 = x_even_2 - W_8^2 * x_odd_2
            let out6 = _mm256_sub_ps(x_even_2, w8_2_x_odd_2);

            // out3 = x_even_3 + W_8^3 * x_odd_3
            // W_8^3 = (-1-i)/√2 as interleaved complex: [re, im, re, im, ...] = [-1/√2, -1/√2, ...]
            let w8_3 = _mm256_setr_ps(
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
            );
            let w8_3_x_odd_3 = complex_mul_avx(w8_3, x_odd_3);
            let out3 = _mm256_add_ps(x_even_3, w8_3_x_odd_3);
            // out7 = x_even_3 - W_8^3 * x_odd_3
            let out7 = _mm256_sub_ps(x_even_3, w8_3_x_odd_3);

            // Interleave 8 outputs for sequential storage: [out0[0], out1[0], ..., out7[0], out0[1], ...]
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);
            let out4_pd = _mm256_castps_pd(out4);
            let out5_pd = _mm256_castps_pd(out5);
            let out6_pd = _mm256_castps_pd(out6);
            let out7_pd = _mm256_castps_pd(out7);

            // Extract 128-bit lanes.
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

            // Build 8 result vectors by interleaving pairs from the low and high lanes.
            // result0: [out0[0], out1[0], out2[0], out3[0]]
            let temp0_lo = _mm_unpacklo_pd(out0_lo, out1_lo);
            let temp0_hi = _mm_unpacklo_pd(out2_lo, out3_lo);
            let result0 = _mm256_castpd_ps(_mm256_set_m128d(temp0_hi, temp0_lo));

            // result1: [out4[0], out5[0], out6[0], out7[0]]
            let temp1_lo = _mm_unpacklo_pd(out4_lo, out5_lo);
            let temp1_hi = _mm_unpacklo_pd(out6_lo, out7_lo);
            let result1 = _mm256_castpd_ps(_mm256_set_m128d(temp1_hi, temp1_lo));

            // result2: [out0[1], out1[1], out2[1], out3[1]]
            let temp2_lo = _mm_unpackhi_pd(out0_lo, out1_lo);
            let temp2_hi = _mm_unpackhi_pd(out2_lo, out3_lo);
            let result2 = _mm256_castpd_ps(_mm256_set_m128d(temp2_hi, temp2_lo));

            // result3: [out4[1], out5[1], out6[1], out7[1]]
            let temp3_lo = _mm_unpackhi_pd(out4_lo, out5_lo);
            let temp3_hi = _mm_unpackhi_pd(out6_lo, out7_lo);
            let result3 = _mm256_castpd_ps(_mm256_set_m128d(temp3_hi, temp3_lo));

            // result4: [out0[2], out1[2], out2[2], out3[2]]
            let temp4_lo = _mm_unpacklo_pd(out0_hi, out1_hi);
            let temp4_hi = _mm_unpacklo_pd(out2_hi, out3_hi);
            let result4 = _mm256_castpd_ps(_mm256_set_m128d(temp4_hi, temp4_lo));

            // result5: [out4[2], out5[2], out6[2], out7[2]]
            let temp5_lo = _mm_unpacklo_pd(out4_hi, out5_hi);
            let temp5_hi = _mm_unpacklo_pd(out6_hi, out7_hi);
            let result5 = _mm256_castpd_ps(_mm256_set_m128d(temp5_hi, temp5_lo));

            // result6: [out0[3], out1[3], out2[3], out3[3]]
            let temp6_lo = _mm_unpackhi_pd(out0_hi, out1_hi);
            let temp6_hi = _mm_unpackhi_pd(out2_hi, out3_hi);
            let result6 = _mm256_castpd_ps(_mm256_set_m128d(temp6_hi, temp6_lo));

            // result7: [out4[3], out5[3], out6[3], out7[3]]
            let temp7_lo = _mm_unpackhi_pd(out4_hi, out5_hi);
            let temp7_hi = _mm_unpackhi_pd(out6_hi, out7_hi);
            let result7 = _mm256_castpd_ps(_mm256_set_m128d(temp7_hi, temp7_lo));

            // Sequential stores.
            let j = 8 * i;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            _mm256_storeu_ps(dst_ptr, result0);
            _mm256_storeu_ps(dst_ptr.add(8), result1);
            _mm256_storeu_ps(dst_ptr.add(16), result2);
            _mm256_storeu_ps(dst_ptr.add(24), result3);
            _mm256_storeu_ps(dst_ptr.add(32), result4);
            _mm256_storeu_ps(dst_ptr.add(40), result5);
            _mm256_storeu_ps(dst_ptr.add(48), result6);
            _mm256_storeu_ps(dst_ptr.add(56), result7);
        }
    }

    for i in simd_iters..eighth_samples {
        let w1 = stage_twiddles[i * 7];
        let w2 = stage_twiddles[i * 7 + 1];
        let w3 = stage_twiddles[i * 7 + 2];
        let w4 = stage_twiddles[i * 7 + 3];
        let w5 = stage_twiddles[i * 7 + 4];
        let w6 = stage_twiddles[i * 7 + 5];
        let w7 = stage_twiddles[i * 7 + 6];

        let z0 = src[i];
        let z1 = src[i + eighth_samples];
        let z2 = src[i + eighth_samples * 2];
        let z3 = src[i + eighth_samples * 3];
        let z4 = src[i + eighth_samples * 4];
        let z5 = src[i + eighth_samples * 5];
        let z6 = src[i + eighth_samples * 6];
        let z7 = src[i + eighth_samples * 7];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);
        let t5 = w5.mul(&z5);
        let t6 = w6.mul(&z6);
        let t7 = w7.mul(&z7);

        let even_a0 = z0.add(&t4);
        let even_a1 = z0.sub(&t4);
        let even_a2 = t2.add(&t6);
        let even_a3_re = t2.im - t6.im;
        let even_a3_im = t6.re - t2.re;

        let x_even_0 = even_a0.add(&even_a2);
        let x_even_2 = even_a0.sub(&even_a2);
        let x_even_1 = Complex32::new(even_a1.re + even_a3_re, even_a1.im + even_a3_im);
        let x_even_3 = Complex32::new(even_a1.re - even_a3_re, even_a1.im - even_a3_im);

        let odd_a0 = t1.add(&t5);
        let odd_a1 = t1.sub(&t5);
        let odd_a2 = t3.add(&t7);
        let odd_a3_re = t3.im - t7.im;
        let odd_a3_im = t7.re - t3.re;

        let x_odd_0 = odd_a0.add(&odd_a2);
        let x_odd_2 = odd_a0.sub(&odd_a2);
        let x_odd_1 = Complex32::new(odd_a1.re + odd_a3_re, odd_a1.im + odd_a3_im);
        let x_odd_3 = Complex32::new(odd_a1.re - odd_a3_re, odd_a1.im - odd_a3_im);

        let j = 8 * i;
        dst[j] = x_even_0.add(&x_odd_0);
        dst[j + 4] = x_even_0.sub(&x_odd_0);

        let w8_1_odd_1 = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd_1.re + x_odd_1.im),
            FRAC_1_SQRT_2 * (x_odd_1.im - x_odd_1.re),
        );
        dst[j + 1] = x_even_1.add(&w8_1_odd_1);
        dst[j + 5] = x_even_1.sub(&w8_1_odd_1);

        let w8_2_odd_2 = Complex32::new(x_odd_2.im, -x_odd_2.re);
        dst[j + 2] = x_even_2.add(&w8_2_odd_2);
        dst[j + 6] = x_even_2.sub(&w8_2_odd_2);

        let w8_3_odd_3 = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd_3.im - x_odd_3.re),
            -FRAC_1_SQRT_2 * (x_odd_3.re + x_odd_3.im),
        );
        dst[j + 3] = x_even_3.add(&w8_3_odd_3);
        dst[j + 7] = x_even_3.sub(&w8_3_odd_3);
    }
}

/// Performs a single radix-8 Stockham butterfly stage for p>1 (out-of-place, AVX+FMA).
///
/// Generic version for p>1 cases. Uses direct SIMD stores, accepting non-sequential
/// stores as the shuffle overhead isn't justified for larger strides.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix8_generic_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    // We convince the compiler here that stride can't be 0 to optimize better.
    if stride == 0 {
        return;
    }

    let samples = src.len();
    let eighth_samples = samples >> 3;
    let simd_iters = (eighth_samples >> 2) << 2;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_avx();
        let frac_1_sqrt2_scalar = FRAC_1_SQRT_2;

        for i in (0..simd_iters).step_by(4) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;
            let k2 = k + 2 - ((k + 2 >= stride) as usize) * stride;
            let k3 = k + 3 - ((k + 3 >= stride) as usize) * stride;

            // Load 8 input values from the 8 "eighths" of the input array.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + eighth_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + eighth_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + eighth_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + eighth_samples * 4) as *const f32;
            let z4 = _mm256_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + eighth_samples * 5) as *const f32;
            let z5 = _mm256_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + eighth_samples * 6) as *const f32;
            let z6 = _mm256_loadu_ps(z6_ptr);

            let z7_ptr = src.as_ptr().add(i + eighth_samples * 7) as *const f32;
            let z7 = _mm256_loadu_ps(z7_ptr);

            // Load prepackaged twiddles in packed format.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 7) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]
            let w3 = _mm256_loadu_ps(tw_ptr.add(16)); // w3[i..i+4]
            let w4 = _mm256_loadu_ps(tw_ptr.add(24)); // w4[i..i+4]
            let w5 = _mm256_loadu_ps(tw_ptr.add(32)); // w5[i..i+4]
            let w6 = _mm256_loadu_ps(tw_ptr.add(40)); // w6[i..i+4]
            let w7 = _mm256_loadu_ps(tw_ptr.add(48)); // w7[i..i+4]

            // Apply twiddle factors
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);
            let t3 = complex_mul_avx(w3, z3);
            let t4 = complex_mul_avx(w4, z4);
            let t5 = complex_mul_avx(w5, z5);
            let t6 = complex_mul_avx(w6, z6);
            let t7 = complex_mul_avx(w7, z7);

            // Split-radix decomposition: compute even radix-4 on (z0, t2, t4, t6)
            let even_a0 = _mm256_add_ps(z0, t4);
            let even_a1 = _mm256_sub_ps(z0, t4);
            let even_a2 = _mm256_add_ps(t2, t6);
            let t2_sub_t6 = _mm256_sub_ps(t2, t6);
            let even_a3 = complex_mul_i_avx(t2_sub_t6, neg_imag_mask);

            let x_even_0 = _mm256_add_ps(even_a0, even_a2);
            let x_even_2 = _mm256_sub_ps(even_a0, even_a2);
            let x_even_1 = _mm256_add_ps(even_a1, even_a3);
            let x_even_3 = _mm256_sub_ps(even_a1, even_a3);

            // Compute odd radix-4 on (t1, t3, t5, t7)
            let odd_a0 = _mm256_add_ps(t1, t5);
            let odd_a1 = _mm256_sub_ps(t1, t5);
            let odd_a2 = _mm256_add_ps(t3, t7);
            let t3_sub_t7 = _mm256_sub_ps(t3, t7);
            let odd_a3 = complex_mul_i_avx(t3_sub_t7, neg_imag_mask);

            let x_odd_0 = _mm256_add_ps(odd_a0, odd_a2);
            let x_odd_2 = _mm256_sub_ps(odd_a0, odd_a2);
            let x_odd_1 = _mm256_add_ps(odd_a1, odd_a3);
            let x_odd_3 = _mm256_sub_ps(odd_a1, odd_a3);

            // Apply W_8 twiddles and combine
            let out0 = _mm256_add_ps(x_even_0, x_odd_0);
            let out4 = _mm256_sub_ps(x_even_0, x_odd_0);

            // W_8^1 = (1-i)/√2 as interleaved complex: [1/√2, -1/√2, ...]
            let w8_1 = _mm256_setr_ps(
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
            );
            let w8_1_x_odd_1 = complex_mul_avx(w8_1, x_odd_1);
            let out1 = _mm256_add_ps(x_even_1, w8_1_x_odd_1);
            let out5 = _mm256_sub_ps(x_even_1, w8_1_x_odd_1);

            // W_8^2 = -i: (a+bi) -> (b, -a)
            let w8_2_x_odd_2 = complex_mul_i_avx(x_odd_2, neg_imag_mask);
            let out2 = _mm256_add_ps(x_even_2, w8_2_x_odd_2);
            let out6 = _mm256_sub_ps(x_even_2, w8_2_x_odd_2);

            // W_8^3 = (-1-i)/√2 as interleaved complex: [-1/√2, -1/√2, ...]
            let w8_3 = _mm256_setr_ps(
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
            );
            let w8_3_x_odd_3 = complex_mul_avx(w8_3, x_odd_3);
            let out3 = _mm256_add_ps(x_even_3, w8_3_x_odd_3);
            let out7 = _mm256_sub_ps(x_even_3, w8_3_x_odd_3);

            // Calculate output indices: j = 8*i - 7*k
            let j0 = 8 * i - 7 * k0;
            let j1 = 8 * (i + 1) - 7 * k1;
            let j2 = 8 * (i + 2) - 7 * k2;
            let j3 = 8 * (i + 3) - 7 * k3;

            // Direct SIMD stores using 64-bit operations.
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);
            let out4_pd = _mm256_castps_pd(out4);
            let out5_pd = _mm256_castps_pd(out5);
            let out6_pd = _mm256_castps_pd(out6);
            let out7_pd = _mm256_castps_pd(out7);

            // Extract 128-bit lanes.
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

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            _mm_storel_pd(dst_ptr.add(j0), out0_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 3), out3_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 4), out4_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 5), out5_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 6), out6_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 7), out7_lo);

            _mm_storeh_pd(dst_ptr.add(j1), out0_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 3), out3_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 4), out4_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 5), out5_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 6), out6_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 7), out7_lo);

            _mm_storel_pd(dst_ptr.add(j2), out0_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride), out1_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 2), out2_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 3), out3_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 4), out4_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 5), out5_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 6), out6_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 7), out7_hi);

            _mm_storeh_pd(dst_ptr.add(j3), out0_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride), out1_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 2), out2_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 3), out3_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 4), out4_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 5), out5_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 6), out6_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 7), out7_hi);
        }
    }

    for i in simd_iters..eighth_samples {
        let k = i % stride;
        let w1 = stage_twiddles[i * 7];
        let w2 = stage_twiddles[i * 7 + 1];
        let w3 = stage_twiddles[i * 7 + 2];
        let w4 = stage_twiddles[i * 7 + 3];
        let w5 = stage_twiddles[i * 7 + 4];
        let w6 = stage_twiddles[i * 7 + 5];
        let w7 = stage_twiddles[i * 7 + 6];

        let z0 = src[i];
        let z1 = src[i + eighth_samples];
        let z2 = src[i + eighth_samples * 2];
        let z3 = src[i + eighth_samples * 3];
        let z4 = src[i + eighth_samples * 4];
        let z5 = src[i + eighth_samples * 5];
        let z6 = src[i + eighth_samples * 6];
        let z7 = src[i + eighth_samples * 7];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);
        let t5 = w5.mul(&z5);
        let t6 = w6.mul(&z6);
        let t7 = w7.mul(&z7);

        let even_a0 = z0.add(&t4);
        let even_a1 = z0.sub(&t4);
        let even_a2 = t2.add(&t6);
        let even_a3_re = t2.im - t6.im;
        let even_a3_im = t6.re - t2.re;

        let x_even_0 = even_a0.add(&even_a2);
        let x_even_2 = even_a0.sub(&even_a2);
        let x_even_1 = Complex32::new(even_a1.re + even_a3_re, even_a1.im + even_a3_im);
        let x_even_3 = Complex32::new(even_a1.re - even_a3_re, even_a1.im - even_a3_im);

        let odd_a0 = t1.add(&t5);
        let odd_a1 = t1.sub(&t5);
        let odd_a2 = t3.add(&t7);
        let odd_a3_re = t3.im - t7.im;
        let odd_a3_im = t7.re - t3.re;

        let x_odd_0 = odd_a0.add(&odd_a2);
        let x_odd_2 = odd_a0.sub(&odd_a2);
        let x_odd_1 = Complex32::new(odd_a1.re + odd_a3_re, odd_a1.im + odd_a3_im);
        let x_odd_3 = Complex32::new(odd_a1.re - odd_a3_re, odd_a1.im - odd_a3_im);

        let j = 8 * i - 7 * k;
        dst[j] = x_even_0.add(&x_odd_0);
        dst[j + stride * 4] = x_even_0.sub(&x_odd_0);

        let w8_1_odd_1 = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd_1.re + x_odd_1.im),
            FRAC_1_SQRT_2 * (x_odd_1.im - x_odd_1.re),
        );
        dst[j + stride] = x_even_1.add(&w8_1_odd_1);
        dst[j + stride * 5] = x_even_1.sub(&w8_1_odd_1);

        let w8_2_odd_2 = Complex32::new(x_odd_2.im, -x_odd_2.re);
        dst[j + stride * 2] = x_even_2.add(&w8_2_odd_2);
        dst[j + stride * 6] = x_even_2.sub(&w8_2_odd_2);

        let w8_3_odd_3 = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd_3.im - x_odd_3.re),
            -FRAC_1_SQRT_2 * (x_odd_3.re + x_odd_3.im),
        );
        dst[j + stride * 3] = x_even_3.add(&w8_3_odd_3);
        dst[j + stride * 7] = x_even_3.sub(&w8_3_odd_3);
    }
}
