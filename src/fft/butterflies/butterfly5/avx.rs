use core::arch::x86_64::*;

use super::{COS_2PI_5, COS_4PI_5, SIN_2PI_5, SIN_4PI_5};
use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_avx, complex_mul_i_avx, load_neg_imag_mask_avx},
};

/// Performs a single radix-5 Stockham butterfly stage (out-of-place, AVX+FMA) for stride=1 (first stage).
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix5_stride1_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let fifth_samples = samples / 5;
    let simd_iters = (fifth_samples >> 2) << 2;

    unsafe {
        let cos_2pi_5_vec = _mm256_set1_ps(COS_2PI_5);
        let sin_2pi_5_vec = _mm256_set1_ps(SIN_2PI_5);
        let cos_4pi_5_vec = _mm256_set1_ps(COS_4PI_5);
        let sin_4pi_5_vec = _mm256_set1_ps(SIN_4PI_5);

        let neg_imag_mask = load_neg_imag_mask_avx();

        for i in (0..simd_iters).step_by(4) {
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + fifth_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + fifth_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + fifth_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + fifth_samples * 4) as *const f32;
            let z4 = _mm256_loadu_ps(z4_ptr);

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 4) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]
            let w3 = _mm256_loadu_ps(tw_ptr.add(16)); // w3[i..i+4]
            let w4 = _mm256_loadu_ps(tw_ptr.add(24)); // w4[i..i+4]

            // Complex multiply: t1 = w1 * z1, t2 = w2 * z2, t3 = w3 * z3, t4 = w4 * z4
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);
            let t3 = complex_mul_avx(w3, z3);
            let t4 = complex_mul_avx(w4, z4);

            // Radix-5 DFT.
            let sum_all = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(t1, t2), t3), t4);

            let a1 = _mm256_add_ps(t1, t4);
            let a2 = _mm256_add_ps(t2, t3);

            // b1 = i * (t1 - t4), b2 = i * (t2 - t3)
            let t1_sub_t4 = _mm256_sub_ps(t1, t4);
            let t2_sub_t3 = _mm256_sub_ps(t2, t3);
            let b1 = complex_mul_i_avx(t1_sub_t4, neg_imag_mask);
            let b2 = complex_mul_i_avx(t2_sub_t3, neg_imag_mask);

            // c1 = z0 + COS_2PI_5 * a1 + COS_4PI_5 * a2
            let c1 = _mm256_add_ps(
                z0,
                _mm256_add_ps(
                    _mm256_mul_ps(cos_2pi_5_vec, a1),
                    _mm256_mul_ps(cos_4pi_5_vec, a2),
                ),
            );

            // c2 = z0 + COS_4PI_5 * a1 + COS_2PI_5 * a2
            let c2 = _mm256_add_ps(
                z0,
                _mm256_add_ps(
                    _mm256_mul_ps(cos_4pi_5_vec, a1),
                    _mm256_mul_ps(cos_2pi_5_vec, a2),
                ),
            );

            // d1 = SIN_2PI_5 * b1 + SIN_4PI_5 * b2
            let d1 = _mm256_add_ps(
                _mm256_mul_ps(sin_2pi_5_vec, b1),
                _mm256_mul_ps(sin_4pi_5_vec, b2),
            );

            // d2 = SIN_4PI_5 * b1 - SIN_2PI_5 * b2
            let d2 = _mm256_sub_ps(
                _mm256_mul_ps(sin_4pi_5_vec, b1),
                _mm256_mul_ps(sin_2pi_5_vec, b2),
            );

            // Final outputs.
            let out0 = _mm256_add_ps(z0, sum_all);
            let out1 = _mm256_add_ps(c1, d1);
            let out4 = _mm256_sub_ps(c1, d1);
            let out2 = _mm256_add_ps(c2, d2);
            let out3 = _mm256_sub_ps(c2, d2);

            // Interleave for sequential stores: [out0[0], out1[0], out2[0], out3[0], out4[0], ...]
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);
            let out4_pd = _mm256_castps_pd(out4);

            // Extract 128-bit lanes (each containing 2 complex numbers).
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

            // Build result0: [out0[0], out1[0], out2[0], out3[0]]
            let temp0_lo = _mm_unpacklo_pd(out0_lo, out1_lo); // [out0[0], out1[0]]
            let temp0_hi = _mm_unpacklo_pd(out2_lo, out3_lo); // [out2[0], out3[0]]
            let result0 = _mm256_castpd_ps(_mm256_set_m128d(temp0_hi, temp0_lo));

            // Build result1: [out4[0], out0[1], out1[1], out2[1]]
            let temp1_lo = _mm_shuffle_pd::<2>(out4_lo, out0_lo); // [out4[0], out0[1]]
            let temp1_hi = _mm_unpackhi_pd(out1_lo, out2_lo); // [out1[1], out2[1]]
            let result1 = _mm256_castpd_ps(_mm256_set_m128d(temp1_hi, temp1_lo));

            // Build result2: [out3[1], out4[1], out0[2], out1[2]]
            let temp2_lo = _mm_unpackhi_pd(out3_lo, out4_lo); // [out3[1], out4[1]]
            let temp2_hi = _mm_unpacklo_pd(out0_hi, out1_hi); // [out0[2], out1[2]]
            let result2 = _mm256_castpd_ps(_mm256_set_m128d(temp2_hi, temp2_lo));

            // Build result3: [out2[2], out3[2], out4[2], out0[3]]
            let temp3_lo = _mm_unpacklo_pd(out2_hi, out3_hi); // [out2[2], out3[2]]
            let temp3_hi = _mm_shuffle_pd::<2>(out4_hi, out0_hi); // [out4[2], out0[3]]
            let result3 = _mm256_castpd_ps(_mm256_set_m128d(temp3_hi, temp3_lo));

            // Build result4: [out1[3], out2[3], out3[3], out4[3]]
            let temp4_lo = _mm_unpackhi_pd(out1_hi, out2_hi); // [out1[3], out2[3]]
            let temp4_hi = _mm_unpackhi_pd(out3_hi, out4_hi); // [out3[3], out4[3]]
            let result4 = _mm256_castpd_ps(_mm256_set_m128d(temp4_hi, temp4_lo));

            // Sequential stores.
            let j = 5 * i;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            _mm256_storeu_ps(dst_ptr, result0);
            _mm256_storeu_ps(dst_ptr.add(8), result1);
            _mm256_storeu_ps(dst_ptr.add(16), result2);
            _mm256_storeu_ps(dst_ptr.add(24), result3);
            _mm256_storeu_ps(dst_ptr.add(32), result4);
        }
    }

    for i in simd_iters..fifth_samples {
        let w1 = stage_twiddles[i * 4];
        let w2 = stage_twiddles[i * 4 + 1];
        let w3 = stage_twiddles[i * 4 + 2];
        let w4 = stage_twiddles[i * 4 + 3];

        let z0 = src[i];
        let z1 = src[i + fifth_samples];
        let z2 = src[i + fifth_samples * 2];
        let z3 = src[i + fifth_samples * 3];
        let z4 = src[i + fifth_samples * 4];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);

        let sum_all = t1.add(&t2).add(&t3).add(&t4);

        let a1 = t1.add(&t4);
        let a2 = t2.add(&t3);
        let b1_re = t1.im - t4.im;
        let b1_im = t4.re - t1.re;
        let b2_re = t2.im - t3.im;
        let b2_im = t3.re - t2.re;

        let c1_re = z0.re + COS_2PI_5 * a1.re + COS_4PI_5 * a2.re;
        let c1_im = z0.im + COS_2PI_5 * a1.im + COS_4PI_5 * a2.im;
        let c2_re = z0.re + COS_4PI_5 * a1.re + COS_2PI_5 * a2.re;
        let c2_im = z0.im + COS_4PI_5 * a1.im + COS_2PI_5 * a2.im;

        let d1_re = SIN_2PI_5 * b1_re + SIN_4PI_5 * b2_re;
        let d1_im = SIN_2PI_5 * b1_im + SIN_4PI_5 * b2_im;
        let d2_re = SIN_4PI_5 * b1_re - SIN_2PI_5 * b2_re;
        let d2_im = SIN_4PI_5 * b1_im - SIN_2PI_5 * b2_im;

        let j = 5 * i;
        dst[j] = z0.add(&sum_all);
        dst[j + 1] = Complex32::new(c1_re + d1_re, c1_im + d1_im);
        dst[j + 4] = Complex32::new(c1_re - d1_re, c1_im - d1_im);
        dst[j + 2] = Complex32::new(c2_re + d2_re, c2_im + d2_im);
        dst[j + 3] = Complex32::new(c2_re - d2_re, c2_im - d2_im);
    }
}

/// Performs a single radix-5 Stockham butterfly stage (out-of-place, AVX+FMA) for generic p.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix5_generic_avx_fma(
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
    let fifth_samples = samples / 5;
    let simd_iters = (fifth_samples >> 2) << 2;

    unsafe {
        let cos_2pi_5_vec = _mm256_set1_ps(COS_2PI_5);
        let sin_2pi_5_vec = _mm256_set1_ps(SIN_2PI_5);
        let cos_4pi_5_vec = _mm256_set1_ps(COS_4PI_5);
        let sin_4pi_5_vec = _mm256_set1_ps(SIN_4PI_5);

        let neg_imag_mask = load_neg_imag_mask_avx();

        for i in (0..simd_iters).step_by(4) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;
            let k2 = k + 2 - ((k + 2 >= stride) as usize) * stride;
            let k3 = k + 3 - ((k + 3 >= stride) as usize) * stride;

            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + fifth_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + fifth_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + fifth_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + fifth_samples * 4) as *const f32;
            let z4 = _mm256_loadu_ps(z4_ptr);

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 4) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]
            let w3 = _mm256_loadu_ps(tw_ptr.add(16)); // w3[i..i+4]
            let w4 = _mm256_loadu_ps(tw_ptr.add(24)); // w4[i..i+4]

            // Complex multiply: t1 = w1 * z1, t2 = w2 * z2, t3 = w3 * z3, t4 = w4 * z4
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);
            let t3 = complex_mul_avx(w3, z3);
            let t4 = complex_mul_avx(w4, z4);

            // Radix-5 DFT.
            let sum_all = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(t1, t2), t3), t4);

            let a1 = _mm256_add_ps(t1, t4);
            let a2 = _mm256_add_ps(t2, t3);

            // b1 = i * (t1 - t4), b2 = i * (t2 - t3)
            let t1_sub_t4 = _mm256_sub_ps(t1, t4);
            let t2_sub_t3 = _mm256_sub_ps(t2, t3);
            let b1 = complex_mul_i_avx(t1_sub_t4, neg_imag_mask);
            let b2 = complex_mul_i_avx(t2_sub_t3, neg_imag_mask);

            // c1 = z0 + COS_2PI_5 * a1 + COS_4PI_5 * a2
            let c1 = _mm256_add_ps(
                z0,
                _mm256_add_ps(
                    _mm256_mul_ps(cos_2pi_5_vec, a1),
                    _mm256_mul_ps(cos_4pi_5_vec, a2),
                ),
            );

            // c2 = z0 + COS_4PI_5 * a1 + COS_2PI_5 * a2
            let c2 = _mm256_add_ps(
                z0,
                _mm256_add_ps(
                    _mm256_mul_ps(cos_4pi_5_vec, a1),
                    _mm256_mul_ps(cos_2pi_5_vec, a2),
                ),
            );

            // d1 = SIN_2PI_5 * b1 + SIN_4PI_5 * b2
            let d1 = _mm256_add_ps(
                _mm256_mul_ps(sin_2pi_5_vec, b1),
                _mm256_mul_ps(sin_4pi_5_vec, b2),
            );

            // d2 = SIN_4PI_5 * b1 - SIN_2PI_5 * b2
            let d2 = _mm256_sub_ps(
                _mm256_mul_ps(sin_4pi_5_vec, b1),
                _mm256_mul_ps(sin_2pi_5_vec, b2),
            );

            // Final outputs.
            let out0 = _mm256_add_ps(z0, sum_all);
            let out1 = _mm256_add_ps(c1, d1);
            let out4 = _mm256_sub_ps(c1, d1);
            let out2 = _mm256_add_ps(c2, d2);
            let out3 = _mm256_sub_ps(c2, d2);

            let j0 = 5 * i - 4 * k0;
            let j1 = 5 * (i + 1) - 4 * k1;
            let j2 = 5 * (i + 2) - 4 * k2;
            let j3 = 5 * (i + 3) - 4 * k3;

            // Direct SIMD stores.
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);
            let out4_pd = _mm256_castps_pd(out4);

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

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            _mm_storel_pd(dst_ptr.add(j0), out0_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 3), out3_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 4), out4_lo);

            _mm_storeh_pd(dst_ptr.add(j1), out0_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 3), out3_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 4), out4_lo);

            _mm_storel_pd(dst_ptr.add(j2), out0_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride), out1_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 2), out2_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 3), out3_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 4), out4_hi);

            _mm_storeh_pd(dst_ptr.add(j3), out0_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride), out1_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 2), out2_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 3), out3_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 4), out4_hi);
        }
    }

    for i in simd_iters..fifth_samples {
        let k = i % stride;
        let w1 = stage_twiddles[i * 4];
        let w2 = stage_twiddles[i * 4 + 1];
        let w3 = stage_twiddles[i * 4 + 2];
        let w4 = stage_twiddles[i * 4 + 3];

        let z0 = src[i];
        let z1 = src[i + fifth_samples];
        let z2 = src[i + fifth_samples * 2];
        let z3 = src[i + fifth_samples * 3];
        let z4 = src[i + fifth_samples * 4];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);

        let sum_all = t1.add(&t2).add(&t3).add(&t4);

        let a1 = t1.add(&t4);
        let a2 = t2.add(&t3);
        let b1_re = t1.im - t4.im;
        let b1_im = t4.re - t1.re;
        let b2_re = t2.im - t3.im;
        let b2_im = t3.re - t2.re;

        let c1_re = z0.re + COS_2PI_5 * a1.re + COS_4PI_5 * a2.re;
        let c1_im = z0.im + COS_2PI_5 * a1.im + COS_4PI_5 * a2.im;
        let c2_re = z0.re + COS_4PI_5 * a1.re + COS_2PI_5 * a2.re;
        let c2_im = z0.im + COS_4PI_5 * a1.im + COS_2PI_5 * a2.im;

        let d1_re = SIN_2PI_5 * b1_re + SIN_4PI_5 * b2_re;
        let d1_im = SIN_2PI_5 * b1_im + SIN_4PI_5 * b2_im;
        let d2_re = SIN_4PI_5 * b1_re - SIN_2PI_5 * b2_re;
        let d2_im = SIN_4PI_5 * b1_im - SIN_2PI_5 * b2_im;

        let j = 5 * i - 4 * k;
        dst[j] = z0.add(&sum_all);
        dst[j + stride] = Complex32::new(c1_re + d1_re, c1_im + d1_im);
        dst[j + stride * 4] = Complex32::new(c1_re - d1_re, c1_im - d1_im);
        dst[j + stride * 2] = Complex32::new(c2_re + d2_re, c2_im + d2_im);
        dst[j + stride * 3] = Complex32::new(c2_re - d2_re, c2_im - d2_im);
    }
}
