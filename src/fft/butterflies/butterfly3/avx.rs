use core::arch::x86_64::*;

use super::SQRT3_2;
use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_avx, complex_mul_sqrt3_i_avx},
};

/// Performs a single radix-3 Stockham butterfly stage (out-of-place, AVX+FMA) for stride=1 (first stage).
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix3_stride1_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let third_samples = samples / 3;
    let simd_iters = (third_samples >> 2) << 2;

    unsafe {
        let sqrt3_2_vec = _mm256_set_ps(
            -SQRT3_2, SQRT3_2, -SQRT3_2, SQRT3_2, -SQRT3_2, SQRT3_2, -SQRT3_2, SQRT3_2,
        );

        let half_vec = _mm256_set1_ps(0.5);

        for i in (0..simd_iters).step_by(4) {
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + third_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + third_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 2) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]

            // Complex multiply: t1 = w1 * z1, t2 = w2 * z2
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);

            // sum_t = t1 + t2, diff_t = t1 - t2
            let sum_t = _mm256_add_ps(t1, t2);
            let diff_t = _mm256_sub_ps(t1, t2);

            // out0 = z0 + sum_t
            let out0 = _mm256_add_ps(z0, sum_t);

            // re_part = z0 - 0.5 * sum_t
            let half_sum_t = _mm256_mul_ps(sum_t, half_vec);
            let re_im_part = _mm256_sub_ps(z0, half_sum_t);

            // sqrt3 multiplication
            let sqrt3_diff = complex_mul_sqrt3_i_avx(diff_t, sqrt3_2_vec);

            let out1 = _mm256_add_ps(re_im_part, sqrt3_diff);
            let out2 = _mm256_sub_ps(re_im_part, sqrt3_diff);

            // Interleave for sequential stores: [out0[0], out1[0], out2[0], out0[1], ...]
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);

            // Extract 128-bit lanes (each containing 2 complex numbers).
            let out0_lo = _mm256_castpd256_pd128(out0_pd);
            let out0_hi = _mm256_extractf128_pd(out0_pd, 1);
            let out1_lo = _mm256_castpd256_pd128(out1_pd);
            let out1_hi = _mm256_extractf128_pd(out1_pd, 1);
            let out2_lo = _mm256_castpd256_pd128(out2_pd);
            let out2_hi = _mm256_extractf128_pd(out2_pd, 1);

            // Build result0: [out0[0], out1[0], out2[0], out0[1]]
            let temp0_lo = _mm_unpacklo_pd(out0_lo, out1_lo); // [out0[0], out1[0]]
            let temp0_hi = _mm_shuffle_pd::<2>(out2_lo, out0_lo); // [out2[0], out0[1]]
            let result0 = _mm256_castpd_ps(_mm256_set_m128d(temp0_hi, temp0_lo));

            // Build result1: [out1[1], out2[1], out0[2], out1[2]]
            let temp1_lo = _mm_unpackhi_pd(out1_lo, out2_lo); // [out1[1], out2[1]]
            let temp1_hi = _mm_unpacklo_pd(out0_hi, out1_hi); // [out0[2], out1[2]]
            let result1 = _mm256_castpd_ps(_mm256_set_m128d(temp1_hi, temp1_lo));

            // Build result2: [out2[2], out0[3], out1[3], out2[3]]
            let temp2_lo = _mm_shuffle_pd::<2>(out2_hi, out0_hi); // [out2[2], out0[3]]
            let temp2_hi = _mm_unpackhi_pd(out1_hi, out2_hi); // [out1[3], out2[3]]
            let result2 = _mm256_castpd_ps(_mm256_set_m128d(temp2_hi, temp2_lo));

            // Sequential stores.
            let j = 3 * i;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            _mm256_storeu_ps(dst_ptr, result0);
            _mm256_storeu_ps(dst_ptr.add(8), result1);
            _mm256_storeu_ps(dst_ptr.add(16), result2);
        }
    }

    for i in simd_iters..third_samples {
        let w1 = stage_twiddles[i * 2];
        let w2 = stage_twiddles[i * 2 + 1];

        let z0 = src[i];
        let z1 = src[i + third_samples];
        let z2 = src[i + third_samples * 2];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);

        let sum_t = t1.add(&t2);
        let diff_t = t1.sub(&t2);

        let j = 3 * i;
        dst[j] = z0.add(&sum_t);

        let re_part = z0.re - 0.5 * sum_t.re;
        let im_part = z0.im - 0.5 * sum_t.im;
        let sqrt3_diff_re = SQRT3_2 * diff_t.im;
        let sqrt3_diff_im = -SQRT3_2 * diff_t.re;

        dst[j + 1] = Complex32::new(re_part + sqrt3_diff_re, im_part + sqrt3_diff_im);
        dst[j + 2] = Complex32::new(re_part - sqrt3_diff_re, im_part - sqrt3_diff_im);
    }
}

/// Performs a single radix-3 Stockham butterfly stage (out-of-place, AVX+FMA) for generic p.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix3_generic_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    // We convince the compiler here that stride can't be 0 to optimize better.
    let stride = stride.max(1);

    let samples = src.len();
    let third_samples = samples / 3;
    let simd_iters = (third_samples >> 2) << 2;

    unsafe {
        let sqrt3_2_vec = _mm256_set_ps(
            -SQRT3_2, SQRT3_2, -SQRT3_2, SQRT3_2, -SQRT3_2, SQRT3_2, -SQRT3_2, SQRT3_2,
        );

        let half_vec = _mm256_set1_ps(0.5);

        for i in (0..simd_iters).step_by(4) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;
            let k2 = k + 2 - ((k + 2 >= stride) as usize) * stride;
            let k3 = k + 3 - ((k + 3 >= stride) as usize) * stride;

            // Load z0
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            // Load z1, z2
            let z1_ptr = src.as_ptr().add(i + third_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + third_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 2) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]

            // Complex multiply: t1 = w1 * z1, t2 = w2 * z2
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);

            // sum_t = t1 + t2, diff_t = t1 - t2
            let sum_t = _mm256_add_ps(t1, t2);
            let diff_t = _mm256_sub_ps(t1, t2);

            // out0 = z0 + sum_t
            let out0 = _mm256_add_ps(z0, sum_t);

            // re_part = z0 - 0.5 * sum_t
            let half_sum_t = _mm256_mul_ps(sum_t, half_vec);
            let re_im_part = _mm256_sub_ps(z0, half_sum_t);

            // sqrt3 multiplication
            let sqrt3_diff = complex_mul_sqrt3_i_avx(diff_t, sqrt3_2_vec);

            let out1 = _mm256_add_ps(re_im_part, sqrt3_diff);
            let out2 = _mm256_sub_ps(re_im_part, sqrt3_diff);

            // Calculate output indices.
            let j0 = 3 * i - 2 * k0;
            let j1 = 3 * (i + 1) - 2 * k1;
            let j2 = 3 * (i + 2) - 2 * k2;
            let j3 = 3 * (i + 3) - 2 * k3;

            // Direct SIMD stores (no stack spills)
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);

            let out0_lo = _mm256_castpd256_pd128(out0_pd);
            let out0_hi = _mm256_extractf128_pd(out0_pd, 1);
            let out1_lo = _mm256_castpd256_pd128(out1_pd);
            let out1_hi = _mm256_extractf128_pd(out1_pd, 1);
            let out2_lo = _mm256_castpd256_pd128(out2_pd);
            let out2_hi = _mm256_extractf128_pd(out2_pd, 1);

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            // Store iteration 0
            _mm_storel_pd(dst_ptr.add(j0), out0_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_lo);

            // Store iteration 1
            _mm_storeh_pd(dst_ptr.add(j1), out0_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_lo);

            // Store iteration 2
            _mm_storel_pd(dst_ptr.add(j2), out0_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride), out1_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 2), out2_hi);

            // Store iteration 3
            _mm_storeh_pd(dst_ptr.add(j3), out0_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride), out1_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 2), out2_hi);
        }
    }

    for i in simd_iters..third_samples {
        let k = i % stride;
        let w1 = stage_twiddles[i * 2];
        let w2 = stage_twiddles[i * 2 + 1];

        let z0 = src[i];
        let z1 = src[i + third_samples];
        let z2 = src[i + third_samples * 2];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);

        let sum_t = t1.add(&t2);
        let diff_t = t1.sub(&t2);

        let j = 3 * i - 2 * k;
        dst[j] = z0.add(&sum_t);

        let re_part = z0.re - 0.5 * sum_t.re;
        let im_part = z0.im - 0.5 * sum_t.im;
        let sqrt3_diff_re = SQRT3_2 * diff_t.im;
        let sqrt3_diff_im = -SQRT3_2 * diff_t.re;

        dst[j + stride] = Complex32::new(re_part + sqrt3_diff_re, im_part + sqrt3_diff_im);
        dst[j + stride * 2] = Complex32::new(re_part - sqrt3_diff_re, im_part - sqrt3_diff_im);
    }
}
