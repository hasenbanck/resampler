use super::SQRT3_2;
use crate::fft::Complex32;

/// Performs a single radix-3 Stockham butterfly stage for stride=1 (out-of-place, SSE2).
///
/// This is a specialized version for the stride=1 case (first stage) that stores
/// outputs sequentially for better cache utilization.
#[target_feature(enable = "sse2")]
pub(super) unsafe fn butterfly_radix3_stride1_sse2(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let third_samples = samples / 3;
    let simd_iters = (third_samples >> 1) << 1;

    unsafe {
        let half_vec = _mm_set1_ps(0.5);
        let sqrt3_vec = _mm_set_ps(-SQRT3_2, SQRT3_2, -SQRT3_2, SQRT3_2);

        for i in (0..simd_iters).step_by(2) {
            // Load 2 complex numbers from each third.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + third_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + third_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            // Load 4 twiddles contiguously: [w1[0], w2[0], w1[1], w2[1]].
            let tw_ptr = stage_twiddles.as_ptr().add(i * 2) as *const f32;
            let tw01 = _mm_loadu_ps(tw_ptr); // [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            let tw23 = _mm_loadu_ps(tw_ptr.add(4)); // [w1[1].re, w1[1].im, w2[1].re, w2[1].im]

            // Extract w1 and w2 using shuffle.
            // tw01 = [w1[0].re, w1[0].im, w2[0].re, w2[0].im]
            // tw23 = [w1[1].re, w1[1].im, w2[1].re, w2[1].im]
            // We want w1 = [w1[0].re, w1[0].im, w1[1].re, w1[1].im]
            //         w2 = [w2[0].re, w2[0].im, w2[1].re, w2[1].im]
            let w1 = _mm_shuffle_ps(tw01, tw23, 0b01_00_01_00); // [w1[0], w1[1]]
            let w2 = _mm_shuffle_ps(tw01, tw23, 0b11_10_11_10); // [w2[0], w2[1]]

            // Complex multiply: t1 = w1 * z1
            let z1_re = _mm_shuffle_ps(z1, z1, 0b10_10_00_00);
            let z1_im = _mm_shuffle_ps(z1, z1, 0b11_11_01_01);
            let w1_swap = _mm_shuffle_ps(w1, w1, 0b10_11_00_01);
            let prod1_re = _mm_mul_ps(w1, z1_re);
            let prod1_im = _mm_mul_ps(w1_swap, z1_im);
            let sub1 = _mm_sub_ps(prod1_re, prod1_im);
            let add1 = _mm_add_ps(prod1_re, prod1_im);
            let select_odd = _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));
            let t1 = _mm_or_ps(
                _mm_and_ps(select_odd, add1),
                _mm_andnot_ps(select_odd, sub1),
            );

            // Complex multiply: t2 = w2 * z2
            let z2_re = _mm_shuffle_ps(z2, z2, 0b10_10_00_00);
            let z2_im = _mm_shuffle_ps(z2, z2, 0b11_11_01_01);
            let w2_swap = _mm_shuffle_ps(w2, w2, 0b10_11_00_01);
            let prod2_re = _mm_mul_ps(w2, z2_re);
            let prod2_im = _mm_mul_ps(w2_swap, z2_im);
            let sub2 = _mm_sub_ps(prod2_re, prod2_im);
            let add2 = _mm_add_ps(prod2_re, prod2_im);
            let t2 = _mm_or_ps(
                _mm_and_ps(select_odd, add2),
                _mm_andnot_ps(select_odd, sub2),
            );

            // sum_t = t1 + t2, diff_t = t1 - t2
            let sum_t = _mm_add_ps(t1, t2);
            let diff_t = _mm_sub_ps(t1, t2);

            // out0 = z0 + sum_t
            let out0 = _mm_add_ps(z0, sum_t);

            // re_im_part = z0 - 0.5 * sum_t
            let half_sum_t = _mm_mul_ps(sum_t, half_vec);
            let re_im_part = _mm_sub_ps(z0, half_sum_t);

            // sqrt3_diff = SQRT3_2 * [diff_t.im, -diff_t.re]
            let diff_t_swap = _mm_shuffle_ps(diff_t, diff_t, 0b10_11_00_01);
            let sqrt3_diff = _mm_mul_ps(diff_t_swap, sqrt3_vec);

            let out1 = _mm_add_ps(re_im_part, sqrt3_diff);
            let out2 = _mm_sub_ps(re_im_part, sqrt3_diff);

            // Sequential 128-bit stores for stride=1 (matching radix-2/radix-4 pattern).
            // We need to store [out0[0], out1[0], out2[0], out0[1], out1[1], out2[1]].
            // Using 3x 128-bit stores: [out0[0], out1[0]], [out2[0], out0[1]], [out1[1], out2[1]]
            let j = 6 * i; // In units of f32 (2 f32 per complex)
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            let out0_pd = _mm_castps_pd(out0);
            let out1_pd = _mm_castps_pd(out1);
            let out2_pd = _mm_castps_pd(out2);

            // Interleave outputs for sequential 128-bit stores
            let out01_lo = _mm_castpd_ps(_mm_unpacklo_pd(out0_pd, out1_pd)); // [out0[0], out1[0]]
            let out20_cross = _mm_castpd_ps(_mm_shuffle_pd(out2_pd, out0_pd, 0b10)); // [out2[0], out0[1]]
            let out12_hi = _mm_castpd_ps(_mm_unpackhi_pd(out1_pd, out2_pd)); // [out1[1], out2[1]]

            _mm_storeu_ps(dst_ptr.add(j), out01_lo);
            _mm_storeu_ps(dst_ptr.add(j + 4), out20_cross);
            _mm_storeu_ps(dst_ptr.add(j + 8), out12_hi);
        }
    }

    // Scalar cleanup loop.
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

/// Performs a single radix-3 Stockham butterfly stage for p>1 (out-of-place, SSE2).
///
/// This is the generic version for p>1 cases.
#[target_feature(enable = "sse2")]
pub(super) unsafe fn butterfly_radix3_generic_sse2(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let third_samples = samples / 3;
    let simd_iters = (third_samples >> 1) << 1;

    unsafe {
        let half_vec = _mm_set1_ps(0.5);
        let sqrt3_vec = _mm_set_ps(-SQRT3_2, SQRT3_2, -SQRT3_2, SQRT3_2);

        for i in (0..simd_iters).step_by(2) {
            let k0 = i % stride;
            let k1 = (i + 1) % stride;

            // Load 2 complex numbers from each third.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + third_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + third_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            // Load 4 twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 2) as *const f32;
            let tw01 = _mm_loadu_ps(tw_ptr);
            let tw23 = _mm_loadu_ps(tw_ptr.add(4));

            // Extract w1 and w2.
            let w1 = _mm_shuffle_ps(tw01, tw23, 0b01_00_01_00);
            let w2 = _mm_shuffle_ps(tw01, tw23, 0b11_10_11_10);

            // Complex multiply: t1 = w1 * z1
            let z1_re = _mm_shuffle_ps(z1, z1, 0b10_10_00_00);
            let z1_im = _mm_shuffle_ps(z1, z1, 0b11_11_01_01);
            let w1_swap = _mm_shuffle_ps(w1, w1, 0b10_11_00_01);
            let prod1_re = _mm_mul_ps(w1, z1_re);
            let prod1_im = _mm_mul_ps(w1_swap, z1_im);
            let sub1 = _mm_sub_ps(prod1_re, prod1_im);
            let add1 = _mm_add_ps(prod1_re, prod1_im);
            let select_odd = _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));
            let t1 = _mm_or_ps(
                _mm_and_ps(select_odd, add1),
                _mm_andnot_ps(select_odd, sub1),
            );

            // Complex multiply: t2 = w2 * z2
            let z2_re = _mm_shuffle_ps(z2, z2, 0b10_10_00_00);
            let z2_im = _mm_shuffle_ps(z2, z2, 0b11_11_01_01);
            let w2_swap = _mm_shuffle_ps(w2, w2, 0b10_11_00_01);
            let prod2_re = _mm_mul_ps(w2, z2_re);
            let prod2_im = _mm_mul_ps(w2_swap, z2_im);
            let sub2 = _mm_sub_ps(prod2_re, prod2_im);
            let add2 = _mm_add_ps(prod2_re, prod2_im);
            let t2 = _mm_or_ps(
                _mm_and_ps(select_odd, add2),
                _mm_andnot_ps(select_odd, sub2),
            );

            // sum_t = t1 + t2, diff_t = t1 - t2
            let sum_t = _mm_add_ps(t1, t2);
            let diff_t = _mm_sub_ps(t1, t2);

            // out0 = z0 + sum_t
            let out0 = _mm_add_ps(z0, sum_t);

            // re_im_part = z0 - 0.5 * sum_t
            let half_sum_t = _mm_mul_ps(sum_t, half_vec);
            let re_im_part = _mm_sub_ps(z0, half_sum_t);

            // sqrt3_diff = SQRT3_2 * [diff_t.im, -diff_t.re]
            let diff_t_swap = _mm_shuffle_ps(diff_t, diff_t, 0b10_11_00_01);
            let sqrt3_diff = _mm_mul_ps(diff_t_swap, sqrt3_vec);

            let out1 = _mm_add_ps(re_im_part, sqrt3_diff);
            let out2 = _mm_sub_ps(re_im_part, sqrt3_diff);

            // Calculate output indices.
            let j0 = 3 * i - 2 * k0;
            let j1 = 3 * (i + 1) - 2 * k1;

            // Direct SIMD stores.
            let out0_pd = _mm_castps_pd(out0);
            let out1_pd = _mm_castps_pd(out1);
            let out2_pd = _mm_castps_pd(out2);

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            _mm_storel_pd(dst_ptr.add(j0), out0_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_pd);

            _mm_storeh_pd(dst_ptr.add(j1), out0_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_pd);
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
