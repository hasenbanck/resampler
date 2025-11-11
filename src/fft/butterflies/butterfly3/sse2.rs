use core::arch::x86_64::*;

use super::SQRT3_2;
use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_sqrt3_i_sse2, complex_mul_sse2},
};

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
    let samples = src.len();
    let third_samples = samples / 3;
    let simd_iters = (third_samples >> 1) << 1;

    unsafe {
        let half_vec = _mm_set1_ps(0.5);

        for i in (0..simd_iters).step_by(2) {
            // Load 2 complex numbers from each third.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + third_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + third_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            // Identity twiddles: t1 = (1+0i) * z1 = z1, t2 = (1+0i) * z2 = z2 (skip twiddle load and multiply)
            let t1 = z1;
            let t2 = z2;

            // sum_t = t1 + t2, diff_t = t1 - t2
            let sum_t = _mm_add_ps(t1, t2);
            let diff_t = _mm_sub_ps(t1, t2);

            // out0 = z0 + sum_t
            let out0 = _mm_add_ps(z0, sum_t);

            // re_im_part = z0 - 0.5 * sum_t
            let half_sum_t = _mm_mul_ps(sum_t, half_vec);
            let re_im_part = _mm_sub_ps(z0, half_sum_t);

            // sqrt3_diff = SQRT3_2 * i * diff_t
            let sqrt3_diff = complex_mul_sqrt3_i_sse2(diff_t, SQRT3_2);

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

    super::butterfly_radix3_scalar::<2>(src, dst, stage_twiddles, 1, simd_iters);
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
    // We convince the compiler here that stride can't be 0 to optimize better.
    if stride == 0 {
        return;
    }

    let samples = src.len();
    let third_samples = samples / 3;
    let simd_iters = (third_samples >> 1) << 1;

    unsafe {
        let half_vec = _mm_set1_ps(0.5);

        for i in (0..simd_iters).step_by(2) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;

            // Load 2 complex numbers from each third.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + third_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + third_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 2) as *const f32;
            let w1 = _mm_loadu_ps(tw_ptr); // w1[i], w1[i+1]
            let w2 = _mm_loadu_ps(tw_ptr.add(4)); // w2[i], w2[i+1]

            // Complex multiply: t1 = w1 * z1, t2 = w2 * z2
            let t1 = complex_mul_sse2(w1, z1);
            let t2 = complex_mul_sse2(w2, z2);

            // sum_t = t1 + t2, diff_t = t1 - t2
            let sum_t = _mm_add_ps(t1, t2);
            let diff_t = _mm_sub_ps(t1, t2);

            // out0 = z0 + sum_t
            let out0 = _mm_add_ps(z0, sum_t);

            // re_im_part = z0 - 0.5 * sum_t
            let half_sum_t = _mm_mul_ps(sum_t, half_vec);
            let re_im_part = _mm_sub_ps(z0, half_sum_t);

            // sqrt3_diff = SQRT3_2 * i * diff_t
            let sqrt3_diff = complex_mul_sqrt3_i_sse2(diff_t, SQRT3_2);

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

    super::butterfly_radix3_scalar::<2>(src, dst, stage_twiddles, stride, simd_iters);
}
