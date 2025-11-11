use core::arch::x86_64::*;

use super::{COS_2PI_5, COS_4PI_5, SIN_2PI_5, SIN_4PI_5};
use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_sse2, load_neg_imag_mask_sse2},
};

/// Performs a single radix-5 Stockham butterfly stage for stride=1 (out-of-place, SSE2).
///
/// This is a specialized version for the stride=1 case (first stage) that stores
/// outputs sequentially for better cache utilization.
#[target_feature(enable = "sse2")]
pub(super) unsafe fn butterfly_radix5_stride1_sse2(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let fifth_samples = samples / 5;
    let simd_iters = (fifth_samples >> 1) << 1;

    unsafe {
        let cos_2pi_5 = _mm_set1_ps(COS_2PI_5);
        let sin_2pi_5 = _mm_set1_ps(SIN_2PI_5);
        let cos_4pi_5 = _mm_set1_ps(COS_4PI_5);
        let sin_4pi_5 = _mm_set1_ps(SIN_4PI_5);

        let negate_im = load_neg_imag_mask_sse2();

        for i in (0..simd_iters).step_by(2) {
            // Load 2 complex numbers from each fifth.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + fifth_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + fifth_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + fifth_samples * 3) as *const f32;
            let z3 = _mm_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + fifth_samples * 4) as *const f32;
            let z4 = _mm_loadu_ps(z4_ptr);

            // Stride=1 optimization: Skip twiddle loads and multiplications (identity twiddles).
            // t_k = (1+0i) * z_k = z_k
            let t1 = z1;
            let t2 = z2;
            let t3 = z3;
            let t4 = z4;

            // Radix-5 DFT decomposition.
            // sum_all = t1 + t2 + t3 + t4
            let sum_all = _mm_add_ps(_mm_add_ps(t1, t2), _mm_add_ps(t3, t4));

            // a1 = t1 + t4
            let a1 = _mm_add_ps(t1, t4);
            // a2 = t2 + t3
            let a2 = _mm_add_ps(t2, t3);

            // b1_re = t1.im - t4.im, b1_im = t4.re - t1.re
            let t1_swap = _mm_shuffle_ps(t1, t1, 0b10_11_00_01);
            let t4_swap = _mm_shuffle_ps(t4, t4, 0b10_11_00_01);
            let b1_temp = _mm_sub_ps(t1_swap, t4_swap);
            let b1 = _mm_xor_ps(b1_temp, negate_im); // [b1_re, b1_im, b1_re, b1_im]

            // b2_re = t2.im - t3.im, b2_im = t3.re - t2.re
            let t2_swap = _mm_shuffle_ps(t2, t2, 0b10_11_00_01);
            let t3_swap = _mm_shuffle_ps(t3, t3, 0b10_11_00_01);
            let b2_temp = _mm_sub_ps(t2_swap, t3_swap);
            let b2 = _mm_xor_ps(b2_temp, negate_im);

            // c1 = z0 + COS_2PI_5 * a1 + COS_4PI_5 * a2
            let c1 = _mm_add_ps(
                z0,
                _mm_add_ps(_mm_mul_ps(cos_2pi_5, a1), _mm_mul_ps(cos_4pi_5, a2)),
            );

            // c2 = z0 + COS_4PI_5 * a1 + COS_2PI_5 * a2
            let c2 = _mm_add_ps(
                z0,
                _mm_add_ps(_mm_mul_ps(cos_4pi_5, a1), _mm_mul_ps(cos_2pi_5, a2)),
            );

            // d1 = SIN_2PI_5 * b1 + SIN_4PI_5 * b2
            let d1 = _mm_add_ps(_mm_mul_ps(sin_2pi_5, b1), _mm_mul_ps(sin_4pi_5, b2));

            // d2 = SIN_4PI_5 * b1 - SIN_2PI_5 * b2
            let d2 = _mm_sub_ps(_mm_mul_ps(sin_4pi_5, b1), _mm_mul_ps(sin_2pi_5, b2));

            // out0 = z0 + sum_all
            let out0 = _mm_add_ps(z0, sum_all);

            // out1 = c1 + d1
            let out1 = _mm_add_ps(c1, d1);

            // out4 = c1 - d1
            let out4 = _mm_sub_ps(c1, d1);

            // out2 = c2 + d2
            let out2 = _mm_add_ps(c2, d2);

            // out3 = c2 - d2
            let out3 = _mm_sub_ps(c2, d2);

            // Sequential 128-bit stores for stride=1 (matching radix-2/radix-4 pattern).
            // Store [out0[0], out1[0], out2[0], out3[0], out4[0], out0[1], out1[1], out2[1], out3[1], out4[1]].
            // Using 5x 128-bit stores: [out0[0], out1[0]], [out2[0], out3[0]], [out4[0], out0[1]], [out1[1], out2[1]], [out3[1], out4[1]]
            let j = 10 * i; // In units of f32 (2 f32 per complex)
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            let out0_pd = _mm_castps_pd(out0);
            let out1_pd = _mm_castps_pd(out1);
            let out2_pd = _mm_castps_pd(out2);
            let out3_pd = _mm_castps_pd(out3);
            let out4_pd = _mm_castps_pd(out4);

            // Interleave outputs for sequential 128-bit stores
            let out01_lo = _mm_castpd_ps(_mm_unpacklo_pd(out0_pd, out1_pd)); // [out0[0], out1[0]]
            let out23_lo = _mm_castpd_ps(_mm_unpacklo_pd(out2_pd, out3_pd)); // [out2[0], out3[0]]
            let out40_cross = _mm_castpd_ps(_mm_shuffle_pd(out4_pd, out0_pd, 0b10)); // [out4[0], out0[1]]
            let out12_hi = _mm_castpd_ps(_mm_unpackhi_pd(out1_pd, out2_pd)); // [out1[1], out2[1]]
            let out34_hi = _mm_castpd_ps(_mm_unpackhi_pd(out3_pd, out4_pd)); // [out3[1], out4[1]]

            _mm_storeu_ps(dst_ptr.add(j), out01_lo);
            _mm_storeu_ps(dst_ptr.add(j + 4), out23_lo);
            _mm_storeu_ps(dst_ptr.add(j + 8), out40_cross);
            _mm_storeu_ps(dst_ptr.add(j + 12), out12_hi);
            _mm_storeu_ps(dst_ptr.add(j + 16), out34_hi);
        }
    }

    super::butterfly_radix5_scalar::<2>(src, dst, stage_twiddles, 1, simd_iters);
}

/// Performs a single radix-5 Stockham butterfly stage for p>1 (out-of-place, SSE2).
///
/// This is the generic version for p>1 cases.
#[target_feature(enable = "sse2")]
pub(super) unsafe fn butterfly_radix5_generic_sse2(
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
    let simd_iters = (fifth_samples >> 1) << 1;

    unsafe {
        let cos_2pi_5 = _mm_set1_ps(COS_2PI_5);
        let sin_2pi_5 = _mm_set1_ps(SIN_2PI_5);
        let cos_4pi_5 = _mm_set1_ps(COS_4PI_5);
        let sin_4pi_5 = _mm_set1_ps(SIN_4PI_5);

        let negate_im = load_neg_imag_mask_sse2();

        for i in (0..simd_iters).step_by(2) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;

            // Load 2 complex numbers from each fifth.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + fifth_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + fifth_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + fifth_samples * 3) as *const f32;
            let z3 = _mm_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + fifth_samples * 4) as *const f32;
            let z4 = _mm_loadu_ps(z4_ptr);

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 4) as *const f32;
            let w1 = _mm_loadu_ps(tw_ptr); // w1[i], w1[i+1]
            let w2 = _mm_loadu_ps(tw_ptr.add(4)); // w2[i], w2[i+1]
            let w3 = _mm_loadu_ps(tw_ptr.add(8)); // w3[i], w3[i+1]
            let w4 = _mm_loadu_ps(tw_ptr.add(12)); // w4[i], w4[i+1]

            // Complex multiply: t1 = w1 * z1, t2 = w2 * z2, t3 = w3 * z3, t4 = w4 * z4
            let t1 = complex_mul_sse2(w1, z1);
            let t2 = complex_mul_sse2(w2, z2);
            let t3 = complex_mul_sse2(w3, z3);
            let t4 = complex_mul_sse2(w4, z4);

            // Radix-5 DFT decomposition
            let sum_all = _mm_add_ps(_mm_add_ps(t1, t2), _mm_add_ps(t3, t4));

            let a1 = _mm_add_ps(t1, t4);
            let a2 = _mm_add_ps(t2, t3);

            let t1_swap = _mm_shuffle_ps(t1, t1, 0b10_11_00_01);
            let t4_swap = _mm_shuffle_ps(t4, t4, 0b10_11_00_01);
            let b1_temp = _mm_sub_ps(t1_swap, t4_swap);
            let b1 = _mm_xor_ps(b1_temp, negate_im);

            let t2_swap = _mm_shuffle_ps(t2, t2, 0b10_11_00_01);
            let t3_swap = _mm_shuffle_ps(t3, t3, 0b10_11_00_01);
            let b2_temp = _mm_sub_ps(t2_swap, t3_swap);
            let b2 = _mm_xor_ps(b2_temp, negate_im);

            let c1 = _mm_add_ps(
                z0,
                _mm_add_ps(_mm_mul_ps(cos_2pi_5, a1), _mm_mul_ps(cos_4pi_5, a2)),
            );

            let c2 = _mm_add_ps(
                z0,
                _mm_add_ps(_mm_mul_ps(cos_4pi_5, a1), _mm_mul_ps(cos_2pi_5, a2)),
            );

            let d1 = _mm_add_ps(_mm_mul_ps(sin_2pi_5, b1), _mm_mul_ps(sin_4pi_5, b2));

            let d2 = _mm_sub_ps(_mm_mul_ps(sin_4pi_5, b1), _mm_mul_ps(sin_2pi_5, b2));

            let out0 = _mm_add_ps(z0, sum_all);
            let out1 = _mm_add_ps(c1, d1);
            let out4 = _mm_sub_ps(c1, d1);
            let out2 = _mm_add_ps(c2, d2);
            let out3 = _mm_sub_ps(c2, d2);

            // Calculate output indices.
            let j0 = 5 * i - 4 * k0;
            let j1 = 5 * (i + 1) - 4 * k1;

            // Direct SIMD stores.
            let out0_pd = _mm_castps_pd(out0);
            let out1_pd = _mm_castps_pd(out1);
            let out2_pd = _mm_castps_pd(out2);
            let out3_pd = _mm_castps_pd(out3);
            let out4_pd = _mm_castps_pd(out4);

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            _mm_storel_pd(dst_ptr.add(j0), out0_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 3), out3_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 4), out4_pd);

            _mm_storeh_pd(dst_ptr.add(j1), out0_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 3), out3_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 4), out4_pd);
        }
    }

    super::butterfly_radix5_scalar::<2>(src, dst, stage_twiddles, stride, simd_iters);
}
