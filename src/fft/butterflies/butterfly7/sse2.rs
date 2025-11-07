use super::{COS_2PI_7, COS_4PI_7, COS_6PI_7, SIN_2PI_7, SIN_4PI_7, SIN_6PI_7};
use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_sse2, load_neg_imag_mask_sse2},
};

/// Performs a single radix-7 Stockham butterfly stage for stride=1 (out-of-place, SSE2).
///
/// This is a specialized version for the stride=1 case (first stage) that stores
/// outputs sequentially for better cache utilization.
#[target_feature(enable = "sse2")]
pub(super) unsafe fn butterfly_radix7_stride1_sse2(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let seventh_samples = samples / 7;
    let simd_iters = (seventh_samples >> 1) << 1;

    unsafe {
        let cos_2pi_7 = _mm_set1_ps(COS_2PI_7);
        let sin_2pi_7 = _mm_set1_ps(SIN_2PI_7);
        let cos_4pi_7 = _mm_set1_ps(COS_4PI_7);
        let sin_4pi_7 = _mm_set1_ps(SIN_4PI_7);
        let cos_6pi_7 = _mm_set1_ps(COS_6PI_7);
        let sin_6pi_7 = _mm_set1_ps(SIN_6PI_7);

        let negate_im = load_neg_imag_mask_sse2();

        for i in (0..simd_iters).step_by(2) {
            // Load 2 complex numbers from each seventh.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + seventh_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + seventh_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + seventh_samples * 3) as *const f32;
            let z3 = _mm_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + seventh_samples * 4) as *const f32;
            let z4 = _mm_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + seventh_samples * 5) as *const f32;
            let z5 = _mm_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + seventh_samples * 6) as *const f32;
            let z6 = _mm_loadu_ps(z6_ptr);

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 6) as *const f32;
            let w1 = _mm_loadu_ps(tw_ptr); // w1[i], w1[i+1]
            let w2 = _mm_loadu_ps(tw_ptr.add(4)); // w2[i], w2[i+1]
            let w3 = _mm_loadu_ps(tw_ptr.add(8)); // w3[i], w3[i+1]
            let w4 = _mm_loadu_ps(tw_ptr.add(12)); // w4[i], w4[i+1]
            let w5 = _mm_loadu_ps(tw_ptr.add(16)); // w5[i], w5[i+1]
            let w6 = _mm_loadu_ps(tw_ptr.add(20)); // w6[i], w6[i+1]

            // Complex multiply: t1 = w1 * z1, t2 = w2 * z2, ..., t6 = w6 * z6
            let t1 = complex_mul_sse2(w1, z1);
            let t2 = complex_mul_sse2(w2, z2);
            let t3 = complex_mul_sse2(w3, z3);
            let t4 = complex_mul_sse2(w4, z4);
            let t5 = complex_mul_sse2(w5, z5);
            let t6 = complex_mul_sse2(w6, z6);

            // Radix-7 DFT decomposition
            let sum_all = _mm_add_ps(
                _mm_add_ps(_mm_add_ps(t1, t2), _mm_add_ps(t3, t4)),
                _mm_add_ps(t5, t6),
            );

            // a1 = t1 + t6
            let a1 = _mm_add_ps(t1, t6);
            // a2 = t2 + t5
            let a2 = _mm_add_ps(t2, t5);
            // a3 = t3 + t4
            let a3 = _mm_add_ps(t3, t4);

            // b1_re = t1.im - t6.im, b1_im = t6.re - t1.re
            let t1_swap = _mm_shuffle_ps(t1, t1, 0b10_11_00_01);
            let t6_swap = _mm_shuffle_ps(t6, t6, 0b10_11_00_01);
            let b1_temp = _mm_sub_ps(t1_swap, t6_swap);
            let b1 = _mm_xor_ps(b1_temp, negate_im);

            // b2_re = t2.im - t5.im, b2_im = t5.re - t2.re
            let t2_swap = _mm_shuffle_ps(t2, t2, 0b10_11_00_01);
            let t5_swap = _mm_shuffle_ps(t5, t5, 0b10_11_00_01);
            let b2_temp = _mm_sub_ps(t2_swap, t5_swap);
            let b2 = _mm_xor_ps(b2_temp, negate_im);

            // b3_re = t3.im - t4.im, b3_im = t4.re - t3.re
            let t3_swap = _mm_shuffle_ps(t3, t3, 0b10_11_00_01);
            let t4_swap = _mm_shuffle_ps(t4, t4, 0b10_11_00_01);
            let b3_temp = _mm_sub_ps(t3_swap, t4_swap);
            let b3 = _mm_xor_ps(b3_temp, negate_im);

            // out0 = z0 + sum_all
            let out0 = _mm_add_ps(z0, sum_all);

            // Compute outputs 1-6 using unrolled loop from scalar version

            // out1: cos1=COS_2PI_7, sin1=SIN_2PI_7, cos2=COS_4PI_7, sin2=SIN_4PI_7, cos3=COS_6PI_7, sin3=SIN_6PI_7
            let c1 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_2pi_7, a1), _mm_mul_ps(cos_4pi_7, a2)),
                    _mm_mul_ps(cos_6pi_7, a3),
                ),
            );
            let d1 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(sin_2pi_7, b1), _mm_mul_ps(sin_4pi_7, b2)),
                _mm_mul_ps(sin_6pi_7, b3),
            );
            let out1 = _mm_add_ps(c1, d1);

            // out2: cos1=COS_4PI_7, sin1=SIN_4PI_7, cos2=COS_6PI_7, sin2=-SIN_6PI_7, cos3=COS_2PI_7, sin3=-SIN_2PI_7
            let neg_sin_6pi_7 = _mm_set1_ps(-SIN_6PI_7);
            let neg_sin_2pi_7 = _mm_set1_ps(-SIN_2PI_7);
            let c2 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_4pi_7, a1), _mm_mul_ps(cos_6pi_7, a2)),
                    _mm_mul_ps(cos_2pi_7, a3),
                ),
            );
            let d2 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(sin_4pi_7, b1), _mm_mul_ps(neg_sin_6pi_7, b2)),
                _mm_mul_ps(neg_sin_2pi_7, b3),
            );
            let out2 = _mm_add_ps(c2, d2);

            // out3: cos1=COS_6PI_7, sin1=SIN_6PI_7, cos2=COS_2PI_7, sin2=-SIN_2PI_7, cos3=COS_4PI_7, sin3=SIN_4PI_7
            let c3 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_6pi_7, a1), _mm_mul_ps(cos_2pi_7, a2)),
                    _mm_mul_ps(cos_4pi_7, a3),
                ),
            );
            let d3 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(sin_6pi_7, b1), _mm_mul_ps(neg_sin_2pi_7, b2)),
                _mm_mul_ps(sin_4pi_7, b3),
            );
            let out3 = _mm_add_ps(c3, d3);

            // out4: cos1=COS_6PI_7, sin1=-SIN_6PI_7, cos2=COS_2PI_7, sin2=SIN_2PI_7, cos3=COS_4PI_7, sin3=-SIN_4PI_7
            let neg_sin_4pi_7 = _mm_set1_ps(-SIN_4PI_7);
            let c4 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_6pi_7, a1), _mm_mul_ps(cos_2pi_7, a2)),
                    _mm_mul_ps(cos_4pi_7, a3),
                ),
            );
            let d4 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(neg_sin_6pi_7, b1), _mm_mul_ps(sin_2pi_7, b2)),
                _mm_mul_ps(neg_sin_4pi_7, b3),
            );
            let out4 = _mm_add_ps(c4, d4);

            // out5: cos1=COS_4PI_7, sin1=-SIN_4PI_7, cos2=COS_6PI_7, sin2=SIN_6PI_7, cos3=COS_2PI_7, sin3=SIN_2PI_7
            let c5 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_4pi_7, a1), _mm_mul_ps(cos_6pi_7, a2)),
                    _mm_mul_ps(cos_2pi_7, a3),
                ),
            );
            let d5 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(neg_sin_4pi_7, b1), _mm_mul_ps(sin_6pi_7, b2)),
                _mm_mul_ps(sin_2pi_7, b3),
            );
            let out5 = _mm_add_ps(c5, d5);

            // out6: cos1=COS_2PI_7, sin1=-SIN_2PI_7, cos2=COS_4PI_7, sin2=-SIN_4PI_7, cos3=COS_6PI_7, sin3=-SIN_6PI_7
            let c6 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_2pi_7, a1), _mm_mul_ps(cos_4pi_7, a2)),
                    _mm_mul_ps(cos_6pi_7, a3),
                ),
            );
            let d6 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(neg_sin_2pi_7, b1), _mm_mul_ps(neg_sin_4pi_7, b2)),
                _mm_mul_ps(neg_sin_6pi_7, b3),
            );
            let out6 = _mm_add_ps(c6, d6);

            // Sequential 128-bit stores for stride=1 (matching radix-2/radix-4 pattern).
            // Using 7x 128-bit stores instead of 14x 64-bit stores
            let j = 14 * i; // In units of f32 (2 f32 per complex)
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            let out0_pd = _mm_castps_pd(out0);
            let out1_pd = _mm_castps_pd(out1);
            let out2_pd = _mm_castps_pd(out2);
            let out3_pd = _mm_castps_pd(out3);
            let out4_pd = _mm_castps_pd(out4);
            let out5_pd = _mm_castps_pd(out5);
            let out6_pd = _mm_castps_pd(out6);

            // Interleave outputs for sequential 128-bit stores
            let out01_lo = _mm_castpd_ps(_mm_unpacklo_pd(out0_pd, out1_pd)); // [out0[0], out1[0]]
            let out23_lo = _mm_castpd_ps(_mm_unpacklo_pd(out2_pd, out3_pd)); // [out2[0], out3[0]]
            let out45_lo = _mm_castpd_ps(_mm_unpacklo_pd(out4_pd, out5_pd)); // [out4[0], out5[0]]
            let out60_cross = _mm_castpd_ps(_mm_shuffle_pd(out6_pd, out0_pd, 0b10)); // [out6[0], out0[1]]
            let out12_hi = _mm_castpd_ps(_mm_unpackhi_pd(out1_pd, out2_pd)); // [out1[1], out2[1]]
            let out34_hi = _mm_castpd_ps(_mm_unpackhi_pd(out3_pd, out4_pd)); // [out3[1], out4[1]]
            let out56_hi = _mm_castpd_ps(_mm_unpackhi_pd(out5_pd, out6_pd)); // [out5[1], out6[1]]

            _mm_storeu_ps(dst_ptr.add(j), out01_lo);
            _mm_storeu_ps(dst_ptr.add(j + 4), out23_lo);
            _mm_storeu_ps(dst_ptr.add(j + 8), out45_lo);
            _mm_storeu_ps(dst_ptr.add(j + 12), out60_cross);
            _mm_storeu_ps(dst_ptr.add(j + 16), out12_hi);
            _mm_storeu_ps(dst_ptr.add(j + 20), out34_hi);
            _mm_storeu_ps(dst_ptr.add(j + 24), out56_hi);
        }
    }

    // Scalar cleanup loop.
    for i in simd_iters..seventh_samples {
        let w1 = stage_twiddles[i * 6];
        let w2 = stage_twiddles[i * 6 + 1];
        let w3 = stage_twiddles[i * 6 + 2];
        let w4 = stage_twiddles[i * 6 + 3];
        let w5 = stage_twiddles[i * 6 + 4];
        let w6 = stage_twiddles[i * 6 + 5];

        let z0 = src[i];
        let z1 = src[i + seventh_samples];
        let z2 = src[i + seventh_samples * 2];
        let z3 = src[i + seventh_samples * 3];
        let z4 = src[i + seventh_samples * 4];
        let z5 = src[i + seventh_samples * 5];
        let z6 = src[i + seventh_samples * 6];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);
        let t5 = w5.mul(&z5);
        let t6 = w6.mul(&z6);

        let sum_all = t1.add(&t2).add(&t3).add(&t4).add(&t5).add(&t6);

        let a1 = t1.add(&t6);
        let a2 = t2.add(&t5);
        let a3 = t3.add(&t4);

        let b1_re = t1.im - t6.im;
        let b1_im = t6.re - t1.re;
        let b2_re = t2.im - t5.im;
        let b2_im = t5.re - t2.re;
        let b3_re = t3.im - t4.im;
        let b3_im = t4.re - t3.re;

        let j = 7 * i;
        dst[j] = z0.add(&sum_all);

        for idx in 1..7 {
            let (cos1, sin1, cos2, sin2, cos3, sin3) = match idx {
                1 => (
                    COS_2PI_7, SIN_2PI_7, COS_4PI_7, SIN_4PI_7, COS_6PI_7, SIN_6PI_7,
                ),
                2 => (
                    COS_4PI_7, SIN_4PI_7, COS_6PI_7, -SIN_6PI_7, COS_2PI_7, -SIN_2PI_7,
                ),
                3 => (
                    COS_6PI_7, SIN_6PI_7, COS_2PI_7, -SIN_2PI_7, COS_4PI_7, SIN_4PI_7,
                ),
                4 => (
                    COS_6PI_7, -SIN_6PI_7, COS_2PI_7, SIN_2PI_7, COS_4PI_7, -SIN_4PI_7,
                ),
                5 => (
                    COS_4PI_7, -SIN_4PI_7, COS_6PI_7, SIN_6PI_7, COS_2PI_7, SIN_2PI_7,
                ),
                6 => (
                    COS_2PI_7, -SIN_2PI_7, COS_4PI_7, -SIN_4PI_7, COS_6PI_7, -SIN_6PI_7,
                ),
                _ => unreachable!(),
            };

            let c_re = z0.re + cos1 * a1.re + cos2 * a2.re + cos3 * a3.re;
            let c_im = z0.im + cos1 * a1.im + cos2 * a2.im + cos3 * a3.im;
            let d_re = sin1 * b1_re + sin2 * b2_re + sin3 * b3_re;
            let d_im = sin1 * b1_im + sin2 * b2_im + sin3 * b3_im;

            dst[j + idx] = Complex32::new(c_re + d_re, c_im + d_im);
        }
    }
}

/// Performs a single radix-7 Stockham butterfly stage for p>1 (out-of-place, SSE2).
///
/// This is the generic version for p>1 cases.
#[target_feature(enable = "sse2")]
pub(super) unsafe fn butterfly_radix7_generic_sse2(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let seventh_samples = samples / 7;
    let simd_iters = (seventh_samples >> 1) << 1;

    unsafe {
        let cos_2pi_7 = _mm_set1_ps(COS_2PI_7);
        let sin_2pi_7 = _mm_set1_ps(SIN_2PI_7);
        let cos_4pi_7 = _mm_set1_ps(COS_4PI_7);
        let sin_4pi_7 = _mm_set1_ps(SIN_4PI_7);
        let cos_6pi_7 = _mm_set1_ps(COS_6PI_7);
        let sin_6pi_7 = _mm_set1_ps(SIN_6PI_7);

        let negate_im = load_neg_imag_mask_sse2();

        for i in (0..simd_iters).step_by(2) {
            let k0 = i % stride;
            let k1 = (i + 1) % stride;

            // Load 2 complex numbers from each seventh.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + seventh_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + seventh_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + seventh_samples * 3) as *const f32;
            let z3 = _mm_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + seventh_samples * 4) as *const f32;
            let z4 = _mm_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + seventh_samples * 5) as *const f32;
            let z5 = _mm_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + seventh_samples * 6) as *const f32;
            let z6 = _mm_loadu_ps(z6_ptr);

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 6) as *const f32;
            let w1 = _mm_loadu_ps(tw_ptr); // w1[i], w1[i+1]
            let w2 = _mm_loadu_ps(tw_ptr.add(4)); // w2[i], w2[i+1]
            let w3 = _mm_loadu_ps(tw_ptr.add(8)); // w3[i], w3[i+1]
            let w4 = _mm_loadu_ps(tw_ptr.add(12)); // w4[i], w4[i+1]
            let w5 = _mm_loadu_ps(tw_ptr.add(16)); // w5[i], w5[i+1]
            let w6 = _mm_loadu_ps(tw_ptr.add(20)); // w6[i], w6[i+1]

            // Complex multiply: t1 = w1 * z1, t2 = w2 * z2, ..., t6 = w6 * z6
            let t1 = complex_mul_sse2(w1, z1);
            let t2 = complex_mul_sse2(w2, z2);
            let t3 = complex_mul_sse2(w3, z3);
            let t4 = complex_mul_sse2(w4, z4);
            let t5 = complex_mul_sse2(w5, z5);
            let t6 = complex_mul_sse2(w6, z6);

            // Radix-7 DFT decomposition
            let sum_all = _mm_add_ps(
                _mm_add_ps(_mm_add_ps(t1, t2), _mm_add_ps(t3, t4)),
                _mm_add_ps(t5, t6),
            );

            let a1 = _mm_add_ps(t1, t6);
            let a2 = _mm_add_ps(t2, t5);
            let a3 = _mm_add_ps(t3, t4);

            let t1_swap = _mm_shuffle_ps(t1, t1, 0b10_11_00_01);
            let t6_swap = _mm_shuffle_ps(t6, t6, 0b10_11_00_01);
            let b1_temp = _mm_sub_ps(t1_swap, t6_swap);
            let b1 = _mm_xor_ps(b1_temp, negate_im);

            let t2_swap = _mm_shuffle_ps(t2, t2, 0b10_11_00_01);
            let t5_swap = _mm_shuffle_ps(t5, t5, 0b10_11_00_01);
            let b2_temp = _mm_sub_ps(t2_swap, t5_swap);
            let b2 = _mm_xor_ps(b2_temp, negate_im);

            let t3_swap = _mm_shuffle_ps(t3, t3, 0b10_11_00_01);
            let t4_swap = _mm_shuffle_ps(t4, t4, 0b10_11_00_01);
            let b3_temp = _mm_sub_ps(t3_swap, t4_swap);
            let b3 = _mm_xor_ps(b3_temp, negate_im);

            let out0 = _mm_add_ps(z0, sum_all);

            // Compute outputs 1-6
            let neg_sin_6pi_7 = _mm_set1_ps(-SIN_6PI_7);
            let neg_sin_2pi_7 = _mm_set1_ps(-SIN_2PI_7);
            let neg_sin_4pi_7 = _mm_set1_ps(-SIN_4PI_7);

            let c1 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_2pi_7, a1), _mm_mul_ps(cos_4pi_7, a2)),
                    _mm_mul_ps(cos_6pi_7, a3),
                ),
            );
            let d1 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(sin_2pi_7, b1), _mm_mul_ps(sin_4pi_7, b2)),
                _mm_mul_ps(sin_6pi_7, b3),
            );
            let out1 = _mm_add_ps(c1, d1);

            let c2 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_4pi_7, a1), _mm_mul_ps(cos_6pi_7, a2)),
                    _mm_mul_ps(cos_2pi_7, a3),
                ),
            );
            let d2 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(sin_4pi_7, b1), _mm_mul_ps(neg_sin_6pi_7, b2)),
                _mm_mul_ps(neg_sin_2pi_7, b3),
            );
            let out2 = _mm_add_ps(c2, d2);

            let c3 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_6pi_7, a1), _mm_mul_ps(cos_2pi_7, a2)),
                    _mm_mul_ps(cos_4pi_7, a3),
                ),
            );
            let d3 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(sin_6pi_7, b1), _mm_mul_ps(neg_sin_2pi_7, b2)),
                _mm_mul_ps(sin_4pi_7, b3),
            );
            let out3 = _mm_add_ps(c3, d3);

            let c4 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_6pi_7, a1), _mm_mul_ps(cos_2pi_7, a2)),
                    _mm_mul_ps(cos_4pi_7, a3),
                ),
            );
            let d4 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(neg_sin_6pi_7, b1), _mm_mul_ps(sin_2pi_7, b2)),
                _mm_mul_ps(neg_sin_4pi_7, b3),
            );
            let out4 = _mm_add_ps(c4, d4);

            let c5 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_4pi_7, a1), _mm_mul_ps(cos_6pi_7, a2)),
                    _mm_mul_ps(cos_2pi_7, a3),
                ),
            );
            let d5 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(neg_sin_4pi_7, b1), _mm_mul_ps(sin_6pi_7, b2)),
                _mm_mul_ps(sin_2pi_7, b3),
            );
            let out5 = _mm_add_ps(c5, d5);

            let c6 = _mm_add_ps(
                z0,
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cos_2pi_7, a1), _mm_mul_ps(cos_4pi_7, a2)),
                    _mm_mul_ps(cos_6pi_7, a3),
                ),
            );
            let d6 = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(neg_sin_2pi_7, b1), _mm_mul_ps(neg_sin_4pi_7, b2)),
                _mm_mul_ps(neg_sin_6pi_7, b3),
            );
            let out6 = _mm_add_ps(c6, d6);

            // Calculate output indices.
            let j0 = 7 * i - 6 * k0;
            let j1 = 7 * (i + 1) - 6 * k1;

            // Direct SIMD stores.
            let out0_pd = _mm_castps_pd(out0);
            let out1_pd = _mm_castps_pd(out1);
            let out2_pd = _mm_castps_pd(out2);
            let out3_pd = _mm_castps_pd(out3);
            let out4_pd = _mm_castps_pd(out4);
            let out5_pd = _mm_castps_pd(out5);
            let out6_pd = _mm_castps_pd(out6);

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            _mm_storel_pd(dst_ptr.add(j0), out0_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 3), out3_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 4), out4_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 5), out5_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 6), out6_pd);

            _mm_storeh_pd(dst_ptr.add(j1), out0_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 3), out3_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 4), out4_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 5), out5_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 6), out6_pd);
        }
    }

    for i in simd_iters..seventh_samples {
        let k = i % stride;
        let w1 = stage_twiddles[i * 6];
        let w2 = stage_twiddles[i * 6 + 1];
        let w3 = stage_twiddles[i * 6 + 2];
        let w4 = stage_twiddles[i * 6 + 3];
        let w5 = stage_twiddles[i * 6 + 4];
        let w6 = stage_twiddles[i * 6 + 5];

        let z0 = src[i];
        let z1 = src[i + seventh_samples];
        let z2 = src[i + seventh_samples * 2];
        let z3 = src[i + seventh_samples * 3];
        let z4 = src[i + seventh_samples * 4];
        let z5 = src[i + seventh_samples * 5];
        let z6 = src[i + seventh_samples * 6];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);
        let t5 = w5.mul(&z5);
        let t6 = w6.mul(&z6);

        let sum_all = t1.add(&t2).add(&t3).add(&t4).add(&t5).add(&t6);

        let a1 = t1.add(&t6);
        let a2 = t2.add(&t5);
        let a3 = t3.add(&t4);

        let b1_re = t1.im - t6.im;
        let b1_im = t6.re - t1.re;
        let b2_re = t2.im - t5.im;
        let b2_im = t5.re - t2.re;
        let b3_re = t3.im - t4.im;
        let b3_im = t4.re - t3.re;

        let j = 7 * i - 6 * k;
        dst[j] = z0.add(&sum_all);

        for idx in 1..7 {
            let (cos1, sin1, cos2, sin2, cos3, sin3) = match idx {
                1 => (
                    COS_2PI_7, SIN_2PI_7, COS_4PI_7, SIN_4PI_7, COS_6PI_7, SIN_6PI_7,
                ),
                2 => (
                    COS_4PI_7, SIN_4PI_7, COS_6PI_7, -SIN_6PI_7, COS_2PI_7, -SIN_2PI_7,
                ),
                3 => (
                    COS_6PI_7, SIN_6PI_7, COS_2PI_7, -SIN_2PI_7, COS_4PI_7, SIN_4PI_7,
                ),
                4 => (
                    COS_6PI_7, -SIN_6PI_7, COS_2PI_7, SIN_2PI_7, COS_4PI_7, -SIN_4PI_7,
                ),
                5 => (
                    COS_4PI_7, -SIN_4PI_7, COS_6PI_7, SIN_6PI_7, COS_2PI_7, SIN_2PI_7,
                ),
                6 => (
                    COS_2PI_7, -SIN_2PI_7, COS_4PI_7, -SIN_4PI_7, COS_6PI_7, -SIN_6PI_7,
                ),
                _ => unreachable!(),
            };

            let c_re = z0.re + cos1 * a1.re + cos2 * a2.re + cos3 * a3.re;
            let c_im = z0.im + cos1 * a1.im + cos2 * a2.im + cos3 * a3.im;
            let d_re = sin1 * b1_re + sin2 * b2_re + sin3 * b3_re;
            let d_im = sin1 * b1_im + sin2 * b2_im + sin3 * b3_im;

            dst[j + stride * idx] = Complex32::new(c_re + d_re, c_im + d_im);
        }
    }
}
