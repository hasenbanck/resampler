use super::{COS_2PI_7, COS_4PI_7, COS_6PI_7, SIN_2PI_7, SIN_4PI_7, SIN_6PI_7};
use crate::fft::Complex32;

/// Performs a single radix-7 Stockham butterfly stage (out-of-place, AVX+FMA) for stride=1 (first stage).
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix7_stride1_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let seventh_samples = samples / 7;
    let simd_iters = (seventh_samples / 4) * 4;

    // Macro for complex multiplication: result = w * z.
    macro_rules! cmul_avx {
        ($w:expr, $z:expr) => {{
            let z_re = _mm256_moveldup_ps($z);
            let z_im = _mm256_movehdup_ps($z);
            let w_swap = _mm256_permute_ps($w, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w_swap, z_im);
            _mm256_fmaddsub_ps($w, z_re, prod_im)
        }};
    }

    // Macro for computing radix-7 outputs using pure SIMD FMA chains.
    macro_rules! compute_output_avx {
        ($z0:expr, $cos1:expr, $sin1:expr, $cos2:expr, $sin2:expr, $cos3:expr, $sin3:expr, $a1:expr, $a2:expr, $a3:expr, $b1:expr, $b2:expr, $b3:expr) => {{
            let c = _mm256_fmadd_ps(
                $cos3,
                $a3,
                _mm256_fmadd_ps($cos2, $a2, _mm256_fmadd_ps($cos1, $a1, $z0)),
            );
            let d = _mm256_fmadd_ps(
                $sin3,
                $b3,
                _mm256_fmadd_ps($sin2, $b2, _mm256_mul_ps($sin1, $b1)),
            );
            _mm256_add_ps(c, d)
        }};
    }

    unsafe {
        for i in (0..simd_iters).step_by(4) {
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + seventh_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + seventh_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + seventh_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + seventh_samples * 4) as *const f32;
            let z4 = _mm256_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + seventh_samples * 5) as *const f32;
            let z5 = _mm256_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + seventh_samples * 6) as *const f32;
            let z6 = _mm256_loadu_ps(z6_ptr);

            // Load 24 twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 6) as *const f32;
            let tw_0 = _mm256_loadu_ps(tw_ptr);
            let tw_1 = _mm256_loadu_ps(tw_ptr.add(8));
            let tw_2 = _mm256_loadu_ps(tw_ptr.add(16));
            let tw_3 = _mm256_loadu_ps(tw_ptr.add(24));
            let tw_4 = _mm256_loadu_ps(tw_ptr.add(32));
            let tw_5 = _mm256_loadu_ps(tw_ptr.add(40));

            // Extract w1, w2, w3, w4, w5, w6 from contiguous loads.
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

            // Preload all trigonometric constants.
            let cos_2pi_7 = _mm256_set1_ps(COS_2PI_7);
            let sin_2pi_7 = _mm256_set1_ps(SIN_2PI_7);
            let cos_4pi_7 = _mm256_set1_ps(COS_4PI_7);
            let sin_4pi_7 = _mm256_set1_ps(SIN_4PI_7);
            let cos_6pi_7 = _mm256_set1_ps(COS_6PI_7);
            let sin_6pi_7 = _mm256_set1_ps(SIN_6PI_7);
            let neg_sin_2pi_7 = _mm256_set1_ps(-SIN_2PI_7);
            let neg_sin_4pi_7 = _mm256_set1_ps(-SIN_4PI_7);
            let neg_sin_6pi_7 = _mm256_set1_ps(-SIN_6PI_7);

            // Complex multiply all twiddles.
            let t1 = cmul_avx!(w1, z1);
            let t2 = cmul_avx!(w2, z2);
            let t3 = cmul_avx!(w3, z3);
            let t4 = cmul_avx!(w4, z4);
            let t5 = cmul_avx!(w5, z5);
            let t6 = cmul_avx!(w6, z6);

            // DC component.
            let sum_12 = _mm256_add_ps(t1, t2);
            let sum_34 = _mm256_add_ps(t3, t4);
            let sum_56 = _mm256_add_ps(t5, t6);
            let sum_1234 = _mm256_add_ps(sum_12, sum_34);
            let sum_all = _mm256_add_ps(sum_1234, sum_56);

            // Radix-7 DFT decomposition.
            let a1 = _mm256_add_ps(t1, t6);
            let a2 = _mm256_add_ps(t2, t5);
            let a3 = _mm256_add_ps(t3, t4);

            let t1_sub_t6 = _mm256_sub_ps(t1, t6);
            let t2_sub_t5 = _mm256_sub_ps(t2, t5);
            let t3_sub_t4 = _mm256_sub_ps(t3, t4);

            let sign_mask = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
            let b1_swapped = _mm256_shuffle_ps(t1_sub_t6, t1_sub_t6, 0b10_11_00_01);
            let b1 = _mm256_xor_ps(b1_swapped, sign_mask);
            let b2_swapped = _mm256_shuffle_ps(t2_sub_t5, t2_sub_t5, 0b10_11_00_01);
            let b2 = _mm256_xor_ps(b2_swapped, sign_mask);
            let b3_swapped = _mm256_shuffle_ps(t3_sub_t4, t3_sub_t4, 0b10_11_00_01);
            let b3 = _mm256_xor_ps(b3_swapped, sign_mask);

            let out0 = _mm256_add_ps(z0, sum_all);

            let out1 = compute_output_avx!(
                z0, cos_2pi_7, sin_2pi_7, cos_4pi_7, sin_4pi_7, cos_6pi_7, sin_6pi_7, a1, a2, a3,
                b1, b2, b3
            );
            let out2 = compute_output_avx!(
                z0,
                cos_4pi_7,
                sin_4pi_7,
                cos_6pi_7,
                neg_sin_6pi_7,
                cos_2pi_7,
                neg_sin_2pi_7,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3
            );
            let out3 = compute_output_avx!(
                z0,
                cos_6pi_7,
                sin_6pi_7,
                cos_2pi_7,
                neg_sin_2pi_7,
                cos_4pi_7,
                sin_4pi_7,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3
            );
            let out4 = compute_output_avx!(
                z0,
                cos_6pi_7,
                neg_sin_6pi_7,
                cos_2pi_7,
                sin_2pi_7,
                cos_4pi_7,
                neg_sin_4pi_7,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3
            );
            let out5 = compute_output_avx!(
                z0,
                cos_4pi_7,
                neg_sin_4pi_7,
                cos_6pi_7,
                sin_6pi_7,
                cos_2pi_7,
                sin_2pi_7,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3
            );
            let out6 = compute_output_avx!(
                z0,
                cos_2pi_7,
                neg_sin_2pi_7,
                cos_4pi_7,
                neg_sin_4pi_7,
                cos_6pi_7,
                neg_sin_6pi_7,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3
            );

            // Interleave for sequential stores: [out0[0], out1[0], out2[0], out3[0], out4[0], out5[0], out6[0], ...]
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);
            let out4_pd = _mm256_castps_pd(out4);
            let out5_pd = _mm256_castps_pd(out5);
            let out6_pd = _mm256_castps_pd(out6);

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
            let out5_lo = _mm256_castpd256_pd128(out5_pd);
            let out5_hi = _mm256_extractf128_pd(out5_pd, 1);
            let out6_lo = _mm256_castpd256_pd128(out6_pd);
            let out6_hi = _mm256_extractf128_pd(out6_pd, 1);

            // Build result0: [out0[0], out1[0], out2[0], out3[0]]
            let temp0_lo = _mm_unpacklo_pd(out0_lo, out1_lo); // [out0[0], out1[0]]
            let temp0_hi = _mm_unpacklo_pd(out2_lo, out3_lo); // [out2[0], out3[0]]
            let result0 = _mm256_castpd_ps(_mm256_set_m128d(temp0_hi, temp0_lo));

            // Build result1: [out4[0], out5[0], out6[0], out0[1]]
            let temp1_lo = _mm_unpacklo_pd(out4_lo, out5_lo); // [out4[0], out5[0]]
            let temp1_hi = _mm_shuffle_pd::<2>(out6_lo, out0_lo); // [out6[0], out0[1]]
            let result1 = _mm256_castpd_ps(_mm256_set_m128d(temp1_hi, temp1_lo));

            // Build result2: [out1[1], out2[1], out3[1], out4[1]]
            let temp2_lo = _mm_unpackhi_pd(out1_lo, out2_lo); // [out1[1], out2[1]]
            let temp2_hi = _mm_unpackhi_pd(out3_lo, out4_lo); // [out3[1], out4[1]]
            let result2 = _mm256_castpd_ps(_mm256_set_m128d(temp2_hi, temp2_lo));

            // Build result3: [out5[1], out6[1], out0[2], out1[2]]
            let temp3_lo = _mm_unpackhi_pd(out5_lo, out6_lo); // [out5[1], out6[1]]
            let temp3_hi = _mm_unpacklo_pd(out0_hi, out1_hi); // [out0[2], out1[2]]
            let result3 = _mm256_castpd_ps(_mm256_set_m128d(temp3_hi, temp3_lo));

            // Build result4: [out2[2], out3[2], out4[2], out5[2]]
            let temp4_lo = _mm_unpacklo_pd(out2_hi, out3_hi); // [out2[2], out3[2]]
            let temp4_hi = _mm_unpacklo_pd(out4_hi, out5_hi); // [out4[2], out5[2]]
            let result4 = _mm256_castpd_ps(_mm256_set_m128d(temp4_hi, temp4_lo));

            // Build result5: [out6[2], out0[3], out1[3], out2[3]]
            let temp5_lo = _mm_shuffle_pd::<2>(out6_hi, out0_hi); // [out6[2], out0[3]]
            let temp5_hi = _mm_unpackhi_pd(out1_hi, out2_hi); // [out1[3], out2[3]]
            let result5 = _mm256_castpd_ps(_mm256_set_m128d(temp5_hi, temp5_lo));

            // Build result6: [out3[3], out4[3], out5[3], out6[3]]
            let temp6_lo = _mm_unpackhi_pd(out3_hi, out4_hi); // [out3[3], out4[3]]
            let temp6_hi = _mm_unpackhi_pd(out5_hi, out6_hi); // [out5[3], out6[3]]
            let result6 = _mm256_castpd_ps(_mm256_set_m128d(temp6_hi, temp6_lo));

            // Sequential stores.
            let j = 7 * i;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            _mm256_storeu_ps(dst_ptr, result0);
            _mm256_storeu_ps(dst_ptr.add(8), result1);
            _mm256_storeu_ps(dst_ptr.add(16), result2);
            _mm256_storeu_ps(dst_ptr.add(24), result3);
            _mm256_storeu_ps(dst_ptr.add(32), result4);
            _mm256_storeu_ps(dst_ptr.add(40), result5);
            _mm256_storeu_ps(dst_ptr.add(48), result6);
        }
    }

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

/// Performs a single radix-7 Stockham butterfly stage (out-of-place, AVX+FMA) for generic p.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix7_generic_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let seventh_samples = samples / 7;
    let simd_iters = (seventh_samples / 4) * 4;

    // Macro for complex multiplication: result = w * z
    macro_rules! cmul_avx {
        ($w:expr, $z:expr) => {{
            let z_re = _mm256_moveldup_ps($z);
            let z_im = _mm256_movehdup_ps($z);
            let w_swap = _mm256_permute_ps($w, 0b10_11_00_01);
            let prod_im = _mm256_mul_ps(w_swap, z_im);
            _mm256_fmaddsub_ps($w, z_re, prod_im)
        }};
    }

    // Macro for computing radix-7 outputs using pure SIMD FMA chains.
    // out = (z0 + cos1*a1 + cos2*a2 + cos3*a3) + (sin1*b1 + sin2*b2 + sin3*b3)
    macro_rules! compute_output_avx {
        ($z0:expr, $cos1:expr, $sin1:expr, $cos2:expr, $sin2:expr, $cos3:expr, $sin3:expr, $a1:expr, $a2:expr, $a3:expr, $b1:expr, $b2:expr, $b3:expr) => {{
            let c = _mm256_fmadd_ps(
                $cos3,
                $a3,
                _mm256_fmadd_ps($cos2, $a2, _mm256_fmadd_ps($cos1, $a1, $z0)),
            );
            let d = _mm256_fmadd_ps(
                $sin3,
                $b3,
                _mm256_fmadd_ps($sin2, $b2, _mm256_mul_ps($sin1, $b1)),
            );
            _mm256_add_ps(c, d)
        }};
    }

    unsafe {
        for i in (0..simd_iters).step_by(4) {
            let k0 = i % stride;
            let k1 = (i + 1) % stride;
            let k2 = (i + 2) % stride;
            let k3 = (i + 3) % stride;

            // Load z0.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + seventh_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + seventh_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + seventh_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + seventh_samples * 4) as *const f32;
            let z4 = _mm256_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + seventh_samples * 5) as *const f32;
            let z5 = _mm256_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + seventh_samples * 6) as *const f32;
            let z6 = _mm256_loadu_ps(z6_ptr);

            // Load 24 twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 6) as *const f32;
            let tw_0 = _mm256_loadu_ps(tw_ptr);
            let tw_1 = _mm256_loadu_ps(tw_ptr.add(8));
            let tw_2 = _mm256_loadu_ps(tw_ptr.add(16));
            let tw_3 = _mm256_loadu_ps(tw_ptr.add(24));
            let tw_4 = _mm256_loadu_ps(tw_ptr.add(32));
            let tw_5 = _mm256_loadu_ps(tw_ptr.add(40));

            // Extract w1, w2, w3, w4, w5, w6 from contiguous loads.
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

            // Preload all trigonometric constants early to mask memory latency.
            let cos_2pi_7 = _mm256_set1_ps(COS_2PI_7);
            let sin_2pi_7 = _mm256_set1_ps(SIN_2PI_7);
            let cos_4pi_7 = _mm256_set1_ps(COS_4PI_7);
            let sin_4pi_7 = _mm256_set1_ps(SIN_4PI_7);
            let cos_6pi_7 = _mm256_set1_ps(COS_6PI_7);
            let sin_6pi_7 = _mm256_set1_ps(SIN_6PI_7);
            let neg_sin_2pi_7 = _mm256_set1_ps(-SIN_2PI_7);
            let neg_sin_4pi_7 = _mm256_set1_ps(-SIN_4PI_7);
            let neg_sin_6pi_7 = _mm256_set1_ps(-SIN_6PI_7);

            // Complex multiply all twiddles using macro.
            let t1 = cmul_avx!(w1, z1);
            let t2 = cmul_avx!(w2, z2);
            let t3 = cmul_avx!(w3, z3);
            let t4 = cmul_avx!(w4, z4);
            let t5 = cmul_avx!(w5, z5);
            let t6 = cmul_avx!(w6, z6);

            // DC component - balanced tree for lower latency.
            let sum_12 = _mm256_add_ps(t1, t2);
            let sum_34 = _mm256_add_ps(t3, t4);
            let sum_56 = _mm256_add_ps(t5, t6);
            let sum_1234 = _mm256_add_ps(sum_12, sum_34);
            let sum_all = _mm256_add_ps(sum_1234, sum_56);

            // Radix-7 DFT decomposition.
            let a1 = _mm256_add_ps(t1, t6);
            let a2 = _mm256_add_ps(t2, t5);
            let a3 = _mm256_add_ps(t3, t4);

            // b = i * (t1 - t6), etc.
            let t1_sub_t6 = _mm256_sub_ps(t1, t6);
            let t2_sub_t5 = _mm256_sub_ps(t2, t5);
            let t3_sub_t4 = _mm256_sub_ps(t3, t4);

            let sign_mask = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
            let b1_swapped = _mm256_shuffle_ps(t1_sub_t6, t1_sub_t6, 0b10_11_00_01);
            let b1 = _mm256_xor_ps(b1_swapped, sign_mask);
            let b2_swapped = _mm256_shuffle_ps(t2_sub_t5, t2_sub_t5, 0b10_11_00_01);
            let b2 = _mm256_xor_ps(b2_swapped, sign_mask);
            let b3_swapped = _mm256_shuffle_ps(t3_sub_t4, t3_sub_t4, 0b10_11_00_01);
            let b3 = _mm256_xor_ps(b3_swapped, sign_mask);

            // Output indices.
            let j0 = 7 * i - 6 * k0;
            let j1 = 7 * (i + 1) - 6 * k1;
            let j2 = 7 * (i + 2) - 6 * k2;
            let j3 = 7 * (i + 3) - 6 * k3;

            let out0 = _mm256_add_ps(z0, sum_all);

            let out1 = compute_output_avx!(
                z0, cos_2pi_7, sin_2pi_7, cos_4pi_7, sin_4pi_7, cos_6pi_7, sin_6pi_7, a1, a2, a3,
                b1, b2, b3
            );
            let out2 = compute_output_avx!(
                z0,
                cos_4pi_7,
                sin_4pi_7,
                cos_6pi_7,
                neg_sin_6pi_7,
                cos_2pi_7,
                neg_sin_2pi_7,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3
            );
            let out3 = compute_output_avx!(
                z0,
                cos_6pi_7,
                sin_6pi_7,
                cos_2pi_7,
                neg_sin_2pi_7,
                cos_4pi_7,
                sin_4pi_7,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3
            );
            let out4 = compute_output_avx!(
                z0,
                cos_6pi_7,
                neg_sin_6pi_7,
                cos_2pi_7,
                sin_2pi_7,
                cos_4pi_7,
                neg_sin_4pi_7,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3
            );
            let out5 = compute_output_avx!(
                z0,
                cos_4pi_7,
                neg_sin_4pi_7,
                cos_6pi_7,
                sin_6pi_7,
                cos_2pi_7,
                sin_2pi_7,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3
            );
            let out6 = compute_output_avx!(
                z0,
                cos_2pi_7,
                neg_sin_2pi_7,
                cos_4pi_7,
                neg_sin_4pi_7,
                cos_6pi_7,
                neg_sin_6pi_7,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3
            );

            // Direct SIMD stores.
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);
            let out4_pd = _mm256_castps_pd(out4);
            let out5_pd = _mm256_castps_pd(out5);
            let out6_pd = _mm256_castps_pd(out6);

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

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            // Store iteration 0.
            _mm_storel_pd(dst_ptr.add(j0), out0_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 3), out3_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 4), out4_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 5), out5_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 6), out6_lo);

            // Store iteration 1.
            _mm_storeh_pd(dst_ptr.add(j1), out0_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 3), out3_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 4), out4_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 5), out5_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 6), out6_lo);

            // Store iteration 2.
            _mm_storel_pd(dst_ptr.add(j2), out0_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride), out1_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 2), out2_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 3), out3_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 4), out4_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 5), out5_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 6), out6_hi);

            // Store iteration 3.
            _mm_storeh_pd(dst_ptr.add(j3), out0_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride), out1_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 2), out2_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 3), out3_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 4), out4_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 5), out5_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 6), out6_hi);
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
