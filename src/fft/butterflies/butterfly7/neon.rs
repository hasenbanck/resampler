use core::arch::aarch64::*;

use super::{
    super::ops::{complex_mul, complex_mul_i, load_neg_imag_mask},
    COS_2PI_7, COS_4PI_7, COS_6PI_7, SIN_2PI_7, SIN_4PI_7, SIN_6PI_7,
};
use crate::fft::Complex32;

/// Performs a single radix-7 Stockham butterfly stage for stride=1 (out-of-place, NEON).
///
/// This is a specialized version for the stride=1 case (first stage) that uses sequential stores
/// instead of scattered stores. When stride==1, output indices are sequential: j = 7*i, j+1, ..., j+6.
/// This provides significant performance benefits through write-combining and better cache utilization.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix7_stride1_neon(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let seventh_samples = samples / 7;
    let simd_iters = (seventh_samples / 2) * 2;

    unsafe {
        let cos_2pi_7 = vdupq_n_f32(COS_2PI_7);
        let sin_2pi_7 = vdupq_n_f32(SIN_2PI_7);
        let cos_4pi_7 = vdupq_n_f32(COS_4PI_7);
        let sin_4pi_7 = vdupq_n_f32(SIN_4PI_7);
        let cos_6pi_7 = vdupq_n_f32(COS_6PI_7);
        let sin_6pi_7 = vdupq_n_f32(SIN_6PI_7);

        let neg_sin_2pi_7 = vdupq_n_f32(-SIN_2PI_7);
        let neg_sin_4pi_7 = vdupq_n_f32(-SIN_4PI_7);
        let neg_sin_6pi_7 = vdupq_n_f32(-SIN_6PI_7);

        let neg_imag = load_neg_imag_mask();

        for i in (0..simd_iters).step_by(2) {
            // Load z0 from first seventh (contiguous).
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = vld1q_f32(z0_ptr);

            // Load z1-z6 from other sevenths using contiguous loads.
            let z1_ptr = src.as_ptr().add(i + seventh_samples) as *const f32;
            let z1 = vld1q_f32(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + seventh_samples * 2) as *const f32;
            let z2 = vld1q_f32(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + seventh_samples * 3) as *const f32;
            let z3 = vld1q_f32(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + seventh_samples * 4) as *const f32;
            let z4 = vld1q_f32(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + seventh_samples * 5) as *const f32;
            let z5 = vld1q_f32(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + seventh_samples * 6) as *const f32;
            let z6 = vld1q_f32(z6_ptr);

            // Load pre-packaged twiddles: [w1[i..i+2], w2[i..i+2], ..., w6[i..i+2]]
            let tw_ptr = stage_twiddles.as_ptr().add(i * 6) as *const f32;
            let w1 = vld1q_f32(tw_ptr); // w1[i], w1[i+1]
            let w2 = vld1q_f32(tw_ptr.add(4)); // w2[i], w2[i+1]
            let w3 = vld1q_f32(tw_ptr.add(8)); // w3[i], w3[i+1]
            let w4 = vld1q_f32(tw_ptr.add(12)); // w4[i], w4[i+1]
            let w5 = vld1q_f32(tw_ptr.add(16)); // w5[i], w5[i+1]
            let w6 = vld1q_f32(tw_ptr.add(20)); // w6[i], w6[i+1]

            let t1 = complex_mul(w1, z1);
            let t2 = complex_mul(w2, z2);
            let t3 = complex_mul(w3, z3);
            let t4 = complex_mul(w4, z4);
            let t5 = complex_mul(w5, z5);
            let t6 = complex_mul(w6, z6);

            // Interleave sum and difference computations to reduce dependency chains.
            // Compute a1, a2, a3 (sums) in parallel with dependency preparation.
            let a1 = vaddq_f32(t1, t6);
            let a2 = vaddq_f32(t2, t5);
            let a3 = vaddq_f32(t3, t4);

            // Compute differences for b1, b2, b3 (independent of sums).
            let t1_sub_t6 = vsubq_f32(t1, t6);
            let t2_sub_t5 = vsubq_f32(t2, t5);
            let t3_sub_t4 = vsubq_f32(t3, t4);

            // Compute sum_all using balanced tree to reduce latency.
            let sum_12 = vaddq_f32(t1, t2);
            let sum_34 = vaddq_f32(t3, t4);
            let sum_56 = vaddq_f32(t5, t6);
            let sum_1234 = vaddq_f32(sum_12, sum_34);
            let sum_all = vaddq_f32(sum_1234, sum_56);

            // b_k = i * (t_k - t_{7-k}) for k=1,2,3
            let b1 = complex_mul_i(t1_sub_t6, neg_imag);
            let b2 = complex_mul_i(t2_sub_t5, neg_imag);
            let b3 = complex_mul_i(t3_sub_t4, neg_imag);

            // Macro to compute each output in NEON.
            macro_rules! compute_output {
                ($cos1:expr, $sin1:expr, $cos2:expr, $sin2:expr, $cos3:expr, $sin3:expr) => {{
                    let c = vmlaq_f32(vmlaq_f32(vmlaq_f32(z0, $cos1, a1), $cos2, a2), $cos3, a3);
                    let d = vmlaq_f32(vmlaq_f32(vmulq_f32($sin1, b1), $sin2, b2), $sin3, b3);
                    vaddq_f32(c, d)
                }};
            }

            let out0 = vaddq_f32(z0, sum_all);
            let out1 = compute_output!(
                cos_2pi_7, sin_2pi_7, cos_4pi_7, sin_4pi_7, cos_6pi_7, sin_6pi_7
            );
            let out2 = compute_output!(
                cos_4pi_7,
                sin_4pi_7,
                cos_6pi_7,
                neg_sin_6pi_7,
                cos_2pi_7,
                neg_sin_2pi_7
            );
            let out3 = compute_output!(
                cos_6pi_7,
                sin_6pi_7,
                cos_2pi_7,
                neg_sin_2pi_7,
                cos_4pi_7,
                sin_4pi_7
            );
            let out4 = compute_output!(
                cos_6pi_7,
                neg_sin_6pi_7,
                cos_2pi_7,
                sin_2pi_7,
                cos_4pi_7,
                neg_sin_4pi_7
            );
            let out5 = compute_output!(
                cos_4pi_7,
                neg_sin_4pi_7,
                cos_6pi_7,
                sin_6pi_7,
                cos_2pi_7,
                sin_2pi_7
            );
            let out6 = compute_output!(
                cos_2pi_7,
                neg_sin_2pi_7,
                cos_4pi_7,
                neg_sin_4pi_7,
                cos_6pi_7,
                neg_sin_6pi_7
            );

            // Sequential stores for stride=1 case.
            // For iteration i, outputs go to indices [7*i, 7*i+1, ..., 7*i+6]
            let j = 7 * i;
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            let out0_0 = vget_low_f32(out0);
            let out1_0 = vget_low_f32(out1);
            let out2_0 = vget_low_f32(out2);
            let out3_0 = vget_low_f32(out3);
            let out4_0 = vget_low_f32(out4);
            let out5_0 = vget_low_f32(out5);
            let out6_0 = vget_low_f32(out6);

            vst1_f32(dst_ptr.add(j * 2), out0_0);
            vst1_f32(dst_ptr.add((j + 1) * 2), out1_0);
            vst1_f32(dst_ptr.add((j + 2) * 2), out2_0);
            vst1_f32(dst_ptr.add((j + 3) * 2), out3_0);
            vst1_f32(dst_ptr.add((j + 4) * 2), out4_0);
            vst1_f32(dst_ptr.add((j + 5) * 2), out5_0);
            vst1_f32(dst_ptr.add((j + 6) * 2), out6_0);

            let out0_1 = vget_high_f32(out0);
            let out1_1 = vget_high_f32(out1);
            let out2_1 = vget_high_f32(out2);
            let out3_1 = vget_high_f32(out3);
            let out4_1 = vget_high_f32(out4);
            let out5_1 = vget_high_f32(out5);
            let out6_1 = vget_high_f32(out6);

            let j1 = 7 * (i + 1);
            vst1_f32(dst_ptr.add(j1 * 2), out0_1);
            vst1_f32(dst_ptr.add((j1 + 1) * 2), out1_1);
            vst1_f32(dst_ptr.add((j1 + 2) * 2), out2_1);
            vst1_f32(dst_ptr.add((j1 + 3) * 2), out3_1);
            vst1_f32(dst_ptr.add((j1 + 4) * 2), out4_1);
            vst1_f32(dst_ptr.add((j1 + 5) * 2), out5_1);
            vst1_f32(dst_ptr.add((j1 + 6) * 2), out6_1);
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

        macro_rules! compute_out_scalar {
            ($idx:expr, $cos1:expr, $sin1:expr, $cos2:expr, $sin2:expr, $cos3:expr, $sin3:expr) => {
                let c_re = z0.re + $cos1 * a1.re + $cos2 * a2.re + $cos3 * a3.re;
                let c_im = z0.im + $cos1 * a1.im + $cos2 * a2.im + $cos3 * a3.im;
                let d_re = $sin1 * b1_re + $sin2 * b2_re + $sin3 * b3_re;
                let d_im = $sin1 * b1_im + $sin2 * b2_im + $sin3 * b3_im;
                dst[j + $idx] = Complex32::new(c_re + d_re, c_im + d_im);
            };
        }

        compute_out_scalar!(
            1, COS_2PI_7, SIN_2PI_7, COS_4PI_7, SIN_4PI_7, COS_6PI_7, SIN_6PI_7
        );
        compute_out_scalar!(
            2, COS_4PI_7, SIN_4PI_7, COS_6PI_7, -SIN_6PI_7, COS_2PI_7, -SIN_2PI_7
        );
        compute_out_scalar!(
            3, COS_6PI_7, SIN_6PI_7, COS_2PI_7, -SIN_2PI_7, COS_4PI_7, SIN_4PI_7
        );
        compute_out_scalar!(
            4, COS_6PI_7, -SIN_6PI_7, COS_2PI_7, SIN_2PI_7, COS_4PI_7, -SIN_4PI_7
        );
        compute_out_scalar!(
            5, COS_4PI_7, -SIN_4PI_7, COS_6PI_7, SIN_6PI_7, COS_2PI_7, SIN_2PI_7
        );
        compute_out_scalar!(
            6, COS_2PI_7, -SIN_2PI_7, COS_4PI_7, -SIN_4PI_7, COS_6PI_7, -SIN_6PI_7
        );
    }
}

/// Performs a single radix-7 Stockham butterfly stage for p>1 (out-of-place, NEON).
///
/// This is the generic version for stride>1 cases. Uses direct SIMD stores,
/// accepting non-sequential stores as the shuffle overhead isn't justified.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix7_generic_neon(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    // We convince the compiler here that stride can't be 0 to optimize better.
    let stride = stride.max(1);

    let samples = src.len();
    let seventh_samples = samples / 7;
    let simd_iters = (seventh_samples / 2) * 2;

    unsafe {
        let cos_2pi_7 = vdupq_n_f32(COS_2PI_7);
        let sin_2pi_7 = vdupq_n_f32(SIN_2PI_7);
        let cos_4pi_7 = vdupq_n_f32(COS_4PI_7);
        let sin_4pi_7 = vdupq_n_f32(SIN_4PI_7);
        let cos_6pi_7 = vdupq_n_f32(COS_6PI_7);
        let sin_6pi_7 = vdupq_n_f32(SIN_6PI_7);

        let neg_sin_2pi_7 = vdupq_n_f32(-SIN_2PI_7);
        let neg_sin_4pi_7 = vdupq_n_f32(-SIN_4PI_7);
        let neg_sin_6pi_7 = vdupq_n_f32(-SIN_6PI_7);

        let neg_imag = load_neg_imag_mask();

        for i in (0..simd_iters).step_by(2) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;

            // Load z0 from first seventh.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = vld1q_f32(z0_ptr);

            // Load z1-z6 from other sevenths using contiguous loads.
            let z1_ptr = src.as_ptr().add(i + seventh_samples) as *const f32;
            let z1 = vld1q_f32(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + seventh_samples * 2) as *const f32;
            let z2 = vld1q_f32(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + seventh_samples * 3) as *const f32;
            let z3 = vld1q_f32(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + seventh_samples * 4) as *const f32;
            let z4 = vld1q_f32(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + seventh_samples * 5) as *const f32;
            let z5 = vld1q_f32(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + seventh_samples * 6) as *const f32;
            let z6 = vld1q_f32(z6_ptr);

            // Load pre-packaged twiddles: [w1[i..i+2], w2[i..i+2], ..., w6[i..i+2]]
            let tw_ptr = stage_twiddles.as_ptr().add(i * 6) as *const f32;
            let w1 = vld1q_f32(tw_ptr); // w1[i], w1[i+1]
            let w2 = vld1q_f32(tw_ptr.add(4)); // w2[i], w2[i+1]
            let w3 = vld1q_f32(tw_ptr.add(8)); // w3[i], w3[i+1]
            let w4 = vld1q_f32(tw_ptr.add(12)); // w4[i], w4[i+1]
            let w5 = vld1q_f32(tw_ptr.add(16)); // w5[i], w5[i+1]
            let w6 = vld1q_f32(tw_ptr.add(20)); // w6[i], w6[i+1]

            let t1 = complex_mul(z1, w1);
            let t2 = complex_mul(z2, w2);
            let t3 = complex_mul(z3, w3);
            let t4 = complex_mul(z4, w4);
            let t5 = complex_mul(z5, w5);
            let t6 = complex_mul(z6, w6);

            // Interleave sum and difference computations to reduce dependency chains.
            let a1 = vaddq_f32(t1, t6);
            let a2 = vaddq_f32(t2, t5);
            let a3 = vaddq_f32(t3, t4);

            // Compute differences for b1, b2, b3 (independent of sums).
            let t1_sub_t6 = vsubq_f32(t1, t6);
            let t2_sub_t5 = vsubq_f32(t2, t5);
            let t3_sub_t4 = vsubq_f32(t3, t4);

            // Compute sum_all using balanced tree to reduce latency.
            let sum_12 = vaddq_f32(t1, t2);
            let sum_34 = vaddq_f32(t3, t4);
            let sum_56 = vaddq_f32(t5, t6);
            let sum_1234 = vaddq_f32(sum_12, sum_34);
            let sum_all = vaddq_f32(sum_1234, sum_56);

            // b = i * (t - t'), compute for later use.
            let b1 = complex_mul_i(t1_sub_t6, neg_imag);
            let b2 = complex_mul_i(t2_sub_t5, neg_imag);
            let b3 = complex_mul_i(t3_sub_t4, neg_imag);

            // Output indices.
            let j0 = 7 * i - 6 * k0;
            let j1 = 7 * (i + 1) - 6 * k1;

            // Macro to compute each output in NEON.
            // Formula: out = (z0 + cos1*a1 + cos2*a2 + cos3*a3) + (sin1*b1 + sin2*b2 + sin3*b3)
            macro_rules! compute_output {
                ($cos1:expr, $sin1:expr, $cos2:expr, $sin2:expr, $cos3:expr, $sin3:expr) => {{
                    // Compute c = z0 + cos1*a1 + cos2*a2 + cos3*a3
                    let c = vmlaq_f32(vmlaq_f32(vmlaq_f32(z0, $cos1, a1), $cos2, a2), $cos3, a3);
                    // Compute d = sin1*b1 + sin2*b2 + sin3*b3
                    let d = vmlaq_f32(vmlaq_f32(vmulq_f32($sin1, b1), $sin2, b2), $sin3, b3);
                    // Result: out = c + d
                    vaddq_f32(c, d)
                }};
            }

            // out0 = z0 + sum_all
            let out0 = vaddq_f32(z0, sum_all);

            // out1-6 using the macro.
            let out1 = compute_output!(
                cos_2pi_7, sin_2pi_7, cos_4pi_7, sin_4pi_7, cos_6pi_7, sin_6pi_7
            );
            let out2 = compute_output!(
                cos_4pi_7,
                sin_4pi_7,
                cos_6pi_7,
                neg_sin_6pi_7,
                cos_2pi_7,
                neg_sin_2pi_7
            );
            let out3 = compute_output!(
                cos_6pi_7,
                sin_6pi_7,
                cos_2pi_7,
                neg_sin_2pi_7,
                cos_4pi_7,
                sin_4pi_7
            );
            let out4 = compute_output!(
                cos_6pi_7,
                neg_sin_6pi_7,
                cos_2pi_7,
                sin_2pi_7,
                cos_4pi_7,
                neg_sin_4pi_7
            );
            let out5 = compute_output!(
                cos_4pi_7,
                neg_sin_4pi_7,
                cos_6pi_7,
                sin_6pi_7,
                cos_2pi_7,
                sin_2pi_7
            );
            let out6 = compute_output!(
                cos_2pi_7,
                neg_sin_2pi_7,
                cos_4pi_7,
                neg_sin_4pi_7,
                cos_6pi_7,
                neg_sin_6pi_7
            );

            // Store all 7 outputs using direct f32 stores.
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            vst1_f32(dst_ptr.add(j0 << 1), vget_low_f32(out0));
            vst1_f32(dst_ptr.add(j1 << 1), vget_high_f32(out0));

            vst1_f32(dst_ptr.add((j0 + stride) << 1), vget_low_f32(out1));
            vst1_f32(dst_ptr.add((j1 + stride) << 1), vget_high_f32(out1));

            vst1_f32(dst_ptr.add((j0 + stride * 2) << 1), vget_low_f32(out2));
            vst1_f32(dst_ptr.add((j1 + stride * 2) << 1), vget_high_f32(out2));

            vst1_f32(dst_ptr.add((j0 + stride * 3) << 1), vget_low_f32(out3));
            vst1_f32(dst_ptr.add((j1 + stride * 3) << 1), vget_high_f32(out3));

            vst1_f32(dst_ptr.add((j0 + stride * 4) << 1), vget_low_f32(out4));
            vst1_f32(dst_ptr.add((j1 + stride * 4) << 1), vget_high_f32(out4));

            vst1_f32(dst_ptr.add((j0 + stride * 5) << 1), vget_low_f32(out5));
            vst1_f32(dst_ptr.add((j1 + stride * 5) << 1), vget_high_f32(out5));

            vst1_f32(dst_ptr.add((j0 + stride * 6) << 1), vget_low_f32(out6));
            vst1_f32(dst_ptr.add((j1 + stride * 6) << 1), vget_high_f32(out6));
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
