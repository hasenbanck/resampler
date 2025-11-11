use core::arch::aarch64::*;

use super::{
    super::ops::{complex_mul, complex_mul_i, load_neg_imag_mask},
    COS_2PI_5, COS_4PI_5, SIN_2PI_5, SIN_4PI_5,
};
use crate::fft::Complex32;

/// Performs a single radix-5 Stockham butterfly stage for stride=1 (out-of-place, NEON).
///
/// This is a specialized version for the stride=1 case (first stage) that uses sequential stores
/// instead of scattered stores. When stride==1, output indices are sequential: j = 5*i, j+1, ..., j+4.
/// This provides significant performance benefits through write-combining and better cache utilization.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix5_stride1_neon(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let fifth_samples = samples / 5;
    let simd_iters = (fifth_samples >> 1) << 1;

    unsafe {
        let cos_2pi_5_vec = vdupq_n_f32(COS_2PI_5);
        let sin_2pi_5_vec = vdupq_n_f32(SIN_2PI_5);
        let cos_4pi_5_vec = vdupq_n_f32(COS_4PI_5);
        let sin_4pi_5_vec = vdupq_n_f32(SIN_4PI_5);

        let neg_imag = load_neg_imag_mask();

        for i in (0..simd_iters).step_by(2) {
            // Load z0 from first fifth.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = vld1q_f32(z0_ptr);

            // Load z1, z2, z3, z4 from other fifths using contiguous loads.
            let z1_ptr = src.as_ptr().add(i + fifth_samples) as *const f32;
            let z1 = vld1q_f32(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + fifth_samples * 2) as *const f32;
            let z2 = vld1q_f32(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + fifth_samples * 3) as *const f32;
            let z3 = vld1q_f32(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + fifth_samples * 4) as *const f32;
            let z4 = vld1q_f32(z4_ptr);

            // Stride=1 optimization: Skip twiddle loads and multiplications (identity twiddles).
            // t_k = (1+0i) * z_k = z_k
            let t1 = z1;
            let t2 = z2;
            let t3 = z3;
            let t4 = z4;

            // Radix-5 DFT.
            let sum_all = vaddq_f32(vaddq_f32(vaddq_f32(t1, t2), t3), t4);

            let a1 = vaddq_f32(t1, t4);
            let a2 = vaddq_f32(t2, t3);

            // b1 = i * (t1 - t4), b2 = i * (t2 - t3)
            let t1_sub_t4 = vsubq_f32(t1, t4);
            let t2_sub_t3 = vsubq_f32(t2, t3);
            let b1 = complex_mul_i(t1_sub_t4, neg_imag);
            let b2 = complex_mul_i(t2_sub_t3, neg_imag);

            // c1 = z0 + COS_2PI_5 * a1 + COS_4PI_5 * a2
            let c1 = vaddq_f32(
                z0,
                vaddq_f32(vmulq_f32(cos_2pi_5_vec, a1), vmulq_f32(cos_4pi_5_vec, a2)),
            );

            // c2 = z0 + COS_4PI_5 * a1 + COS_2PI_5 * a2
            let c2 = vaddq_f32(
                z0,
                vaddq_f32(vmulq_f32(cos_4pi_5_vec, a1), vmulq_f32(cos_2pi_5_vec, a2)),
            );

            // d1 = SIN_2PI_5 * b1 + SIN_4PI_5 * b2
            let d1 = vaddq_f32(vmulq_f32(sin_2pi_5_vec, b1), vmulq_f32(sin_4pi_5_vec, b2));

            // d2 = SIN_4PI_5 * b1 - SIN_2PI_5 * b2
            let d2 = vsubq_f32(vmulq_f32(sin_4pi_5_vec, b1), vmulq_f32(sin_2pi_5_vec, b2));

            // Final outputs.
            let out0 = vaddq_f32(z0, sum_all);
            let out1 = vaddq_f32(c1, d1);
            let out4 = vsubq_f32(c1, d1);
            let out2 = vaddq_f32(c2, d2);
            let out3 = vsubq_f32(c2, d2);

            let j = 5 * i;
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            let out0_0 = vget_low_f32(out0);
            let out1_0 = vget_low_f32(out1);
            let out2_0 = vget_low_f32(out2);
            let out3_0 = vget_low_f32(out3);
            let out4_0 = vget_low_f32(out4);

            vst1_f32(dst_ptr.add(j * 2), out0_0);
            vst1_f32(dst_ptr.add((j + 1) * 2), out1_0);
            vst1_f32(dst_ptr.add((j + 2) * 2), out2_0);
            vst1_f32(dst_ptr.add((j + 3) * 2), out3_0);
            vst1_f32(dst_ptr.add((j + 4) * 2), out4_0);

            let out0_1 = vget_high_f32(out0);
            let out1_1 = vget_high_f32(out1);
            let out2_1 = vget_high_f32(out2);
            let out3_1 = vget_high_f32(out3);
            let out4_1 = vget_high_f32(out4);

            let j1 = 5 * (i + 1);
            vst1_f32(dst_ptr.add(j1 * 2), out0_1);
            vst1_f32(dst_ptr.add((j1 + 1) * 2), out1_1);
            vst1_f32(dst_ptr.add((j1 + 2) * 2), out2_1);
            vst1_f32(dst_ptr.add((j1 + 3) * 2), out3_1);
            vst1_f32(dst_ptr.add((j1 + 4) * 2), out4_1);
        }
    }

    super::butterfly_radix5_scalar::<2>(src, dst, stage_twiddles, 1, simd_iters);
}

/// Performs a single radix-5 Stockham butterfly stage for p>1 (out-of-place, NEON).
///
/// This is the generic version for stride>1 cases. Uses direct SIMD stores,
/// accepting non-sequential stores as the shuffle overhead isn't justified.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix5_generic_neon(
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
        let cos_2pi_5_vec = vdupq_n_f32(COS_2PI_5);
        let sin_2pi_5_vec = vdupq_n_f32(SIN_2PI_5);
        let cos_4pi_5_vec = vdupq_n_f32(COS_4PI_5);
        let sin_4pi_5_vec = vdupq_n_f32(SIN_4PI_5);

        let neg_imag = load_neg_imag_mask();

        for i in (0..simd_iters).step_by(2) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;

            // Load z0 from first fifth.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = vld1q_f32(z0_ptr);

            // Load z1, z2, z3, z4 from other fifths using contiguous loads.
            let z1_ptr = src.as_ptr().add(i + fifth_samples) as *const f32;
            let z1 = vld1q_f32(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + fifth_samples * 2) as *const f32;
            let z2 = vld1q_f32(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + fifth_samples * 3) as *const f32;
            let z3 = vld1q_f32(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + fifth_samples * 4) as *const f32;
            let z4 = vld1q_f32(z4_ptr);

            // Load pre-packaged twiddles: [w1[i..i+2], w2[i..i+2], w3[i..i+2], w4[i..i+2]]
            let tw_ptr = stage_twiddles.as_ptr().add(i * 4) as *const f32;
            let w1 = vld1q_f32(tw_ptr); // w1[i], w1[i+1]
            let w2 = vld1q_f32(tw_ptr.add(4)); // w2[i], w2[i+1]
            let w3 = vld1q_f32(tw_ptr.add(8)); // w3[i], w3[i+1]
            let w4 = vld1q_f32(tw_ptr.add(12)); // w4[i], w4[i+1]

            // Complex multiply using optimized helper function.
            let t1 = complex_mul(z1, w1);
            let t2 = complex_mul(z2, w2);
            let t3 = complex_mul(z3, w3);
            let t4 = complex_mul(z4, w4);

            // Radix-5 DFT.
            let sum_all = vaddq_f32(vaddq_f32(vaddq_f32(t1, t2), t3), t4);

            let a1 = vaddq_f32(t1, t4);
            let a2 = vaddq_f32(t2, t3);

            // b1 = i * (t1 - t4), b2 = i * (t2 - t3)
            let t1_sub_t4 = vsubq_f32(t1, t4);
            let t2_sub_t3 = vsubq_f32(t2, t3);
            let b1 = complex_mul_i(t1_sub_t4, neg_imag);
            let b2 = complex_mul_i(t2_sub_t3, neg_imag);

            // c1 = z0 + COS_2PI_5 * a1 + COS_4PI_5 * a2
            let c1 = vaddq_f32(
                z0,
                vaddq_f32(vmulq_f32(cos_2pi_5_vec, a1), vmulq_f32(cos_4pi_5_vec, a2)),
            );

            // c2 = z0 + COS_4PI_5 * a1 + COS_2PI_5 * a2
            let c2 = vaddq_f32(
                z0,
                vaddq_f32(vmulq_f32(cos_4pi_5_vec, a1), vmulq_f32(cos_2pi_5_vec, a2)),
            );

            // d1 = SIN_2PI_5 * b1 + SIN_4PI_5 * b2
            let d1 = vaddq_f32(vmulq_f32(sin_2pi_5_vec, b1), vmulq_f32(sin_4pi_5_vec, b2));

            // d2 = SIN_4PI_5 * b1 - SIN_2PI_5 * b2
            let d2 = vsubq_f32(vmulq_f32(sin_4pi_5_vec, b1), vmulq_f32(sin_2pi_5_vec, b2));

            // Final outputs.
            let out0 = vaddq_f32(z0, sum_all);
            let out1 = vaddq_f32(c1, d1);
            let out4 = vsubq_f32(c1, d1);
            let out2 = vaddq_f32(c2, d2);
            let out3 = vsubq_f32(c2, d2);

            // Calculate output indices: j = 5*i - 4*k
            let j0 = 5 * i - 4 * k0;
            let j1 = 5 * (i + 1) - 4 * k1;

            // Extract and store each complex number separately using direct f32 stores.
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            let out0_0 = vget_low_f32(out0);
            let out1_0 = vget_low_f32(out1);
            let out2_0 = vget_low_f32(out2);
            let out3_0 = vget_low_f32(out3);
            let out4_0 = vget_low_f32(out4);

            vst1_f32(dst_ptr.add(j0 << 1), out0_0);
            vst1_f32(dst_ptr.add((j0 + stride) << 1), out1_0);
            vst1_f32(dst_ptr.add((j0 + stride * 2) << 1), out2_0);
            vst1_f32(dst_ptr.add((j0 + stride * 3) << 1), out3_0);
            vst1_f32(dst_ptr.add((j0 + stride * 4) << 1), out4_0);

            let out0_1 = vget_high_f32(out0);
            let out1_1 = vget_high_f32(out1);
            let out2_1 = vget_high_f32(out2);
            let out3_1 = vget_high_f32(out3);
            let out4_1 = vget_high_f32(out4);

            vst1_f32(dst_ptr.add(j1 << 1), out0_1);
            vst1_f32(dst_ptr.add((j1 + stride) << 1), out1_1);
            vst1_f32(dst_ptr.add((j1 + stride * 2) << 1), out2_1);
            vst1_f32(dst_ptr.add((j1 + stride * 3) << 1), out3_1);
            vst1_f32(dst_ptr.add((j1 + stride * 4) << 1), out4_1);
        }
    }

    super::butterfly_radix5_scalar::<2>(src, dst, stage_twiddles, stride, simd_iters);
}
