use super::{COS_2PI_5, COS_4PI_5, SIN_2PI_5, SIN_4PI_5};
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
    use core::arch::aarch64::*;

    let samples = src.len();
    let fifth_samples = samples / 5;
    let simd_iters = (fifth_samples >> 1) << 1;

    // Sign flip mask for complex multiplication: [-0.0, +0.0, -0.0, +0.0].
    #[repr(align(16))]
    struct AlignedMask([u32; 4]);
    const SIGN_FLIP_MASK: AlignedMask =
        AlignedMask([0x80000000, 0x00000000, 0x80000000, 0x00000000]);
    const NEG_IMAG_MASK: AlignedMask =
        AlignedMask([0x00000000, 0x80000000, 0x00000000, 0x80000000]);

    unsafe {
        let sign_flip = vreinterpretq_f32_u32(vld1q_u32(SIGN_FLIP_MASK.0.as_ptr()));
        let neg_imag = vreinterpretq_f32_u32(vld1q_u32(NEG_IMAG_MASK.0.as_ptr()));

        let cos_2pi_5_vec = vdupq_n_f32(COS_2PI_5);
        let sin_2pi_5_vec = vdupq_n_f32(SIN_2PI_5);
        let cos_4pi_5_vec = vdupq_n_f32(COS_4PI_5);
        let sin_4pi_5_vec = vdupq_n_f32(SIN_4PI_5);

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

            // Load 8 twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 4) as *const f32;
            let tw_01 = vld1q_f32(tw_ptr); // w1[0], w2[0]
            let tw_23 = vld1q_f32(tw_ptr.add(4)); // w3[0], w4[0]
            let tw_45 = vld1q_f32(tw_ptr.add(8)); // w1[1], w2[1]
            let tw_67 = vld1q_f32(tw_ptr.add(12)); // w3[1], w4[1]

            // Extract w1, w2, w3, w4.
            let w1_low = vget_low_f32(tw_01); // w1[0]
            let w1_high = vget_low_f32(tw_45); // w1[1]
            let w1 = vcombine_f32(w1_low, w1_high);

            let w2_low = vget_high_f32(tw_01); // w2[0]
            let w2_high = vget_high_f32(tw_45); // w2[1]
            let w2 = vcombine_f32(w2_low, w2_high);

            let w3_low = vget_low_f32(tw_23); // w3[0]
            let w3_high = vget_low_f32(tw_67); // w3[1]
            let w3 = vcombine_f32(w3_low, w3_high);

            let w4_low = vget_high_f32(tw_23); // w4[0]
            let w4_high = vget_high_f32(tw_67); // w4[1]
            let w4 = vcombine_f32(w4_low, w4_high);

            // Complex multiply: t1 = w1 * z1
            // TODO: Once ARMv8.3+ FCMLA instructions are stabilized, use vcmlaq_f32/vcmlaq_rot90_f32.
            let w1_transposed = vtrnq_f32(w1, w1);
            let w1_re_dup = w1_transposed.0;
            let w1_im_dup = w1_transposed.1;
            let z1_swap = vrev64q_f32(z1);
            let w1_im_signed = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(w1_im_dup),
                vreinterpretq_u32_f32(sign_flip),
            ));
            let t1 = vmlaq_f32(vmulq_f32(w1_re_dup, z1), w1_im_signed, z1_swap);

            // Complex multiply: t2 = w2 * z2
            // TODO: Once ARMv8.3+ FCMLA instructions are stabilized, use vcmlaq_f32/vcmlaq_rot90_f32.
            let w2_transposed = vtrnq_f32(w2, w2);
            let w2_re_dup = w2_transposed.0;
            let w2_im_dup = w2_transposed.1;
            let z2_swap = vrev64q_f32(z2);
            let w2_im_signed = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(w2_im_dup),
                vreinterpretq_u32_f32(sign_flip),
            ));
            let t2 = vmlaq_f32(vmulq_f32(w2_re_dup, z2), w2_im_signed, z2_swap);

            // Complex multiply: t3 = w3 * z3
            // TODO: Once ARMv8.3+ FCMLA instructions are stabilized, use vcmlaq_f32/vcmlaq_rot90_f32
            let w3_transposed = vtrnq_f32(w3, w3);
            let w3_re_dup = w3_transposed.0;
            let w3_im_dup = w3_transposed.1;
            let z3_swap = vrev64q_f32(z3);
            let w3_im_signed = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(w3_im_dup),
                vreinterpretq_u32_f32(sign_flip),
            ));
            let t3 = vmlaq_f32(vmulq_f32(w3_re_dup, z3), w3_im_signed, z3_swap);

            // Complex multiply: t4 = w4 * z4
            // TODO: Once ARMv8.3+ FCMLA instructions are stabilized, use vcmlaq_f32/vcmlaq_rot90_f32
            let w4_transposed = vtrnq_f32(w4, w4);
            let w4_re_dup = w4_transposed.0;
            let w4_im_dup = w4_transposed.1;
            let z4_swap = vrev64q_f32(z4);
            let w4_im_signed = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(w4_im_dup),
                vreinterpretq_u32_f32(sign_flip),
            ));
            let t4 = vmlaq_f32(vmulq_f32(w4_re_dup, z4), w4_im_signed, z4_swap);

            // Radix-5 DFT.
            let sum_all = vaddq_f32(vaddq_f32(vaddq_f32(t1, t2), t3), t4);

            let a1 = vaddq_f32(t1, t4);
            let a2 = vaddq_f32(t2, t3);

            // b1 = i * (t1 - t4), b2 = i * (t2 - t3)
            // i * (a + bi) = -b + ai, so swap and negate imaginary
            let t1_sub_t4 = vsubq_f32(t1, t4);
            let t2_sub_t3 = vsubq_f32(t2, t3);
            let b1_swapped = vrev64q_f32(t1_sub_t4);
            let b1 = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(b1_swapped),
                vreinterpretq_u32_f32(neg_imag),
            ));
            let b2_swapped = vrev64q_f32(t2_sub_t3);
            let b2 = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(b2_swapped),
                vreinterpretq_u32_f32(neg_imag),
            ));

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
    use core::arch::aarch64::*;

    let samples = src.len();
    let fifth_samples = samples / 5;
    let simd_iters = (fifth_samples >> 1) << 1;

    // Sign flip mask for complex multiplication: [-0.0, +0.0, -0.0, +0.0].
    #[repr(align(16))]
    struct AlignedMask([u32; 4]);
    const SIGN_FLIP_MASK: AlignedMask =
        AlignedMask([0x80000000, 0x00000000, 0x80000000, 0x00000000]);
    const NEG_IMAG_MASK: AlignedMask =
        AlignedMask([0x00000000, 0x80000000, 0x00000000, 0x80000000]);

    unsafe {
        let sign_flip = vreinterpretq_f32_u32(vld1q_u32(SIGN_FLIP_MASK.0.as_ptr()));
        let neg_imag = vreinterpretq_f32_u32(vld1q_u32(NEG_IMAG_MASK.0.as_ptr()));

        let cos_2pi_5_vec = vdupq_n_f32(COS_2PI_5);
        let sin_2pi_5_vec = vdupq_n_f32(SIN_2PI_5);
        let cos_4pi_5_vec = vdupq_n_f32(COS_4PI_5);
        let sin_4pi_5_vec = vdupq_n_f32(SIN_4PI_5);

        for i in (0..simd_iters).step_by(2) {
            let k0 = i % stride;
            let k1 = (i + 1) % stride;

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

            // Load 8 twiddles contiguously (2 iterations × 4 twiddles each).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 4) as *const f32;
            let tw_01 = vld1q_f32(tw_ptr); // w1[0], w2[0]
            let tw_23 = vld1q_f32(tw_ptr.add(4)); // w3[0], w4[0]
            let tw_45 = vld1q_f32(tw_ptr.add(8)); // w1[1], w2[1]
            let tw_67 = vld1q_f32(tw_ptr.add(12)); // w3[1], w4[1]

            // Extract w1, w2, w3, w4.
            let w1_low = vget_low_f32(tw_01); // w1[0]
            let w1_high = vget_low_f32(tw_45); // w1[1]
            let w1 = vcombine_f32(w1_low, w1_high);

            let w2_low = vget_high_f32(tw_01); // w2[0]
            let w2_high = vget_high_f32(tw_45); // w2[1]
            let w2 = vcombine_f32(w2_low, w2_high);

            let w3_low = vget_low_f32(tw_23); // w3[0]
            let w3_high = vget_low_f32(tw_67); // w3[1]
            let w3 = vcombine_f32(w3_low, w3_high);

            let w4_low = vget_high_f32(tw_23); // w4[0]
            let w4_high = vget_high_f32(tw_67); // w4[1]
            let w4 = vcombine_f32(w4_low, w4_high);

            // Complex multiply: t1 = w1 * z1
            // TODO: Once ARMv8.3+ FCMLA instructions are stabilized, use vcmlaq_f32/vcmlaq_rot90_f32.
            let w1_transposed = vtrnq_f32(w1, w1);
            let w1_re_dup = w1_transposed.0;
            let w1_im_dup = w1_transposed.1;
            let z1_swap = vrev64q_f32(z1);
            let w1_im_signed = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(w1_im_dup),
                vreinterpretq_u32_f32(sign_flip),
            ));
            let t1 = vmlaq_f32(vmulq_f32(w1_re_dup, z1), w1_im_signed, z1_swap);

            // Complex multiply: t2 = w2 * z2
            // TODO: Once ARMv8.3+ FCMLA instructions are stabilized, use vcmlaq_f32/vcmlaq_rot90_f32.
            let w2_transposed = vtrnq_f32(w2, w2);
            let w2_re_dup = w2_transposed.0;
            let w2_im_dup = w2_transposed.1;
            let z2_swap = vrev64q_f32(z2);
            let w2_im_signed = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(w2_im_dup),
                vreinterpretq_u32_f32(sign_flip),
            ));
            let t2 = vmlaq_f32(vmulq_f32(w2_re_dup, z2), w2_im_signed, z2_swap);

            // Complex multiply: t3 = w3 * z3
            // TODO: Once ARMv8.3+ FCMLA instructions are stabilized, use vcmlaq_f32/vcmlaq_rot90_f32
            let w3_transposed = vtrnq_f32(w3, w3);
            let w3_re_dup = w3_transposed.0;
            let w3_im_dup = w3_transposed.1;
            let z3_swap = vrev64q_f32(z3);
            let w3_im_signed = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(w3_im_dup),
                vreinterpretq_u32_f32(sign_flip),
            ));
            let t3 = vmlaq_f32(vmulq_f32(w3_re_dup, z3), w3_im_signed, z3_swap);

            // Complex multiply: t4 = w4 * z4
            // TODO: Once ARMv8.3+ FCMLA instructions are stabilized, use vcmlaq_f32/vcmlaq_rot90_f32
            let w4_transposed = vtrnq_f32(w4, w4);
            let w4_re_dup = w4_transposed.0;
            let w4_im_dup = w4_transposed.1;
            let z4_swap = vrev64q_f32(z4);
            let w4_im_signed = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(w4_im_dup),
                vreinterpretq_u32_f32(sign_flip),
            ));
            let t4 = vmlaq_f32(vmulq_f32(w4_re_dup, z4), w4_im_signed, z4_swap);

            // Radix-5 DFT.
            let sum_all = vaddq_f32(vaddq_f32(vaddq_f32(t1, t2), t3), t4);

            let a1 = vaddq_f32(t1, t4);
            let a2 = vaddq_f32(t2, t3);

            // b1 = i * (t1 - t4), b2 = i * (t2 - t3)
            // i * (a + bi) = -b + ai, so swap and negate imaginary
            let t1_sub_t4 = vsubq_f32(t1, t4);
            let t2_sub_t3 = vsubq_f32(t2, t3);
            let b1_swapped = vrev64q_f32(t1_sub_t4);
            let b1 = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(b1_swapped),
                vreinterpretq_u32_f32(neg_imag),
            ));
            let b2_swapped = vrev64q_f32(t2_sub_t3);
            let b2 = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(b2_swapped),
                vreinterpretq_u32_f32(neg_imag),
            ));

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
