use crate::fft::Complex32;

/// Performs a single radix-4 Stockham butterfly stage for stride=1 (out-of-place, NEON).
///
/// This is a specialized version for the stride=1 case (first stage) that uses the zip pattern
/// to enable sequential SIMD stores. For stride=1, output indices are sequential: j=4*i.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix4_stride1_neon(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    use core::arch::aarch64::*;

    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples >> 1) << 1;

    // Sign flip mask for complex multiplication.
    #[repr(align(16))]
    struct AlignedMask([u32; 4]);
    const SIGN_FLIP_MASK: AlignedMask =
        AlignedMask([0x80000000, 0x00000000, 0x80000000, 0x00000000]);
    const NEG_IMAG_MASK: AlignedMask =
        AlignedMask([0x00000000, 0x80000000, 0x00000000, 0x80000000]);

    unsafe {
        let sign_flip = vreinterpretq_f32_u32(vld1q_u32(SIGN_FLIP_MASK.0.as_ptr()));
        let neg_imag = vreinterpretq_f32_u32(vld1q_u32(NEG_IMAG_MASK.0.as_ptr()));

        for i in (0..simd_iters).step_by(2) {
            // For stride=1: k=0, so j = 4*i (simplified indexing, outputs are sequential).

            // Load z0 from first quarter (2 complex numbers).
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = vld1q_f32(z0_ptr);

            // Load z1, z2, z3 from other quarters using contiguous loads.
            let z1_ptr = src.as_ptr().add(i + quarter_samples) as *const f32;
            let z1 = vld1q_f32(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + quarter_samples * 2) as *const f32;
            let z2 = vld1q_f32(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + quarter_samples * 3) as *const f32;
            let z3 = vld1q_f32(z3_ptr);

            // Load 6 twiddles contiguously (2 iterations × 3 twiddles each).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 3) as *const f32;
            let tw_01 = vld1q_f32(tw_ptr); // w1[0], w2[0]
            let tw_23 = vld1q_f32(tw_ptr.add(4)); // w3[0], w1[1]
            let tw_45 = vld1q_f32(tw_ptr.add(8)); // w2[1], w3[1]

            // Extract individual twiddles.
            let w1_low = vget_low_f32(tw_01); // w1[0]
            let w1_high = vget_high_f32(tw_23); // w1[1]
            let w1 = vcombine_f32(w1_low, w1_high);

            let w2_low = vget_high_f32(tw_01); // w2[0]
            let w2_high = vget_low_f32(tw_45); // w2[1]
            let w2 = vcombine_f32(w2_low, w2_high);

            let w3_low = vget_low_f32(tw_23); // w3[0]
            let w3_high = vget_high_f32(tw_45); // w3[1]
            let w3 = vcombine_f32(w3_low, w3_high);

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
            // TODO: Once ARMv8.3+ FCMLA instructions are stabilized, use vcmlaq_f32/vcmlaq_rot90_f32.
            let w3_transposed = vtrnq_f32(w3, w3);
            let w3_re_dup = w3_transposed.0;
            let w3_im_dup = w3_transposed.1;
            let z3_swap = vrev64q_f32(z3);
            let w3_im_signed = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(w3_im_dup),
                vreinterpretq_u32_f32(sign_flip),
            ));
            let t3 = vmlaq_f32(vmulq_f32(w3_re_dup, z3), w3_im_signed, z3_swap);

            // Radix-4 butterfly.
            let a0 = vaddq_f32(z0, t2);
            let a1 = vsubq_f32(z0, t2);
            let a2 = vaddq_f32(t1, t3);

            // a3 = i * (t1 - t3) = swap(t1 - t3) * [-1, 1, -1, 1]
            let t1_sub_t3 = vsubq_f32(t1, t3);
            let a3_swapped = vrev64q_f32(t1_sub_t3); // Swap re and im
            let a3 = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(a3_swapped),
                vreinterpretq_u32_f32(neg_imag),
            ));

            // Final butterfly outputs.
            let out0 = vaddq_f32(a0, a2);
            let out2 = vsubq_f32(a0, a2);
            let out1 = vaddq_f32(a1, a3);
            let out3 = vsubq_f32(a1, a3);

            // Apply zip pattern for sequential stores.
            // We have 4 outputs, each containing 2 complex numbers.
            // Need to interleave: [out0[0], out1[0], out2[0], out3[0], out0[1], out1[1], out2[1], out3[1]]

            let j = 4 * i;

            // Cast to f64 view to treat each complex number as a single 64-bit unit.
            let out0_f64 = vreinterpretq_f64_f32(out0);
            let out1_f64 = vreinterpretq_f64_f32(out1);
            let out2_f64 = vreinterpretq_f64_f32(out2);
            let out3_f64 = vreinterpretq_f64_f32(out3);

            // Perform 2x2 matrix transpose to get the interleaving we need.
            // First level: interleave pairs (out0, out1) and (out2, out3)
            let pair01_lo = vzip1q_f64(out0_f64, out1_f64); // [out0[0], out1[0]]
            let pair01_hi = vzip2q_f64(out0_f64, out1_f64); // [out0[1], out1[1]]
            let pair23_lo = vzip1q_f64(out2_f64, out3_f64); // [out2[0], out3[0]]
            let pair23_hi = vzip2q_f64(out2_f64, out3_f64); // [out2[1], out3[1]]

            // Now combine the lower and upper halves into 128-bit groups
            // We want: [out0[0], out1[0], out2[0], out3[0]] in first two stores
            // and:     [out0[1], out1[1], out2[1], out3[1]] in second two stores

            // Cast back to f32 to access as 128-bit vectors
            let pair01_lo_f32 = vreinterpretq_f32_f64(pair01_lo); // [out0[0].re, out0[0].im, out1[0].re, out1[0].im]
            let pair23_lo_f32 = vreinterpretq_f32_f64(pair23_lo); // [out2[0].re, out2[0].im, out3[0].re, out3[0].im]
            let pair01_hi_f32 = vreinterpretq_f32_f64(pair01_hi); // [out0[1].re, out0[1].im, out1[1].re, out1[1].im]
            let pair23_hi_f32 = vreinterpretq_f32_f64(pair23_hi); // [out2[1].re, out2[1].im, out3[1].re, out3[1].im]

            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            vst1q_f32(dst_ptr, pair01_lo_f32); // Store out0[0], out1[0]
            vst1q_f32(dst_ptr.add(4), pair23_lo_f32); // Store out2[0], out3[0]
            vst1q_f32(dst_ptr.add(8), pair01_hi_f32); // Store out0[1], out1[1]
            vst1q_f32(dst_ptr.add(12), pair23_hi_f32); // Store out2[1], out3[1]
        }
    }

    for i in simd_iters..quarter_samples {
        let w1 = stage_twiddles[i * 3];
        let w2 = stage_twiddles[i * 3 + 1];
        let w3 = stage_twiddles[i * 3 + 2];

        let z0 = src[i];
        let z1 = src[i + quarter_samples];
        let z2 = src[i + quarter_samples * 2];
        let z3 = src[i + quarter_samples * 3];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);

        let a0 = z0.add(&t2);
        let a1 = z0.sub(&t2);
        let a2 = t1.add(&t3);
        let a3_re = t1.im - t3.im;
        let a3_im = t3.re - t1.re;

        let j = 4 * i;
        dst[j] = a0.add(&a2);
        dst[j + 2] = a0.sub(&a2);
        dst[j + 1] = Complex32::new(a1.re + a3_re, a1.im + a3_im);
        dst[j + 3] = Complex32::new(a1.re - a3_re, a1.im - a3_im);
    }
}

/// Performs a single radix-4 Stockham butterfly stage for p>1 (out-of-place, NEON).
///
/// Generic version for p>1 cases. Uses direct SIMD stores, accepting non-sequential
/// stores as the shuffle overhead isn't justified for larger strides.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix4_generic_neon(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    use core::arch::aarch64::*;

    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples >> 1) << 1;

    // Sign flip mask for complex multiplication.
    #[repr(align(16))]
    struct AlignedMask([u32; 4]);
    const SIGN_FLIP_MASK: AlignedMask =
        AlignedMask([0x80000000, 0x00000000, 0x80000000, 0x00000000]);
    const NEG_IMAG_MASK: AlignedMask =
        AlignedMask([0x00000000, 0x80000000, 0x00000000, 0x80000000]);

    unsafe {
        let sign_flip = vreinterpretq_f32_u32(vld1q_u32(SIGN_FLIP_MASK.0.as_ptr()));
        let neg_imag = vreinterpretq_f32_u32(vld1q_u32(NEG_IMAG_MASK.0.as_ptr()));

        for i in (0..simd_iters).step_by(2) {
            // Calculate twiddle indices.
            let k0 = i % stride;
            let k1 = (i + 1) % stride;

            // Load z0 from first quarter.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = vld1q_f32(z0_ptr);

            // Load z1, z2, z3 from other quarters using contiguous loads.
            let z1_ptr = src.as_ptr().add(i + quarter_samples) as *const f32;
            let z1 = vld1q_f32(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + quarter_samples * 2) as *const f32;
            let z2 = vld1q_f32(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + quarter_samples * 3) as *const f32;
            let z3 = vld1q_f32(z3_ptr);

            // Load 6 twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 3) as *const f32;
            let tw_01 = vld1q_f32(tw_ptr);
            let tw_23 = vld1q_f32(tw_ptr.add(4));
            let tw_45 = vld1q_f32(tw_ptr.add(8));

            // Extract individual twiddles.
            let w1_low = vget_low_f32(tw_01);
            let w1_high = vget_high_f32(tw_23);
            let w1 = vcombine_f32(w1_low, w1_high);

            let w2_low = vget_high_f32(tw_01);
            let w2_high = vget_low_f32(tw_45);
            let w2 = vcombine_f32(w2_low, w2_high);

            let w3_low = vget_low_f32(tw_23);
            let w3_high = vget_high_f32(tw_45);
            let w3 = vcombine_f32(w3_low, w3_high);

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
            // TODO: Once ARMv8.3+ FCMLA instructions are stabilized, use vcmlaq_f32/vcmlaq_rot90_f32.
            let w3_transposed = vtrnq_f32(w3, w3);
            let w3_re_dup = w3_transposed.0;
            let w3_im_dup = w3_transposed.1;
            let z3_swap = vrev64q_f32(z3);
            let w3_im_signed = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(w3_im_dup),
                vreinterpretq_u32_f32(sign_flip),
            ));
            let t3 = vmlaq_f32(vmulq_f32(w3_re_dup, z3), w3_im_signed, z3_swap);

            // Radix-4 butterfly.
            let a0 = vaddq_f32(z0, t2);
            let a1 = vsubq_f32(z0, t2);
            let a2 = vaddq_f32(t1, t3);

            // a3 = i * (t1 - t3)
            let t1_sub_t3 = vsubq_f32(t1, t3);
            let a3_swapped = vrev64q_f32(t1_sub_t3);
            let a3 = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(a3_swapped),
                vreinterpretq_u32_f32(neg_imag),
            ));

            // Final butterfly outputs.
            let out0 = vaddq_f32(a0, a2);
            let out2 = vsubq_f32(a0, a2);
            let out1 = vaddq_f32(a1, a3);
            let out3 = vsubq_f32(a1, a3);

            // Calculate output indices.
            let j0 = 4 * i - 3 * k0;
            let j1 = 4 * (i + 1) - 3 * k1;

            // Extract and store each complex number separately using direct f32 stores.
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            let out0_0 = vget_low_f32(out0);
            let out1_0 = vget_low_f32(out1);
            let out2_0 = vget_low_f32(out2);
            let out3_0 = vget_low_f32(out3);

            vst1_f32(dst_ptr.add(j0 << 1), out0_0);
            vst1_f32(dst_ptr.add((j0 + stride) << 1), out1_0);
            vst1_f32(dst_ptr.add((j0 + stride * 2) << 1), out2_0);
            vst1_f32(dst_ptr.add((j0 + stride * 3) << 1), out3_0);

            let out0_1 = vget_high_f32(out0);
            let out1_1 = vget_high_f32(out1);
            let out2_1 = vget_high_f32(out2);
            let out3_1 = vget_high_f32(out3);

            vst1_f32(dst_ptr.add(j1 << 1), out0_1);
            vst1_f32(dst_ptr.add((j1 + stride) << 1), out1_1);
            vst1_f32(dst_ptr.add((j1 + stride * 2) << 1), out2_1);
            vst1_f32(dst_ptr.add((j1 + stride * 3) << 1), out3_1);
        }
    }

    for i in simd_iters..quarter_samples {
        let k = i % stride;
        let w1 = stage_twiddles[i * 3];
        let w2 = stage_twiddles[i * 3 + 1];
        let w3 = stage_twiddles[i * 3 + 2];

        let z0 = src[i];
        let z1 = src[i + quarter_samples];
        let z2 = src[i + quarter_samples * 2];
        let z3 = src[i + quarter_samples * 3];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);

        let a0 = z0.add(&t2);
        let a1 = z0.sub(&t2);
        let a2 = t1.add(&t3);
        let a3_re = t1.im - t3.im;
        let a3_im = t3.re - t1.re;

        let j = 4 * i - 3 * k;
        dst[j] = a0.add(&a2);
        dst[j + stride * 2] = a0.sub(&a2);
        dst[j + stride] = Complex32::new(a1.re + a3_re, a1.im + a3_im);
        dst[j + stride * 3] = Complex32::new(a1.re - a3_re, a1.im - a3_im);
    }
}
