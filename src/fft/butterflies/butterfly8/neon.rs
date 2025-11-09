use core::{arch::aarch64::*, f32::consts::FRAC_1_SQRT_2};

use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul, complex_mul_i, load_neg_imag_mask},
};

/// Performs a single radix-8 Stockham butterfly stage for stride=1 (out-of-place, NEON).
///
/// This is a specialized version for the stride=1 case (first stage) that uses sequential
/// stores. For stride=1, output indices are sequential: j=8*i.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix8_stride1_neon(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;
    let simd_iters = (eighth_samples >> 1) << 1;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask();

        let w8_1 = vreinterpretq_f32_u32(vld1q_u32(
            [
                FRAC_1_SQRT_2.to_bits(),
                (-FRAC_1_SQRT_2).to_bits(),
                FRAC_1_SQRT_2.to_bits(),
                (-FRAC_1_SQRT_2).to_bits(),
            ]
            .as_ptr(),
        ));

        let w8_3 = vdupq_n_f32(-FRAC_1_SQRT_2);

        for i in (0..simd_iters).step_by(2) {
            // Load 2 complex numbers from each eighth.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = vld1q_f32(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + eighth_samples) as *const f32;
            let z1 = vld1q_f32(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + eighth_samples * 2) as *const f32;
            let z2 = vld1q_f32(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + eighth_samples * 3) as *const f32;
            let z3 = vld1q_f32(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + eighth_samples * 4) as *const f32;
            let z4 = vld1q_f32(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + eighth_samples * 5) as *const f32;
            let z5 = vld1q_f32(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + eighth_samples * 6) as *const f32;
            let z6 = vld1q_f32(z6_ptr);

            let z7_ptr = src.as_ptr().add(i + eighth_samples * 7) as *const f32;
            let z7 = vld1q_f32(z7_ptr);

            // Load prepackaged twiddles in packed format (2 complex per load, stride 4 floats).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 7) as *const f32;
            let w1 = vld1q_f32(tw_ptr); // w1[i], w1[i+1]
            let w2 = vld1q_f32(tw_ptr.add(4)); // w2[i], w2[i+1]
            let w3 = vld1q_f32(tw_ptr.add(8)); // w3[i], w3[i+1]
            let w4 = vld1q_f32(tw_ptr.add(12)); // w4[i], w4[i+1]
            let w5 = vld1q_f32(tw_ptr.add(16)); // w5[i], w5[i+1]
            let w6 = vld1q_f32(tw_ptr.add(20)); // w6[i], w6[i+1]
            let w7 = vld1q_f32(tw_ptr.add(24)); // w7[i], w7[i+1]

            // Apply twiddle factors.
            let t1 = complex_mul(w1, z1);
            let t2 = complex_mul(w2, z2);
            let t3 = complex_mul(w3, z3);
            let t4 = complex_mul(w4, z4);
            let t5 = complex_mul(w5, z5);
            let t6 = complex_mul(w6, z6);
            let t7 = complex_mul(w7, z7);

            // Compute radix-4 DFT on even indices (z0, t2, t4, t6).
            let even_a0 = vaddq_f32(z0, t4);
            let even_a1 = vsubq_f32(z0, t4);
            let even_a2 = vaddq_f32(t2, t6);
            let t2_sub_t6 = vsubq_f32(t2, t6);
            let even_a3 = complex_mul_i(t2_sub_t6, neg_imag_mask);

            let x_even_0 = vaddq_f32(even_a0, even_a2);
            let x_even_2 = vsubq_f32(even_a0, even_a2);
            let x_even_1 = vaddq_f32(even_a1, even_a3);
            let x_even_3 = vsubq_f32(even_a1, even_a3);

            // Compute radix-4 DFT on odd indices (t1, t3, t5, t7).
            let odd_a0 = vaddq_f32(t1, t5);
            let odd_a1 = vsubq_f32(t1, t5);
            let odd_a2 = vaddq_f32(t3, t7);
            let t3_sub_t7 = vsubq_f32(t3, t7);
            let odd_a3 = complex_mul_i(t3_sub_t7, neg_imag_mask);

            let x_odd_0 = vaddq_f32(odd_a0, odd_a2);
            let x_odd_2 = vsubq_f32(odd_a0, odd_a2);
            let x_odd_1 = vaddq_f32(odd_a1, odd_a3);
            let x_odd_3 = vsubq_f32(odd_a1, odd_a3);

            // Combine even and odd parts with W_8 twiddles.
            // out[0] = x_even[0] + x_odd[0]
            // out[4] = x_even[0] - x_odd[0]
            let out0 = vaddq_f32(x_even_0, x_odd_0);
            let out4 = vsubq_f32(x_even_0, x_odd_0);

            // out[1] = x_even[1] + W_8^1 * x_odd[1]
            // out[5] = x_even[1] - W_8^1 * x_odd[1]
            let w8_1_odd_1 = complex_mul(w8_1, x_odd_1);
            let out1 = vaddq_f32(x_even_1, w8_1_odd_1);
            let out5 = vsubq_f32(x_even_1, w8_1_odd_1);

            // out[2] = x_even[2] + W_8^2 * x_odd[2]
            // out[6] = x_even[2] - W_8^2 * x_odd[2]
            // W_8^2 = -i, so multiply by -i: (a+bi)*(-i) = (b, -a)
            let w8_2_odd_2 = complex_mul_i(x_odd_2, neg_imag_mask);
            let out2 = vaddq_f32(x_even_2, w8_2_odd_2);
            let out6 = vsubq_f32(x_even_2, w8_2_odd_2);

            // out[3] = x_even[3] + W_8^3 * x_odd[3]
            // out[7] = x_even[3] - W_8^3 * x_odd[3]
            let w8_3_odd_3 = complex_mul(w8_3, x_odd_3);
            let out3 = vaddq_f32(x_even_3, w8_3_odd_3);
            let out7 = vsubq_f32(x_even_3, w8_3_odd_3);

            // Sequential stores for stride=1.
            // We have 8 outputs (out0-out7), each containing 2 complex numbers.
            // Need to store them sequentially: [out0[0], out1[0], out2[0], out3[0], out4[0], out5[0], out6[0], out7[0], out0[1], ...]

            let j = 8 * i;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;

            // Extract low and high halves of each output.
            let out0_0 = vget_low_f32(out0);
            let out0_1 = vget_high_f32(out0);
            let out1_0 = vget_low_f32(out1);
            let out1_1 = vget_high_f32(out1);
            let out2_0 = vget_low_f32(out2);
            let out2_1 = vget_high_f32(out2);
            let out3_0 = vget_low_f32(out3);
            let out3_1 = vget_high_f32(out3);
            let out4_0 = vget_low_f32(out4);
            let out4_1 = vget_high_f32(out4);
            let out5_0 = vget_low_f32(out5);
            let out5_1 = vget_high_f32(out5);
            let out6_0 = vget_low_f32(out6);
            let out6_1 = vget_high_f32(out6);
            let out7_0 = vget_low_f32(out7);
            let out7_1 = vget_high_f32(out7);

            // Store first iteration outputs.
            vst1_f32(dst_ptr, out0_0);
            vst1_f32(dst_ptr.add(2), out1_0);
            vst1_f32(dst_ptr.add(4), out2_0);
            vst1_f32(dst_ptr.add(6), out3_0);
            vst1_f32(dst_ptr.add(8), out4_0);
            vst1_f32(dst_ptr.add(10), out5_0);
            vst1_f32(dst_ptr.add(12), out6_0);
            vst1_f32(dst_ptr.add(14), out7_0);

            // Store second iteration outputs.
            vst1_f32(dst_ptr.add(16), out0_1);
            vst1_f32(dst_ptr.add(18), out1_1);
            vst1_f32(dst_ptr.add(20), out2_1);
            vst1_f32(dst_ptr.add(22), out3_1);
            vst1_f32(dst_ptr.add(24), out4_1);
            vst1_f32(dst_ptr.add(26), out5_1);
            vst1_f32(dst_ptr.add(28), out6_1);
            vst1_f32(dst_ptr.add(30), out7_1);
        }
    }

    super::butterfly_radix8_scalar::<2>(src, dst, stage_twiddles, 1, simd_iters);
}

/// Performs a single radix-8 Stockham butterfly stage for p>1 (out-of-place, NEON).
///
/// Generic version for p>1 cases. Uses direct SIMD stores with scattered outputs.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix8_generic_neon(
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
    let eighth_samples = samples >> 3;
    let simd_iters = (eighth_samples >> 1) << 1;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask();

        let w8_1 = vreinterpretq_f32_u32(vld1q_u32(
            [
                FRAC_1_SQRT_2.to_bits(),
                (-FRAC_1_SQRT_2).to_bits(),
                FRAC_1_SQRT_2.to_bits(),
                (-FRAC_1_SQRT_2).to_bits(),
            ]
            .as_ptr(),
        ));

        let w8_3 = vdupq_n_f32(-FRAC_1_SQRT_2);

        for i in (0..simd_iters).step_by(2) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;

            // Load 2 complex numbers from each eighth.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = vld1q_f32(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + eighth_samples) as *const f32;
            let z1 = vld1q_f32(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + eighth_samples * 2) as *const f32;
            let z2 = vld1q_f32(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + eighth_samples * 3) as *const f32;
            let z3 = vld1q_f32(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + eighth_samples * 4) as *const f32;
            let z4 = vld1q_f32(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + eighth_samples * 5) as *const f32;
            let z5 = vld1q_f32(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + eighth_samples * 6) as *const f32;
            let z6 = vld1q_f32(z6_ptr);

            let z7_ptr = src.as_ptr().add(i + eighth_samples * 7) as *const f32;
            let z7 = vld1q_f32(z7_ptr);

            // Load prepackaged twiddles in packed format.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 7) as *const f32;
            let w1 = vld1q_f32(tw_ptr);
            let w2 = vld1q_f32(tw_ptr.add(4));
            let w3 = vld1q_f32(tw_ptr.add(8));
            let w4 = vld1q_f32(tw_ptr.add(12));
            let w5 = vld1q_f32(tw_ptr.add(16));
            let w6 = vld1q_f32(tw_ptr.add(20));
            let w7 = vld1q_f32(tw_ptr.add(24));

            // Apply twiddle factors.
            let t1 = complex_mul(w1, z1);
            let t2 = complex_mul(w2, z2);
            let t3 = complex_mul(w3, z3);
            let t4 = complex_mul(w4, z4);
            let t5 = complex_mul(w5, z5);
            let t6 = complex_mul(w6, z6);
            let t7 = complex_mul(w7, z7);

            // Compute radix-4 DFT on even indices.
            let even_a0 = vaddq_f32(z0, t4);
            let even_a1 = vsubq_f32(z0, t4);
            let even_a2 = vaddq_f32(t2, t6);
            let t2_sub_t6 = vsubq_f32(t2, t6);
            let even_a3 = complex_mul_i(t2_sub_t6, neg_imag_mask);

            let x_even_0 = vaddq_f32(even_a0, even_a2);
            let x_even_2 = vsubq_f32(even_a0, even_a2);
            let x_even_1 = vaddq_f32(even_a1, even_a3);
            let x_even_3 = vsubq_f32(even_a1, even_a3);

            // Compute radix-4 DFT on odd indices.
            let odd_a0 = vaddq_f32(t1, t5);
            let odd_a1 = vsubq_f32(t1, t5);
            let odd_a2 = vaddq_f32(t3, t7);
            let t3_sub_t7 = vsubq_f32(t3, t7);
            let odd_a3 = complex_mul_i(t3_sub_t7, neg_imag_mask);

            let x_odd_0 = vaddq_f32(odd_a0, odd_a2);
            let x_odd_2 = vsubq_f32(odd_a0, odd_a2);
            let x_odd_1 = vaddq_f32(odd_a1, odd_a3);
            let x_odd_3 = vsubq_f32(odd_a1, odd_a3);

            // Combine even and odd parts.
            let out0 = vaddq_f32(x_even_0, x_odd_0);
            let out4 = vsubq_f32(x_even_0, x_odd_0);

            let w8_1_odd_1 = complex_mul(w8_1, x_odd_1);
            let out1 = vaddq_f32(x_even_1, w8_1_odd_1);
            let out5 = vsubq_f32(x_even_1, w8_1_odd_1);

            let w8_2_odd_2 = complex_mul_i(x_odd_2, neg_imag_mask);
            let out2 = vaddq_f32(x_even_2, w8_2_odd_2);
            let out6 = vsubq_f32(x_even_2, w8_2_odd_2);

            let w8_3_odd_3 = complex_mul(w8_3, x_odd_3);
            let out3 = vaddq_f32(x_even_3, w8_3_odd_3);
            let out7 = vsubq_f32(x_even_3, w8_3_odd_3);

            // Calculate output indices with wraparound.
            let j0 = 8 * i - 7 * k0;
            let j1 = 8 * (i + 1) - 7 * k1;

            // Direct scattered stores using 64-bit operations.
            let dst_ptr = dst.as_mut_ptr().add(j0) as *mut f32;

            // Store first iteration (k0) outputs.
            vst1_f32(dst_ptr, vget_low_f32(out0));
            vst1_f32(dst_ptr.add(stride * 2), vget_low_f32(out1));
            vst1_f32(dst_ptr.add(stride * 4), vget_low_f32(out2));
            vst1_f32(dst_ptr.add(stride * 6), vget_low_f32(out3));
            vst1_f32(dst_ptr.add(stride * 8), vget_low_f32(out4));
            vst1_f32(dst_ptr.add(stride * 10), vget_low_f32(out5));
            vst1_f32(dst_ptr.add(stride * 12), vget_low_f32(out6));
            vst1_f32(dst_ptr.add(stride * 14), vget_low_f32(out7));

            // Store second iteration (k1) outputs.
            let dst_ptr1 = dst.as_mut_ptr().add(j1) as *mut f32;
            vst1_f32(dst_ptr1, vget_high_f32(out0));
            vst1_f32(dst_ptr1.add(stride * 2), vget_high_f32(out1));
            vst1_f32(dst_ptr1.add(stride * 4), vget_high_f32(out2));
            vst1_f32(dst_ptr1.add(stride * 6), vget_high_f32(out3));
            vst1_f32(dst_ptr1.add(stride * 8), vget_high_f32(out4));
            vst1_f32(dst_ptr1.add(stride * 10), vget_high_f32(out5));
            vst1_f32(dst_ptr1.add(stride * 12), vget_high_f32(out6));
            vst1_f32(dst_ptr1.add(stride * 14), vget_high_f32(out7));
        }
    }

    super::butterfly_radix8_scalar::<2>(src, dst, stage_twiddles, stride, simd_iters);
}
