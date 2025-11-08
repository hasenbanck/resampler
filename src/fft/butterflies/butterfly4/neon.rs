use core::arch::aarch64::*;

use super::super::ops::{complex_mul, complex_mul_i, load_neg_imag_mask};
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
    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples >> 1) << 1;

    unsafe {
        let neg_imag = load_neg_imag_mask();

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

            // Load pre-packaged twiddles: [w1[i..i+2], w2[i..i+2], w3[i..i+2]]
            let tw_ptr = stage_twiddles.as_ptr().add(i * 3) as *const f32;
            let w1 = vld1q_f32(tw_ptr); // w1[i], w1[i+1]
            let w2 = vld1q_f32(tw_ptr.add(4)); // w2[i], w2[i+1]
            let w3 = vld1q_f32(tw_ptr.add(8)); // w3[i], w3[i+1]

            let t1 = complex_mul(z1, w1);
            let t2 = complex_mul(z2, w2);
            let t3 = complex_mul(z3, w3);

            // Radix-4 butterfly.
            let a0 = vaddq_f32(z0, t2);
            let a1 = vsubq_f32(z0, t2);
            let a2 = vaddq_f32(t1, t3);

            // a3 = i * (t1 - t3)
            let t1_sub_t3 = vsubq_f32(t1, t3);
            let a3 = complex_mul_i(t1_sub_t3, neg_imag);

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
    // We convince the compiler here that stride can't be 0 to optimize better.
    if stride == 0 {
        return;
    }

    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples >> 1) << 1;

    unsafe {
        let neg_imag = load_neg_imag_mask();

        for i in (0..simd_iters).step_by(2) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;

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

            // Load pre-packaged twiddles: [w1[i..i+2], w2[i..i+2], w3[i..i+2]]
            let tw_ptr = stage_twiddles.as_ptr().add(i * 3) as *const f32;
            let w1 = vld1q_f32(tw_ptr); // w1[i], w1[i+1]
            let w2 = vld1q_f32(tw_ptr.add(4)); // w2[i], w2[i+1]
            let w3 = vld1q_f32(tw_ptr.add(8)); // w3[i], w3[i+1]

            let t1 = complex_mul(z1, w1);
            let t2 = complex_mul(z2, w2);
            let t3 = complex_mul(z3, w3);

            // Radix-4 butterfly.
            let a0 = vaddq_f32(z0, t2);
            let a1 = vsubq_f32(z0, t2);
            let a2 = vaddq_f32(t1, t3);

            // a3 = i * (t1 - t3)
            let t1_sub_t3 = vsubq_f32(t1, t3);
            let a3 = complex_mul_i(t1_sub_t3, neg_imag);

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
