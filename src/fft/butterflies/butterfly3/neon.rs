use core::arch::aarch64::*;

use super::{super::ops::complex_mul, SQRT3_2};
use crate::fft::Complex32;

/// Performs a single radix-3 Stockham butterfly stage for stride=1 (out-of-place, NEON).
///
/// This is a specialized version for the stride=1 case (first stage) that uses sequential stores
/// instead of scattered stores. When stride==1, output indices are sequential: j = 3*i, j+1, j+2.
/// This provides significant performance benefits through write-combining and better cache utilization.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix3_stride1_neon(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let third_samples = samples / 3;
    let simd_iters = (third_samples >> 1) << 1;

    #[repr(align(16))]
    struct AlignedPattern([f32; 4]);
    const SQRT3_PATTERN: AlignedPattern = AlignedPattern([SQRT3_2, -SQRT3_2, SQRT3_2, -SQRT3_2]);

    unsafe {
        let half_vec = vdupq_n_f32(0.5);
        let sqrt3_signs = vld1q_f32(SQRT3_PATTERN.0.as_ptr());

        for i in (0..simd_iters).step_by(2) {
            // Load z0 from first third (contiguous).
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = vld1q_f32(z0_ptr);

            // Load z1, z2 from other thirds using contiguous loads.
            let z1_ptr = src.as_ptr().add(i + third_samples) as *const f32;
            let z1 = vld1q_f32(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + third_samples * 2) as *const f32;
            let z2 = vld1q_f32(z2_ptr);

            // Load pre-packaged twiddles: [w1[i..i+2], w2[i..i+2]]
            let tw_ptr = stage_twiddles.as_ptr().add(i * 2) as *const f32;
            let w1 = vld1q_f32(tw_ptr); // w1[i], w1[i+1]
            let w2 = vld1q_f32(tw_ptr.add(4)); // w2[i], w2[i+1]

            let t1 = complex_mul(z1, w1);
            let t2 = complex_mul(z2, w2);

            // Radix-3 DFT.
            // sum_t = t1 + t2, diff_t = t1 - t2
            let sum_t = vaddq_f32(t1, t2);
            let diff_t = vsubq_f32(t1, t2);

            // out0 = z0 + sum_t
            let out0 = vaddq_f32(z0, sum_t);

            // re_im_part = z0 - 0.5 * sum_t
            let half_sum_t = vmulq_f32(sum_t, half_vec);
            let re_im_part = vsubq_f32(z0, half_sum_t);

            // sqrt3_diff = SQRT3_2 * [diff_t.im, -diff_t.re]
            let diff_t_swap = vrev64q_f32(diff_t);
            let sqrt3_diff = vmulq_f32(diff_t_swap, sqrt3_signs);

            let out1 = vaddq_f32(re_im_part, sqrt3_diff);
            let out2 = vsubq_f32(re_im_part, sqrt3_diff);

            // Sequential stores for stride=1 case.
            // For iteration i, outputs go to indices [3*i, 3*i+1, 3*i+2]
            // For iteration i+1, outputs go to indices [3*(i+1), 3*(i+1)+1, 3*(i+1)+2]
            let j = 3 * i;
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            let out0_0 = vget_low_f32(out0);
            let out1_0 = vget_low_f32(out1);
            let out2_0 = vget_low_f32(out2);

            vst1_f32(dst_ptr.add(j * 2), out0_0);
            vst1_f32(dst_ptr.add((j + 1) * 2), out1_0);
            vst1_f32(dst_ptr.add((j + 2) * 2), out2_0);

            let out0_1 = vget_high_f32(out0);
            let out1_1 = vget_high_f32(out1);
            let out2_1 = vget_high_f32(out2);

            let j1 = 3 * (i + 1);
            vst1_f32(dst_ptr.add(j1 * 2), out0_1);
            vst1_f32(dst_ptr.add((j1 + 1) * 2), out1_1);
            vst1_f32(dst_ptr.add((j1 + 2) * 2), out2_1);
        }
    }

    for i in simd_iters..third_samples {
        let w1 = stage_twiddles[i * 2];
        let w2 = stage_twiddles[i * 2 + 1];

        let z0 = src[i];
        let z1 = src[i + third_samples];
        let z2 = src[i + third_samples * 2];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);

        let sum_t = t1.add(&t2);
        let diff_t = t1.sub(&t2);

        let j = 3 * i;
        dst[j] = z0.add(&sum_t);

        let re_part = z0.re - 0.5 * sum_t.re;
        let im_part = z0.im - 0.5 * sum_t.im;
        let sqrt3_diff_re = SQRT3_2 * diff_t.im;
        let sqrt3_diff_im = -SQRT3_2 * diff_t.re;

        dst[j + 1] = Complex32::new(re_part + sqrt3_diff_re, im_part + sqrt3_diff_im);
        dst[j + 2] = Complex32::new(re_part - sqrt3_diff_re, im_part - sqrt3_diff_im);
    }
}

/// Performs a single radix-3 Stockham butterfly stage for p>1 (out-of-place, NEON).
///
/// This is the generic version for stride>1 cases. Uses direct SIMD stores,
/// accepting non-sequential stores as the shuffle overhead isn't justified.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix3_generic_neon(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    // We convince the compiler here that stride can't be 0 to optimize better.
    let stride = stride.max(1);

    let samples = src.len();
    let third_samples = samples / 3;
    let simd_iters = (third_samples >> 1) << 1;

    #[repr(align(16))]
    struct AlignedPattern([f32; 4]);
    const SQRT3_PATTERN: AlignedPattern = AlignedPattern([SQRT3_2, -SQRT3_2, SQRT3_2, -SQRT3_2]);

    unsafe {
        let half_vec = vdupq_n_f32(0.5);
        let sqrt3_signs = vld1q_f32(SQRT3_PATTERN.0.as_ptr());

        for i in (0..simd_iters).step_by(2) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;

            // Load z0 from first third (contiguous).
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = vld1q_f32(z0_ptr);

            // Load z1, z2 from other thirds using contiguous loads.
            let z1_ptr = src.as_ptr().add(i + third_samples) as *const f32;
            let z1 = vld1q_f32(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + third_samples * 2) as *const f32;
            let z2 = vld1q_f32(z2_ptr);

            // Load pre-packaged twiddles: [w1[i..i+2], w2[i..i+2]]
            let tw_ptr = stage_twiddles.as_ptr().add(i * 2) as *const f32;
            let w1 = vld1q_f32(tw_ptr); // w1[i], w1[i+1]
            let w2 = vld1q_f32(tw_ptr.add(4)); // w2[i], w2[i+1]

            // Complex multiply using optimized helper function.
            let t1 = complex_mul(z1, w1);
            let t2 = complex_mul(z2, w2);

            // Radix-3 DFT.
            // sum_t = t1 + t2, diff_t = t1 - t2
            let sum_t = vaddq_f32(t1, t2);
            let diff_t = vsubq_f32(t1, t2);

            // out0 = z0 + sum_t
            let out0 = vaddq_f32(z0, sum_t);

            // re_im_part = z0 - 0.5 * sum_t
            let half_sum_t = vmulq_f32(sum_t, half_vec);
            let re_im_part = vsubq_f32(z0, half_sum_t);

            // sqrt3_diff = SQRT3_2 * [diff_t.im, -diff_t.re]
            // diff_t = [re0, im0, re1, im1]
            // diff_t_swap = [im0, re0, im1, re1]
            let diff_t_swap = vrev64q_f32(diff_t);
            // Multiply by [SQRT3_2, -SQRT3_2, SQRT3_2, -SQRT3_2]
            // to get [SQRT3_2 * im0, -SQRT3_2 * re0, SQRT3_2 * im1, -SQRT3_2 * re1]
            let sqrt3_diff = vmulq_f32(diff_t_swap, sqrt3_signs);

            let out1 = vaddq_f32(re_im_part, sqrt3_diff);
            let out2 = vsubq_f32(re_im_part, sqrt3_diff);

            // Calculate output indices: j = 3*i - 2*k
            let j0 = 3 * i - 2 * k0;
            let j1 = 3 * (i + 1) - 2 * k1;

            // Extract and store each complex number separately.
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            let out0_0 = vget_low_f32(out0);
            let out1_0 = vget_low_f32(out1);
            let out2_0 = vget_low_f32(out2);

            vst1_f32(dst_ptr.add(j0 << 1), out0_0);
            vst1_f32(dst_ptr.add((j0 + stride) << 1), out1_0);
            vst1_f32(dst_ptr.add((j0 + stride * 2) << 1), out2_0);

            let out0_1 = vget_high_f32(out0);
            let out1_1 = vget_high_f32(out1);
            let out2_1 = vget_high_f32(out2);

            vst1_f32(dst_ptr.add(j1 << 1), out0_1);
            vst1_f32(dst_ptr.add((j1 + stride) << 1), out1_1);
            vst1_f32(dst_ptr.add((j1 + stride * 2) << 1), out2_1);
        }
    }

    for i in simd_iters..third_samples {
        let k = i % stride;
        let w1 = stage_twiddles[i * 2];
        let w2 = stage_twiddles[i * 2 + 1];

        let z0 = src[i];
        let z1 = src[i + third_samples];
        let z2 = src[i + third_samples * 2];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);

        let sum_t = t1.add(&t2);
        let diff_t = t1.sub(&t2);

        let j = 3 * i - 2 * k;
        dst[j] = z0.add(&sum_t);

        let re_part = z0.re - 0.5 * sum_t.re;
        let im_part = z0.im - 0.5 * sum_t.im;
        let sqrt3_diff_re = SQRT3_2 * diff_t.im;
        let sqrt3_diff_im = -SQRT3_2 * diff_t.re;

        dst[j + stride] = Complex32::new(re_part + sqrt3_diff_re, im_part + sqrt3_diff_im);
        dst[j + stride * 2] = Complex32::new(re_part - sqrt3_diff_re, im_part - sqrt3_diff_im);
    }
}
