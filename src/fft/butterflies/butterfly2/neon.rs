use core::arch::aarch64::*;

use super::super::ops::complex_mul;
use crate::fft::Complex32;

/// Performs a single radix-2 Stockham butterfly stage for stride=1 (out-of-place, NEON).
///
/// This is a specialized version for the stride=1 case (first stage) that uses the zip pattern
/// to enable sequential SIMD stores instead of scattered scalar stores.
/// This provides significant performance benefits through write-combining and better cache utilization.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix2_stride1_neon(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let half_samples = samples >> 1;
    let simd_iters = (half_samples >> 1) << 1;

    unsafe {
        for i in (0..simd_iters).step_by(2) {
            // Load 2 complex numbers from first half.
            let a_ptr = src.as_ptr().add(i) as *const f32;
            let a = vld1q_f32(a_ptr);

            // Load 2 complex numbers from second half.
            let b_ptr = src.as_ptr().add(i + half_samples) as *const f32;
            let b = vld1q_f32(b_ptr);

            // Identity twiddles: t = (1+0i) * b = b (skip twiddle load and multiply)

            // Butterfly: out_top = a + b, out_bot = a - b
            let out_top = vaddq_f32(a, b); // Results for even indices
            let out_bot = vsubq_f32(a, b); // Results for odd indices

            // Apply zip pattern for sequential stores.
            // Cast to f64 view to treat each complex number as a single 64-bit unit.
            let out_top_f64 = vreinterpretq_f64_f32(out_top);
            let out_bot_f64 = vreinterpretq_f64_f32(out_bot);

            // Interleave: [top0, bot0] and [top1, bot1].
            let result_lo = vzip1q_f64(out_top_f64, out_bot_f64); // [top0, bot0]
            let result_hi = vzip2q_f64(out_top_f64, out_bot_f64); // [top1, bot1]

            // Cast back to f32 and store sequentially.
            let result_lo_f32 = vreinterpretq_f32_f64(result_lo);
            let result_hi_f32 = vreinterpretq_f32_f64(result_hi);

            let j = i << 1;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            vst1q_f32(dst_ptr, result_lo_f32);
            vst1q_f32(dst_ptr.add(4), result_hi_f32);
        }
    }

    super::butterfly_radix2_scalar::<2>(src, dst, stage_twiddles, 1, simd_iters);
}

/// Performs a single radix-2 Stockham butterfly stage for p>1 (out-of-place, NEON).
///
/// This is the generic version for p>1 cases. Uses direct SIMD stores instead of
/// temporary arrays, accepting non-sequential stores as the shuffle overhead isn't justified.
#[target_feature(enable = "neon")]
pub(super) unsafe fn butterfly_radix2_generic_neon(
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
    let half_samples = samples >> 1;
    let simd_iters = (half_samples >> 1) << 1;

    unsafe {
        for i in (0..simd_iters).step_by(2) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;

            // Load 2 complex numbers from first half.
            let a_ptr = src.as_ptr().add(i) as *const f32;
            let a = vld1q_f32(a_ptr);

            // Load 2 complex numbers from second half.
            let b_ptr = src.as_ptr().add(i + half_samples) as *const f32;
            let b = vld1q_f32(b_ptr);

            // Load twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i) as *const f32;
            let tw = vld1q_f32(tw_ptr);

            let t = complex_mul(b, tw);

            // Butterfly: out_top = a + t, out_bot = a - t
            let out_top = vaddq_f32(a, t);
            let out_bot = vsubq_f32(a, t);

            // Calculate output indices: j = (i << 1) - k
            let j0 = (i << 1) - k0;
            let j1 = ((i + 1) << 1) - k1;

            // Extract and store each complex number separately using direct f32 stores.
            let dst_ptr = dst.as_mut_ptr() as *mut f32;

            let top0 = vget_low_f32(out_top);
            let bot0 = vget_low_f32(out_bot);
            vst1_f32(dst_ptr.add(j0 << 1), top0);
            vst1_f32(dst_ptr.add((j0 + stride) << 1), bot0);

            let top1 = vget_high_f32(out_top);
            let bot1 = vget_high_f32(out_bot);
            vst1_f32(dst_ptr.add(j1 << 1), top1);
            vst1_f32(dst_ptr.add((j1 + stride) << 1), bot1);
        }
    }

    super::butterfly_radix2_scalar::<2>(src, dst, stage_twiddles, stride, simd_iters);
}
