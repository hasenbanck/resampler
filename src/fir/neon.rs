//! NEON optimized FIR convolution implementation for aarch64.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// NEON implementation of FIR convolution (dot product).
///
/// Uses 128-bit SIMD registers to process 4 f32 values at a time with fused multiply-add.
/// For a 64-tap filter, this performs 16 iterations instead of 64.
/// NEON always has FMA support via `vmlaq_f32`, unlike x86 where FMA is optional.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn convolve_neon(input: &[f32], coeffs: &[f32], taps: usize) -> f32 {
    unsafe {
        const SIMD_WIDTH: usize = 4;
        let simd_iterations = taps / SIMD_WIDTH;

        // Initialize accumulator to zero.
        let mut acc = vdupq_n_f32(0.0);

        // Process 4 taps at a time using FMA.
        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            // Load 4 input samples (may be unaligned) and 4 coefficients (aligned).
            let input_vec = vld1q_f32(input.as_ptr().add(offset));
            let coeffs_vec = vld1q_f32(coeffs.as_ptr().add(offset));

            // Fused multiply-add: acc = acc + (coeffs_vec * input_vec).
            acc = vmlaq_f32(acc, coeffs_vec, input_vec);
        }

        // Horizontal sum: reduce 4-element vector to single scalar.
        vaddvq_f32(acc)
    }
}
