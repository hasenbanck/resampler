//! NEON optimized FIR convolution implementation for aarch64.

use core::arch::aarch64::*;

/// NEON implementation of dual-phase FIR convolution with interpolation.
#[target_feature(enable = "neon")]
pub(super) unsafe fn convolve_interp_neon(
    input: &[f32],
    coeffs1: &[f32],
    coeffs2: &[f32],
    frac: f32,
    taps: usize,
) -> f32 {
    unsafe {
        const SIMD_WIDTH: usize = 4;
        let simd_iterations = taps / SIMD_WIDTH;

        assert_eq!(coeffs1.as_ptr().addr() % 64, 0);
        assert_eq!(coeffs2.as_ptr().addr() % 64, 0);

        // Initialize dual accumulators to zero.
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);

        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            // Load 4 input samples once.
            let input_vec = vld1q_f32(input.as_ptr().add(offset));

            // Load both coefficient phases (aligned).
            let coeffs_vec1 = vld1q_f32(coeffs1.as_ptr().add(offset));
            let coeffs_vec2 = vld1q_f32(coeffs2.as_ptr().add(offset));

            // Fused multiply-add for both phases.
            acc1 = vmlaq_f32(acc1, coeffs_vec1, input_vec);
            acc2 = vmlaq_f32(acc2, coeffs_vec2, input_vec);
        }

        // Interpolate before horizontal reduction to avoid two reductions.
        let frac_vec = vdupq_n_f32(frac);
        let one_minus_frac = vdupq_n_f32(1.0 - frac);
        let weighted1 = vmulq_f32(acc1, one_minus_frac);
        let weighted2 = vmulq_f32(acc2, frac_vec);
        let interpolated = vaddq_f32(weighted1, weighted2);

        // Horizontal sum: reduce 4-element vector to single scalar.
        vaddvq_f32(interpolated)
    }
}
