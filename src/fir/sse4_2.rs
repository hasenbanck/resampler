//! SSE4.2 optimized FIR convolution implementations.

/// SSE4.2 implementation of dual-phase FIR convolution with interpolation.
#[target_feature(enable = "sse4.2")]
pub(crate) unsafe fn convolve_interp_sse4_2(
    input: &[f32],
    coeffs1: &[f32],
    coeffs2: &[f32],
    frac: f32,
    taps: usize,
) -> f32 {
    use core::arch::x86_64::*;

    unsafe {
        const SIMD_WIDTH: usize = 4;
        let simd_iterations = taps / SIMD_WIDTH;

        assert_eq!(coeffs1.as_ptr().addr() % 64, 0);
        assert_eq!(coeffs2.as_ptr().addr() % 64, 0);

        // Initialize dual accumulators to zero.
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();

        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            // Load 4 input samples (unaligned) once.
            let input_vec = _mm_loadu_ps(input.as_ptr().add(offset));

            // Load both coefficient phases (aligned).
            let coeffs_vec1 = _mm_load_ps(coeffs1.as_ptr().add(offset));
            let coeffs_vec2 = _mm_load_ps(coeffs2.as_ptr().add(offset));

            // Multiply and accumulate for both phases.
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(coeffs_vec1, input_vec));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(coeffs_vec2, input_vec));
        }

        // Interpolate before horizontal reduction to avoid two reductions.
        let frac_vec = _mm_set1_ps(frac);
        let one_minus_frac = _mm_set1_ps(1.0 - frac);
        let weighted1 = _mm_mul_ps(acc1, one_minus_frac);
        let weighted2 = _mm_mul_ps(acc2, frac_vec);
        let interpolated = _mm_add_ps(weighted1, weighted2);

        // Horizontal sum: reduce 4-element vector to single scalar.
        let sum1 = _mm_hadd_ps(interpolated, interpolated);
        let sum2 = _mm_hadd_ps(sum1, sum1);
        _mm_cvtss_f32(sum2)
    }
}
