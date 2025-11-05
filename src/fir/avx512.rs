//! AVX-512 optimized FIR convolution implementation with aligned loads.

/// AVX-512 implementation of dual-phase FIR convolution with interpolation.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn convolve_interp_avx512(
    input: &[f32],
    coeffs1: &[f32],
    coeffs2: &[f32],
    frac: f32,
    taps: usize,
) -> f32 {
    use core::arch::x86_64::*;

    unsafe {
        const SIMD_WIDTH: usize = 16;
        let simd_iterations = taps / SIMD_WIDTH;

        assert_eq!(coeffs1.as_ptr().addr() % 64, 0);
        assert_eq!(coeffs2.as_ptr().addr() % 64, 0);

        // Initialize dual accumulators to zero.
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();

        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            // Regular unaligned load for input (benefits from cache) - load once.
            let input_vec = _mm512_loadu_ps(input.as_ptr().add(offset));

            // Aligned loads for both coefficient phases.
            let coeffs_vec1 = _mm512_load_ps(coeffs1.as_ptr().add(offset));
            let coeffs_vec2 = _mm512_load_ps(coeffs2.as_ptr().add(offset));

            // Fused multiply-add for both phases.
            acc1 = _mm512_fmadd_ps(coeffs_vec1, input_vec, acc1);
            acc2 = _mm512_fmadd_ps(coeffs_vec2, input_vec, acc2);
        }

        // Interpolate before horizontal reduction to avoid two reductions.
        let frac_vec = _mm512_set1_ps(frac);
        let one_minus_frac = _mm512_set1_ps(1.0 - frac);
        let weighted1 = _mm512_mul_ps(acc1, one_minus_frac);
        let weighted2 = _mm512_mul_ps(acc2, frac_vec);
        let interpolated = _mm512_add_ps(weighted1, weighted2);

        // Horizontal sum.
        _mm512_reduce_add_ps(interpolated)
    }
}
