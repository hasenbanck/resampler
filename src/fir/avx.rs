//! AVX+FMA optimized FIR convolution implementations.

/// AVX+FMA implementation of dual-phase FIR convolution with interpolation.
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn convolve_interp_avx_fma(
    input: &[f32],
    coeffs1: &[f32],
    coeffs2: &[f32],
    frac: f32,
    taps: usize,
) -> f32 {
    use core::arch::x86_64::*;

    unsafe {
        const SIMD_WIDTH: usize = 8;
        let simd_iterations = taps / SIMD_WIDTH;

        assert_eq!(coeffs1.as_ptr().addr() % 64, 0);
        assert_eq!(coeffs2.as_ptr().addr() % 64, 0);

        // Initialize dual accumulators to zero.
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();

        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            // Load 8 input samples (unaligned) once.
            let input_vec = _mm256_loadu_ps(input.as_ptr().add(offset));

            // Load both coefficient phases (aligned).
            let coeffs_vec1 = _mm256_load_ps(coeffs1.as_ptr().add(offset));
            let coeffs_vec2 = _mm256_load_ps(coeffs2.as_ptr().add(offset));

            // Fused multiply-add for both phases.
            acc1 = _mm256_fmadd_ps(coeffs_vec1, input_vec, acc1);
            acc2 = _mm256_fmadd_ps(coeffs_vec2, input_vec, acc2);
        }

        // Interpolate before horizontal reduction to avoid two reductions.
        let frac_vec = _mm256_set1_ps(frac);
        let one_minus_frac = _mm256_set1_ps(1.0 - frac);
        let weighted1 = _mm256_mul_ps(acc1, one_minus_frac);
        let weighted2 = _mm256_mul_ps(acc2, frac_vec);
        let interpolated = _mm256_add_ps(weighted1, weighted2);

        // Horizontal sum.

        // Step 1: Extract and add high/low 128-bit lanes to get a single 128-bit vector.
        let high = _mm256_extractf128_ps(interpolated, 1);
        let low = _mm256_castps256_ps128(interpolated);
        let sum128 = _mm_add_ps(high, low);

        // Step 2: Horizontal sum within 128-bit vector using shuffle.
        let shuf = _mm_shuffle_ps(sum128, sum128, 0b01_00_11_10);
        let sum1 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sum1, sum1, 0b00_00_00_01);
        let sum2 = _mm_add_ps(sum1, shuf2);
        _mm_cvtss_f32(sum2)
    }
}
