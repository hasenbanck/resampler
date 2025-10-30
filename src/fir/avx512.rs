//! AVX-512 optimized FIR convolution implementation.

/// AVX-512 implementation of FIR convolution (dot product) with FMA.
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx512f", feature = "no_std"),
    not(feature = "no_std")
))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn convolve_avx512(input: &[f32], coeffs: &[f32], taps: usize) -> f32 {
    use core::arch::x86_64::*;

    unsafe {
        const SIMD_WIDTH: usize = 16;
        let simd_iterations = taps / SIMD_WIDTH;

        // Initialize accumulator to zero.
        let mut acc = _mm512_setzero_ps();

        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            let input_vec = _mm512_loadu_ps(input.as_ptr().add(offset));
            let coeffs_vec = _mm512_loadu_ps(coeffs.as_ptr().add(offset));

            // Fused multiply-add: acc = acc + (coeffs_vec * input_vec).
            acc = _mm512_fmadd_ps(coeffs_vec, input_vec, acc);
        }

        // Horizontal sum.
        _mm512_reduce_add_ps(acc)
    }
}
