//! AVX-512 optimized FIR convolution implementation with streaming loads.

/// AVX-512 implementation of FIR convolution (dot product) with FMA.
/// Uses streaming loads for coefficients to reduce cache pollution.
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

        debug_assert_eq!(
            coeffs.as_ptr().addr() % 64,
            0,
            "Coefficient data must be 64-byte aligned for streaming loads"
        );

        // Initialize accumulator to zero.
        let mut acc = _mm512_setzero_ps();

        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            // Regular unaligned load for input (benefits from cache).
            let input_vec = _mm512_loadu_ps(input.as_ptr().add(offset));

            // Streaming load for coefficients (bypasses cache to reduce pollution).
            // _mm512_stream_load_si512 requires __m512i*, so we cast and then convert.
            let coeffs_ptr = coeffs.as_ptr().add(offset) as *const __m512i;
            let coeffs_vec = _mm512_castsi512_ps(_mm512_stream_load_si512(coeffs_ptr));

            // Fused multiply-add: acc = acc + (coeffs_vec * input_vec).
            acc = _mm512_fmadd_ps(coeffs_vec, input_vec, acc);
        }

        // Horizontal sum.
        _mm512_reduce_add_ps(acc)
    }
}
