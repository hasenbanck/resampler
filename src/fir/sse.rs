//! SSE and SSE3 optimized FIR convolution implementations.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// SSE implementation of FIR convolution (dot product).
///
/// Uses 128-bit SIMD registers to process 4 f32 values at a time.
/// For a 64-tap filter, this performs 16 iterations instead of 64.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    not(target_feature = "sse3"),
    not(target_feature = "avx")
))]
#[target_feature(enable = "sse")]
pub(super) unsafe fn convolve_sse(input: &[f32], coeffs: &[f32], taps: usize) -> f32 {
    unsafe {
        const SIMD_WIDTH: usize = 4;
        let simd_iterations = taps / SIMD_WIDTH;

        // Initialize accumulator to zero.
        let mut acc = _mm_setzero_ps();

        // Process 4 taps at a time.
        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            // Load 4 input samples and 4 coefficients (unaligned load).
            let input_vec = _mm_loadu_ps(input.as_ptr().add(offset));
            let coeffs_vec = _mm_loadu_ps(coeffs.as_ptr().add(offset));

            // Multiply and add: acc = acc + (coeffs_vec * input_vec).
            let prod = _mm_mul_ps(coeffs_vec, input_vec);
            acc = _mm_add_ps(acc, prod);
        }

        // Horizontal sum: reduce 4-element vector to single scalar.
        // SSE doesn't have hadd, so we manually shuffle and add.
        // Step 1: Shuffle to get [a2, a3, a0, a1] and add to get [a0+a2, a1+a3, a2+a0, a3+a1].
        let shuf = _mm_shuffle_ps(acc, acc, 0b01_00_11_10); // [a2, a3, a0, a1]
        let sum1 = _mm_add_ps(acc, shuf); // [a0+a2, a1+a3, a2+a0, a3+a1]
        // Step 2: Shuffle again to get [a1+a3, ...] in position 0 and add.
        let shuf2 = _mm_shuffle_ps(sum1, sum1, 0b00_00_00_01); // [a1+a3, ...]
        let sum2 = _mm_add_ps(sum1, shuf2); // [a0+a2+a1+a3, ...]
        // Step 3: Extract the final scalar result.
        _mm_cvtss_f32(sum2)
    }
}

/// SSE3 implementation of FIR convolution (dot product).
///
/// Uses 128-bit SIMD registers to process 4 f32 values at a time.
/// For a 64-tap filter, this performs 16 iterations instead of 64.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse3",
    not(target_feature = "avx")
))]
#[target_feature(enable = "sse3")]
pub(super) unsafe fn convolve_sse3(input: &[f32], coeffs: &[f32], taps: usize) -> f32 {
    unsafe {
        const SIMD_WIDTH: usize = 4;
        let simd_iterations = taps / SIMD_WIDTH;

        // Initialize accumulator to zero.
        let mut acc = _mm_setzero_ps();

        // Process 4 taps at a time.
        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            // Load 4 input samples and 4 coefficients (unaligned load).
            let input_vec = _mm_loadu_ps(input.as_ptr().add(offset));
            let coeffs_vec = _mm_loadu_ps(coeffs.as_ptr().add(offset));

            // Multiply and add: acc = acc + (coeffs_vec * input_vec).
            let prod = _mm_mul_ps(coeffs_vec, input_vec);
            acc = _mm_add_ps(acc, prod);
        }

        // Horizontal sum: reduce 4-element vector to single scalar.
        // Step 1: Add pairs: [a0+a1, a2+a3, a0+a1, a2+a3].
        let sum1 = _mm_hadd_ps(acc, acc);
        // Step 2: Add pairs again: [a0+a1+a2+a3, a0+a1+a2+a3, ...].
        let sum2 = _mm_hadd_ps(sum1, sum1);
        // Step 3: Extract the final scalar result.
        _mm_cvtss_f32(sum2)
    }
}
