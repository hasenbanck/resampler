//! AVX and AVX+FMA optimized FIR convolution implementations.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// AVX implementation of FIR convolution (dot product) without FMA.
///
/// Uses 256-bit SIMD registers to process 8 f32 values at a time with separate multiply and add.
/// For a 64-tap filter, this performs 8 iterations instead of 64.
/// Slightly slower than AVX+FMA due to separate multiply and add operations.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx",
    not(target_feature = "fma")
))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn convolve_avx(input: &[f32], coeffs: &[f32], taps: usize) -> f32 {
    unsafe {
        const SIMD_WIDTH: usize = 8;
        let simd_iterations = taps / SIMD_WIDTH;

        // Initialize accumulator to zero.
        let mut acc = _mm256_setzero_ps();

        // Process 8 taps at a time using separate multiply and add.
        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            // Load 8 input samples and 8 coefficients (unaligned load).
            let input_vec = _mm256_loadu_ps(input.as_ptr().add(offset));
            let coeffs_vec = _mm256_loadu_ps(coeffs.as_ptr().add(offset));

            // Separate multiply and add: acc = acc + (coeffs_vec * input_vec).
            let prod = _mm256_mul_ps(coeffs_vec, input_vec);
            acc = _mm256_add_ps(acc, prod);
        }

        // Horizontal sum: reduce 8-element vector to single scalar.
        // Step 1: Add pairs within 128-bit lanes: [a0+a1, a2+a3, a4+a5, a6+a7, ...].
        let sum1 = _mm256_hadd_ps(acc, acc);
        // Step 2: Add pairs again: [a0+a1+a2+a3, a4+a5+a6+a7, ...].
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        // Step 3: Extract high and low 128-bit lanes and add them.
        let high = _mm256_extractf128_ps(sum2, 1);
        let low = _mm256_castps256_ps128(sum2);
        let sum3 = _mm_add_ps(high, low);
        // Step 4: Extract the final scalar result.
        _mm_cvtss_f32(sum3)
    }
}

/// AVX+FMA implementation of FIR convolution (dot product).
///
/// Uses 256-bit SIMD registers to process 8 f32 values at a time with fused multiply-add.
/// For a 64-tap filter, this performs 8 iterations instead of 64.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn convolve_avx_fma(input: &[f32], coeffs: &[f32], taps: usize) -> f32 {
    unsafe {
        const SIMD_WIDTH: usize = 8;
        let simd_iterations = taps / SIMD_WIDTH;

        // Initialize accumulator to zero.
        let mut acc = _mm256_setzero_ps();

        // Process 8 taps at a time using FMA.
        for i in 0..simd_iterations {
            let offset = i * SIMD_WIDTH;

            // Load 8 input samples and 8 coefficients (unaligned load).
            let input_vec = _mm256_loadu_ps(input.as_ptr().add(offset));
            let coeffs_vec = _mm256_loadu_ps(coeffs.as_ptr().add(offset));

            // Fused multiply-add: acc = acc + (coeffs_vec * input_vec).
            acc = _mm256_fmadd_ps(coeffs_vec, input_vec, acc);
        }

        // Horizontal sum: reduce 8-element vector to single scalar.
        // Step 1: Add pairs within 128-bit lanes: [a0+a1, a2+a3, a4+a5, a6+a7, ...].
        let sum1 = _mm256_hadd_ps(acc, acc);
        // Step 2: Add pairs again: [a0+a1+a2+a3, a4+a5+a6+a7, ...].
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        // Step 3: Extract high and low 128-bit lanes and add them.
        let high = _mm256_extractf128_ps(sum2, 1);
        let low = _mm256_castps256_ps128(sum2);
        let sum3 = _mm_add_ps(high, low);
        // Step 4: Extract the final scalar result.
        _mm_cvtss_f32(sum3)
    }
}
