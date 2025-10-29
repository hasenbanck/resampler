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

        // Horizontal sum
        // Step 1: Extract and add high/low 128-bit lanes to get a single 128-bit vector.
        let high = _mm256_extractf128_ps(acc, 1);
        let low = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(high, low);
        // Step 2: Horizontal sum within 128-bit vector using shuffle.
        // Shuffle to get [a2, a3, a0, a1] and add: [a0+a2, a1+a3, a2+a0, a3+a1].
        let shuf = _mm_shuffle_ps(sum128, sum128, 0b01_00_11_10);
        let sum1 = _mm_add_ps(sum128, shuf);
        // Shuffle again to get [a1+a3, ...] in position 0 and add: [a0+a2+a1+a3, ...].
        let shuf2 = _mm_shuffle_ps(sum1, sum1, 0b00_00_00_01);
        let sum2 = _mm_add_ps(sum1, shuf2);
        // Step 3: Extract the final scalar result.
        _mm_cvtss_f32(sum2)
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

        // Horizontal sum
        // Step 1: Extract and add high/low 128-bit lanes to get a single 128-bit vector.
        let high = _mm256_extractf128_ps(acc, 1);
        let low = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(high, low);
        // Step 2: Horizontal sum within 128-bit vector using shuffle.
        // Shuffle to get [a2, a3, a0, a1] and add: [a0+a2, a1+a3, a2+a0, a3+a1].
        let shuf = _mm_shuffle_ps(sum128, sum128, 0b01_00_11_10);
        let sum1 = _mm_add_ps(sum128, shuf);
        // Shuffle again to get [a1+a3, ...] in position 0 and add: [a0+a2+a1+a3, ...].
        let shuf2 = _mm_shuffle_ps(sum1, sum1, 0b00_00_00_01);
        let sum2 = _mm_add_ps(sum1, shuf2);
        // Step 3: Extract the final scalar result.
        _mm_cvtss_f32(sum2)
    }
}
