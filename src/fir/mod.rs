#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
mod avx;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    not(target_feature = "avx")
))]
mod sse;

#[cfg(target_arch = "aarch64")]
mod neon;

/// Scalar implementation of FIR convolution (dot product).
#[inline(always)]
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "sse"),
    target_arch = "aarch64"
)))]
fn convolve_scalar(input: &[f32], coeffs: &[f32], taps: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..taps {
        sum += coeffs[i] * input[i];
    }
    sum
}

/// Dispatch function for FIR convolution with compile-time SIMD selection.
///
/// Automatically selects the best available implementation based on compile-time target features.
/// Only one SIMD implementation is compiled per target configuration:
///
/// - **AVX+FMA**: Best performance - 8 f32 per iteration with fused multiply-add (x86_64)
/// - **AVX**: 8 f32 per iteration with separate multiply/add (x86_64, no FMA)
/// - **SSE3**: 4 f32 per iteration with hadd instruction (x86_64, no AVX)
/// - **SSE**: 4 f32 per iteration with manual shuffle (x86_64, no SSE3/AVX)
/// - **NEON**: 4 f32 per iteration with fused multiply-add (aarch64, always available)
/// - **Scalar**: Fallback for all other architectures
#[inline(always)]
pub(crate) fn convolve(input: &[f32], coeffs: &[f32], taps: usize) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        // Safety: We've checked that avx and fma features are enabled at compile time.
        unsafe { avx::convolve_avx_fma(input, coeffs, taps) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        not(target_feature = "fma")
    ))]
    {
        // Safety: We've checked that avx feature is enabled at compile time.
        unsafe { avx::convolve_avx(input, coeffs, taps) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse3",
        not(target_feature = "avx")
    ))]
    {
        // Safety: We've checked that sse3 feature is enabled at compile time.
        unsafe { sse::convolve_sse3(input, coeffs, taps) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse",
        not(target_feature = "sse3"),
        not(target_feature = "avx")
    ))]
    {
        // Safety: We've checked that sse feature is enabled at compile time.
        unsafe { sse::convolve_sse(input, coeffs, taps) }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is mandatory on aarch64, so it's always available.
        unsafe { neon::convolve_neon(input, coeffs, taps) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "sse"),
        target_arch = "aarch64"
    )))]
    convolve_scalar(input, coeffs, taps)
}
