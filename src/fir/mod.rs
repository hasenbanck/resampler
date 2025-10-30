#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx;

#[cfg(target_arch = "x86_64")]
pub(crate) mod sse;

#[cfg(target_arch = "aarch64")]
mod neon;

/// Scalar implementation of FIR convolution (dot product).
#[inline(always)]
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "sse", feature = "no_std"),
    target_arch = "aarch64"
)))]
pub(crate) fn convolve_scalar(input: &[f32], coeffs: &[f32], taps: usize) -> f32 {
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
/// - **AVX-512+FMA**: Best performance - 16 f32 per iteration with fused multiply-add (x86_64)
/// - **AVX+FMA**: 8 f32 per iteration with fused multiply-add (x86_64, no AVX-512)
/// - **AVX**: 8 f32 per iteration with separate multiply/add (x86_64, no FMA/AVX-512)
/// - **SSE3**: 4 f32 per iteration with hadd instruction (x86_64, no AVX/AVX-512)
/// - **SSE**: 4 f32 per iteration with manual shuffle (x86_64, no SSE3/AVX/AVX-512)
/// - **NEON**: 4 f32 per iteration with fused multiply-add (aarch64, always available)
/// - **Scalar**: Fallback for all other architectures
#[cfg(any(
    all(target_arch = "x86_64", feature = "no_std"),
    not(target_arch = "x86_64")
))]
pub(crate) fn convolve(input: &[f32], coeffs: &[f32], taps: usize) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        // Safety: We've checked that avx512f feature is enabled at compile time.
        unsafe { avx512::convolve_avx512(input, coeffs, taps) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        target_feature = "fma",
        not(target_feature = "avx512f")
    ))]
    {
        // Safety: We've checked that avx and fma features are enabled at compile time.
        unsafe { avx::convolve_avx_fma(input, coeffs, taps) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx",
        not(target_feature = "fma"),
        not(target_feature = "avx512f")
    ))]
    {
        // Safety: We've checked that avx feature is enabled at compile time.
        unsafe { avx::convolve_avx(input, coeffs, taps) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse3",
        not(target_feature = "avx"),
        not(target_feature = "avx512f")
    ))]
    {
        // Safety: We've checked that sse3 feature is enabled at compile time.
        unsafe { sse::convolve_sse3(input, coeffs, taps) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse",
        not(target_feature = "sse3"),
        not(target_feature = "avx"),
        not(target_feature = "avx512f")
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
