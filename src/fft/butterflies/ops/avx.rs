use core::arch::x86_64::*;

/// Complex multiplication.
#[target_feature(enable = "avx,fma")]
pub(crate) fn complex_mul_avx(left: __m256, right: __m256) -> __m256 {
    let right_re = _mm256_moveldup_ps(right);
    let right_im = _mm256_movehdup_ps(right);
    let left_swap = _mm256_permute_ps(left, 0b10_11_00_01);
    let prod_im = _mm256_mul_ps(left_swap, right_im);
    _mm256_fmaddsub_ps(left, right_re, prod_im)
}

/// Returns a mask for negating imaginary parts: [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0].
#[target_feature(enable = "avx")]
pub(crate) fn load_neg_imag_mask_avx() -> __m256 {
    _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0)
}

/// Multiplies a complex vector by i (90-degree rotation).
///
/// Performs i * (a + bi) = -b + ai by swapping real/imaginary parts
/// and negating the new imaginary part (which is the old real part).
#[target_feature(enable = "avx")]
pub(crate) fn complex_mul_i_avx(vec: __m256, neg_imag_mask: __m256) -> __m256 {
    let swapped = _mm256_shuffle_ps(vec, vec, 0b10_11_00_01);
    _mm256_xor_ps(swapped, neg_imag_mask)
}

/// Multiplies a complex vector by sqrt(3)/2 * i.
#[target_feature(enable = "avx")]
pub(crate) fn complex_mul_sqrt3_i_avx(vec: __m256, sqrt3_pattern: __m256) -> __m256 {
    let swapped = _mm256_shuffle_ps(vec, vec, 0b10_11_00_01);
    _mm256_mul_ps(swapped, sqrt3_pattern)
}
