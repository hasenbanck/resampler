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

/// Returns the scale constant 1/√2 as a 256-bit SIMD vector.
#[target_feature(enable = "avx")]
pub(crate) fn load_scale_avx() -> __m256 {
    const SQRT_HALF: f32 = core::f32::consts::FRAC_1_SQRT_2;
    _mm256_set1_ps(SQRT_HALF)
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

/// Optimized W₈¹ multiplication: (1-i)/√2 * (x+iy).
///
/// Computes ((x+y)/√2, (y-x)/√2) using fewer operations than full complex multiplication:
/// - W₈¹ × (x+iy) = ((x+y) + i(y-x))/√2
#[target_feature(enable = "avx")]
pub(crate) fn w8x_avx(xy: __m256, sign_mask: __m256, scale: __m256) -> __m256 {
    let yx = _mm256_shuffle_ps(xy, xy, 0b10_11_00_01);
    let ymx = _mm256_xor_ps(yx, sign_mask);
    let sum = _mm256_add_ps(xy, ymx);
    _mm256_mul_ps(scale, sum)
}

/// Optimized W₈³ multiplication: (-1-i)/√2 * (x+iy).
///
/// Computes ((y-x)/√2, -(x+y)/√2) using fewer operations than full complex multiplication:
/// - W₈³ × (x+iy) = ((y-x) - i(x+y))/√2
#[target_feature(enable = "avx")]
pub(crate) fn v8x_avx(xy: __m256, sign_mask: __m256, scale: __m256) -> __m256 {
    let yx = _mm256_shuffle_ps(xy, xy, 0b10_11_00_01);
    let ymx = _mm256_xor_ps(yx, sign_mask);
    let diff = _mm256_sub_ps(ymx, xy);
    _mm256_mul_ps(scale, diff)
}
