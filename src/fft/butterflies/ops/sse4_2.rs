use core::arch::x86_64::*;

/// Complex multiplication.
#[target_feature(enable = "sse4.2")]
pub(crate) fn complex_mul_sse4_2(left: __m128, right: __m128) -> __m128 {
    let right_re = _mm_moveldup_ps(right);
    let right_im = _mm_movehdup_ps(right);
    let left_swap = _mm_shuffle_ps(left, left, 0b10_11_00_01);
    let prod_re = _mm_mul_ps(left, right_re);
    let prod_im = _mm_mul_ps(left_swap, right_im);
    _mm_addsub_ps(prod_re, prod_im)
}

/// Returns a mask for negating imaginary parts: [-0.0, 0.0, -0.0, 0.0].
#[target_feature(enable = "sse4.2")]
pub(crate) fn load_neg_imag_mask_sse4_2() -> __m128 {
    _mm_set_ps(-0.0, 0.0, -0.0, 0.0)
}

/// Multiplies a complex vector by i (90-degree rotation).
///
/// Performs i * (a + bi) = -b + ai by swapping real/imaginary parts
/// and negating the new imaginary part (which is the old real part).
#[target_feature(enable = "sse4.2")]
pub(crate) fn complex_mul_i_sse4_2(vec: __m128, neg_imag_mask: __m128) -> __m128 {
    let swapped = _mm_shuffle_ps(vec, vec, 0b10_11_00_01);
    _mm_xor_ps(swapped, neg_imag_mask)
}

/// Multiplies a complex vector by sqrt(3)/2 * i.
#[target_feature(enable = "sse4.2")]
pub(crate) fn complex_mul_sqrt3_i_sse4_2(vec: __m128, sqrt3_2: f32) -> __m128 {
    let swapped = _mm_shuffle_ps(vec, vec, 0b10_11_00_01);
    _mm_mul_ps(swapped, _mm_set_ps(-sqrt3_2, sqrt3_2, -sqrt3_2, sqrt3_2))
}
