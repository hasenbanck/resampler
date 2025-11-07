use core::arch::x86_64::*;

/// Complex multiplication.
#[target_feature(enable = "sse2")]
pub(crate) fn complex_mul_sse2(left: __m128, right: __m128) -> __m128 {
    let right_re = _mm_shuffle_ps(right, right, 0b10_10_00_00);
    let right_im = _mm_shuffle_ps(right, right, 0b11_11_01_01);
    let left_swap = _mm_shuffle_ps(left, left, 0b10_11_00_01);

    let prod_re = _mm_mul_ps(left, right_re);
    let prod_im = _mm_mul_ps(left_swap, right_im);

    // Emulate addsub: result[i] = (i%2==0) ? prod_re[i]-prod_im[i] : prod_re[i]+prod_im[i]
    let sub_result = _mm_sub_ps(prod_re, prod_im);
    let add_result = _mm_add_ps(prod_re, prod_im);
    let select_odd = _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));
    _mm_or_ps(
        _mm_and_ps(select_odd, add_result),
        _mm_andnot_ps(select_odd, sub_result),
    )
}

/// Returns a mask for negating imaginary parts: [-0.0, 0.0, -0.0, 0.0].
#[target_feature(enable = "sse2")]
pub(crate) fn load_neg_imag_mask_sse2() -> __m128 {
    _mm_set_ps(-0.0, 0.0, -0.0, 0.0)
}

/// Multiplies a complex vector by i (90-degree rotation).
///
/// Performs i * (a + bi) = -b + ai by swapping real/imaginary parts
/// and negating the new imaginary part (which is the old real part).
#[target_feature(enable = "sse2")]
pub(crate) fn complex_mul_i_sse2(vec: __m128, neg_imag_mask: __m128) -> __m128 {
    let swapped = _mm_shuffle_ps(vec, vec, 0b10_11_00_01);
    _mm_xor_ps(swapped, neg_imag_mask)
}

/// Multiplies a complex vector by sqrt(3)/2 * i.
#[target_feature(enable = "sse2")]
pub(crate) fn complex_mul_sqrt3_i_sse2(vec: __m128, sqrt3_2: f32) -> __m128 {
    let swapped = _mm_shuffle_ps(vec, vec, 0b10_11_00_01);
    _mm_mul_ps(swapped, _mm_set_ps(-sqrt3_2, sqrt3_2, -sqrt3_2, sqrt3_2))
}
