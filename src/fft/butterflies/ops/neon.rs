use core::arch::aarch64::*;

/// Complex multiplication.
#[target_feature(enable = "neon")]
pub(crate) fn complex_mul(left: float32x4_t, right: float32x4_t) -> float32x4_t {
    let right_re = vtrn1q_f32(right, right);
    let right_im_neg = vtrn2q_f32(right, vnegq_f32(right));
    let prod_im = vmulq_f32(right_im_neg, left);
    let prod_im_swap = vrev64q_f32(prod_im);
    vfmaq_f32(prod_im_swap, right_re, left)
}

/// Returns a mask for negating imaginary parts: [0, -0, 0, -0].
#[target_feature(enable = "neon")]
pub(crate) fn load_neg_imag_mask() -> float32x4_t {
    #[repr(align(16))]
    struct AlignedMask([u32; 4]);
    const NEG_IMAG_MASK: AlignedMask =
        AlignedMask([0x00000000, 0x80000000, 0x00000000, 0x80000000]);
    unsafe { vreinterpretq_f32_u32(vld1q_u32(NEG_IMAG_MASK.0.as_ptr())) }
}

/// Multiplies a complex vector by i (90-degree rotation).
///
/// Performs i * (a + bi) = -b + ai by swapping real/imaginary parts
/// and negating the new imaginary part.
#[target_feature(enable = "neon")]
pub(crate) fn complex_mul_i(vec: float32x4_t, neg_imag_mask: float32x4_t) -> float32x4_t {
    let swapped = vrev64q_f32(vec);
    vreinterpretq_f32_u32(veorq_u32(
        vreinterpretq_u32_f32(swapped),
        vreinterpretq_u32_f32(neg_imag_mask),
    ))
}
