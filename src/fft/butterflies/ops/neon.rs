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

/// Returns the scale constant 1/√2 as a NEON SIMD vector.
#[target_feature(enable = "neon")]
pub(crate) fn load_scale_neon() -> float32x4_t {
    const SQRT_HALF: f32 = core::f32::consts::FRAC_1_SQRT_2;
    vdupq_n_f32(SQRT_HALF)
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

/// Optimized W₈¹ multiplication: (1-i)/√2 * (x+iy).
///
/// Computes ((x+y)/√2, (y-x)/√2) using fewer operations than full complex multiplication.
#[target_feature(enable = "neon")]
pub(crate) fn w8x_neon(xy: float32x4_t, sign_mask: float32x4_t, scale: float32x4_t) -> float32x4_t {
    let yx = vrev64q_f32(xy);
    let ymx = vreinterpretq_f32_u32(veorq_u32(
        vreinterpretq_u32_f32(yx),
        vreinterpretq_u32_f32(sign_mask),
    ));
    let sum = vaddq_f32(xy, ymx);
    vmulq_f32(scale, sum)
}

/// Optimized W₈³ multiplication: (-1-i)/√2 * (x+iy).
///
/// Computes ((y-x)/√2, -(x+y)/√2) using fewer operations than full complex multiplication.
#[target_feature(enable = "neon")]
pub(crate) fn v8x_neon(xy: float32x4_t, sign_mask: float32x4_t, scale: float32x4_t) -> float32x4_t {
    let yx = vrev64q_f32(xy);
    let ymx = vreinterpretq_f32_u32(veorq_u32(
        vreinterpretq_u32_f32(yx),
        vreinterpretq_u32_f32(sign_mask),
    ));
    let diff = vsubq_f32(ymx, xy);
    vmulq_f32(scale, diff)
}
