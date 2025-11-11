use crate::fft::Complex32;

#[cfg(all(
    target_arch = "x86_64",
    any(
        not(feature = "no_std"),
        all(target_feature = "avx", target_feature = "fma")
    )
))]
mod avx;

#[cfg(all(
    target_arch = "x86_64",
    any(
        test,
        not(feature = "no_std"),
        all(
            feature = "no_std",
            target_feature = "sse2",
            not(target_feature = "sse4.2"),
            not(all(target_feature = "avx", target_feature = "fma"))
        )
    ),
))]
mod sse2;

#[cfg(all(
    target_arch = "x86_64",
    any(
        test,
        not(feature = "no_std"),
        all(
            feature = "no_std",
            target_feature = "sse4.2",
            not(all(target_feature = "avx", target_feature = "fma"))
        )
    ),
))]
mod sse4_2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;

/// Dispatch function for radix-8 Stockham butterfly.
#[inline(always)]
pub(crate) fn butterfly_radix8_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    #[allow(unused)]
    let samples = src.len();
    #[allow(unused)]
    let eighth_samples = samples >> 3;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    unsafe {
        if eighth_samples >= 4 {
            return match stride {
                1 => avx::butterfly_radix8_stride1_avx_fma(src, dst, stage_twiddles),
                _ => avx::butterfly_radix8_generic_avx_fma(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse4.2",
        not(all(target_feature = "avx", target_feature = "fma"))
    ))]
    unsafe {
        if eighth_samples >= 2 {
            return match stride {
                1 => sse4_2::butterfly_radix8_stride1_sse4_2(src, dst, stage_twiddles),
                _ => sse4_2::butterfly_radix8_generic_sse4_2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse4.2")
    ))]
    unsafe {
        if eighth_samples >= 2 {
            return match stride {
                1 => sse2::butterfly_radix8_stride1_sse2(src, dst, stage_twiddles),
                _ => sse2::butterfly_radix8_generic_sse2(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        if eighth_samples >= 2 {
            return match stride {
                1 => neon::butterfly_radix8_stride1_neon(src, dst, stage_twiddles),
                _ => neon::butterfly_radix8_generic_neon(src, dst, stage_twiddles, stride),
            };
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    butterfly_radix8_scalar::<4>(src, dst, stage_twiddles, stride, 0);
    #[cfg(all(
        not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")),
        not(target_arch = "aarch64")
    ))]
    butterfly_radix8_scalar::<2>(src, dst, stage_twiddles, stride, 0);
    #[cfg(target_arch = "aarch64")]
    butterfly_radix8_scalar::<2>(src, dst, stage_twiddles, stride, 0);
}

/// AVX+FMA dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix8_stride1_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;

    if eighth_samples >= 4 {
        return unsafe { avx::butterfly_radix8_stride1_avx_fma(src, dst, stage_twiddles) };
    }

    butterfly_radix8_scalar::<4>(src, dst, stage_twiddles, 1, 0);
}

/// AVX+FMA dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix8_generic_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;

    if eighth_samples >= 4 {
        return unsafe { avx::butterfly_radix8_generic_avx_fma(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix8_scalar::<4>(src, dst, stage_twiddles, stride, 0);
}

/// SSE2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix8_stride1_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;

    if eighth_samples >= 2 {
        return unsafe { sse2::butterfly_radix8_stride1_sse2(src, dst, stage_twiddles) };
    }

    butterfly_radix8_scalar::<2>(src, dst, stage_twiddles, 1, 0);
}

/// SSE2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix8_generic_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;

    if eighth_samples >= 2 {
        return unsafe { sse2::butterfly_radix8_generic_sse2(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix8_scalar::<2>(src, dst, stage_twiddles, stride, 0);
}

/// SSE4.2 dispatcher for p1 (stride=1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix8_stride1_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;

    if eighth_samples >= 2 {
        return unsafe { sse4_2::butterfly_radix8_stride1_sse4_2(src, dst, stage_twiddles) };
    }

    butterfly_radix8_scalar::<2>(src, dst, stage_twiddles, 1, 0);
}

/// SSE4.2 dispatcher for generic (stride>1) variant
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix8_generic_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;

    if eighth_samples >= 2 {
        return unsafe {
            sse4_2::butterfly_radix8_generic_sse4_2(src, dst, stage_twiddles, stride)
        };
    }

    butterfly_radix8_scalar::<2>(src, dst, stage_twiddles, stride, 0);
}

/// Performs a single radix-8 Stockham butterfly stage (out-of-place, scalar).
///
/// This implementation uses a split-radix approach, decomposing the 8-point DFT
/// into two 4-point DFTs (even and odd indexed inputs) and combining the results.
///
/// Expects twiddles in packed format matching SIMD code:
/// - Packed portion (for groups of iterations): [w1[i], w1[i+1], ..., w2[i], w2[i+1], ...]
/// - Scalar tail (if any): [w1[i], w2[i], w3[i], w4[i], w5[i], w6[i], w7[i], ...] (interleaved)
#[allow(dead_code)]
#[inline(always)]
pub(super) fn butterfly_radix8_scalar<const WIDTH: usize>(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
    start_index: usize,
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;
    let simd_iters = (eighth_samples / WIDTH) * WIDTH;

    // Stride=1 optimization: skip identity twiddle multiplications.
    if stride == 1 {
        // Process iterations with identity twiddles (no multiplication needed).
        for i in start_index..simd_iters {
            // Load 8 input values.
            let z0 = src[i];
            let z1 = src[i + eighth_samples];
            let z2 = src[i + eighth_samples * 2];
            let z3 = src[i + eighth_samples * 3];
            let z4 = src[i + eighth_samples * 4];
            let z5 = src[i + eighth_samples * 5];
            let z6 = src[i + eighth_samples * 6];
            let z7 = src[i + eighth_samples * 7];

            // Identity twiddles: t_k = (1+0i) * z_k = z_k
            let t1 = z1;
            let t2 = z2;
            let t3 = z3;
            let t4 = z4;
            let t5 = z5;
            let t6 = z6;
            let t7 = z7;

            // Split-radix decomposition:
            // First, compute radix-4 DFT on even indices (z0, t2, t4, t6)
            let even_a0 = z0.add(&t4);
            let even_a1 = z0.sub(&t4);
            let even_a2 = t2.add(&t6);
            let even_a3_re = t2.im - t6.im;
            let even_a3_im = t6.re - t2.re;

            let x_even_0 = even_a0.add(&even_a2);
            let x_even_2 = even_a0.sub(&even_a2);
            let x_even_1 = Complex32::new(even_a1.re + even_a3_re, even_a1.im + even_a3_im);
            let x_even_3 = Complex32::new(even_a1.re - even_a3_re, even_a1.im - even_a3_im);

            // Compute radix-4 DFT on odd indices (t1, t3, t5, t7)
            let odd_a0 = t1.add(&t5);
            let odd_a1 = t1.sub(&t5);
            let odd_a2 = t3.add(&t7);
            let odd_a3_re = t3.im - t7.im;
            let odd_a3_im = t7.re - t3.re;

            let x_odd_0 = odd_a0.add(&odd_a2);
            let x_odd_2 = odd_a0.sub(&odd_a2);
            let x_odd_1 = Complex32::new(odd_a1.re + odd_a3_re, odd_a1.im + odd_a3_im);
            let x_odd_3 = Complex32::new(odd_a1.re - odd_a3_re, odd_a1.im - odd_a3_im);

            // Combine even and odd parts with additional twiddle factors.
            use core::f32::consts::FRAC_1_SQRT_2;

            // For stride=1, k=0, so j = 8*i
            let j = 8 * i;
            dst[j] = x_even_0.add(&x_odd_0);
            dst[j + 4] = x_even_0.sub(&x_odd_0);

            let w8_1_odd_1 = Complex32::new(
                FRAC_1_SQRT_2 * (x_odd_1.re + x_odd_1.im),
                FRAC_1_SQRT_2 * (x_odd_1.im - x_odd_1.re),
            );
            dst[j + 1] = x_even_1.add(&w8_1_odd_1);
            dst[j + 5] = x_even_1.sub(&w8_1_odd_1);

            let w8_2_odd_2 = Complex32::new(x_odd_2.im, -x_odd_2.re);
            dst[j + 2] = x_even_2.add(&w8_2_odd_2);
            dst[j + 6] = x_even_2.sub(&w8_2_odd_2);

            let w8_3_odd_3 = Complex32::new(
                FRAC_1_SQRT_2 * (x_odd_3.im - x_odd_3.re),
                -FRAC_1_SQRT_2 * (x_odd_3.re + x_odd_3.im),
            );
            dst[j + 3] = x_even_3.add(&w8_3_odd_3);
            dst[j + 7] = x_even_3.sub(&w8_3_odd_3);
        }

        // Process scalar tail with identity twiddles.
        for i in simd_iters..eighth_samples {
            // Load 8 input values
            let z0 = src[i];
            let z1 = src[i + eighth_samples];
            let z2 = src[i + eighth_samples * 2];
            let z3 = src[i + eighth_samples * 3];
            let z4 = src[i + eighth_samples * 4];
            let z5 = src[i + eighth_samples * 5];
            let z6 = src[i + eighth_samples * 6];
            let z7 = src[i + eighth_samples * 7];

            // Identity twiddles: t_k = z_k
            let t1 = z1;
            let t2 = z2;
            let t3 = z3;
            let t4 = z4;
            let t5 = z5;
            let t6 = z6;
            let t7 = z7;

            // Split-radix decomposition:
            // First, compute radix-4 DFT on even indices (z0, t2, t4, t6)
            let even_a0 = z0.add(&t4);
            let even_a1 = z0.sub(&t4);
            let even_a2 = t2.add(&t6);
            let even_a3_re = t2.im - t6.im;
            let even_a3_im = t6.re - t2.re;

            let x_even_0 = even_a0.add(&even_a2);
            let x_even_2 = even_a0.sub(&even_a2);
            let x_even_1 = Complex32::new(even_a1.re + even_a3_re, even_a1.im + even_a3_im);
            let x_even_3 = Complex32::new(even_a1.re - even_a3_re, even_a1.im - even_a3_im);

            // Compute radix-4 DFT on odd indices (t1, t3, t5, t7)
            let odd_a0 = t1.add(&t5);
            let odd_a1 = t1.sub(&t5);
            let odd_a2 = t3.add(&t7);
            let odd_a3_re = t3.im - t7.im;
            let odd_a3_im = t7.re - t3.re;

            let x_odd_0 = odd_a0.add(&odd_a2);
            let x_odd_2 = odd_a0.sub(&odd_a2);
            let x_odd_1 = Complex32::new(odd_a1.re + odd_a3_re, odd_a1.im + odd_a3_im);
            let x_odd_3 = Complex32::new(odd_a1.re - odd_a3_re, odd_a1.im - odd_a3_im);

            // Combine even and odd parts with additional twiddle factors
            use core::f32::consts::FRAC_1_SQRT_2;

            let j = 8 * i;
            dst[j] = x_even_0.add(&x_odd_0);
            dst[j + 4] = x_even_0.sub(&x_odd_0);

            let w8_1_odd_1 = Complex32::new(
                FRAC_1_SQRT_2 * (x_odd_1.re + x_odd_1.im),
                FRAC_1_SQRT_2 * (x_odd_1.im - x_odd_1.re),
            );
            dst[j + 1] = x_even_1.add(&w8_1_odd_1);
            dst[j + 5] = x_even_1.sub(&w8_1_odd_1);

            let w8_2_odd_2 = Complex32::new(x_odd_2.im, -x_odd_2.re);
            dst[j + 2] = x_even_2.add(&w8_2_odd_2);
            dst[j + 6] = x_even_2.sub(&w8_2_odd_2);

            let w8_3_odd_3 = Complex32::new(
                FRAC_1_SQRT_2 * (x_odd_3.im - x_odd_3.re),
                -FRAC_1_SQRT_2 * (x_odd_3.re + x_odd_3.im),
            );
            dst[j + 3] = x_even_3.add(&w8_3_odd_3);
            dst[j + 7] = x_even_3.sub(&w8_3_odd_3);
        }

        return;
    }

    // Process iterations using packed twiddle format.
    for i in start_index..simd_iters {
        let k = i % stride;

        // Calculate twiddle indices in packed format (width-aware).
        let group_idx = i / WIDTH;
        let offset_in_group = i % WIDTH;
        let tw_base = group_idx * (7 * WIDTH) + offset_in_group;
        let w1 = stage_twiddles[tw_base];
        let w2 = stage_twiddles[tw_base + WIDTH];
        let w3 = stage_twiddles[tw_base + WIDTH * 2];
        let w4 = stage_twiddles[tw_base + WIDTH * 3];
        let w5 = stage_twiddles[tw_base + WIDTH * 4];
        let w6 = stage_twiddles[tw_base + WIDTH * 5];
        let w7 = stage_twiddles[tw_base + WIDTH * 6];

        // Load 8 input values.
        let z0 = src[i];
        let z1 = src[i + eighth_samples];
        let z2 = src[i + eighth_samples * 2];
        let z3 = src[i + eighth_samples * 3];
        let z4 = src[i + eighth_samples * 4];
        let z5 = src[i + eighth_samples * 5];
        let z6 = src[i + eighth_samples * 6];
        let z7 = src[i + eighth_samples * 7];

        // Apply twiddle factors.
        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);
        let t5 = w5.mul(&z5);
        let t6 = w6.mul(&z6);
        let t7 = w7.mul(&z7);

        // Split-radix decomposition:
        // First, compute radix-4 DFT on even indices (z0, t2, t4, t6)
        let even_a0 = z0.add(&t4);
        let even_a1 = z0.sub(&t4);
        let even_a2 = t2.add(&t6);
        let even_a3_re = t2.im - t6.im;
        let even_a3_im = t6.re - t2.re;

        let x_even_0 = even_a0.add(&even_a2);
        let x_even_2 = even_a0.sub(&even_a2);
        let x_even_1 = Complex32::new(even_a1.re + even_a3_re, even_a1.im + even_a3_im);
        let x_even_3 = Complex32::new(even_a1.re - even_a3_re, even_a1.im - even_a3_im);

        // Compute radix-4 DFT on odd indices (t1, t3, t5, t7)
        let odd_a0 = t1.add(&t5);
        let odd_a1 = t1.sub(&t5);
        let odd_a2 = t3.add(&t7);
        let odd_a3_re = t3.im - t7.im;
        let odd_a3_im = t7.re - t3.re;

        let x_odd_0 = odd_a0.add(&odd_a2);
        let x_odd_2 = odd_a0.sub(&odd_a2);
        let x_odd_1 = Complex32::new(odd_a1.re + odd_a3_re, odd_a1.im + odd_a3_im);
        let x_odd_3 = Complex32::new(odd_a1.re - odd_a3_re, odd_a1.im - odd_a3_im);

        // Combine even and odd parts with additional twiddle factors.
        // For split-radix radix-8, we need to apply W_8^k = exp(-i*2π*k/8) to odd parts
        // W_8^0 = 1
        // W_8^1 = exp(-i*π/4) = (1-i)/sqrt(2) ≈ 0.707 - 0.707i
        // W_8^2 = exp(-i*π/2) = -i
        // W_8^3 = exp(-i*3π/4) = (-1-i)/sqrt(2) ≈ -0.707 - 0.707i

        use core::f32::consts::FRAC_1_SQRT_2;

        // X[0] = X_even[0] + X_odd[0]
        // X[4] = X_even[0] - X_odd[0]
        let j = 8 * i - 7 * k;
        dst[j] = x_even_0.add(&x_odd_0);
        dst[j + stride * 4] = x_even_0.sub(&x_odd_0);

        // X[1] = X_even[1] + W_8^1 * X_odd[1]
        // X[5] = X_even[1] - W_8^1 * X_odd[1]
        // W_8^1 = (1-i)/sqrt(2)
        let w8_1_odd_1 = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd_1.re + x_odd_1.im),
            FRAC_1_SQRT_2 * (x_odd_1.im - x_odd_1.re),
        );
        dst[j + stride] = x_even_1.add(&w8_1_odd_1);
        dst[j + stride * 5] = x_even_1.sub(&w8_1_odd_1);

        // X[2] = X_even[2] + W_8^2 * X_odd[2]
        // X[6] = X_even[2] - W_8^2 * X_odd[2]
        // W_8^2 = -i, so multiply by -i: (a+bi)*(-i) = b - ai
        let w8_2_odd_2 = Complex32::new(x_odd_2.im, -x_odd_2.re);
        dst[j + stride * 2] = x_even_2.add(&w8_2_odd_2);
        dst[j + stride * 6] = x_even_2.sub(&w8_2_odd_2);

        // X[3] = X_even[3] + W_8^3 * X_odd[3]
        // X[7] = X_even[3] - W_8^3 * X_odd[3]
        // W_8^3 = (-1-i)/sqrt(2) = (-1/√2, -1/√2)
        // Multiplication: (-1-i)/√2 * (a+bi) = [(b-a)/√2, -(a+b)/√2]
        let w8_3_odd_3 = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd_3.im - x_odd_3.re),
            -FRAC_1_SQRT_2 * (x_odd_3.re + x_odd_3.im),
        );
        dst[j + stride * 3] = x_even_3.add(&w8_3_odd_3);
        dst[j + stride * 7] = x_even_3.sub(&w8_3_odd_3);
    }

    // Process scalar tail using interleaved twiddle format
    let tail_offset = (simd_iters / WIDTH) * (7 * WIDTH);
    for i in simd_iters..eighth_samples {
        let k = i % stride;
        let tail_idx = i - simd_iters;
        let w1 = stage_twiddles[tail_offset + tail_idx * 7];
        let w2 = stage_twiddles[tail_offset + tail_idx * 7 + 1];
        let w3 = stage_twiddles[tail_offset + tail_idx * 7 + 2];
        let w4 = stage_twiddles[tail_offset + tail_idx * 7 + 3];
        let w5 = stage_twiddles[tail_offset + tail_idx * 7 + 4];
        let w6 = stage_twiddles[tail_offset + tail_idx * 7 + 5];
        let w7 = stage_twiddles[tail_offset + tail_idx * 7 + 6];

        // Load 8 input values
        let z0 = src[i];
        let z1 = src[i + eighth_samples];
        let z2 = src[i + eighth_samples * 2];
        let z3 = src[i + eighth_samples * 3];
        let z4 = src[i + eighth_samples * 4];
        let z5 = src[i + eighth_samples * 5];
        let z6 = src[i + eighth_samples * 6];
        let z7 = src[i + eighth_samples * 7];

        // Apply twiddle factors
        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);
        let t4 = w4.mul(&z4);
        let t5 = w5.mul(&z5);
        let t6 = w6.mul(&z6);
        let t7 = w7.mul(&z7);

        // Split-radix decomposition:
        // First, compute radix-4 DFT on even indices (z0, t2, t4, t6)
        let even_a0 = z0.add(&t4);
        let even_a1 = z0.sub(&t4);
        let even_a2 = t2.add(&t6);
        let even_a3_re = t2.im - t6.im;
        let even_a3_im = t6.re - t2.re;

        let x_even_0 = even_a0.add(&even_a2);
        let x_even_2 = even_a0.sub(&even_a2);
        let x_even_1 = Complex32::new(even_a1.re + even_a3_re, even_a1.im + even_a3_im);
        let x_even_3 = Complex32::new(even_a1.re - even_a3_re, even_a1.im - even_a3_im);

        // Compute radix-4 DFT on odd indices (t1, t3, t5, t7)
        let odd_a0 = t1.add(&t5);
        let odd_a1 = t1.sub(&t5);
        let odd_a2 = t3.add(&t7);
        let odd_a3_re = t3.im - t7.im;
        let odd_a3_im = t7.re - t3.re;

        let x_odd_0 = odd_a0.add(&odd_a2);
        let x_odd_2 = odd_a0.sub(&odd_a2);
        let x_odd_1 = Complex32::new(odd_a1.re + odd_a3_re, odd_a1.im + odd_a3_im);
        let x_odd_3 = Complex32::new(odd_a1.re - odd_a3_re, odd_a1.im - odd_a3_im);

        // Combine even and odd parts with additional twiddle factors
        use core::f32::consts::FRAC_1_SQRT_2;

        let j = 8 * i - 7 * k;
        dst[j] = x_even_0.add(&x_odd_0);
        dst[j + stride * 4] = x_even_0.sub(&x_odd_0);

        let w8_1_odd_1 = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd_1.re + x_odd_1.im),
            FRAC_1_SQRT_2 * (x_odd_1.im - x_odd_1.re),
        );
        dst[j + stride] = x_even_1.add(&w8_1_odd_1);
        dst[j + stride * 5] = x_even_1.sub(&w8_1_odd_1);

        let w8_2_odd_2 = Complex32::new(x_odd_2.im, -x_odd_2.re);
        dst[j + stride * 2] = x_even_2.add(&w8_2_odd_2);
        dst[j + stride * 6] = x_even_2.sub(&w8_2_odd_2);

        let w8_3_odd_3 = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd_3.im - x_odd_3.re),
            -FRAC_1_SQRT_2 * (x_odd_3.re + x_odd_3.im),
        );
        dst[j + stride * 3] = x_even_3.add(&w8_3_odd_3);
        dst[j + stride * 7] = x_even_3.sub(&w8_3_odd_3);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    /// Naive complex-to-complex DFT for validation.
    /// Computes X[k] = Σ(n=0 to N-1) x[n] * exp(-i*2π*k*n/N)
    fn naive_complex_dft(input: &[Complex32]) -> alloc::vec::Vec<Complex32> {
        let n = input.len();
        let mut output = vec![Complex32::zero(); n];

        for k in 0..n {
            let mut sum = Complex32::zero();
            for (i, &x) in input.iter().enumerate() {
                let angle = -2.0 * core::f32::consts::PI * (k * i) as f32 / n as f32;
                #[cfg(not(feature = "no_std"))]
                let twiddle = Complex32::new(angle.cos(), angle.sin());
                #[cfg(feature = "no_std")]
                let twiddle = Complex32::new(libm::cosf(angle), libm::sinf(angle));
                sum = sum.add(&twiddle.mul(&x));
            }
            output[k] = sum;
        }

        output
    }

    #[test]
    fn test_radix8_vs_naive_dft() {
        let test_cases = [
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
            ],
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(-1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(-1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(-1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(-1.0, 0.0),
            ],
            vec![
                Complex32::new(1.0, 0.5),
                Complex32::new(0.5, -1.0),
                Complex32::new(-1.0, 0.5),
                Complex32::new(0.5, 1.0),
                Complex32::new(1.0, -0.5),
                Complex32::new(-0.5, -1.0),
                Complex32::new(-1.0, -0.5),
                Complex32::new(-0.5, 1.0),
            ],
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
            ],
        ];

        for (test_idx, input) in test_cases.iter().enumerate() {
            // For stride=1, single-stage radix-8 FFT: all twiddles are identity.
            let stride = 1;
            let stage_size = 8;
            let mut twiddles = vec![Complex32::zero(); 7];
            for k in 1..8 {
                // col = 0 for the single iteration when stride=1
                let col = 0;
                let angle = -2.0 * core::f32::consts::PI * (col * k) as f32 / stage_size as f32;
                #[cfg(not(feature = "no_std"))]
                let tw = Complex32::new(angle.cos(), angle.sin());
                #[cfg(feature = "no_std")]
                let tw = Complex32::new(libm::cosf(angle), libm::sinf(angle));
                twiddles[k - 1] = tw;
            }

            let mut dst = vec![Complex32::zero(); 8];
            butterfly_radix8_scalar::<2>(input, &mut dst, &twiddles, stride, 0);

            let expected = naive_complex_dft(input);

            for i in 0..8 {
                let diff_re = (dst[i].re - expected[i].re).abs();
                let diff_im = (dst[i].im - expected[i].im).abs();
                assert!(
                    diff_re < 1e-5 && diff_im < 1e-5,
                    "Test case {test_idx}: Mismatch at index {i}: butterfly=({}, {}), dft=({}, {}), diff=({}, {})",
                    dst[i].re,
                    dst[i].im,
                    expected[i].re,
                    expected[i].im,
                    diff_re,
                    diff_im
                );
            }
        }
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        all(target_feature = "avx", target_feature = "fma")
    ))]
    fn test_butterfly_radix8_avx_fma_vs_scalar() {
        use crate::fft::butterflies::tests::{TestSimdWidth, test_butterfly_against_scalar};
        test_butterfly_against_scalar(
            butterfly_radix8_scalar::<4>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    avx::butterfly_radix8_stride1_avx_fma(src, dst, twiddles);
                } else {
                    avx::butterfly_radix8_generic_avx_fma(src, dst, twiddles, p);
                }
            },
            8,
            7,
            TestSimdWidth::Width4,
            "butterfly_radix8_avx_fma",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    fn test_butterfly_radix8_sse2_vs_scalar() {
        use crate::fft::butterflies::tests::{TestSimdWidth, test_butterfly_against_scalar};
        test_butterfly_against_scalar(
            butterfly_radix8_scalar::<2>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse2::butterfly_radix8_stride1_sse2(src, dst, twiddles);
                } else {
                    sse2::butterfly_radix8_generic_sse2(src, dst, twiddles, p);
                }
            },
            8,
            7,
            TestSimdWidth::Width2,
            "butterfly_radix8_sse2",
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn test_butterfly_radix8_sse4_2_vs_scalar() {
        use crate::fft::butterflies::tests::{TestSimdWidth, test_butterfly_against_scalar};
        test_butterfly_against_scalar(
            butterfly_radix8_scalar::<2>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    sse4_2::butterfly_radix8_stride1_sse4_2(src, dst, twiddles);
                } else {
                    sse4_2::butterfly_radix8_generic_sse4_2(src, dst, twiddles, p);
                }
            },
            8,
            7,
            TestSimdWidth::Width2,
            "butterfly_radix8_sse4_2",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn test_butterfly_radix8_neon_vs_scalar() {
        use crate::fft::butterflies::tests::{TestSimdWidth, test_butterfly_against_scalar};
        test_butterfly_against_scalar(
            butterfly_radix8_scalar::<2>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    neon::butterfly_radix8_stride1_neon(src, dst, twiddles);
                } else {
                    neon::butterfly_radix8_generic_neon(src, dst, twiddles, p);
                }
            },
            8,
            7,
            TestSimdWidth::Width2,
            "butterfly_radix8_neon",
        );
    }
}
