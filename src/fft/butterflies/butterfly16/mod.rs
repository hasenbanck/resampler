use crate::fft::Complex32;

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
mod avx;

const W16_1_RE: f32 = 0.9238795;
const W16_1_IM: f32 = -0.38268343;
const W16_3_RE: f32 = 0.38268343;
const W16_3_IM: f32 = -0.9238795;

/// Dispatch function for radix-16 Stockham butterfly.
#[inline(always)]
pub(crate) fn butterfly_radix16_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    #[allow(unused)]
    let samples = src.len();
    #[allow(unused)]
    let sixteenth_samples = samples >> 4;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        if sixteenth_samples >= 4 {
            return unsafe { butterfly_radix16_avx_fma_dispatch(src, dst, stage_twiddles, stride) };
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    butterfly_radix16_scalar::<4>(src, dst, stage_twiddles, stride);
    #[cfg(all(
        not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")),
        not(target_arch = "aarch64")
    ))]
    butterfly_radix16_scalar::<2>(src, dst, stage_twiddles, stride);
    #[cfg(target_arch = "aarch64")]
    butterfly_radix16_scalar::<2>(src, dst, stage_twiddles, stride);
}

/// Dispatch to AVX implementations (stride-1 vs generic).
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline(always)]
unsafe fn butterfly_radix16_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    if stride == 1 {
        unsafe { avx::butterfly_radix16_stride1_avx_fma(src, dst, stage_twiddles) };
    } else {
        unsafe { avx::butterfly_radix16_generic_avx_fma(src, dst, stage_twiddles, stride) };
    }
}

/// AVX+FMA dispatcher for p1 (stride=1) variant.
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix16_stride1_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let sixteenth_samples = samples >> 4;

    if sixteenth_samples >= 4 {
        return unsafe { avx::butterfly_radix16_stride1_avx_fma(src, dst, stage_twiddles) };
    }

    butterfly_radix16_scalar::<4>(src, dst, stage_twiddles, 1);
}

/// AVX+FMA dispatcher for generic (stride>1) variant.
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix16_generic_avx_fma_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let sixteenth_samples = samples >> 4;

    if sixteenth_samples >= 4 {
        return unsafe { avx::butterfly_radix16_generic_avx_fma(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix16_scalar::<4>(src, dst, stage_twiddles, stride);
}

/// SSE2 dispatcher for p1 (stride=1) variant.
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix16_stride1_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let sixteenth_samples = samples >> 4;

    if sixteenth_samples >= 2 {
        return unsafe { avx::butterfly_radix16_stride1_avx_fma(src, dst, stage_twiddles) };
    }

    butterfly_radix16_scalar::<2>(src, dst, stage_twiddles, 1);
}

/// SSE2 dispatcher for generic (stride>1) variant.
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix16_generic_sse2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let sixteenth_samples = samples >> 4;

    if sixteenth_samples >= 2 {
        return unsafe { avx::butterfly_radix16_generic_avx_fma(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix16_scalar::<2>(src, dst, stage_twiddles, stride);
}

/// SSE4.2 dispatcher for p1 (stride=1) variant.
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix16_stride1_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let sixteenth_samples = samples >> 4;

    if sixteenth_samples >= 2 {
        return unsafe { avx::butterfly_radix16_stride1_avx_fma(src, dst, stage_twiddles) };
    }

    butterfly_radix16_scalar::<2>(src, dst, stage_twiddles, 1);
}

/// SSE4.2 dispatcher for generic (stride>1) variant.
#[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
#[inline(always)]
pub(crate) fn butterfly_radix16_generic_sse4_2_dispatch(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let sixteenth_samples = samples >> 4;

    if sixteenth_samples >= 2 {
        return unsafe { avx::butterfly_radix16_generic_avx_fma(src, dst, stage_twiddles, stride) };
    }

    butterfly_radix16_scalar::<2>(src, dst, stage_twiddles, stride);
}

/// Performs a single radix-16 Stockham butterfly stage (out-of-place, scalar).
///
/// This implementation uses split-radix decomposition (2-way split):
/// - Split 16 inputs into EVEN and ODD indexed elements
/// - Apply radix-8 split-radix to each 8-element group
/// - Combine using W_16 twiddle factors
///
/// Expects twiddles in packed format matching SIMD code:
/// - Packed portion (for groups of iterations): [w1[i], w1[i+1], ..., w2[i], w2[i+1], ...]
/// - Scalar tail (if any): [w1[i], w2[i], ..., w15[i], ...] (interleaved)
#[allow(dead_code)]
#[inline(always)]
pub(super) fn butterfly_radix16_scalar<const WIDTH: usize>(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    let samples = src.len();
    let sixteenth_samples = samples >> 4;
    let simd_iters = (sixteenth_samples / WIDTH) * WIDTH;

    // Process iterations using packed twiddle format.
    for i in 0..simd_iters {
        let k = i % stride;

        // Calculate twiddle indices in packed format (width-aware).
        let group_idx = i / WIDTH;
        let offset_in_group = i % WIDTH;
        let tw_base = group_idx * (15 * WIDTH) + offset_in_group;

        // Load all 15 twiddle factors.
        let w = [
            stage_twiddles[tw_base],
            stage_twiddles[tw_base + WIDTH],
            stage_twiddles[tw_base + WIDTH * 2],
            stage_twiddles[tw_base + WIDTH * 3],
            stage_twiddles[tw_base + WIDTH * 4],
            stage_twiddles[tw_base + WIDTH * 5],
            stage_twiddles[tw_base + WIDTH * 6],
            stage_twiddles[tw_base + WIDTH * 7],
            stage_twiddles[tw_base + WIDTH * 8],
            stage_twiddles[tw_base + WIDTH * 9],
            stage_twiddles[tw_base + WIDTH * 10],
            stage_twiddles[tw_base + WIDTH * 11],
            stage_twiddles[tw_base + WIDTH * 12],
            stage_twiddles[tw_base + WIDTH * 13],
            stage_twiddles[tw_base + WIDTH * 14],
        ];

        // Load 16 input values.
        let z = [
            src[i],
            src[i + sixteenth_samples],
            src[i + sixteenth_samples * 2],
            src[i + sixteenth_samples * 3],
            src[i + sixteenth_samples * 4],
            src[i + sixteenth_samples * 5],
            src[i + sixteenth_samples * 6],
            src[i + sixteenth_samples * 7],
            src[i + sixteenth_samples * 8],
            src[i + sixteenth_samples * 9],
            src[i + sixteenth_samples * 10],
            src[i + sixteenth_samples * 11],
            src[i + sixteenth_samples * 12],
            src[i + sixteenth_samples * 13],
            src[i + sixteenth_samples * 14],
            src[i + sixteenth_samples * 15],
        ];

        // Apply twiddle factors.
        let t = [
            w[0].mul(&z[1]),
            w[1].mul(&z[2]),
            w[2].mul(&z[3]),
            w[3].mul(&z[4]),
            w[4].mul(&z[5]),
            w[5].mul(&z[6]),
            w[6].mul(&z[7]),
            w[7].mul(&z[8]),
            w[8].mul(&z[9]),
            w[9].mul(&z[10]),
            w[10].mul(&z[11]),
            w[11].mul(&z[12]),
            w[12].mul(&z[13]),
            w[13].mul(&z[14]),
            w[14].mul(&z[15]),
        ];

        // Split-radix 2-way decomposition:
        // EVEN indices: 0, 2, 4, 6, 8, 10, 12, 14
        // ODD indices: 1, 3, 5, 7, 9, 11, 13, 15

        // Process EVEN group (8 elements) using radix-8 split-radix.
        // Further split EVEN into even-even and even-odd.
        // Even-even: [z0, z4, z8, z12] = [z0, t[3], t[7], t[11]]
        // Even-odd: [z2, z6, z10, z14] = [t[1], t[5], t[9], t[13]]

        let z0 = z[0];
        let z2 = t[1];
        let z4 = t[3];
        let z6 = t[5];
        let z8 = t[7];
        let z10 = t[9];
        let z12 = t[11];
        let z14 = t[13];

        // Radix-4 on even-even: [z0, z4, z8, z12]
        let ee_a0 = z0.add(&z8);
        let ee_a1 = z0.sub(&z8);
        let ee_a2 = z4.add(&z12);
        let ee_a3_re = z4.im - z12.im;
        let ee_a3_im = z12.re - z4.re;

        let x_ee_0 = ee_a0.add(&ee_a2);
        let x_ee_2 = ee_a0.sub(&ee_a2);
        let x_ee_1 = Complex32::new(ee_a1.re + ee_a3_re, ee_a1.im + ee_a3_im);
        let x_ee_3 = Complex32::new(ee_a1.re - ee_a3_re, ee_a1.im - ee_a3_im);

        // Radix-4 on even-odd: [z2, z6, z10, z14]
        let eo_a0 = z2.add(&z10);
        let eo_a1 = z2.sub(&z10);
        let eo_a2 = z6.add(&z14);
        let eo_a3_re = z6.im - z14.im;
        let eo_a3_im = z14.re - z6.re;

        let x_eo_0 = eo_a0.add(&eo_a2);
        let x_eo_2 = eo_a0.sub(&eo_a2);
        let x_eo_1 = Complex32::new(eo_a1.re + eo_a3_re, eo_a1.im + eo_a3_im);
        let x_eo_3 = Complex32::new(eo_a1.re - eo_a3_re, eo_a1.im - eo_a3_im);

        // Combine with W_8 twiddles to get X_even[0..7]
        use core::f32::consts::FRAC_1_SQRT_2;

        let x_even = [
            x_ee_0.add(&x_eo_0),
            Complex32::new(
                x_ee_1.re + FRAC_1_SQRT_2 * (x_eo_1.re + x_eo_1.im),
                x_ee_1.im + FRAC_1_SQRT_2 * (x_eo_1.im - x_eo_1.re),
            ),
            x_ee_2.add(&Complex32::new(x_eo_2.im, -x_eo_2.re)),
            Complex32::new(
                x_ee_3.re + FRAC_1_SQRT_2 * (x_eo_3.im - x_eo_3.re),
                x_ee_3.im - FRAC_1_SQRT_2 * (x_eo_3.re + x_eo_3.im),
            ),
            x_ee_0.sub(&x_eo_0),
            Complex32::new(
                x_ee_1.re - FRAC_1_SQRT_2 * (x_eo_1.re + x_eo_1.im),
                x_ee_1.im - FRAC_1_SQRT_2 * (x_eo_1.im - x_eo_1.re),
            ),
            x_ee_2.sub(&Complex32::new(x_eo_2.im, -x_eo_2.re)),
            Complex32::new(
                x_ee_3.re - FRAC_1_SQRT_2 * (x_eo_3.im - x_eo_3.re),
                x_ee_3.im + FRAC_1_SQRT_2 * (x_eo_3.re + x_eo_3.im),
            ),
        ];

        // Process ODD group (8 elements) using radix-8 split-radix.
        // Further split ODD into odd-even and odd-odd.
        // Odd-even: [t[0], t[4], t[8], t[12]] = [t[0], t[4], t[8], t[12]]
        // Odd-odd: [t[2], t[6], t[10], t[14]] = [t[2], t[6], t[10], t[14]]

        let z1 = t[0];
        let z3 = t[2];
        let z5 = t[4];
        let z7 = t[6];
        let z9 = t[8];
        let z11 = t[10];
        let z13 = t[12];
        let z15 = t[14];

        // Radix-4 on odd-even: [z1, z5, z9, z13]
        let oe_a0 = z1.add(&z9);
        let oe_a1 = z1.sub(&z9);
        let oe_a2 = z5.add(&z13);
        let oe_a3_re = z5.im - z13.im;
        let oe_a3_im = z13.re - z5.re;

        let x_oe_0 = oe_a0.add(&oe_a2);
        let x_oe_2 = oe_a0.sub(&oe_a2);
        let x_oe_1 = Complex32::new(oe_a1.re + oe_a3_re, oe_a1.im + oe_a3_im);
        let x_oe_3 = Complex32::new(oe_a1.re - oe_a3_re, oe_a1.im - oe_a3_im);

        // Radix-4 on odd-odd: [z3, z7, z11, z15]
        let oo_a0 = z3.add(&z11);
        let oo_a1 = z3.sub(&z11);
        let oo_a2 = z7.add(&z15);
        let oo_a3_re = z7.im - z15.im;
        let oo_a3_im = z15.re - z7.re;

        let x_oo_0 = oo_a0.add(&oo_a2);
        let x_oo_2 = oo_a0.sub(&oo_a2);
        let x_oo_1 = Complex32::new(oo_a1.re + oo_a3_re, oo_a1.im + oo_a3_im);
        let x_oo_3 = Complex32::new(oo_a1.re - oo_a3_re, oo_a1.im - oo_a3_im);

        // Combine with W_8 twiddles to get X_odd[0..7]
        let x_odd = [
            x_oe_0.add(&x_oo_0),
            Complex32::new(
                x_oe_1.re + FRAC_1_SQRT_2 * (x_oo_1.re + x_oo_1.im),
                x_oe_1.im + FRAC_1_SQRT_2 * (x_oo_1.im - x_oo_1.re),
            ),
            x_oe_2.add(&Complex32::new(x_oo_2.im, -x_oo_2.re)),
            Complex32::new(
                x_oe_3.re + FRAC_1_SQRT_2 * (x_oo_3.im - x_oo_3.re),
                x_oe_3.im - FRAC_1_SQRT_2 * (x_oo_3.re + x_oo_3.im),
            ),
            x_oe_0.sub(&x_oo_0),
            Complex32::new(
                x_oe_1.re - FRAC_1_SQRT_2 * (x_oo_1.re + x_oo_1.im),
                x_oe_1.im - FRAC_1_SQRT_2 * (x_oo_1.im - x_oo_1.re),
            ),
            x_oe_2.sub(&Complex32::new(x_oo_2.im, -x_oo_2.re)),
            Complex32::new(
                x_oe_3.re - FRAC_1_SQRT_2 * (x_oo_3.im - x_oo_3.re),
                x_oe_3.im + FRAC_1_SQRT_2 * (x_oo_3.re + x_oo_3.im),
            ),
        ];

        // Final combining with W_16 twiddles.
        // For k=0..7:
        // X[k] = X_even[k] + W_16^k * X_odd[k]
        // X[k+8] = X_even[k] - W_16^k * X_odd[k]

        let j = 16 * i - 15 * k;

        // k=0: W_16^0 = 1
        let w16_0_odd = x_odd[0];
        dst[j] = x_even[0].add(&w16_0_odd);
        dst[j + stride * 8] = x_even[0].sub(&w16_0_odd);

        // k=1: W_16^1
        let w16_1_odd = Complex32::new(
            W16_1_RE * x_odd[1].re - W16_1_IM * x_odd[1].im,
            W16_1_RE * x_odd[1].im + W16_1_IM * x_odd[1].re,
        );
        dst[j + stride] = x_even[1].add(&w16_1_odd);
        dst[j + stride * 9] = x_even[1].sub(&w16_1_odd);

        // k=2: W_16^2 = W_8^1 = (1-i)/sqrt(2)
        let w16_2_odd = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd[2].re + x_odd[2].im),
            FRAC_1_SQRT_2 * (x_odd[2].im - x_odd[2].re),
        );
        dst[j + stride * 2] = x_even[2].add(&w16_2_odd);
        dst[j + stride * 10] = x_even[2].sub(&w16_2_odd);

        // k=3: W_16^3
        let w16_3_odd = Complex32::new(
            W16_3_RE * x_odd[3].re - W16_3_IM * x_odd[3].im,
            W16_3_RE * x_odd[3].im + W16_3_IM * x_odd[3].re,
        );
        dst[j + stride * 3] = x_even[3].add(&w16_3_odd);
        dst[j + stride * 11] = x_even[3].sub(&w16_3_odd);

        // k=4: W_16^4 = W_8^2 = -i
        let w16_4_odd = Complex32::new(x_odd[4].im, -x_odd[4].re);
        dst[j + stride * 4] = x_even[4].add(&w16_4_odd);
        dst[j + stride * 12] = x_even[4].sub(&w16_4_odd);

        // k=5: W_16^5 = -W_16^3 = -0.38268343 - 0.92387953i
        let w16_5_odd = Complex32::new(
            -W16_3_RE * x_odd[5].re - W16_3_IM * x_odd[5].im,
            -W16_3_RE * x_odd[5].im + W16_3_IM * x_odd[5].re,
        );
        dst[j + stride * 5] = x_even[5].add(&w16_5_odd);
        dst[j + stride * 13] = x_even[5].sub(&w16_5_odd);

        // k=6: W_16^6 = W_8^3 = (-1-i)/sqrt(2)
        let w16_6_odd = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd[6].im - x_odd[6].re),
            -FRAC_1_SQRT_2 * (x_odd[6].re + x_odd[6].im),
        );
        dst[j + stride * 6] = x_even[6].add(&w16_6_odd);
        dst[j + stride * 14] = x_even[6].sub(&w16_6_odd);

        // k=7: W_16^7 = -W_16^1 = -0.92387953 - 0.38268343i
        let w16_7_odd = Complex32::new(
            -W16_1_RE * x_odd[7].re - W16_1_IM * x_odd[7].im,
            -W16_1_RE * x_odd[7].im + W16_1_IM * x_odd[7].re,
        );
        dst[j + stride * 7] = x_even[7].add(&w16_7_odd);
        dst[j + stride * 15] = x_even[7].sub(&w16_7_odd);
    }

    // Process scalar tail using interleaved twiddle format.
    let tail_offset = (simd_iters / WIDTH) * (15 * WIDTH);
    for i in simd_iters..sixteenth_samples {
        let k = i % stride;
        let tail_idx = i - simd_iters;

        // Load twiddles
        let w: [Complex32; 15] = [
            stage_twiddles[tail_offset + tail_idx * 15],
            stage_twiddles[tail_offset + tail_idx * 15 + 1],
            stage_twiddles[tail_offset + tail_idx * 15 + 2],
            stage_twiddles[tail_offset + tail_idx * 15 + 3],
            stage_twiddles[tail_offset + tail_idx * 15 + 4],
            stage_twiddles[tail_offset + tail_idx * 15 + 5],
            stage_twiddles[tail_offset + tail_idx * 15 + 6],
            stage_twiddles[tail_offset + tail_idx * 15 + 7],
            stage_twiddles[tail_offset + tail_idx * 15 + 8],
            stage_twiddles[tail_offset + tail_idx * 15 + 9],
            stage_twiddles[tail_offset + tail_idx * 15 + 10],
            stage_twiddles[tail_offset + tail_idx * 15 + 11],
            stage_twiddles[tail_offset + tail_idx * 15 + 12],
            stage_twiddles[tail_offset + tail_idx * 15 + 13],
            stage_twiddles[tail_offset + tail_idx * 15 + 14],
        ];

        // Load 16 input values.
        let z = [
            src[i],
            src[i + sixteenth_samples],
            src[i + sixteenth_samples * 2],
            src[i + sixteenth_samples * 3],
            src[i + sixteenth_samples * 4],
            src[i + sixteenth_samples * 5],
            src[i + sixteenth_samples * 6],
            src[i + sixteenth_samples * 7],
            src[i + sixteenth_samples * 8],
            src[i + sixteenth_samples * 9],
            src[i + sixteenth_samples * 10],
            src[i + sixteenth_samples * 11],
            src[i + sixteenth_samples * 12],
            src[i + sixteenth_samples * 13],
            src[i + sixteenth_samples * 14],
            src[i + sixteenth_samples * 15],
        ];

        // Apply twiddle factors.
        let t = [
            w[0].mul(&z[1]),
            w[1].mul(&z[2]),
            w[2].mul(&z[3]),
            w[3].mul(&z[4]),
            w[4].mul(&z[5]),
            w[5].mul(&z[6]),
            w[6].mul(&z[7]),
            w[7].mul(&z[8]),
            w[8].mul(&z[9]),
            w[9].mul(&z[10]),
            w[10].mul(&z[11]),
            w[11].mul(&z[12]),
            w[12].mul(&z[13]),
            w[13].mul(&z[14]),
            w[14].mul(&z[15]),
        ];

        // Same processing as packed loop.
        let z0 = z[0];
        let z2 = t[1];
        let z4 = t[3];
        let z6 = t[5];
        let z8 = t[7];
        let z10 = t[9];
        let z12 = t[11];
        let z14 = t[13];

        let ee_a0 = z0.add(&z8);
        let ee_a1 = z0.sub(&z8);
        let ee_a2 = z4.add(&z12);
        let ee_a3_re = z4.im - z12.im;
        let ee_a3_im = z12.re - z4.re;

        let x_ee_0 = ee_a0.add(&ee_a2);
        let x_ee_2 = ee_a0.sub(&ee_a2);
        let x_ee_1 = Complex32::new(ee_a1.re + ee_a3_re, ee_a1.im + ee_a3_im);
        let x_ee_3 = Complex32::new(ee_a1.re - ee_a3_re, ee_a1.im - ee_a3_im);

        let z1 = t[0];
        let z3 = t[2];
        let z5 = t[4];
        let z7 = t[6];
        let z9 = t[8];
        let z11 = t[10];
        let z13 = t[12];
        let z15 = t[14];

        let oe_a0 = z1.add(&z9);
        let oe_a1 = z1.sub(&z9);
        let oe_a2 = z5.add(&z13);
        let oe_a3_re = z5.im - z13.im;
        let oe_a3_im = z13.re - z5.re;

        let x_oe_0 = oe_a0.add(&oe_a2);
        let x_oe_2 = oe_a0.sub(&oe_a2);
        let x_oe_1 = Complex32::new(oe_a1.re + oe_a3_re, oe_a1.im + oe_a3_im);
        let x_oe_3 = Complex32::new(oe_a1.re - oe_a3_re, oe_a1.im - oe_a3_im);

        let eo_a0 = z2.add(&z10);
        let eo_a1 = z2.sub(&z10);
        let eo_a2 = z6.add(&z14);
        let eo_a3_re = z6.im - z14.im;
        let eo_a3_im = z14.re - z6.re;

        let x_eo_0 = eo_a0.add(&eo_a2);
        let x_eo_2 = eo_a0.sub(&eo_a2);
        let x_eo_1 = Complex32::new(eo_a1.re + eo_a3_re, eo_a1.im + eo_a3_im);
        let x_eo_3 = Complex32::new(eo_a1.re - eo_a3_re, eo_a1.im - eo_a3_im);

        let oo_a0 = z3.add(&z11);
        let oo_a1 = z3.sub(&z11);
        let oo_a2 = z7.add(&z15);
        let oo_a3_re = z7.im - z15.im;
        let oo_a3_im = z15.re - z7.re;

        let x_oo_0 = oo_a0.add(&oo_a2);
        let x_oo_2 = oo_a0.sub(&oo_a2);
        let x_oo_1 = Complex32::new(oo_a1.re + oo_a3_re, oo_a1.im + oo_a3_im);
        let x_oo_3 = Complex32::new(oo_a1.re - oo_a3_re, oo_a1.im - oo_a3_im);

        use core::f32::consts::FRAC_1_SQRT_2;

        let x_even = [
            x_ee_0.add(&x_eo_0),
            Complex32::new(
                x_ee_1.re + FRAC_1_SQRT_2 * (x_eo_1.re + x_eo_1.im),
                x_ee_1.im + FRAC_1_SQRT_2 * (x_eo_1.im - x_eo_1.re),
            ),
            x_ee_2.add(&Complex32::new(x_eo_2.im, -x_eo_2.re)),
            Complex32::new(
                x_ee_3.re + FRAC_1_SQRT_2 * (x_eo_3.im - x_eo_3.re),
                x_ee_3.im - FRAC_1_SQRT_2 * (x_eo_3.re + x_eo_3.im),
            ),
            x_ee_0.sub(&x_eo_0),
            Complex32::new(
                x_ee_1.re - FRAC_1_SQRT_2 * (x_eo_1.re + x_eo_1.im),
                x_ee_1.im - FRAC_1_SQRT_2 * (x_eo_1.im - x_eo_1.re),
            ),
            x_ee_2.sub(&Complex32::new(x_eo_2.im, -x_eo_2.re)),
            Complex32::new(
                x_ee_3.re - FRAC_1_SQRT_2 * (x_eo_3.im - x_eo_3.re),
                x_ee_3.im + FRAC_1_SQRT_2 * (x_eo_3.re + x_eo_3.im),
            ),
        ];

        let x_odd = [
            x_oe_0.add(&x_oo_0),
            Complex32::new(
                x_oe_1.re + FRAC_1_SQRT_2 * (x_oo_1.re + x_oo_1.im),
                x_oe_1.im + FRAC_1_SQRT_2 * (x_oo_1.im - x_oo_1.re),
            ),
            x_oe_2.add(&Complex32::new(x_oo_2.im, -x_oo_2.re)),
            Complex32::new(
                x_oe_3.re + FRAC_1_SQRT_2 * (x_oo_3.im - x_oo_3.re),
                x_oe_3.im - FRAC_1_SQRT_2 * (x_oo_3.re + x_oo_3.im),
            ),
            x_oe_0.sub(&x_oo_0),
            Complex32::new(
                x_oe_1.re - FRAC_1_SQRT_2 * (x_oo_1.re + x_oo_1.im),
                x_oe_1.im - FRAC_1_SQRT_2 * (x_oo_1.im - x_oo_1.re),
            ),
            x_oe_2.sub(&Complex32::new(x_oo_2.im, -x_oo_2.re)),
            Complex32::new(
                x_oe_3.re - FRAC_1_SQRT_2 * (x_oo_3.im - x_oo_3.re),
                x_oe_3.im + FRAC_1_SQRT_2 * (x_oo_3.re + x_oo_3.im),
            ),
        ];

        let j = 16 * i - 15 * k;

        let w16_0_odd = x_odd[0];
        dst[j] = x_even[0].add(&w16_0_odd);
        dst[j + stride * 8] = x_even[0].sub(&w16_0_odd);

        let w16_1_odd = Complex32::new(
            W16_1_RE * x_odd[1].re - W16_1_IM * x_odd[1].im,
            W16_1_RE * x_odd[1].im + W16_1_IM * x_odd[1].re,
        );
        dst[j + stride] = x_even[1].add(&w16_1_odd);
        dst[j + stride * 9] = x_even[1].sub(&w16_1_odd);

        let w16_2_odd = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd[2].re + x_odd[2].im),
            FRAC_1_SQRT_2 * (x_odd[2].im - x_odd[2].re),
        );
        dst[j + stride * 2] = x_even[2].add(&w16_2_odd);
        dst[j + stride * 10] = x_even[2].sub(&w16_2_odd);

        let w16_3_odd = Complex32::new(
            W16_3_RE * x_odd[3].re - W16_3_IM * x_odd[3].im,
            W16_3_RE * x_odd[3].im + W16_3_IM * x_odd[3].re,
        );
        dst[j + stride * 3] = x_even[3].add(&w16_3_odd);
        dst[j + stride * 11] = x_even[3].sub(&w16_3_odd);

        let w16_4_odd = Complex32::new(x_odd[4].im, -x_odd[4].re);
        dst[j + stride * 4] = x_even[4].add(&w16_4_odd);
        dst[j + stride * 12] = x_even[4].sub(&w16_4_odd);

        let w16_5_odd = Complex32::new(
            -W16_3_RE * x_odd[5].re - W16_3_IM * x_odd[5].im,
            -W16_3_RE * x_odd[5].im + W16_3_IM * x_odd[5].re,
        );
        dst[j + stride * 5] = x_even[5].add(&w16_5_odd);
        dst[j + stride * 13] = x_even[5].sub(&w16_5_odd);

        let w16_6_odd = Complex32::new(
            FRAC_1_SQRT_2 * (x_odd[6].im - x_odd[6].re),
            -FRAC_1_SQRT_2 * (x_odd[6].re + x_odd[6].im),
        );
        dst[j + stride * 6] = x_even[6].add(&w16_6_odd);
        dst[j + stride * 14] = x_even[6].sub(&w16_6_odd);

        let w16_7_odd = Complex32::new(
            -W16_1_RE * x_odd[7].re - W16_1_IM * x_odd[7].im,
            -W16_1_RE * x_odd[7].im + W16_1_IM * x_odd[7].re,
        );
        dst[j + stride * 7] = x_even[7].add(&w16_7_odd);
        dst[j + stride * 15] = x_even[7].sub(&w16_7_odd);
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::*;

    /// Naive complex-to-complex DFT for validation.
    fn naive_complex_dft(input: &[Complex32]) -> Vec<Complex32> {
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
    fn test_radix16_vs_naive_dft() {
        let test_cases = [
            vec![Complex32::new(1.0, 0.0); 16],
            {
                let mut v = vec![Complex32::zero(); 16];
                for i in 0..16 {
                    v[i] = if i % 2 == 0 {
                        Complex32::new(1.0, 0.0)
                    } else {
                        Complex32::new(-1.0, 0.0)
                    };
                }
                v
            },
            vec![
                Complex32::new(1.0, 0.5),
                Complex32::new(0.5, -1.0),
                Complex32::new(-1.0, 0.5),
                Complex32::new(0.5, 1.0),
                Complex32::new(1.0, -0.5),
                Complex32::new(-0.5, -1.0),
                Complex32::new(-1.0, -0.5),
                Complex32::new(-0.5, 1.0),
                Complex32::new(0.75, 0.25),
                Complex32::new(0.25, -0.75),
                Complex32::new(-0.75, 0.25),
                Complex32::new(0.25, 0.75),
                Complex32::new(0.6, -0.8),
                Complex32::new(-0.8, -0.6),
                Complex32::new(0.3, 0.9),
                Complex32::new(0.9, -0.3),
            ],
            {
                let mut v = vec![Complex32::zero(); 16];
                v[0] = Complex32::new(1.0, 0.0);
                v
            },
        ];

        for (test_idx, input) in test_cases.iter().enumerate() {
            let stride = 1;
            let stage_size = 16;
            let mut twiddles = vec![Complex32::zero(); 15];
            for k in 1..16 {
                let col = 0;
                let angle = -2.0 * core::f32::consts::PI * (col * k) as f32 / stage_size as f32;
                #[cfg(not(feature = "no_std"))]
                let tw = Complex32::new(angle.cos(), angle.sin());
                #[cfg(feature = "no_std")]
                let tw = Complex32::new(libm::cosf(angle), libm::sinf(angle));
                twiddles[k - 1] = tw;
            }

            let mut dst = vec![Complex32::zero(); 16];
            butterfly_radix16_scalar::<2>(input, &mut dst, &twiddles, stride);

            let expected = naive_complex_dft(input);

            for i in 0..16 {
                let diff_re = (dst[i].re - expected[i].re).abs();
                let diff_im = (dst[i].im - expected[i].im).abs();
                assert!(
                    diff_re < 1e-4 && diff_im < 1e-4,
                    "Test case {test_idx}: Mismatch at index {i}: butterfly=({}, {}), dft=({}, {}), diff=({diff_re}, {diff_im})",
                    dst[i].re,
                    dst[i].im,
                    expected[i].re,
                    expected[i].im,
                );
            }
        }
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        all(target_feature = "avx", target_feature = "fma")
    ))]
    fn test_butterfly_radix16_avx_fma_vs_scalar() {
        use crate::fft::butterflies::tests::{TestSimdWidth, test_butterfly_against_scalar};

        test_butterfly_against_scalar(
            butterfly_radix16_scalar::<4>,
            |src, dst, twiddles, p| unsafe {
                if p == 1 {
                    avx::butterfly_radix16_stride1_avx_fma(src, dst, twiddles);
                } else {
                    avx::butterfly_radix16_generic_avx_fma(src, dst, twiddles, p);
                }
            },
            16,
            15,
            TestSimdWidth::Width4,
            "butterfly_radix16_avx_fma",
        );
    }
}
