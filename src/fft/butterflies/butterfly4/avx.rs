use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_avx, complex_mul_i_avx, load_neg_imag_mask_avx},
};

/// Performs a single radix-4 Stockham butterfly stage for stride=1 (out-of-place, AVX+FMA).
///
/// This is a specialized version for the stride=1 case (first stage) that uses the unpack-permute
/// pattern to enable sequential SIMD stores. For stride=1, output indices are sequential: j=4*i.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix4_stride1_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples >> 2) << 2;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_avx();

        for i in (0..simd_iters).step_by(4) {
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + quarter_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + quarter_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + quarter_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 3) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]
            let w3 = _mm256_loadu_ps(tw_ptr.add(16)); // w3[i..i+4]

            // Complex multiply: t1 = w1 * z1, t2 = w2 * z2, t3 = w3 * z3
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);
            let t3 = complex_mul_avx(w3, z3);

            // Radix-4 butterfly
            let a0 = _mm256_add_ps(z0, t2);
            let a1 = _mm256_sub_ps(z0, t2);
            let a2 = _mm256_add_ps(t1, t3);

            // a3 = i * (t1 - t3)
            let t1_sub_t3 = _mm256_sub_ps(t1, t3);
            let a3 = complex_mul_i_avx(t1_sub_t3, neg_imag_mask);

            // Final butterfly outputs
            let out0 = _mm256_add_ps(a0, a2);
            let out2 = _mm256_sub_ps(a0, a2);
            let out1 = _mm256_add_ps(a1, a3);
            let out3 = _mm256_sub_ps(a1, a3);

            // Apply unpack-permute pattern for sequential stores.
            // We have 4 outputs (out0, out1, out2, out3), each containing 4 complex numbers
            // Need to interleave them: [out0[0], out1[0], out2[0], out3[0], out0[1], ...]

            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);

            // Interleave pairs: out0 with out1, out2 with out3.
            let out01_lo = _mm256_castpd_ps(_mm256_unpacklo_pd(out0_pd, out1_pd));
            let out01_hi = _mm256_castpd_ps(_mm256_unpackhi_pd(out0_pd, out1_pd));
            let out23_lo = _mm256_castpd_ps(_mm256_unpacklo_pd(out2_pd, out3_pd));
            let out23_hi = _mm256_castpd_ps(_mm256_unpackhi_pd(out2_pd, out3_pd));

            // Further interleave to get final sequential layout.
            let out0123_0 = _mm256_permute2f128_ps(out01_lo, out23_lo, 0x20);
            let out0123_1 = _mm256_permute2f128_ps(out01_hi, out23_hi, 0x20);
            let out0123_2 = _mm256_permute2f128_ps(out01_lo, out23_lo, 0x31);
            let out0123_3 = _mm256_permute2f128_ps(out01_hi, out23_hi, 0x31);

            // Sequential stores.
            let j = 4 * i;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            _mm256_storeu_ps(dst_ptr, out0123_0);
            _mm256_storeu_ps(dst_ptr.add(8), out0123_1);
            _mm256_storeu_ps(dst_ptr.add(16), out0123_2);
            _mm256_storeu_ps(dst_ptr.add(24), out0123_3);
        }
    }

    for i in simd_iters..quarter_samples {
        let w1 = stage_twiddles[i * 3];
        let w2 = stage_twiddles[i * 3 + 1];
        let w3 = stage_twiddles[i * 3 + 2];

        let z0 = src[i];
        let z1 = src[i + quarter_samples];
        let z2 = src[i + quarter_samples * 2];
        let z3 = src[i + quarter_samples * 3];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);

        let a0 = z0.add(&t2);
        let a1 = z0.sub(&t2);
        let a2 = t1.add(&t3);
        let a3_re = t1.im - t3.im;
        let a3_im = t3.re - t1.re;

        let j = 4 * i;
        dst[j] = a0.add(&a2);
        dst[j + 2] = a0.sub(&a2);
        dst[j + 1] = Complex32::new(a1.re + a3_re, a1.im + a3_im);
        dst[j + 3] = Complex32::new(a1.re - a3_re, a1.im - a3_im);
    }
}

/// Performs a single radix-4 Stockham butterfly stage for p>1 (out-of-place, AVX+FMA).
///
/// Generic version for p>1 cases. Uses direct SIMD stores, accepting non-sequential
/// stores as the shuffle overhead isn't justified for larger strides.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix4_generic_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples >> 2) << 2;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_avx();

        for i in (0..simd_iters).step_by(4) {
            // Calculate twiddle indices.
            let k0 = i % stride;
            let k1 = (i + 1) % stride;
            let k2 = (i + 2) % stride;
            let k3 = (i + 3) % stride;

            // Load z0 from first quarter.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            // Load z1, z2, z3 from other quarters.
            let z1_ptr = src.as_ptr().add(i + quarter_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + quarter_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + quarter_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 3) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]
            let w3 = _mm256_loadu_ps(tw_ptr.add(16)); // w3[i..i+4]

            // Complex multiply: t1 = w1 * z1, t2 = w2 * z2, t3 = w3 * z3
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);
            let t3 = complex_mul_avx(w3, z3);

            // Radix-4 butterfly.
            let a0 = _mm256_add_ps(z0, t2);
            let a1 = _mm256_sub_ps(z0, t2);
            let a2 = _mm256_add_ps(t1, t3);

            // a3 = i * (t1 - t3)
            let t1_sub_t3 = _mm256_sub_ps(t1, t3);
            let a3 = complex_mul_i_avx(t1_sub_t3, neg_imag_mask);

            // Final butterfly outputs.
            let out0 = _mm256_add_ps(a0, a2);
            let out2 = _mm256_sub_ps(a0, a2);
            let out1 = _mm256_add_ps(a1, a3);
            let out3 = _mm256_sub_ps(a1, a3);

            // Calculate output indices.
            let j0 = 4 * i - 3 * k0;
            let j1 = 4 * (i + 1) - 3 * k1;
            let j2 = 4 * (i + 2) - 3 * k2;
            let j3 = 4 * (i + 3) - 3 * k3;

            // Direct SIMD stores using 64-bit operations.
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            // Extract 128-bit lanes.
            let out0_lo = _mm256_castpd256_pd128(out0_pd);
            let out0_hi = _mm256_extractf128_pd(out0_pd, 1);
            let out1_lo = _mm256_castpd256_pd128(out1_pd);
            let out1_hi = _mm256_extractf128_pd(out1_pd, 1);
            let out2_lo = _mm256_castpd256_pd128(out2_pd);
            let out2_hi = _mm256_extractf128_pd(out2_pd, 1);
            let out3_lo = _mm256_castpd256_pd128(out3_pd);
            let out3_hi = _mm256_extractf128_pd(out3_pd, 1);

            // Store iteration 0.
            _mm_storel_pd(dst_ptr.add(j0), out0_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 3), out3_lo);

            // Store iteration 1.
            _mm_storeh_pd(dst_ptr.add(j1), out0_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 3), out3_lo);

            // Store iteration 2.
            _mm_storel_pd(dst_ptr.add(j2), out0_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride), out1_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 2), out2_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 3), out3_hi);

            // Store iteration 3.
            _mm_storeh_pd(dst_ptr.add(j3), out0_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride), out1_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 2), out2_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 3), out3_hi);
        }
    }

    for i in simd_iters..quarter_samples {
        let k = i % stride;
        let w1 = stage_twiddles[i * 3];
        let w2 = stage_twiddles[i * 3 + 1];
        let w3 = stage_twiddles[i * 3 + 2];

        let z0 = src[i];
        let z1 = src[i + quarter_samples];
        let z2 = src[i + quarter_samples * 2];
        let z3 = src[i + quarter_samples * 3];

        let t1 = w1.mul(&z1);
        let t2 = w2.mul(&z2);
        let t3 = w3.mul(&z3);

        let a0 = z0.add(&t2);
        let a1 = z0.sub(&t2);
        let a2 = t1.add(&t3);
        let a3_re = t1.im - t3.im;
        let a3_im = t3.re - t1.re;

        let j = 4 * i - 3 * k;
        dst[j] = a0.add(&a2);
        dst[j + stride * 2] = a0.sub(&a2);
        dst[j + stride] = Complex32::new(a1.re + a3_re, a1.im + a3_im);
        dst[j + stride * 3] = Complex32::new(a1.re - a3_re, a1.im - a3_im);
    }
}
