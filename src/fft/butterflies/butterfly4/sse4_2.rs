use crate::fft::Complex32;

/// Performs a single radix-4 Stockham butterfly stage for stride=1 (out-of-place, SSE4.2).
///
/// This is a specialized version for the stride=1 case (first stage) that uses the unpack-permute
/// pattern to enable sequential SIMD stores. For stride=1, output indices are sequential: j=4*i.
#[target_feature(enable = "sse4.2")]
pub(super) unsafe fn butterfly_radix4_stride1_sse4_2(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples >> 1) << 1;

    unsafe {
        for i in (0..simd_iters).step_by(2) {
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + quarter_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + quarter_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + quarter_samples * 3) as *const f32;
            let z3 = _mm_loadu_ps(z3_ptr);

            // Load 6 twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 3) as *const f32;
            let tw_0 = _mm_loadu_ps(tw_ptr); // [w1[0], w2[0]]
            let tw_1 = _mm_loadu_ps(tw_ptr.add(4)); // [w3[0], w1[1]]
            let tw_2 = _mm_loadu_ps(tw_ptr.add(8)); // [w2[1], w3[1]]

            // Extract w1, w2, w3 using shuffle_pd for 64-bit element selection.
            let tw_0_pd = _mm_castps_pd(tw_0);
            let tw_1_pd = _mm_castps_pd(tw_1);
            let tw_2_pd = _mm_castps_pd(tw_2);

            let w1 = _mm_castpd_ps(_mm_shuffle_pd(tw_0_pd, tw_1_pd, 0b10)); // [w1[0], w1[1]]
            let w2 = _mm_castpd_ps(_mm_shuffle_pd(tw_0_pd, tw_2_pd, 0b01)); // [w2[0], w2[1]]
            let w3 = _mm_castpd_ps(_mm_shuffle_pd(tw_1_pd, tw_2_pd, 0b10)); // [w3[0], w3[1]]

            // Complex multiply: t1 = w1 * z1
            let z1_re = _mm_moveldup_ps(z1);
            let z1_im = _mm_movehdup_ps(z1);
            let w1_swap = _mm_shuffle_ps(w1, w1, 0b10_11_00_01);
            let prod1_re = _mm_mul_ps(w1, z1_re);
            let prod1_im = _mm_mul_ps(w1_swap, z1_im);
            let t1 = _mm_addsub_ps(prod1_re, prod1_im);

            // Complex multiply: t2 = w2 * z2
            let z2_re = _mm_moveldup_ps(z2);
            let z2_im = _mm_movehdup_ps(z2);
            let w2_swap = _mm_shuffle_ps(w2, w2, 0b10_11_00_01);
            let prod2_re = _mm_mul_ps(w2, z2_re);
            let prod2_im = _mm_mul_ps(w2_swap, z2_im);
            let t2 = _mm_addsub_ps(prod2_re, prod2_im);

            // Complex multiply: t3 = w3 * z3
            let z3_re = _mm_moveldup_ps(z3);
            let z3_im = _mm_movehdup_ps(z3);
            let w3_swap = _mm_shuffle_ps(w3, w3, 0b10_11_00_01);
            let prod3_re = _mm_mul_ps(w3, z3_re);
            let prod3_im = _mm_mul_ps(w3_swap, z3_im);
            let t3 = _mm_addsub_ps(prod3_re, prod3_im);

            // Radix-4 butterfly
            let a0 = _mm_add_ps(z0, t2);
            let a1 = _mm_sub_ps(z0, t2);
            let a2 = _mm_add_ps(t1, t3);

            // a3 = i * (t1 - t3) = [im, -re] (swap and negate real part)
            let t1_sub_t3 = _mm_sub_ps(t1, t3);
            let a3_swapped = _mm_shuffle_ps(t1_sub_t3, t1_sub_t3, 0b10_11_00_01);
            let a3_neg = _mm_sub_ps(_mm_setzero_ps(), a3_swapped);
            // Blend: select lanes 0,2 from a3_swapped (positive im), lanes 1,3 from a3_neg (negative re)
            // Blend mask: 0b1010 = select second operand (a3_neg) for lanes 1,3
            let a3 = _mm_blend_ps(a3_swapped, a3_neg, 0b1010);

            // Final butterfly outputs
            let out0 = _mm_add_ps(a0, a2);
            let out2 = _mm_sub_ps(a0, a2);
            let out1 = _mm_add_ps(a1, a3);
            let out3 = _mm_sub_ps(a1, a3);

            // Apply unpack-permute pattern for sequential stores.
            let out0_pd = _mm_castps_pd(out0);
            let out1_pd = _mm_castps_pd(out1);
            let out2_pd = _mm_castps_pd(out2);
            let out3_pd = _mm_castps_pd(out3);

            // Interleave pairs: out0 with out1, out2 with out3.
            let out01_lo = _mm_castpd_ps(_mm_unpacklo_pd(out0_pd, out1_pd));
            let out01_hi = _mm_castpd_ps(_mm_unpackhi_pd(out0_pd, out1_pd));
            let out23_lo = _mm_castpd_ps(_mm_unpacklo_pd(out2_pd, out3_pd));
            let out23_hi = _mm_castpd_ps(_mm_unpackhi_pd(out2_pd, out3_pd));

            let j = 4 * i;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;

            _mm_storeu_ps(dst_ptr, out01_lo);
            _mm_storeu_ps(dst_ptr.add(4), out23_lo);
            _mm_storeu_ps(dst_ptr.add(8), out01_hi);
            _mm_storeu_ps(dst_ptr.add(12), out23_hi);
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

/// Performs a single radix-4 Stockham butterfly stage for p>1 (out-of-place, SSE4.2).
///
/// Generic version for p>1 cases. Uses direct SIMD stores, accepting non-sequential
/// stores as the shuffle overhead isn't justified for larger strides.
#[target_feature(enable = "sse4.2")]
pub(super) unsafe fn butterfly_radix4_generic_sse4_2(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples >> 1) << 1;

    unsafe {
        for i in (0..simd_iters).step_by(2) {
            // Calculate twiddle indices.
            let k0 = i % stride;
            let k1 = (i + 1) % stride;

            // Load z0 from first quarter.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            // Load z1, z2, z3 from other quarters.
            let z1_ptr = src.as_ptr().add(i + quarter_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + quarter_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + quarter_samples * 3) as *const f32;
            let z3 = _mm_loadu_ps(z3_ptr);

            // Load 6 twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 3) as *const f32;
            let tw_0 = _mm_loadu_ps(tw_ptr);
            let tw_1 = _mm_loadu_ps(tw_ptr.add(4));
            let tw_2 = _mm_loadu_ps(tw_ptr.add(8));

            // Extract w1, w2, w3 using shuffle_pd for 64-bit element selection.
            let tw_0_pd = _mm_castps_pd(tw_0);
            let tw_1_pd = _mm_castps_pd(tw_1);
            let tw_2_pd = _mm_castps_pd(tw_2);

            let w1 = _mm_castpd_ps(_mm_shuffle_pd(tw_0_pd, tw_1_pd, 0b10));
            let w2 = _mm_castpd_ps(_mm_shuffle_pd(tw_0_pd, tw_2_pd, 0b01));
            let w3 = _mm_castpd_ps(_mm_shuffle_pd(tw_1_pd, tw_2_pd, 0b10));

            // Complex multiply: t1 = w1 * z1
            let z1_re = _mm_moveldup_ps(z1);
            let z1_im = _mm_movehdup_ps(z1);
            let w1_swap = _mm_shuffle_ps(w1, w1, 0b10_11_00_01);
            let prod1_re = _mm_mul_ps(w1, z1_re);
            let prod1_im = _mm_mul_ps(w1_swap, z1_im);
            let t1 = _mm_addsub_ps(prod1_re, prod1_im);

            // Complex multiply: t2 = w2 * z2
            let z2_re = _mm_moveldup_ps(z2);
            let z2_im = _mm_movehdup_ps(z2);
            let w2_swap = _mm_shuffle_ps(w2, w2, 0b10_11_00_01);
            let prod2_re = _mm_mul_ps(w2, z2_re);
            let prod2_im = _mm_mul_ps(w2_swap, z2_im);
            let t2 = _mm_addsub_ps(prod2_re, prod2_im);

            // Complex multiply: t3 = w3 * z3
            let z3_re = _mm_moveldup_ps(z3);
            let z3_im = _mm_movehdup_ps(z3);
            let w3_swap = _mm_shuffle_ps(w3, w3, 0b10_11_00_01);
            let prod3_re = _mm_mul_ps(w3, z3_re);
            let prod3_im = _mm_mul_ps(w3_swap, z3_im);
            let t3 = _mm_addsub_ps(prod3_re, prod3_im);

            // Radix-4 butterfly.
            let a0 = _mm_add_ps(z0, t2);
            let a1 = _mm_sub_ps(z0, t2);
            let a2 = _mm_add_ps(t1, t3);

            // a3 = i * (t1 - t3) = [im, -re] (swap and negate real part)
            let t1_sub_t3 = _mm_sub_ps(t1, t3);
            let a3_swapped = _mm_shuffle_ps(t1_sub_t3, t1_sub_t3, 0b10_11_00_01);
            let a3_neg = _mm_sub_ps(_mm_setzero_ps(), a3_swapped);
            // Blend: select lanes 0,2 from a3_swapped (positive im), lanes 1,3 from a3_neg (negative re)
            let a3 = _mm_blend_ps(a3_swapped, a3_neg, 0b1010);

            // Final butterfly outputs.
            let out0 = _mm_add_ps(a0, a2);
            let out2 = _mm_sub_ps(a0, a2);
            let out1 = _mm_add_ps(a1, a3);
            let out3 = _mm_sub_ps(a1, a3);

            // Calculate output indices.
            let j0 = 4 * i - 3 * k0;
            let j1 = 4 * (i + 1) - 3 * k1;

            // Direct SIMD stores using 64-bit operations.
            let out0_pd = _mm_castps_pd(out0);
            let out1_pd = _mm_castps_pd(out1);
            let out2_pd = _mm_castps_pd(out2);
            let out3_pd = _mm_castps_pd(out3);

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            _mm_storel_pd(dst_ptr.add(j0), out0_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 3), out3_pd);

            _mm_storeh_pd(dst_ptr.add(j1), out0_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 3), out3_pd);
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
