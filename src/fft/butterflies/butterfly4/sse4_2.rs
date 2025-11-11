use core::arch::x86_64::*;

use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_i_sse4_2, complex_mul_sse4_2, load_neg_imag_mask_sse4_2},
};

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
    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples >> 1) << 1;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_sse4_2();

        for i in (0..simd_iters).step_by(2) {
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + quarter_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + quarter_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + quarter_samples * 3) as *const f32;
            let z3 = _mm_loadu_ps(z3_ptr);

            // Identity twiddles: t1 = (1+0i) * z1 = z1, t2 = z2, t3 = z3 (skip twiddle load and multiply)
            let t1 = z1;
            let t2 = z2;
            let t3 = z3;

            // Radix-4 butterfly
            let a0 = _mm_add_ps(z0, t2);
            let a1 = _mm_sub_ps(z0, t2);
            let a2 = _mm_add_ps(t1, t3);

            // a3 = i * (t1 - t3)
            let t1_sub_t3 = _mm_sub_ps(t1, t3);
            let a3 = complex_mul_i_sse4_2(t1_sub_t3, neg_imag_mask);

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

    super::butterfly_radix4_scalar::<2>(src, dst, stage_twiddles, 1, simd_iters);
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
    // We convince the compiler here that stride can't be 0 to optimize better.
    if stride == 0 {
        return;
    }

    let samples = src.len();
    let quarter_samples = samples >> 2;
    let simd_iters = (quarter_samples >> 1) << 1;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_sse4_2();

        for i in (0..simd_iters).step_by(2) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;

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

            // Load prepackaged twiddles directly (no shuffle needed).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 3) as *const f32;
            let w1 = _mm_loadu_ps(tw_ptr); // w1[i], w1[i+1]
            let w2 = _mm_loadu_ps(tw_ptr.add(4)); // w2[i], w2[i+1]
            let w3 = _mm_loadu_ps(tw_ptr.add(8)); // w3[i], w3[i+1]

            // Complex multiply.
            let t1 = complex_mul_sse4_2(w1, z1);
            let t2 = complex_mul_sse4_2(w2, z2);
            let t3 = complex_mul_sse4_2(w3, z3);

            // Radix-4 butterfly.
            let a0 = _mm_add_ps(z0, t2);
            let a1 = _mm_sub_ps(z0, t2);
            let a2 = _mm_add_ps(t1, t3);

            // a3 = i * (t1 - t3)
            let t1_sub_t3 = _mm_sub_ps(t1, t3);
            let a3 = complex_mul_i_sse4_2(t1_sub_t3, neg_imag_mask);

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

    super::butterfly_radix4_scalar::<2>(src, dst, stage_twiddles, stride, simd_iters);
}
