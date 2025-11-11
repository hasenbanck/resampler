use core::{arch::x86_64::*, f32::consts::FRAC_1_SQRT_2};

use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_i_sse4_2, complex_mul_sse4_2, load_neg_imag_mask_sse4_2},
};

/// Performs a single radix-8 Stockham butterfly stage for stride=1 (out-of-place, SSE4.2).
///
/// This is a specialized version for the stride=1 case (first stage) that uses the unpack-permute
/// pattern to enable sequential SIMD stores. For stride=1, output indices are sequential: j=8*i.
#[target_feature(enable = "sse4.2")]
pub(super) unsafe fn butterfly_radix8_stride1_sse4_2(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;
    let simd_iters = (eighth_samples >> 1) << 1;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_sse4_2();

        let w8_1 = _mm_setr_ps(FRAC_1_SQRT_2, -FRAC_1_SQRT_2, FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
        let w8_3 = _mm_setr_ps(
            -FRAC_1_SQRT_2,
            -FRAC_1_SQRT_2,
            -FRAC_1_SQRT_2,
            -FRAC_1_SQRT_2,
        );

        for i in (0..simd_iters).step_by(2) {
            // Load 2 complex numbers from each eighth.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + eighth_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + eighth_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + eighth_samples * 3) as *const f32;
            let z3 = _mm_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + eighth_samples * 4) as *const f32;
            let z4 = _mm_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + eighth_samples * 5) as *const f32;
            let z5 = _mm_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + eighth_samples * 6) as *const f32;
            let z6 = _mm_loadu_ps(z6_ptr);

            let z7_ptr = src.as_ptr().add(i + eighth_samples * 7) as *const f32;
            let z7 = _mm_loadu_ps(z7_ptr);

            // Stride=1 optimization: all twiddles are identity (1+0i), skip multiplication.
            let t1 = z1;
            let t2 = z2;
            let t3 = z3;
            let t4 = z4;
            let t5 = z5;
            let t6 = z6;
            let t7 = z7;

            // Compute radix-4 DFT on even indices (z0, t2, t4, t6).
            let even_a0 = _mm_add_ps(z0, t4);
            let even_a1 = _mm_sub_ps(z0, t4);
            let even_a2 = _mm_add_ps(t2, t6);
            let t2_sub_t6 = _mm_sub_ps(t2, t6);
            let even_a3 = complex_mul_i_sse4_2(t2_sub_t6, neg_imag_mask);

            let x_even_0 = _mm_add_ps(even_a0, even_a2);
            let x_even_2 = _mm_sub_ps(even_a0, even_a2);
            let x_even_1 = _mm_add_ps(even_a1, even_a3);
            let x_even_3 = _mm_sub_ps(even_a1, even_a3);

            // Compute radix-4 DFT on odd indices (t1, t3, t5, t7).
            let odd_a0 = _mm_add_ps(t1, t5);
            let odd_a1 = _mm_sub_ps(t1, t5);
            let odd_a2 = _mm_add_ps(t3, t7);
            let t3_sub_t7 = _mm_sub_ps(t3, t7);
            let odd_a3 = complex_mul_i_sse4_2(t3_sub_t7, neg_imag_mask);

            let x_odd_0 = _mm_add_ps(odd_a0, odd_a2);
            let x_odd_2 = _mm_sub_ps(odd_a0, odd_a2);
            let x_odd_1 = _mm_add_ps(odd_a1, odd_a3);
            let x_odd_3 = _mm_sub_ps(odd_a1, odd_a3);

            // Combine even and odd parts with W_8 twiddles.
            // out[0] = x_even[0] + x_odd[0]
            // out[4] = x_even[0] - x_odd[0]
            let out0 = _mm_add_ps(x_even_0, x_odd_0);
            let out4 = _mm_sub_ps(x_even_0, x_odd_0);

            // out[1] = x_even[1] + W_8^1 * x_odd[1]
            // out[5] = x_even[1] - W_8^1 * x_odd[1]
            // W_8^1 * (a+bi) = ((a+b)/√2, (b-a)/√2)
            // Using SIMD: W_8^1 = [1/√2, -1/√2, 1/√2, -1/√2]
            let w8_1_odd_1 = complex_mul_sse4_2(w8_1, x_odd_1);
            let out1 = _mm_add_ps(x_even_1, w8_1_odd_1);
            let out5 = _mm_sub_ps(x_even_1, w8_1_odd_1);

            // out[2] = x_even[2] + W_8^2 * x_odd[2]
            // out[6] = x_even[2] - W_8^2 * x_odd[2]
            // W_8^2 = -i, so multiply by -i: (a+bi)*(-i) = (b, -a)
            let w8_2_odd_2 = complex_mul_i_sse4_2(x_odd_2, neg_imag_mask);
            let out2 = _mm_add_ps(x_even_2, w8_2_odd_2);
            let out6 = _mm_sub_ps(x_even_2, w8_2_odd_2);

            // out[3] = x_even[3] + W_8^3 * x_odd[3]
            // out[7] = x_even[3] - W_8^3 * x_odd[3]
            // W_8^3 = (-1-i)/√2 = [-1/√2, -1/√2, -1/√2, -1/√2]
            let w8_3_odd_3 = complex_mul_sse4_2(w8_3, x_odd_3);
            let out3 = _mm_add_ps(x_even_3, w8_3_odd_3);
            let out7 = _mm_sub_ps(x_even_3, w8_3_odd_3);

            // Sequential 128-bit stores for stride=1 using unpack-permute pattern.
            // We have 8 outputs (out0-out7), each containing 2 complex numbers.
            // Need to interleave them: [out0[0], out1[0], out2[0], out3[0], out4[0], out5[0], out6[0], out7[0], out0[1], ...]

            let out0_pd = _mm_castps_pd(out0);
            let out1_pd = _mm_castps_pd(out1);
            let out2_pd = _mm_castps_pd(out2);
            let out3_pd = _mm_castps_pd(out3);
            let out4_pd = _mm_castps_pd(out4);
            let out5_pd = _mm_castps_pd(out5);
            let out6_pd = _mm_castps_pd(out6);
            let out7_pd = _mm_castps_pd(out7);

            // Interleave pairs for sequential stores.
            let out01_lo = _mm_castpd_ps(_mm_unpacklo_pd(out0_pd, out1_pd));
            let out23_lo = _mm_castpd_ps(_mm_unpacklo_pd(out2_pd, out3_pd));
            let out45_lo = _mm_castpd_ps(_mm_unpacklo_pd(out4_pd, out5_pd));
            let out67_lo = _mm_castpd_ps(_mm_unpacklo_pd(out6_pd, out7_pd));

            let out01_hi = _mm_castpd_ps(_mm_unpackhi_pd(out0_pd, out1_pd));
            let out23_hi = _mm_castpd_ps(_mm_unpackhi_pd(out2_pd, out3_pd));
            let out45_hi = _mm_castpd_ps(_mm_unpackhi_pd(out4_pd, out5_pd));
            let out67_hi = _mm_castpd_ps(_mm_unpackhi_pd(out6_pd, out7_pd));

            let j = 8 * i;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            _mm_storeu_ps(dst_ptr, out01_lo);
            _mm_storeu_ps(dst_ptr.add(4), out23_lo);
            _mm_storeu_ps(dst_ptr.add(8), out45_lo);
            _mm_storeu_ps(dst_ptr.add(12), out67_lo);
            _mm_storeu_ps(dst_ptr.add(16), out01_hi);
            _mm_storeu_ps(dst_ptr.add(20), out23_hi);
            _mm_storeu_ps(dst_ptr.add(24), out45_hi);
            _mm_storeu_ps(dst_ptr.add(28), out67_hi);
        }
    }

    super::butterfly_radix8_scalar::<2>(src, dst, stage_twiddles, 1, simd_iters);
}

/// Performs a single radix-8 Stockham butterfly stage for p>1 (out-of-place, SSE4.2).
///
/// Generic version for p>1 cases. Uses direct SIMD stores, accepting non-sequential
/// stores as the shuffle overhead isn't justified for larger strides.
#[target_feature(enable = "sse4.2")]
pub(super) unsafe fn butterfly_radix8_generic_sse4_2(
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
    let eighth_samples = samples >> 3;
    let simd_iters = (eighth_samples >> 1) << 1;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_sse4_2();

        let w8_1 = _mm_setr_ps(FRAC_1_SQRT_2, -FRAC_1_SQRT_2, FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
        let w8_3 = _mm_setr_ps(
            -FRAC_1_SQRT_2,
            -FRAC_1_SQRT_2,
            -FRAC_1_SQRT_2,
            -FRAC_1_SQRT_2,
        );

        for i in (0..simd_iters).step_by(2) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;

            // Load 2 complex numbers from each eighth.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + eighth_samples) as *const f32;
            let z1 = _mm_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + eighth_samples * 2) as *const f32;
            let z2 = _mm_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + eighth_samples * 3) as *const f32;
            let z3 = _mm_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + eighth_samples * 4) as *const f32;
            let z4 = _mm_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + eighth_samples * 5) as *const f32;
            let z5 = _mm_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + eighth_samples * 6) as *const f32;
            let z6 = _mm_loadu_ps(z6_ptr);

            let z7_ptr = src.as_ptr().add(i + eighth_samples * 7) as *const f32;
            let z7 = _mm_loadu_ps(z7_ptr);

            // Load prepackaged twiddles in packed format.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 7) as *const f32;
            let w1 = _mm_loadu_ps(tw_ptr);
            let w2 = _mm_loadu_ps(tw_ptr.add(4));
            let w3 = _mm_loadu_ps(tw_ptr.add(8));
            let w4 = _mm_loadu_ps(tw_ptr.add(12));
            let w5 = _mm_loadu_ps(tw_ptr.add(16));
            let w6 = _mm_loadu_ps(tw_ptr.add(20));
            let w7 = _mm_loadu_ps(tw_ptr.add(24));

            // Apply twiddle factors.
            let t1 = complex_mul_sse4_2(w1, z1);
            let t2 = complex_mul_sse4_2(w2, z2);
            let t3 = complex_mul_sse4_2(w3, z3);
            let t4 = complex_mul_sse4_2(w4, z4);
            let t5 = complex_mul_sse4_2(w5, z5);
            let t6 = complex_mul_sse4_2(w6, z6);
            let t7 = complex_mul_sse4_2(w7, z7);

            // Compute radix-4 DFT on even indices.
            let even_a0 = _mm_add_ps(z0, t4);
            let even_a1 = _mm_sub_ps(z0, t4);
            let even_a2 = _mm_add_ps(t2, t6);
            let t2_sub_t6 = _mm_sub_ps(t2, t6);
            let even_a3 = complex_mul_i_sse4_2(t2_sub_t6, neg_imag_mask);

            let x_even_0 = _mm_add_ps(even_a0, even_a2);
            let x_even_2 = _mm_sub_ps(even_a0, even_a2);
            let x_even_1 = _mm_add_ps(even_a1, even_a3);
            let x_even_3 = _mm_sub_ps(even_a1, even_a3);

            // Compute radix-4 DFT on odd indices.
            let odd_a0 = _mm_add_ps(t1, t5);
            let odd_a1 = _mm_sub_ps(t1, t5);
            let odd_a2 = _mm_add_ps(t3, t7);
            let t3_sub_t7 = _mm_sub_ps(t3, t7);
            let odd_a3 = complex_mul_i_sse4_2(t3_sub_t7, neg_imag_mask);

            let x_odd_0 = _mm_add_ps(odd_a0, odd_a2);
            let x_odd_2 = _mm_sub_ps(odd_a0, odd_a2);
            let x_odd_1 = _mm_add_ps(odd_a1, odd_a3);
            let x_odd_3 = _mm_sub_ps(odd_a1, odd_a3);

            // Combine even and odd parts.
            let out0 = _mm_add_ps(x_even_0, x_odd_0);
            let out4 = _mm_sub_ps(x_even_0, x_odd_0);

            let w8_1_odd_1 = complex_mul_sse4_2(w8_1, x_odd_1);
            let out1 = _mm_add_ps(x_even_1, w8_1_odd_1);
            let out5 = _mm_sub_ps(x_even_1, w8_1_odd_1);

            let w8_2_odd_2 = complex_mul_i_sse4_2(x_odd_2, neg_imag_mask);
            let out2 = _mm_add_ps(x_even_2, w8_2_odd_2);
            let out6 = _mm_sub_ps(x_even_2, w8_2_odd_2);

            let w8_3_odd_3 = complex_mul_sse4_2(w8_3, x_odd_3);
            let out3 = _mm_add_ps(x_even_3, w8_3_odd_3);
            let out7 = _mm_sub_ps(x_even_3, w8_3_odd_3);

            // Calculate output indices with wraparound.
            let j0 = 8 * i - 7 * k0;
            let j1 = 8 * (i + 1) - 7 * k1;

            // Direct SIMD stores using 64-bit operations.
            let out0_pd = _mm_castps_pd(out0);
            let out1_pd = _mm_castps_pd(out1);
            let out2_pd = _mm_castps_pd(out2);
            let out3_pd = _mm_castps_pd(out3);
            let out4_pd = _mm_castps_pd(out4);
            let out5_pd = _mm_castps_pd(out5);
            let out6_pd = _mm_castps_pd(out6);
            let out7_pd = _mm_castps_pd(out7);

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            // Store first iteration (k0) outputs.
            _mm_storel_pd(dst_ptr.add(j0), out0_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 3), out3_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 4), out4_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 5), out5_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 6), out6_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 7), out7_pd);

            // Store second iteration (k1) outputs.
            _mm_storeh_pd(dst_ptr.add(j1), out0_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 3), out3_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 4), out4_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 5), out5_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 6), out6_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 7), out7_pd);
        }
    }

    super::butterfly_radix8_scalar::<2>(src, dst, stage_twiddles, stride, simd_iters);
}
