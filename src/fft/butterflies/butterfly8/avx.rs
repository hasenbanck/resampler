use core::{arch::x86_64::*, f32::consts::FRAC_1_SQRT_2};

use crate::fft::{
    Complex32,
    butterflies::ops::{complex_mul_avx, complex_mul_i_avx, load_neg_imag_mask_avx},
};

/// Performs a single radix-8 Stockham butterfly stage for stride=1 (out-of-place, AVX+FMA).
///
/// This is a specialized version for the stride=1 case (first stage) that uses chunked streaming
/// to reduce register pressure. Processes outputs in two groups of 4 to minimize spilling.
/// For stride=1, output indices are sequential: j=8*i.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix8_stride1_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let eighth_samples = samples >> 3;
    let simd_iters = (eighth_samples >> 2) << 2;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_avx();
        let frac_1_sqrt2_scalar = FRAC_1_SQRT_2;

        for i in (0..simd_iters).step_by(4) {
            // Load 8 input values from the 8 "eighths" of the input array.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + eighth_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + eighth_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + eighth_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + eighth_samples * 4) as *const f32;
            let z4 = _mm256_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + eighth_samples * 5) as *const f32;
            let z5 = _mm256_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + eighth_samples * 6) as *const f32;
            let z6 = _mm256_loadu_ps(z6_ptr);

            let z7_ptr = src.as_ptr().add(i + eighth_samples * 7) as *const f32;
            let z7 = _mm256_loadu_ps(z7_ptr);

            // Load prepackaged twiddles in packed format (7 twiddles per group of 4 lanes).
            let tw_ptr = stage_twiddles.as_ptr().add(i * 7) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]
            let w3 = _mm256_loadu_ps(tw_ptr.add(16)); // w3[i..i+4]
            let w4 = _mm256_loadu_ps(tw_ptr.add(24)); // w4[i..i+4]
            let w5 = _mm256_loadu_ps(tw_ptr.add(32)); // w5[i..i+4]
            let w6 = _mm256_loadu_ps(tw_ptr.add(40)); // w6[i..i+4]
            let w7 = _mm256_loadu_ps(tw_ptr.add(48)); // w7[i..i+4]

            // Apply twiddle factors: t1-t7 = w1-w7 * z1-z7
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);
            let t3 = complex_mul_avx(w3, z3);
            let t4 = complex_mul_avx(w4, z4);
            let t5 = complex_mul_avx(w5, z5);
            let t6 = complex_mul_avx(w6, z6);
            let t7 = complex_mul_avx(w7, z7);

            // Split-radix decomposition: compute even radix-4 on (z0, t2, t4, t6)
            let even_a0 = _mm256_add_ps(z0, t4);
            let even_a1 = _mm256_sub_ps(z0, t4);
            let even_a2 = _mm256_add_ps(t2, t6);
            // even_a3 = i * (t2 - t6): (real, imag) -> (imag, -real)
            let t2_sub_t6 = _mm256_sub_ps(t2, t6);
            let even_a3 = complex_mul_i_avx(t2_sub_t6, neg_imag_mask);

            let x_even_0 = _mm256_add_ps(even_a0, even_a2);
            let x_even_2 = _mm256_sub_ps(even_a0, even_a2);
            let x_even_1 = _mm256_add_ps(even_a1, even_a3);
            let x_even_3 = _mm256_sub_ps(even_a1, even_a3);

            // Compute odd radix-4 on (t1, t3, t5, t7).
            let odd_a0 = _mm256_add_ps(t1, t5);
            let odd_a1 = _mm256_sub_ps(t1, t5);
            let odd_a2 = _mm256_add_ps(t3, t7);
            let t3_sub_t7 = _mm256_sub_ps(t3, t7);
            let odd_a3 = complex_mul_i_avx(t3_sub_t7, neg_imag_mask);

            let x_odd_0 = _mm256_add_ps(odd_a0, odd_a2);
            let x_odd_2 = _mm256_sub_ps(odd_a0, odd_a2);
            let x_odd_1 = _mm256_add_ps(odd_a1, odd_a3);
            let x_odd_3 = _mm256_sub_ps(odd_a1, odd_a3);

            // Apply W_8 twiddles and combine even/odd results.
            // W_8^0 = 1, W_8^1 = (1-i)/√2, W_8^2 = -i, W_8^3 = (-1-i)/√2

            // out0 = x_even_0 + x_odd_0
            let out0 = _mm256_add_ps(x_even_0, x_odd_0);
            // out4 = x_even_0 - x_odd_0
            let out4 = _mm256_sub_ps(x_even_0, x_odd_0);

            // out1 = x_even_1 + W_8^1 * x_odd_1
            // W_8^1 = (1-i)/√2 as interleaved complex: [re, im, re, im, ...] = [1/√2, -1/√2, ...]
            let w8_1 = _mm256_setr_ps(
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
            );
            let w8_1_x_odd_1 = complex_mul_avx(w8_1, x_odd_1);
            let out1 = _mm256_add_ps(x_even_1, w8_1_x_odd_1);
            // out5 = x_even_1 - W_8^1 * x_odd_1
            let out5 = _mm256_sub_ps(x_even_1, w8_1_x_odd_1);

            // out2 = x_even_2 + W_8^2 * x_odd_2
            // W_8^2 = -i = (0, -1), so multiply by -i: (a+bi)*(-i) = (b, -a)
            let w8_2_x_odd_2 = complex_mul_i_avx(x_odd_2, neg_imag_mask);
            let out2 = _mm256_add_ps(x_even_2, w8_2_x_odd_2);
            // out6 = x_even_2 - W_8^2 * x_odd_2
            let out6 = _mm256_sub_ps(x_even_2, w8_2_x_odd_2);

            // out3 = x_even_3 + W_8^3 * x_odd_3
            // W_8^3 = (-1-i)/√2 as interleaved complex: [re, im, re, im, ...] = [-1/√2, -1/√2, ...]
            let w8_3 = _mm256_setr_ps(
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
            );
            let w8_3_x_odd_3 = complex_mul_avx(w8_3, x_odd_3);
            let out3 = _mm256_add_ps(x_even_3, w8_3_x_odd_3);
            // out7 = x_even_3 - W_8^3 * x_odd_3
            let out7 = _mm256_sub_ps(x_even_3, w8_3_x_odd_3);

            // Chunked interleaving: process out0-3 first, then out4-7 to reduce register pressure.
            let j = 8 * i;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;

            // First chunk: Interleave out0-3 using 256-bit operations.
            // Use 256-bit unpacks directly instead of costly 128-bit extract+insert operations.
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);

            // 256-bit unpack operations.
            // unpacklo_pd(a,b) = [a[0],b[0],a[2],b[2]] on 4x 64-bit elements
            // unpackhi_pd(a,b) = [a[1],b[1],a[3],b[3]] on 4x 64-bit elements
            let temp_01_lo = _mm256_unpacklo_pd(out0_pd, out1_pd);
            let temp_01_hi = _mm256_unpackhi_pd(out0_pd, out1_pd);
            let temp_23_lo = _mm256_unpacklo_pd(out2_pd, out3_pd);
            let temp_23_hi = _mm256_unpackhi_pd(out2_pd, out3_pd);

            // Use permute2f128_pd to arrange 128-bit lanes to get sequential layout.
            // permute2f128(a,b,0x20) = [lo128(a), lo128(b)]
            // permute2f128(a,b,0x31) = [hi128(a), hi128(b)]
            let result0 = _mm256_castpd_ps(_mm256_permute2f128_pd(temp_01_lo, temp_23_lo, 0x20));
            let result2 = _mm256_castpd_ps(_mm256_permute2f128_pd(temp_01_hi, temp_23_hi, 0x20));
            let result4 = _mm256_castpd_ps(_mm256_permute2f128_pd(temp_01_lo, temp_23_lo, 0x31));
            let result6 = _mm256_castpd_ps(_mm256_permute2f128_pd(temp_01_hi, temp_23_hi, 0x31));

            // Store first chunk.
            _mm256_storeu_ps(dst_ptr, result0);
            _mm256_storeu_ps(dst_ptr.add(16), result2);
            _mm256_storeu_ps(dst_ptr.add(32), result4);
            _mm256_storeu_ps(dst_ptr.add(48), result6);

            // Second chunk: Interleave out4-7 using 256-bit operations.
            let out4_pd = _mm256_castps_pd(out4);
            let out5_pd = _mm256_castps_pd(out5);
            let out6_pd = _mm256_castps_pd(out6);
            let out7_pd = _mm256_castps_pd(out7);

            // 256-bit unpack operations.
            let temp_45_lo = _mm256_unpacklo_pd(out4_pd, out5_pd);
            let temp_45_hi = _mm256_unpackhi_pd(out4_pd, out5_pd);
            let temp_67_lo = _mm256_unpacklo_pd(out6_pd, out7_pd);
            let temp_67_hi = _mm256_unpackhi_pd(out6_pd, out7_pd);

            // Use permute2f128_pd to arrange 128-bit lanes.
            let result1 = _mm256_castpd_ps(_mm256_permute2f128_pd(temp_45_lo, temp_67_lo, 0x20));
            let result3 = _mm256_castpd_ps(_mm256_permute2f128_pd(temp_45_hi, temp_67_hi, 0x20));
            let result5 = _mm256_castpd_ps(_mm256_permute2f128_pd(temp_45_lo, temp_67_lo, 0x31));
            let result7 = _mm256_castpd_ps(_mm256_permute2f128_pd(temp_45_hi, temp_67_hi, 0x31));

            // Store second chunk.
            _mm256_storeu_ps(dst_ptr.add(8), result1);
            _mm256_storeu_ps(dst_ptr.add(24), result3);
            _mm256_storeu_ps(dst_ptr.add(40), result5);
            _mm256_storeu_ps(dst_ptr.add(56), result7);
        }
    }

    super::butterfly_radix8_scalar::<4>(src, dst, stage_twiddles, 1, simd_iters);
}

/// Performs a single radix-8 Stockham butterfly stage for p>1 (out-of-place, AVX+FMA).
///
/// Generic version for p>1 cases. Uses chunked streaming to reduce register pressure
/// while maintaining scattered stores for non-unit strides.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix8_generic_avx_fma(
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
    let simd_iters = (eighth_samples >> 2) << 2;

    unsafe {
        let neg_imag_mask = load_neg_imag_mask_avx();
        let frac_1_sqrt2_scalar = FRAC_1_SQRT_2;

        for i in (0..simd_iters).step_by(4) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;
            let k2 = k + 2 - ((k + 2 >= stride) as usize) * stride;
            let k3 = k + 3 - ((k + 3 >= stride) as usize) * stride;

            // Load 8 input values from the 8 "eighths" of the input array.
            let z0_ptr = src.as_ptr().add(i) as *const f32;
            let z0 = _mm256_loadu_ps(z0_ptr);

            let z1_ptr = src.as_ptr().add(i + eighth_samples) as *const f32;
            let z1 = _mm256_loadu_ps(z1_ptr);

            let z2_ptr = src.as_ptr().add(i + eighth_samples * 2) as *const f32;
            let z2 = _mm256_loadu_ps(z2_ptr);

            let z3_ptr = src.as_ptr().add(i + eighth_samples * 3) as *const f32;
            let z3 = _mm256_loadu_ps(z3_ptr);

            let z4_ptr = src.as_ptr().add(i + eighth_samples * 4) as *const f32;
            let z4 = _mm256_loadu_ps(z4_ptr);

            let z5_ptr = src.as_ptr().add(i + eighth_samples * 5) as *const f32;
            let z5 = _mm256_loadu_ps(z5_ptr);

            let z6_ptr = src.as_ptr().add(i + eighth_samples * 6) as *const f32;
            let z6 = _mm256_loadu_ps(z6_ptr);

            let z7_ptr = src.as_ptr().add(i + eighth_samples * 7) as *const f32;
            let z7 = _mm256_loadu_ps(z7_ptr);

            // Load prepackaged twiddles in packed format.
            let tw_ptr = stage_twiddles.as_ptr().add(i * 7) as *const f32;
            let w1 = _mm256_loadu_ps(tw_ptr); // w1[i..i+4]
            let w2 = _mm256_loadu_ps(tw_ptr.add(8)); // w2[i..i+4]
            let w3 = _mm256_loadu_ps(tw_ptr.add(16)); // w3[i..i+4]
            let w4 = _mm256_loadu_ps(tw_ptr.add(24)); // w4[i..i+4]
            let w5 = _mm256_loadu_ps(tw_ptr.add(32)); // w5[i..i+4]
            let w6 = _mm256_loadu_ps(tw_ptr.add(40)); // w6[i..i+4]
            let w7 = _mm256_loadu_ps(tw_ptr.add(48)); // w7[i..i+4]

            // Apply twiddle factors
            let t1 = complex_mul_avx(w1, z1);
            let t2 = complex_mul_avx(w2, z2);
            let t3 = complex_mul_avx(w3, z3);
            let t4 = complex_mul_avx(w4, z4);
            let t5 = complex_mul_avx(w5, z5);
            let t6 = complex_mul_avx(w6, z6);
            let t7 = complex_mul_avx(w7, z7);

            // Split-radix decomposition: compute even radix-4 on (z0, t2, t4, t6)
            let even_a0 = _mm256_add_ps(z0, t4);
            let even_a1 = _mm256_sub_ps(z0, t4);
            let even_a2 = _mm256_add_ps(t2, t6);
            let t2_sub_t6 = _mm256_sub_ps(t2, t6);
            let even_a3 = complex_mul_i_avx(t2_sub_t6, neg_imag_mask);

            let x_even_0 = _mm256_add_ps(even_a0, even_a2);
            let x_even_2 = _mm256_sub_ps(even_a0, even_a2);
            let x_even_1 = _mm256_add_ps(even_a1, even_a3);
            let x_even_3 = _mm256_sub_ps(even_a1, even_a3);

            // Compute odd radix-4 on (t1, t3, t5, t7)
            let odd_a0 = _mm256_add_ps(t1, t5);
            let odd_a1 = _mm256_sub_ps(t1, t5);
            let odd_a2 = _mm256_add_ps(t3, t7);
            let t3_sub_t7 = _mm256_sub_ps(t3, t7);
            let odd_a3 = complex_mul_i_avx(t3_sub_t7, neg_imag_mask);

            let x_odd_0 = _mm256_add_ps(odd_a0, odd_a2);
            let x_odd_2 = _mm256_sub_ps(odd_a0, odd_a2);
            let x_odd_1 = _mm256_add_ps(odd_a1, odd_a3);
            let x_odd_3 = _mm256_sub_ps(odd_a1, odd_a3);

            // Apply W_8 twiddles and combine
            let out0 = _mm256_add_ps(x_even_0, x_odd_0);
            let out4 = _mm256_sub_ps(x_even_0, x_odd_0);

            // W_8^1 = (1-i)/√2 as interleaved complex: [1/√2, -1/√2, ...]
            let w8_1 = _mm256_setr_ps(
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
            );
            let w8_1_x_odd_1 = complex_mul_avx(w8_1, x_odd_1);
            let out1 = _mm256_add_ps(x_even_1, w8_1_x_odd_1);
            let out5 = _mm256_sub_ps(x_even_1, w8_1_x_odd_1);

            // W_8^2 = -i: (a+bi) -> (b, -a)
            let w8_2_x_odd_2 = complex_mul_i_avx(x_odd_2, neg_imag_mask);
            let out2 = _mm256_add_ps(x_even_2, w8_2_x_odd_2);
            let out6 = _mm256_sub_ps(x_even_2, w8_2_x_odd_2);

            // W_8^3 = (-1-i)/√2 as interleaved complex: [-1/√2, -1/√2, ...]
            let w8_3 = _mm256_setr_ps(
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
                -frac_1_sqrt2_scalar,
            );
            let w8_3_x_odd_3 = complex_mul_avx(w8_3, x_odd_3);
            let out3 = _mm256_add_ps(x_even_3, w8_3_x_odd_3);
            let out7 = _mm256_sub_ps(x_even_3, w8_3_x_odd_3);

            // Calculate output indices: j = 8*i - 7*k
            let j0 = 8 * i - 7 * k0;
            let j1 = 8 * (i + 1) - 7 * k1;
            let j2 = 8 * (i + 2) - 7 * k2;
            let j3 = 8 * (i + 3) - 7 * k3;

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            // First chunk: out0-3
            let out0_pd = _mm256_castps_pd(out0);
            let out1_pd = _mm256_castps_pd(out1);
            let out2_pd = _mm256_castps_pd(out2);
            let out3_pd = _mm256_castps_pd(out3);

            let out0_lo = _mm256_castpd256_pd128(out0_pd);
            let out0_hi = _mm256_extractf128_pd(out0_pd, 1);
            let out1_lo = _mm256_castpd256_pd128(out1_pd);
            let out1_hi = _mm256_extractf128_pd(out1_pd, 1);
            let out2_lo = _mm256_castpd256_pd128(out2_pd);
            let out2_hi = _mm256_extractf128_pd(out2_pd, 1);
            let out3_lo = _mm256_castpd256_pd128(out3_pd);
            let out3_hi = _mm256_extractf128_pd(out3_pd, 1);

            _mm_storel_pd(dst_ptr.add(j0), out0_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out1_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 2), out2_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 3), out3_lo);

            _mm_storeh_pd(dst_ptr.add(j1), out0_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out1_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 2), out2_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 3), out3_lo);

            _mm_storel_pd(dst_ptr.add(j2), out0_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride), out1_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 2), out2_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 3), out3_hi);

            _mm_storeh_pd(dst_ptr.add(j3), out0_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride), out1_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 2), out2_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 3), out3_hi);

            // Second chunk: out4-7
            let out4_pd = _mm256_castps_pd(out4);
            let out5_pd = _mm256_castps_pd(out5);
            let out6_pd = _mm256_castps_pd(out6);
            let out7_pd = _mm256_castps_pd(out7);

            let out4_lo = _mm256_castpd256_pd128(out4_pd);
            let out4_hi = _mm256_extractf128_pd(out4_pd, 1);
            let out5_lo = _mm256_castpd256_pd128(out5_pd);
            let out5_hi = _mm256_extractf128_pd(out5_pd, 1);
            let out6_lo = _mm256_castpd256_pd128(out6_pd);
            let out6_hi = _mm256_extractf128_pd(out6_pd, 1);
            let out7_lo = _mm256_castpd256_pd128(out7_pd);
            let out7_hi = _mm256_extractf128_pd(out7_pd, 1);

            _mm_storel_pd(dst_ptr.add(j0 + stride * 4), out4_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 5), out5_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 6), out6_lo);
            _mm_storel_pd(dst_ptr.add(j0 + stride * 7), out7_lo);

            _mm_storeh_pd(dst_ptr.add(j1 + stride * 4), out4_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 5), out5_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 6), out6_lo);
            _mm_storeh_pd(dst_ptr.add(j1 + stride * 7), out7_lo);

            _mm_storel_pd(dst_ptr.add(j2 + stride * 4), out4_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 5), out5_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 6), out6_hi);
            _mm_storel_pd(dst_ptr.add(j2 + stride * 7), out7_hi);

            _mm_storeh_pd(dst_ptr.add(j3 + stride * 4), out4_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 5), out5_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 6), out6_hi);
            _mm_storeh_pd(dst_ptr.add(j3 + stride * 7), out7_hi);
        }
    }

    super::butterfly_radix8_scalar::<4>(src, dst, stage_twiddles, stride, simd_iters);
}
