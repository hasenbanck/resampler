use core::arch::x86_64::*;

use crate::fft::{Complex32, butterflies::ops::complex_mul_avx};

/// Performs a single radix-2 Stockham butterfly stage for stride=1 (out-of-place, AVX+FMA).
///
/// This is a specialized version for the stride=1 case (first stage) that uses the unpack-permute
/// pattern to enable sequential SIMD stores instead of scattered scalar stores.
/// This provides significant performance benefits through write-combining and better cache utilization.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix2_stride1_avx_fma(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    let samples = src.len();
    let half_samples = samples >> 1;
    let simd_iters = (half_samples / 4) * 4;

    unsafe {
        for i in (0..simd_iters).step_by(4) {
            // Load 4 complex numbers from first half.
            let a_ptr = src.as_ptr().add(i) as *const f32;
            let a = _mm256_loadu_ps(a_ptr);

            // Load 4 complex numbers from second half.
            let b_ptr = src.as_ptr().add(i + half_samples) as *const f32;
            let b = _mm256_loadu_ps(b_ptr);

            // Load twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i) as *const f32;
            let tw = _mm256_loadu_ps(tw_ptr);

            // Complex multiply: t = tw * b
            let t = complex_mul_avx(tw, b);

            // Butterfly: out_top = a + t, out_bot = a - t
            let out_top = _mm256_add_ps(a, t);
            let out_bot = _mm256_sub_ps(a, t);

            // Apply unpack-permute pattern for sequential stores.
            let out_top_pd = _mm256_castps_pd(out_top);
            let out_bot_pd = _mm256_castps_pd(out_bot);

            // unpacklo_pd: [top0, bot0, top1, bot1] in lower 128-bit lanes.
            let interleaved_lo = _mm256_castpd_ps(_mm256_unpacklo_pd(out_top_pd, out_bot_pd));
            // unpackhi_pd: [top2, bot2, top3, bot3] in higher 128-bit lanes.
            let interleaved_hi = _mm256_castpd_ps(_mm256_unpackhi_pd(out_top_pd, out_bot_pd));

            // Rearrange 128-bit lanes for fully sequential output.
            // permute2f128(a, b, 0x20): [a.lo128, b.lo128]
            let final_0 = _mm256_permute2f128_ps(interleaved_lo, interleaved_hi, 0x20);
            // permute2f128(a, b, 0x31): [a.hi128, b.hi128]
            let final_1 = _mm256_permute2f128_ps(interleaved_lo, interleaved_hi, 0x31);

            // Sequential stores.
            let j = i << 1;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            _mm256_storeu_ps(dst_ptr, final_0);
            _mm256_storeu_ps(dst_ptr.add(8), final_1);
        }
    }

    for i in simd_iters..half_samples {
        let twiddle = stage_twiddles[i];
        let a = src[i];
        let b = twiddle.mul(&src[i + half_samples]);

        let j = i << 1;
        dst[j] = a.add(&b);
        dst[j + 1] = a.sub(&b);
    }
}

/// Performs a single radix-2 Stockham butterfly stage for p>1 (out-of-place, AVX+FMA).
///
/// This is the generic version for p>1 cases.
#[target_feature(enable = "avx,fma")]
pub(super) unsafe fn butterfly_radix2_generic_avx_fma(
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
    let half_samples = samples >> 1;
    let simd_iters = (half_samples / 4) * 4;

    unsafe {
        for i in (0..simd_iters).step_by(4) {
            let k = i % stride;
            let k0 = k;
            let k1 = k + 1 - ((k + 1 >= stride) as usize) * stride;
            let k2 = k + 2 - ((k + 2 >= stride) as usize) * stride;
            let k3 = k + 3 - ((k + 3 >= stride) as usize) * stride;

            // Load 4 complex numbers from first half.
            let a_ptr = src.as_ptr().add(i) as *const f32;
            let a = _mm256_loadu_ps(a_ptr);

            // Load 4 complex numbers from second half.
            let b_ptr = src.as_ptr().add(i + half_samples) as *const f32;
            let b = _mm256_loadu_ps(b_ptr);

            // Load twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i) as *const f32;
            let tw = _mm256_loadu_ps(tw_ptr);

            // Complex multiply: t = tw * b
            let t = complex_mul_avx(tw, b);

            // Butterfly: out_top = a + t, out_bot = a - t
            let out_top = _mm256_add_ps(a, t);
            let out_bot = _mm256_sub_ps(a, t);

            // Calculate output indices: j = (i << 1) - k
            let j0 = (i << 1) - k0;
            let j1 = ((i + 1) << 1) - k1;
            let j2 = ((i + 2) << 1) - k2;
            let j3 = ((i + 3) << 1) - k3;

            // Direct SIMD stores (no temp arrays).
            // Use 64-bit stores for each complex number.
            let out_top_pd = _mm256_castps_pd(out_top);
            let out_bot_pd = _mm256_castps_pd(out_bot);

            // Extract and store each complex number.
            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            // Store complex numbers using _mm_storel_pd.
            let top_lo128 = _mm256_castpd256_pd128(out_top_pd);
            let bot_lo128 = _mm256_castpd256_pd128(out_bot_pd);
            let top_hi128 = _mm256_extractf128_pd(out_top_pd, 1);
            let bot_hi128 = _mm256_extractf128_pd(out_bot_pd, 1);

            _mm_storel_pd(dst_ptr.add(j0), top_lo128);
            _mm_storel_pd(dst_ptr.add(j0 + stride), bot_lo128);

            _mm_storeh_pd(dst_ptr.add(j1), top_lo128);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), bot_lo128);

            _mm_storel_pd(dst_ptr.add(j2), top_hi128);
            _mm_storel_pd(dst_ptr.add(j2 + stride), bot_hi128);

            _mm_storeh_pd(dst_ptr.add(j3), top_hi128);
            _mm_storeh_pd(dst_ptr.add(j3 + stride), bot_hi128);
        }
    }

    for i in simd_iters..half_samples {
        let k = i % stride;
        let twiddle = stage_twiddles[i];

        let a = src[i];
        let b = twiddle.mul(&src[i + half_samples]);

        let j = (i << 1) - k;
        dst[j] = a.add(&b);
        dst[j + stride] = a.sub(&b);
    }
}
