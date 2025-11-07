use crate::fft::{Complex32, butterflies::ops::complex_mul_sse4_2};

/// Performs a single radix-2 Stockham butterfly stage for stride=1 (out-of-place, SSE4.2).
///
/// This is a specialized version for the stride=1 case (first stage) that uses the unpack-permute
/// pattern to enable sequential SIMD stores instead of scattered scalar stores.
/// This provides significant performance benefits through write-combining and better cache utilization.
#[target_feature(enable = "sse4.2")]
pub(super) unsafe fn butterfly_radix2_stride1_sse4_2(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let half_samples = samples >> 1;
    let simd_iters = (half_samples >> 1) << 1;

    unsafe {
        for i in (0..simd_iters).step_by(2) {
            // Load 2 complex numbers from first half.
            let a_ptr = src.as_ptr().add(i) as *const f32;
            let a = _mm_loadu_ps(a_ptr);

            // Load 2 complex numbers from second half.
            let b_ptr = src.as_ptr().add(i + half_samples) as *const f32;
            let b = _mm_loadu_ps(b_ptr);

            // Load twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i) as *const f32;
            let tw = _mm_loadu_ps(tw_ptr);

            // Complex multiply: t = tw * b
            let t = complex_mul_sse4_2(tw, b);

            // Butterfly: out_top = a + t, out_bot = a - t
            let out_top = _mm_add_ps(a, t);
            let out_bot = _mm_sub_ps(a, t);

            // Apply unpack-permute pattern for sequential stores.
            let out_top_pd = _mm_castps_pd(out_top);
            let out_bot_pd = _mm_castps_pd(out_bot);

            // unpacklo_pd: [top0, bot0]
            let interleaved_lo = _mm_castpd_ps(_mm_unpacklo_pd(out_top_pd, out_bot_pd));
            // unpackhi_pd: [top1, bot1]
            let interleaved_hi = _mm_castpd_ps(_mm_unpackhi_pd(out_top_pd, out_bot_pd));

            // Sequential stores.
            let j = i << 1;
            let dst_ptr = dst.as_mut_ptr().add(j) as *mut f32;
            _mm_storeu_ps(dst_ptr, interleaved_lo);
            _mm_storeu_ps(dst_ptr.add(4), interleaved_hi);
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

/// Performs a single radix-2 Stockham butterfly stage for p>1 (out-of-place, SSE4.2).
///
/// This is the generic version for p>1 cases.
#[target_feature(enable = "sse4.2")]
pub(super) unsafe fn butterfly_radix2_generic_sse4_2(
    src: &[Complex32],
    dst: &mut [Complex32],
    stage_twiddles: &[Complex32],
    stride: usize,
) {
    use core::arch::x86_64::*;

    let samples = src.len();
    let half_samples = samples >> 1;
    let simd_iters = (half_samples >> 1) << 1;

    unsafe {
        for i in (0..simd_iters).step_by(2) {
            let k0 = i % stride;
            let k1 = (i + 1) % stride;

            // Load 2 complex numbers from first half.
            let a_ptr = src.as_ptr().add(i) as *const f32;
            let a = _mm_loadu_ps(a_ptr);

            // Load 2 complex numbers from second half.
            let b_ptr = src.as_ptr().add(i + half_samples) as *const f32;
            let b = _mm_loadu_ps(b_ptr);

            // Load twiddles contiguously.
            let tw_ptr = stage_twiddles.as_ptr().add(i) as *const f32;
            let tw = _mm_loadu_ps(tw_ptr);

            // Complex multiply: t = tw * b
            let t = complex_mul_sse4_2(tw, b);

            // Butterfly: out_top = a + t, out_bot = a - t
            let out_top = _mm_add_ps(a, t);
            let out_bot = _mm_sub_ps(a, t);

            // Calculate output indices: j = (i << 1) - k
            let j0 = (i << 1) - k0;
            let j1 = ((i + 1) << 1) - k1;

            // Direct SIMD stores.
            let out_top_pd = _mm_castps_pd(out_top);
            let out_bot_pd = _mm_castps_pd(out_bot);

            let dst_ptr = dst.as_mut_ptr() as *mut f64;

            _mm_storel_pd(dst_ptr.add(j0), out_top_pd);
            _mm_storel_pd(dst_ptr.add(j0 + stride), out_bot_pd);

            _mm_storeh_pd(dst_ptr.add(j1), out_top_pd);
            _mm_storeh_pd(dst_ptr.add(j1 + stride), out_bot_pd);
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
