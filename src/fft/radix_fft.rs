use alloc::{vec, vec::Vec};
use core::{f64, marker::PhantomData, slice};

use super::{
    butterflies::{
        butterfly_radix2_dispatch, butterfly_radix3_dispatch, butterfly_radix4_dispatch,
        butterfly_radix5_dispatch, butterfly_radix7_dispatch, butterfly_radix8_dispatch,
    },
    optimizer::optimize_factors,
    stockham_autosort::OutputLocation,
};
use crate::Complex32;

/// Radix factors supported for mixed-radix FFT decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Radix {
    /// Radix-2
    Factor2,
    /// Radix-3
    Factor3,
    /// Radix-4
    Factor4,
    /// Radix-5
    Factor5,
    /// Radix-7
    Factor7,
    /// Radix-8
    Factor8,
}

impl Radix {
    /// Returns the radix size.
    pub(crate) const fn radix(&self) -> usize {
        match self {
            Radix::Factor2 => 2,
            Radix::Factor3 => 3,
            Radix::Factor4 => 4,
            Radix::Factor5 => 5,
            Radix::Factor7 => 7,
            Radix::Factor8 => 8,
        }
    }
}

/// Marker type for forward FFT direction.
pub(crate) struct Forward;

/// Marker type for inverse FFT direction.
pub(crate) struct Inverse;

type PostprocessFn = fn(&mut [Complex32], &mut [Complex32], &[Complex32]);

type PreprocessFn = fn(&mut [Complex32], &mut [Complex32], &[Complex32]);

type StockhamAutosortFn =
    fn(&mut [Complex32], &[Complex32], &[Radix], &mut [Complex32]) -> OutputLocation;

/// Mixed-radix FFT for real values.
///
/// Generic over direction type (`Forward` or `Inverse`).
///
/// # Normalization
///
/// This FFT implementation does **not** normalize outputs:
///  - Forward FFT produces raw DFT values (no scaling applied)
///  - Inverse FFT produces raw IDFT values (no `1/N` scaling applied)
///  - A forward+inverse round-trip produces `N * input` (where `N` is the FFT length)
///  - Callers must manually normalize results by dividing by `len()` to recover the original signal
///
/// Multiple normalization steps can be merged: when doing forward+inverse FFTs,
/// normalize once by dividing by `len()` instead of normalizing each transform separately.
pub(crate) struct RadixFFT<D> {
    n: usize,
    n2: usize,
    factors: Vec<Radix>,
    stockham_twiddles: Vec<Complex32>,
    real_complex_expansion_twiddles: Vec<Complex32>,
    complex_real_reduction_twiddles: Vec<Complex32>,
    postprocess_fft: PostprocessFn,
    preprocess_ifft: PreprocessFn,
    stockham_autosort_fn: StockhamAutosortFn,
    _direction: PhantomData<D>,
}

/// SIMD width used for twiddle layout optimization.
#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimdWidth {
    /// AVX: 4 complex numbers (256-bit)
    Width4,
    /// SSE + NEON: 2 complex numbers (128-bit).
    Width2,
    /// Scalar: 1 complex number.
    Scalar,
}

impl<D> RadixFFT<D> {
    /// Constructs a new [`RadixFFT`] FFT instance.
    ///
    /// # Arguments
    /// * `factors` - Vector of radix factors defining the FFT stages (e.g., `vec![Factor2; 4]` for size 16)
    ///
    /// # Panics
    /// Panics if the total FFT length is not even since we apply an N/2 optimization double the performance of the FFT.
    pub(crate) fn new(factors: Vec<Radix>) -> Self {
        assert!(!factors.is_empty(), "Factors vector must not be empty");

        let n = factors.iter().map(|f| f.radix()).product();
        assert_eq!(
            n % 2,
            0,
            "FFT length must be even for N/2 optimization, got {n}"
        );
        let n2 = n / 2;

        let factors = Self::compute_factors(&factors);
        let factors = optimize_factors(factors);

        let simd_width = Self::detect_simd_width();

        let stockham_twiddles = if factors.is_empty() || factors.len() == 1 {
            // N/2 = 1 or N/2 is a single radix: no twiddles needed.
            Vec::new()
        } else {
            // Compute pre-packaged twiddles for Stockham autosort optimized for detected SIMD width.
            Self::compute_stockham_twiddles(n2, &factors, simd_width)
        };

        let real_complex_expansion_twiddles = Self::compute_real_complex_expansion_twiddles(n);
        let complex_real_reduction_twiddles = Self::compute_complex_real_reduction_twiddles(n);

        #[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
        let (postprocess_fft, preprocess_ifft): (PostprocessFn, PreprocessFn) =
            if std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("fma")
            {
                (
                    super::real_complex::postprocess_fft_avx_fma_wrapper,
                    super::real_complex::preprocess_ifft_avx_fma_wrapper,
                )
            } else {
                // SSE2 is always available.
                (
                    super::real_complex::postprocess_fft_sse2_wrapper,
                    super::real_complex::preprocess_ifft_sse2_wrapper,
                )
            };

        #[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
        let stockham_autosort_fn: StockhamAutosortFn = if std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("fma")
        {
            super::stockham_autosort::stockham_autosort_avx_fma
        } else if std::arch::is_x86_feature_detected!("sse4.2") {
            super::stockham_autosort::stockham_autosort_sse4_2
        } else {
            // SSE2 is always available.
            super::stockham_autosort::stockham_autosort_sse2
        };

        Self {
            n,
            n2,
            factors,
            stockham_twiddles,
            real_complex_expansion_twiddles,
            complex_real_reduction_twiddles,
            #[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
            postprocess_fft,
            #[cfg(any(not(target_arch = "x86_64"), feature = "no_std"))]
            postprocess_fft: super::real_complex::select_postprocess_fn(),
            #[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
            preprocess_ifft,
            #[cfg(any(not(target_arch = "x86_64"), feature = "no_std"))]
            preprocess_ifft: super::real_complex::select_preprocess_fn(),
            #[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
            stockham_autosort_fn,
            #[cfg(any(not(target_arch = "x86_64"), feature = "no_std"))]
            stockham_autosort_fn: super::stockham_autosort::stockham_autosort,

            _direction: PhantomData,
        }
    }

    fn detect_simd_width() -> SimdWidth {
        #[cfg(all(target_arch = "x86_64", not(feature = "no_std")))]
        let simd_width = if std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("fma")
        {
            SimdWidth::Width4
        } else {
            // SSE2 is always available on x86_64.
            SimdWidth::Width2
        };

        #[cfg(all(
            target_arch = "x86_64",
            feature = "no_std",
            target_feature = "avx",
            target_feature = "fma"
        ))]
        let simd_width = SimdWidth::Width4;

        #[cfg(all(
            target_arch = "x86_64",
            feature = "no_std",
            target_feature = "sse2",
            not(all(target_feature = "avx", target_feature = "fma"))
        ))]
        let simd_width = SimdWidth::Width2;

        #[cfg(target_arch = "aarch64")]
        let simd_width = SimdWidth::Width2;

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let simd_width = SimdWidth::Scalar;

        simd_width
    }

    /// Compute N/2 factors by removing one Factor2 or converting Factor4/Factor8 to appropriate factor.
    fn compute_factors(factors: &[Radix]) -> Vec<Radix> {
        if factors.len() == 1 {
            let factor = factors[0];
            match factor {
                Radix::Factor2 => Vec::new(),
                Radix::Factor4 => vec![Radix::Factor2],
                Radix::Factor8 => vec![Radix::Factor4],
                _ => panic!("Unsupported single factor for N/2 optimization: {factor:?}"),
            }
        } else {
            let mut factors = factors.to_vec();

            if let Some(pos) = factors.iter().position(|&f| f == Radix::Factor2) {
                factors.remove(pos);
            } else if let Some(pos) = factors.iter().position(|&f| f == Radix::Factor8) {
                factors[pos] = Radix::Factor4;
            } else if let Some(pos) = factors.iter().position(|&f| f == Radix::Factor4) {
                factors[pos] = Radix::Factor2;
            } else {
                panic!("Even-length FFT must have at least one Factor2, Factor4, or Factor8");
            }

            factors
        }
    }

    /// Compute a twiddle factor with f64 precision, then convert to f32.
    /// This reduces accumulated floating-point error for large FFTs.
    #[inline(always)]
    fn compute_twiddle_f32(index: usize, fft_len: usize) -> Complex32 {
        let constant = -2.0 * f64::consts::PI / fft_len as f64;
        let angle = constant * index as f64;
        #[cfg(not(feature = "no_std"))]
        return Complex32::new(angle.cos() as f32, angle.sin() as f32);
        #[cfg(feature = "no_std")]
        return Complex32::new(libm::cos(angle) as f32, libm::sin(angle) as f32);
    }

    /// Compute pre-packaged twiddle factors for Stockham autosort algorithm.
    ///
    /// Generates twiddles in a SIMD-optimized layout to eliminate shuffle operations.
    /// For radix-r with (r-1) twiddles per iteration:
    ///
    /// **Width4 (AVX)**: Pack (r-1) twiddles for 4 consecutive iterations, then scalar tail
    /// Example radix-5, stride=2: [w1[0], w1[1], w1[0], w1[1], w2[0], w2[1], w2[0], w2[1], ..., tail]
    ///
    /// **Width2 (SSE/NEON)**: Pack (r-1) twiddles for 2 consecutive iterations, then scalar tail
    /// Example radix-5, stride=2: [w1[0], w1[1], w2[0], w2[1], w3[0], w3[1], ..., tail]
    ///
    /// **Scalar**: Interleaved layout
    /// Example radix-5: [w1[0], w2[0], w3[0], w4[0], w1[1], w2[1], ...]
    ///
    /// This provides 20-30% performance improvement for high-radix butterflies by enabling
    /// direct SIMD loads without shuffling (eliminates 16-24 cycles per 4 samples).
    fn compute_stockham_twiddles(n: usize, factors: &[Radix], width: SimdWidth) -> Vec<Complex32> {
        let mut twiddles = Vec::new();
        let mut stage_size = 1;
        let mut stride = 1;

        for (stage_index, &radix) in factors.iter().enumerate() {
            let r = radix.radix();
            let num_twiddles_per_iter = r - 1;
            stage_size *= r;
            let num_columns = stage_size / r;
            debug_assert_eq!(num_columns, stride);

            // Skip generating twiddles for the first stage (stride=1) since all twiddles
            // are identity values (1+0i) and the butterfly implementations handle this
            // case without twiddle multiplication.
            if stage_index == 0 {
                stride = stage_size;
                continue;
            }

            let iterations = n / r;

            // Generate base twiddles for this stage (one per column).
            let mut base_twiddles = Vec::with_capacity(stride * num_twiddles_per_iter);
            for col in 0..stride {
                for k in 1..r {
                    base_twiddles.push(Self::compute_twiddle_f32(col * k, stage_size));
                }
            }

            // Generate twiddles in the appropriate SIMD-optimized layout.
            match width {
                SimdWidth::Width4 => {
                    // AVX layout: For radix-r, pack r-1 twiddles for 4 consecutive iterations.
                    // Example radix-5: [w1[i], w1[i+1], w1[i+2], w1[i+3], w2[i], w2[i+1], w2[i+2], w2[i+3], ...]
                    let simd_iters = (iterations / 4) * 4;
                    for i in (0..simd_iters).step_by(4) {
                        for tw_idx in 0..num_twiddles_per_iter {
                            for j in 0..4 {
                                let col = (i + j) % stride;
                                let base_idx = col * num_twiddles_per_iter + tw_idx;
                                twiddles.push(base_twiddles[base_idx]);
                            }
                        }
                    }
                    // Scalar tail.
                    for i in simd_iters..iterations {
                        let col = i % stride;
                        for k in 1..r {
                            twiddles.push(base_twiddles[col * num_twiddles_per_iter + (k - 1)]);
                        }
                    }
                }
                SimdWidth::Width2 => {
                    // SSE/NEON layout: For radix-r, pack r-1 twiddles for 2 consecutive iterations.
                    // Example radix-5: [w1[i], w1[i+1], w2[i], w2[i+1], w3[i], w3[i+1], ...]
                    let simd_iters = (iterations / 2) * 2;
                    for i in (0..simd_iters).step_by(2) {
                        for tw_idx in 0..num_twiddles_per_iter {
                            for j in 0..2 {
                                let col = (i + j) % stride;
                                let base_idx = col * num_twiddles_per_iter + tw_idx;
                                twiddles.push(base_twiddles[base_idx]);
                            }
                        }
                    }
                    // Scalar tail.
                    for i in simd_iters..iterations {
                        let col = i % stride;
                        for k in 1..r {
                            twiddles.push(base_twiddles[col * num_twiddles_per_iter + (k - 1)]);
                        }
                    }
                }
                SimdWidth::Scalar => {
                    // Scalar layout: [w1[0], w2[0], ..., w1[1], w2[1], ...]
                    for i in 0..iterations {
                        let col = i % stride;
                        for k in 1..r {
                            twiddles.push(base_twiddles[col * num_twiddles_per_iter + (k - 1)]);
                        }
                    }
                }
            }

            stride = stage_size;
        }

        twiddles
    }

    #[inline(always)]
    fn twiddle_count(n: usize) -> usize {
        if n.is_multiple_of(4) {
            n / 4
        } else {
            n / 4 + 1
        }
    }

    /// Compute post-processing twiddle factors for N/2 optimization.
    ///
    /// These twiddles are used to expand the N/2-point complex FFT result into
    /// the full N/2+1-point real FFT spectrum.
    fn compute_real_complex_expansion_twiddles(n: usize) -> Vec<Complex32> {
        let twiddle_count = Self::twiddle_count(n);
        (1..twiddle_count)
            .map(|k| {
                let twiddle = Self::compute_twiddle_f32(k, n);
                Complex32::new(twiddle.re * 0.5, twiddle.im * 0.5)
            })
            .collect()
    }

    /// Compute inverse preprocessing twiddle factors for N/2 optimization.
    ///
    /// These twiddles are used to reduce the full N/2+1-point real FFT spectrum result into
    /// the N/2-point complex FFT.
    fn compute_complex_real_reduction_twiddles(n: usize) -> Vec<Complex32> {
        let twiddle_count = Self::twiddle_count(n);
        (1..twiddle_count)
            .map(|k| {
                let twiddle = Self::compute_twiddle_f32(k, n);
                Complex32::new(twiddle.re, -twiddle.im)
            })
            .collect()
    }

    /// Returns the required scratchpad size for FFT processing.
    pub(crate) fn scratchpad_size(&self) -> usize {
        // The Stockham autosort algorithm requires 3x N buffer space.
        3 * self.n
    }

    /// For single-element FFTs, twiddles are all the identity value.
    fn apply_single_butterfly(data: &mut [Complex32], scratch: &mut [Complex32], factor: Radix) {
        const STRIDE: usize = 1;

        let radix = factor.radix();
        debug_assert_eq!(data.len(), radix);
        debug_assert!(scratch.len() >= radix);

        match factor {
            Radix::Factor2 => {
                let twiddles = [Complex32::new(1.0, 0.0)];
                butterfly_radix2_dispatch(data, scratch, &twiddles, STRIDE);
            }
            Radix::Factor3 => {
                let twiddles = [Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)];
                butterfly_radix3_dispatch(data, scratch, &twiddles, STRIDE);
            }
            Radix::Factor4 => {
                let twiddles = [
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                ];
                butterfly_radix4_dispatch(data, scratch, &twiddles, STRIDE);
            }
            Radix::Factor5 => {
                let twiddles = [
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                ];
                butterfly_radix5_dispatch(data, scratch, &twiddles, STRIDE);
            }
            Radix::Factor7 => {
                let twiddles = [
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                ];
                butterfly_radix7_dispatch(data, scratch, &twiddles, STRIDE);
            }
            Radix::Factor8 => {
                let twiddles = [
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                ];
                butterfly_radix8_dispatch(data, scratch, &twiddles, STRIDE);
            }
        }

        // Copy result back to data buffer to maintain OutputLocation::Data semantics.
        data.copy_from_slice(&scratch[..radix]);
    }

    /// Perform N/2-point complex FFT using Stockham Autosort algorithm.
    ///
    /// # Arguments
    /// * `scratchpad` - Scratch buffer where first n2 elements contain the FFT data,
    ///   and remaining elements are used for ping-pong buffering
    #[must_use]
    fn process_forward_complex(&self, scratchpad: &mut [Complex32]) -> OutputLocation {
        debug_assert!(scratchpad.len() >= 3 * self.n);

        let (data, scratch_remainder) = scratchpad.split_at_mut(self.n2);

        if self.factors.is_empty() {
            // N/2 = 1, no FFT needed - data already contains input.
            OutputLocation::Data
        } else if self.factors.len() == 1 {
            // Single factor.
            Self::apply_single_butterfly(data, &mut scratch_remainder[..self.n2], self.factors[0]);
            OutputLocation::Data
        } else {
            // Use Stockham autosort for both pure radix-2 and mixed-radix.
            (self.stockham_autosort_fn)(
                data,
                &self.stockham_twiddles,
                &self.factors,
                &mut scratch_remainder[..self.n2],
            )
        }
    }

    /// Post-process N/2 complex FFT output to produce N/2+1.
    fn postprocess_fft(&self, fft_output: &[Complex32], output: &mut [Complex32]) {
        debug_assert_eq!(fft_output.len(), self.n2);
        debug_assert!(output.len() > self.n2);

        output[..self.n2].copy_from_slice(fft_output);

        let split_point = output.len() / 2;
        let (output_left, output_right) = output.split_at_mut(split_point);

        // Handle DC and Nyquist bins first (no twiddles needed).
        // Extract first element's real and imaginary for special processing.
        if let (Some(first_element), Some(last_element)) =
            (output_left.first_mut(), output_right.last_mut())
        {
            let z0 = *first_element;
            *first_element = Complex32::new(z0.re + z0.im, 0.0);
            *last_element = Complex32::new(z0.re - z0.im, 0.0);
        } else {
            return;
        }

        // Get slices excluding first and last elements (the middle elements).
        let output_left_middle = &mut output_left[1..];
        let right_len = output_right.len();
        let output_right_middle = &mut output_right[..right_len - 1];

        (self.postprocess_fft)(
            output_left_middle,
            output_right_middle,
            &self.real_complex_expansion_twiddles,
        );

        // Handle odd-length case: conjugate center element.
        if output.len() % 2 == 1 {
            let center_idx = output.len() / 2;
            output[center_idx].im = -output[center_idx].im;
        }
    }

    /// Internal forward real -> complex FFT using N/2 optimization.
    fn process_forward_internal(
        &self,
        input: &[f32],
        output: &mut [Complex32],
        scratchpad: &mut [Complex32],
    ) {
        assert_eq!(self.n / 2, self.n2);
        assert!(input.len() <= self.n);
        assert!(output.len() >= self.n2);
        assert!(scratchpad.len() >= self.scratchpad_size());

        // Reinterpret N real values as N/2 complex values by pairing consecutive samples.
        let ptr = input.as_ptr() as *const Complex32;
        let buf_in = unsafe { slice::from_raw_parts(ptr, self.n2) };
        scratchpad[..self.n2].copy_from_slice(buf_in);

        let fft_output = match self.process_forward_complex(scratchpad) {
            OutputLocation::Data => &scratchpad[..self.n2],
            OutputLocation::Scratchpad => &scratchpad[self.n2..self.n2 * 2],
        };

        self.postprocess_fft(fft_output, output);
    }

    /// Internal inverse complex -> real FFT using N/2 optimization.
    fn process_inverse_internal(
        &self,
        input: &[Complex32],
        output: &mut [f32],
        scratchpad: &mut [Complex32],
    ) {
        assert_eq!(self.n / 2, self.n2);
        assert!(input.len() <= self.n);
        assert!(output.len() >= self.n2);
        assert!(scratchpad.len() >= self.scratchpad_size());

        // Copy input to scratch buffer (need N/2+1 elements including Nyquist bin).
        scratchpad[..input.len()].copy_from_slice(input);

        self.preprocess_ifft(&mut scratchpad[..self.n2 + 1]);
        self.process_inverse_complex(scratchpad);

        // Reinterpret N real values as N/2 complex values by pairing consecutive samples.
        let ptr = output.as_mut_ptr() as *mut Complex32;
        let buf_out = unsafe { slice::from_raw_parts_mut(ptr, self.n2) };
        buf_out.copy_from_slice(&scratchpad[..self.n2]);

        // Zero-pad remaining output if needed.
        output[self.n..].fill(0.0);
    }

    /// Pre-process real FFT input to prepare for N/2-point inverse complex FFT.
    fn preprocess_ifft(&self, buffer: &mut [Complex32]) {
        let (input_left, input_right) = buffer.split_at_mut(buffer.len() / 2);

        // DC and Nyquist preprocessing.
        if let (Some(first_element), Some(last_element)) =
            (input_left.first_mut(), input_right.last_mut())
        {
            let first_sum = first_element.add(last_element);
            let first_diff = first_element.sub(last_element);
            *first_element =
                Complex32::new(first_sum.re - first_sum.im, first_diff.re - first_diff.im);
        } else {
            return;
        }

        let input_left_middle = &mut input_left[1..];
        let right_len = input_right.len();
        let input_right_middle = &mut input_right[..right_len - 1];

        (self.preprocess_ifft)(
            input_left_middle,
            input_right_middle,
            &self.complex_real_reduction_twiddles,
        );

        // Handle odd-length case: double and conjugate center element.
        if buffer.len() % 2 == 1 {
            let center_idx = buffer.len() / 2;
            let center_element = buffer[center_idx];
            let doubled = center_element.add(&center_element);
            buffer[center_idx] = doubled.conj();
        }
    }

    /// Perform N/2-point complex inverse FFT using Stockham Autosort algorithm.
    ///
    /// # Arguments
    /// * `scratchpad` - Scratch buffer where first n2 elements contain the FFT data,
    ///   and remaining elements are used for ping-pong buffering
    fn process_inverse_complex(&self, scratchpad: &mut [Complex32]) {
        let (data, scratch_remainder) = scratchpad.split_at_mut(self.n2);

        // Conjugate input.
        data.iter_mut().for_each(|x| {
            x.im = -x.im;
        });

        let output_location = if self.factors.is_empty() {
            // N/2 = 1, no FFT needed.
            OutputLocation::Data
        } else if self.factors.len() == 1 {
            // Single factor.
            Self::apply_single_butterfly(data, &mut scratch_remainder[..self.n2], self.factors[0]);
            OutputLocation::Data
        } else {
            // Use Stockham autosort for both pure radix-2 and mixed-radix.
            (self.stockham_autosort_fn)(
                data,
                &self.stockham_twiddles,
                &self.factors,
                &mut scratch_remainder[..self.n2],
            )
        };

        // Conjugate output in the correct buffer and copy to data if needed.
        match output_location {
            OutputLocation::Data => {
                data.iter_mut().for_each(|x| {
                    x.im = -x.im;
                });
            }
            OutputLocation::Scratchpad => {
                scratch_remainder[..self.n2].iter_mut().for_each(|x| {
                    x.im = -x.im;
                });
                data.copy_from_slice(&scratch_remainder[..self.n2]);
            }
        }
    }
}

impl RadixFFT<Forward> {
    /// Forward real -> complex FFT.
    ///
    /// Transforms real-valued input into half-complex packed format.
    ///
    /// **Note:** Output is unnormalized. No scaling is applied to the result.
    ///
    /// # Arguments
    /// * `input` - Real-valued input samples
    /// * `output` - Half-complex output
    /// * `scratchpad` - Workspace for intermediate calculations
    pub(crate) fn process(
        &self,
        input: &[f32],
        output: &mut [Complex32],
        scratchpad: &mut [Complex32],
    ) {
        self.process_forward_internal(input, output, scratchpad);
    }
}

impl RadixFFT<Inverse> {
    /// Inverse complex -> real FFT.
    ///
    /// Transforms half-complex packed format back into real-valued output.
    ///
    /// **Note:** Output is unnormalized (no `1/N` scaling applied).
    ///
    /// # Arguments
    /// * `input` - Half-complex input
    /// * `output` - Real-valued output samples
    /// * `scratchpad` - Workspace for intermediate calculations
    pub(crate) fn process(
        &self,
        input: &[Complex32],
        output: &mut [f32],
        scratchpad: &mut [Complex32],
    ) {
        self.process_inverse_internal(input, output, scratchpad);
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::PI;

    use super::*;

    const EPSILON: f32 = 1.0e-5;

    /// Single-stage radix factors to test (only even lengths for N/2 optimization).
    const SINGLE_STAGE_FACTORS: &[Radix] = &[Radix::Factor2, Radix::Factor4];

    /// Multi-stage radix factor combinations to test (only even lengths for N/2 optimization).
    const MULTI_STAGE_FACTORS: &[(&[Radix], &str)] = &[
        (&[Radix::Factor2, Radix::Factor2], "2x2"),
        (&[Radix::Factor2, Radix::Factor3], "2x3"),
        (&[Radix::Factor3, Radix::Factor2], "3x2"),
        (&[Radix::Factor2, Radix::Factor4], "2x4"),
        (&[Radix::Factor2, Radix::Factor5], "2x5"),
        (&[Radix::Factor5, Radix::Factor2], "5x2"),
        (&[Radix::Factor2, Radix::Factor7], "2x7"),
        (&[Radix::Factor7, Radix::Factor2], "7x2"),
        (&[Radix::Factor4, Radix::Factor3], "4x3"),
        (&[Radix::Factor3, Radix::Factor4], "3x4"),
        (&[Radix::Factor4, Radix::Factor4], "4x4"),
        (&[Radix::Factor2, Radix::Factor2, Radix::Factor3], "2x2x3"),
        (&[Radix::Factor2, Radix::Factor3, Radix::Factor2], "2x3x2"),
        (&[Radix::Factor3, Radix::Factor2, Radix::Factor2], "3x2x2"),
        (&[Radix::Factor2, Radix::Factor2, Radix::Factor4], "2x2x4"),
        (&[Radix::Factor2, Radix::Factor2, Radix::Factor5], "2x2x5"),
        (&[Radix::Factor5, Radix::Factor2, Radix::Factor3], "5x2x3"),
        (&[Radix::Factor5, Radix::Factor3, Radix::Factor2], "5x3x2"),
        (&[Radix::Factor2, Radix::Factor7, Radix::Factor3], "2x7x3"),
        (&[Radix::Factor4, Radix::Factor4, Radix::Factor4], "4x4x4"),
        (&[Radix::Factor2, Radix::Factor2, Radix::Factor7], "2x2x7"),
        (&[Radix::Factor3, Radix::Factor2, Radix::Factor5], "3x2x5"),
        (&[Radix::Factor3, Radix::Factor5, Radix::Factor2], "3x5x2"),
        (&[Radix::Factor2, Radix::Factor3, Radix::Factor7], "2x3x7"),
        (&[Radix::Factor2, Radix::Factor4, Radix::Factor2], "2x4x2"),
        (&[Radix::Factor4, Radix::Factor2, Radix::Factor2], "4x2x2"),
        (&[Radix::Factor4, Radix::Factor3, Radix::Factor2], "4x3x2"),
        (&[Radix::Factor2, Radix::Factor2, Radix::Factor2], "2x2x2"),
        (&[Radix::Factor4, Radix::Factor2, Radix::Factor4], "4x2x4"),
        (&[Radix::Factor2, Radix::Factor4, Radix::Factor4], "2x4x4"),
    ];

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    fn approx_eq_complex(a: Complex32, b: Complex32, epsilon: f32) -> bool {
        approx_eq(a.re, b.re, epsilon) && approx_eq(a.im, b.im, epsilon)
    }

    fn make_multiplier(size: usize) -> usize {
        debug_assert!(size.is_power_of_two());
        size.trailing_zeros() as usize
    }

    #[test]
    fn test_dc_signal() {
        // A constant signal should have all energy in DC bin.
        for &factor in SINGLE_STAGE_FACTORS {
            let size = factor.radix();
            let fft = RadixFFT::<Forward>::new(vec![factor]);
            let input = vec![1.0f32; size];
            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);

            // DC component should equal sum of inputs.
            assert!(
                approx_eq(output[0].re, size as f32, EPSILON),
                "Factor {factor:?}: DC real failed"
            );
            assert!(
                approx_eq(output[0].im, 0.0, EPSILON),
                "Factor {factor:?}: DC imag failed"
            );

            // All other bins should be near zero.
            for i in 1..output.len() {
                assert!(
                    output[i].re.abs() < EPSILON,
                    "Factor {factor:?}: Bin {i} re: {}",
                    output[i].re
                );
                assert!(
                    output[i].im.abs() < EPSILON,
                    "Factor {factor:?}: Bin {i} im: {}",
                    output[i].im
                );
            }
        }
    }

    #[test]
    fn test_impulse() {
        // An impulse should have flat spectrum.
        for &factor in SINGLE_STAGE_FACTORS {
            let size = factor.radix();
            let fft = RadixFFT::<Forward>::new(vec![factor]);
            let mut input = vec![0.0f32; size];
            input[0] = 1.0;

            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);

            // All bins should have magnitude close to 1.
            for i in 0..output.len() {
                let mag = (output[i].re * output[i].re + output[i].im * output[i].im).sqrt();
                assert!(
                    approx_eq(mag, 1.0, EPSILON),
                    "Factor {factor:?}: Bin {i} mag: {mag}"
                );
            }
        }
    }

    #[test]
    fn test_sine_wave() {
        // A sine wave at bin k should have energy only at bin k.
        let size = 32;
        let multiplier = make_multiplier(size);
        let fft = RadixFFT::<Forward>::new(vec![Radix::Factor2; multiplier]);
        let k = 3;

        let input: Vec<f32> = (0..size)
            .map(|i| (2.0 * PI * k as f32 * i as f32 / size as f32).sin())
            .collect();

        let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
        let mut output = vec![Complex32::default(); size / 2 + 1];

        fft.process(&input, &mut output, &mut scratchpad);

        // Check that bin k has most of the energy
        for i in 0..output.len() {
            let mag = (output[i].re * output[i].re + output[i].im * output[i].im).sqrt();
            if i == k {
                assert!(
                    mag > size as f32 * 0.4,
                    "Bin {k} should have high energy, got {mag}"
                );
            } else {
                assert!(mag < 1.0, "Bin {i} should have low energy, got {mag}");
            }
        }
    }

    #[test]
    fn test_cosine_wave() {
        // A cosine wave at bin k should have energy only at bin k.
        let size = 32;
        let multiplier = make_multiplier(size);
        let fft = RadixFFT::<Forward>::new(vec![Radix::Factor2; multiplier]);
        let k = 4; // frequency bin

        let input: Vec<f32> = (0..size)
            .map(|i| (2.0 * PI * k as f32 * i as f32 / size as f32).cos())
            .collect();

        let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
        let mut output = vec![Complex32::default(); size / 2 + 1];

        fft.process(&input, &mut output, &mut scratchpad);

        // Check that bin k has most of the energy.
        for i in 0..output.len() {
            let mag = (output[i].re * output[i].re + output[i].im * output[i].im).sqrt();
            if i == k {
                assert!(
                    mag > size as f32 * 0.4,
                    "Bin {k} should have high energy, got {mag}"
                );
            } else if i == 0 {
                assert!(mag < 1.0, "DC bin should have low energy, got {mag}");
            } else {
                assert!(mag < 1.0, "Bin {i} should have low energy, got {mag}");
            }
        }
    }

    #[test]
    fn test_linearity() {
        // FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
        for &factor in SINGLE_STAGE_FACTORS {
            let size = factor.radix();
            let fft = RadixFFT::<Forward>::new(vec![factor]);

            let input1: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
            let input2: Vec<f32> = (0..size).map(|i| (i as f32).cos()).collect();

            let a = 2.0;
            let b = 3.0;
            let combined: Vec<f32> = input1
                .iter()
                .zip(input2.iter())
                .map(|(x, y)| a * x + b * y)
                .collect();

            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output1 = vec![Complex32::default(); size / 2 + 1];
            let mut output2 = vec![Complex32::default(); size / 2 + 1];
            let mut output_combined = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input1, &mut output1, &mut scratchpad);
            fft.process(&input2, &mut output2, &mut scratchpad);
            fft.process(&combined, &mut output_combined, &mut scratchpad);

            for i in 0..output1.len() {
                let expected = Complex32::new(
                    a * output1[i].re + b * output2[i].re,
                    a * output1[i].im + b * output2[i].im,
                );
                assert!(
                    approx_eq_complex(output_combined[i], expected, EPSILON * 10.0),
                    "Factor {:?}: Bin {i} failed linearity: got ({}, {}), expected ({}, {})",
                    factor,
                    output_combined[i].re,
                    output_combined[i].im,
                    expected.re,
                    expected.im
                );
            }
        }
    }

    #[test]
    fn test_parseval() {
        // Parseval's theorem: sum of |x|^2 = sum of |X|^2 / N
        for &factor in SINGLE_STAGE_FACTORS {
            let size = factor.radix();
            let fft = RadixFFT::<Forward>::new(vec![factor]);

            let input: Vec<f32> = (0..size).map(|i| (i as f32 / 10.0).sin()).collect();

            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);

            let time_energy: f32 = input.iter().map(|x| x * x).sum();

            // For real FFT, we need to account for the symmetry.
            // DC bin is counted once.
            let mut freq_energy = output[0].re * output[0].re;

            // For even sizes, Nyquist bin (at size/2) is also real and counted once.
            // For odd sizes, there is no Nyquist bin.
            if size % 2 == 0 {
                freq_energy += output[size / 2].re * output[size / 2].re;
                // Intermediate bins are counted twice (positive and negative frequencies).
                for i in 1..(size / 2) {
                    freq_energy +=
                        2.0 * (output[i].re * output[i].re + output[i].im * output[i].im);
                }
            } else {
                // For odd sizes, all bins except DC are counted twice.
                for i in 1..=size / 2 {
                    freq_energy +=
                        2.0 * (output[i].re * output[i].re + output[i].im * output[i].im);
                }
            }
            freq_energy /= size as f32;

            assert!(
                approx_eq(time_energy, freq_energy, EPSILON * size as f32),
                "Factor {factor:?}: Parseval's theorem failed: time={time_energy}, freq={freq_energy}"
            );
        }
    }

    #[test]
    fn test_multiple_sizes() {
        // Test various power-of-two sizes.
        for log_size in 2..10 {
            let size = 1 << log_size;
            let multiplier = make_multiplier(size);
            let fft_fwd = RadixFFT::<Forward>::new(vec![Radix::Factor2; multiplier]);
            let fft_inv = RadixFFT::<Inverse>::new(vec![Radix::Factor2; multiplier]);

            let input: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
            let mut scratchpad = vec![Complex32::default(); fft_fwd.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];
            let mut reconstructed = vec![0.0f32; size];

            fft_fwd.process(&input, &mut output, &mut scratchpad);
            fft_inv.process(&output, &mut reconstructed, &mut scratchpad);

            // Manually normalize by 1/N.
            for i in 0..size {
                reconstructed[i] /= size as f32;
            }

            for i in 0..size {
                assert!(
                    approx_eq(input[i], reconstructed[i], EPSILON * 10.0),
                    "Size {size} failed at index {i}: {} != {}",
                    input[i],
                    reconstructed[i]
                );
            }
        }
    }

    #[test]
    fn test_nyquist_frequency() {
        // Test Nyquist frequency (alternating +1, -1).
        // This test only makes sense for even sizes.
        for &factor in SINGLE_STAGE_FACTORS {
            let size = factor.radix();

            // Skip odd sizes, they don't have a Nyquist bin.
            if size % 2 != 0 {
                continue;
            }

            let fft = RadixFFT::<Forward>::new(vec![factor]);

            let input: Vec<f32> = (0..size)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect();

            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);

            // All energy should be at Nyquist frequency.
            let nyquist_idx = size / 2;
            let nyquist_mag = output[nyquist_idx].re.abs();

            assert!(
                nyquist_mag > size as f32 * 0.4,
                "Factor {factor:?}: Nyquist should have high energy"
            );

            for i in 0..output.len() {
                if i != nyquist_idx {
                    let mag = (output[i].re * output[i].re + output[i].im * output[i].im).sqrt();
                    assert!(
                        mag < 1.0,
                        "Factor {factor:?}: Bin {i} should have low energy, got {mag}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_random_signals() {
        // Test with pseudo-random data.
        for &factor in SINGLE_STAGE_FACTORS {
            let size = factor.radix();
            let fft_fwd = RadixFFT::<Forward>::new(vec![factor]);
            let fft_inv = RadixFFT::<Inverse>::new(vec![factor]);

            // Simple pseudo-random generator.
            let mut seed = 12345u32;
            let random = |s: &mut u32| -> f32 {
                *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (*s as f32 / u32::MAX as f32) * 2.0 - 1.0
            };

            let input: Vec<f32> = (0..size).map(|_| random(&mut seed)).collect();

            let mut scratchpad = vec![Complex32::default(); fft_fwd.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];
            let mut reconstructed = vec![0.0f32; size];

            fft_fwd.process(&input, &mut output, &mut scratchpad);
            fft_inv.process(&output, &mut reconstructed, &mut scratchpad);

            // Manually normalize by 1/N.
            for i in 0..size {
                reconstructed[i] /= size as f32;
            }

            for i in 0..size {
                assert!(
                    approx_eq(input[i], reconstructed[i], EPSILON * 10.0),
                    "Factor {factor:?}: Random signal failed at index {i}: {} != {}",
                    input[i],
                    reconstructed[i]
                );
            }
        }
    }

    #[test]
    fn test_zero_signal() {
        // Zero input should give zero output.
        for &factor in SINGLE_STAGE_FACTORS {
            let size = factor.radix();
            let fft = RadixFFT::<Forward>::new(vec![factor]);

            let input = vec![0.0f32; size];
            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);

            for i in 0..output.len() {
                assert!(
                    approx_eq(output[i].re, 0.0, EPSILON),
                    "Factor {factor:?}: Bin {i} re should be 0"
                );
                assert!(
                    approx_eq(output[i].im, 0.0, EPSILON),
                    "Factor {factor:?}: Bin {i} im should be 0"
                );
            }
        }
    }

    #[test]
    fn test_symmetry() {
        // For real input, the output should have Hermitian symmetry.
        // This is implicitly tested by the half-complex format, but let's verify.
        for &factor in SINGLE_STAGE_FACTORS {
            let size = factor.radix();
            let fft = RadixFFT::<Forward>::new(vec![factor]);

            let input: Vec<f32> = (0..size).map(|i| (i as f32 / 2.0).sin()).collect();

            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);

            // DC should always be real.
            assert!(
                approx_eq(output[0].im, 0.0, EPSILON),
                "Factor {factor:?}: DC should be real"
            );

            // Nyquist bin (only exists for even sizes) should also be real.
            if size % 2 == 0 {
                assert!(
                    approx_eq(output[size / 2].im, 0.0, EPSILON),
                    "Factor {factor:?}: Nyquist should be real"
                );
            }
        }
    }

    #[test]
    fn test_two_sines() {
        // Signal with two frequency components.
        let size = 64;
        let multiplier = make_multiplier(size);
        let fft = RadixFFT::<Forward>::new(vec![Radix::Factor2; multiplier]);

        let k1 = 5;
        let k2 = 10;

        let input: Vec<f32> = (0..size)
            .map(|i| {
                (2.0 * PI * k1 as f32 * i as f32 / size as f32).sin()
                    + 0.5 * (2.0 * PI * k2 as f32 * i as f32 / size as f32).sin()
            })
            .collect();

        let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
        let mut output = vec![Complex32::default(); size / 2 + 1];

        fft.process(&input, &mut output, &mut scratchpad);

        // Check that k1 and k2 have high energy.
        let mag_k1 = (output[k1].re * output[k1].re + output[k1].im * output[k1].im).sqrt();
        let mag_k2 = (output[k2].re * output[k2].re + output[k2].im * output[k2].im).sqrt();

        assert!(mag_k1 > 10.0, "Bin {k1} should have high energy",);
        assert!(mag_k2 > 5.0, "Bin {k2} should have moderate energy");
    }

    #[test]
    fn test_inverse_scaling() {
        // Test round-trip with manual normalization.
        // FFT is unnormalized, so forward+inverse produces N*input.
        // User must manually divide by N to recover original signal.
        for &factor in SINGLE_STAGE_FACTORS {
            let size = factor.radix();
            let fft_fwd = RadixFFT::<Forward>::new(vec![factor]);
            let fft_inv = RadixFFT::<Inverse>::new(vec![factor]);

            let input = vec![1.0f32; size];
            let mut scratchpad = vec![Complex32::default(); fft_fwd.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];
            let mut reconstructed = vec![0.0f32; size];

            fft_fwd.process(&input, &mut output, &mut scratchpad);
            fft_inv.process(&output, &mut reconstructed, &mut scratchpad);

            // Manually normalize by 1/N.
            for i in 0..size {
                reconstructed[i] /= size as f32;
            }

            // Should get back exactly 1.0 for all samples after normalization.
            for i in 0..size {
                assert!(
                    approx_eq(reconstructed[i], 1.0, EPSILON),
                    "Factor {factor:?}: Sample {i} = {}, expected 1.0",
                    reconstructed[i]
                );
            }
        }
    }

    // Naive DFT for comparison (only practical for small sizes).
    fn naive_dft(input: &[f32]) -> Vec<Complex32> {
        let n = input.len();
        let mut output = vec![Complex32::default(); n / 2 + 1];

        for k in 0..=n / 2 {
            let mut sum = Complex32::new(0.0, 0.0);
            for (i, &x) in input.iter().enumerate() {
                let angle = -2.0 * PI * k as f32 * i as f32 / n as f32;
                let twiddle = Complex32::new(angle.cos(), angle.sin());
                sum = sum.add(&twiddle.scale(x));
            }
            output[k] = sum;
        }

        output
    }

    #[test]
    fn test_vs_naive_dft() {
        // Compare against naive DFT for small sizes
        for &factor in SINGLE_STAGE_FACTORS {
            let size = factor.radix();
            let fft = RadixFFT::<Forward>::new(vec![factor]);

            let input: Vec<f32> = (0..size).map(|i| (i as f32 / 3.0).sin()).collect();

            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);
            let naive_output = naive_dft(&input);

            for i in 0..output.len() {
                assert!(
                    approx_eq_complex(output[i], naive_output[i], EPSILON * size as f32),
                    "Factor {factor:?}: Size {size} bin {i} failed: got ({}, {}), expected ({}, {})",
                    output[i].re,
                    output[i].im,
                    naive_output[i].re,
                    naive_output[i].im
                );
            }
        }
    }

    #[test]
    fn test_multi_stage_round_trip() {
        // Test forward+inverse reconstruction for all multi-stage configurations.
        for &(factors, desc) in MULTI_STAGE_FACTORS {
            let size: usize = factors.iter().map(|f| f.radix()).product();
            let fft_fwd = RadixFFT::<Forward>::new(factors.to_vec());
            let fft_inv = RadixFFT::<Inverse>::new(factors.to_vec());

            let input: Vec<f32> = (0..size).map(|i| (i as f32 / 2.0).sin()).collect();
            let mut scratchpad = vec![Complex32::default(); fft_fwd.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];
            let mut reconstructed = vec![0.0f32; size];

            fft_fwd.process(&input, &mut output, &mut scratchpad);
            fft_inv.process(&output, &mut reconstructed, &mut scratchpad);

            // Manually normalize by 1/N.
            for i in 0..size {
                reconstructed[i] /= size as f32;
            }

            for i in 0..size {
                assert!(
                    approx_eq(input[i], reconstructed[i], EPSILON * 10.0),
                    "Config {desc} (size {size}) failed at index {i}: {} != {}",
                    input[i],
                    reconstructed[i]
                );
            }
        }
    }

    #[test]
    fn test_multi_stage_dc_signal() {
        // A constant signal should have all energy in DC bin (multi-stage configs).
        for &(factors, desc) in MULTI_STAGE_FACTORS {
            let size: usize = factors.iter().map(|f| f.radix()).product();
            let fft = RadixFFT::<Forward>::new(factors.to_vec());

            let input = vec![1.0f32; size];
            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);

            // DC component should equal sum of inputs.
            assert!(
                approx_eq(output[0].re, size as f32, EPSILON),
                "Config {desc}: DC real failed"
            );
            assert!(
                approx_eq(output[0].im, 0.0, EPSILON),
                "Config {desc}: DC imag failed"
            );

            // All other bins should be near zero.
            for i in 1..output.len() {
                assert!(
                    output[i].re.abs() < EPSILON,
                    "Config {desc}: Bin {i} re: {}",
                    output[i].re
                );
                assert!(
                    output[i].im.abs() < EPSILON,
                    "Config {desc}: Bin {i} im: {}",
                    output[i].im
                );
            }
        }
    }

    #[test]
    fn test_multi_stage_impulse() {
        // An impulse should have flat spectrum (multi-stage configs).
        for &(factors, desc) in MULTI_STAGE_FACTORS {
            let size: usize = factors.iter().map(|f| f.radix()).product();
            let fft = RadixFFT::<Forward>::new(factors.to_vec());

            let mut input = vec![0.0f32; size];
            input[0] = 1.0;

            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);

            // All bins should have magnitude close to 1.
            for i in 0..output.len() {
                let mag = (output[i].re * output[i].re + output[i].im * output[i].im).sqrt();
                assert!(
                    approx_eq(mag, 1.0, EPSILON),
                    "Config {desc}: Bin {i} mag: {mag}"
                );
            }
        }
    }

    #[test]
    fn test_multi_stage_linearity() {
        // FFT(a*x + b*y) = a*FFT(x) + b*FFT(y) (multi-stage configs).
        for &(factors, desc) in MULTI_STAGE_FACTORS {
            let size: usize = factors.iter().map(|f| f.radix()).product();
            let fft = RadixFFT::<Forward>::new(factors.to_vec());

            let input1: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
            let input2: Vec<f32> = (0..size).map(|i| (i as f32).cos()).collect();

            let a = 2.0;
            let b = 3.0;
            let combined: Vec<f32> = input1
                .iter()
                .zip(input2.iter())
                .map(|(x, y)| a * x + b * y)
                .collect();

            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output1 = vec![Complex32::default(); size / 2 + 1];
            let mut output2 = vec![Complex32::default(); size / 2 + 1];
            let mut output_combined = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input1, &mut output1, &mut scratchpad);
            fft.process(&input2, &mut output2, &mut scratchpad);
            fft.process(&combined, &mut output_combined, &mut scratchpad);

            for i in 0..output1.len() {
                let expected = Complex32::new(
                    a * output1[i].re + b * output2[i].re,
                    a * output1[i].im + b * output2[i].im,
                );
                assert!(
                    approx_eq_complex(output_combined[i], expected, EPSILON * 10.0),
                    "Config {desc}: Bin {i} failed linearity: got ({}, {}), expected ({}, {})",
                    output_combined[i].re,
                    output_combined[i].im,
                    expected.re,
                    expected.im
                );
            }
        }
    }

    #[test]
    fn test_multi_stage_parseval() {
        // Parseval's theorem: sum of |x|^2 = sum of |X|^2 / N (multi-stage configs).
        for &(factors, desc) in MULTI_STAGE_FACTORS {
            let size: usize = factors.iter().map(|f| f.radix()).product();
            let fft = RadixFFT::<Forward>::new(factors.to_vec());

            let input: Vec<f32> = (0..size).map(|i| (i as f32 / 10.0).sin()).collect();

            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);

            let time_energy: f32 = input.iter().map(|x| x * x).sum();

            // For real FFT, we need to account for the symmetry.
            // DC bin is counted once.
            let mut freq_energy = output[0].re * output[0].re;

            // For even sizes, Nyquist bin (at size/2) is also real and counted once.
            // For odd sizes, there is no Nyquist bin.
            if size.is_multiple_of(2) {
                freq_energy += output[size / 2].re * output[size / 2].re;
                // Intermediate bins are counted twice (positive and negative frequencies).
                for i in 1..(size / 2) {
                    freq_energy +=
                        2.0 * (output[i].re * output[i].re + output[i].im * output[i].im);
                }
            } else {
                // For odd sizes, all bins except DC are counted twice.
                for i in 1..=size / 2 {
                    freq_energy +=
                        2.0 * (output[i].re * output[i].re + output[i].im * output[i].im);
                }
            }
            freq_energy /= size as f32;

            assert!(
                approx_eq(time_energy, freq_energy, EPSILON * size as f32),
                "Config {desc}: Parseval's theorem failed: time={time_energy}, freq={freq_energy}"
            );
        }
    }

    #[test]
    fn test_multi_stage_vs_naive_dft() {
        // Compare against naive DFT for multi-stage configs.
        for &(factors, desc) in MULTI_STAGE_FACTORS {
            let size: usize = factors.iter().map(|f| f.radix()).product();
            let fft = RadixFFT::<Forward>::new(factors.to_vec());

            let input: Vec<f32> = (0..size).map(|i| (i as f32 / 3.0).sin()).collect();

            let mut scratchpad = vec![Complex32::default(); fft.scratchpad_size()];
            let mut output = vec![Complex32::default(); size / 2 + 1];

            fft.process(&input, &mut output, &mut scratchpad);
            let naive_output = naive_dft(&input);

            for i in 0..output.len() {
                assert!(
                    approx_eq_complex(output[i], naive_output[i], EPSILON * size as f32),
                    "Config {desc}: Size {size} bin {i} failed: got ({}, {}), expected ({}, {})",
                    output[i].re,
                    output[i].im,
                    naive_output[i].re,
                    naive_output[i].im
                );
            }
        }
    }

    #[test]
    fn test_complex32_layout() {
        assert_eq!(
            size_of::<Complex32>(),
            2 * size_of::<f32>(),
            "Complex32 must be exactly 2 f32s in size"
        );

        assert_eq!(
            align_of::<Complex32>(),
            align_of::<f32>(),
            "Complex32 alignment must match f32"
        );

        // Verify that casting is safe.
        let reals = [1.0f32, 2.0, 3.0, 4.0];
        let complex = unsafe { slice::from_raw_parts(reals.as_ptr() as *const Complex32, 2) };

        assert_eq!(complex[0].re, 1.0);
        assert_eq!(complex[0].im, 2.0);
        assert_eq!(complex[1].re, 3.0);
        assert_eq!(complex[1].im, 4.0);
    }
}
