use super::butterflies::{
    butterfly_2, butterfly_3, butterfly_4, butterfly_5, butterfly_7, cooley_tukey_radix_2,
    cooley_tukey_radix_n,
};
use crate::Complex32;

/// Radix factors supported for mixed-radix FFT decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Radix {
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
}

impl Radix {
    /// Returns the radix size.
    pub const fn radix(&self) -> usize {
        match self {
            Radix::Factor2 => 2,
            Radix::Factor3 => 3,
            Radix::Factor4 => 4,
            Radix::Factor5 => 5,
            Radix::Factor7 => 7,
        }
    }
}

/// Marker type for forward FFT direction.
pub struct Forward;

/// Marker type for inverse FFT direction.
pub struct Inverse;

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
pub struct RadixFFT<D> {
    n: usize,
    n2: usize,
    factors: Vec<Radix>,
    twiddles: Vec<Complex32>,
    digit_reversal_permutation: Vec<usize>,
    real_complex_expansion_twiddles: Vec<Complex32>,
    complex_real_reduction_twiddles: Vec<Complex32>,
    is_pure_radix2: bool,
    _direction: std::marker::PhantomData<D>,
}

impl<D> RadixFFT<D> {
    /// Constructs a new [`RadixFFT`] FFT instance.
    ///
    /// # Arguments
    /// * `factors` - Vector of radix factors defining the FFT stages (e.g., `vec![Factor2; 4]` for size 16)
    ///
    /// # Panics
    /// Panics if the total FFT length is not even since we apply an N/2 optimization double the performance of the FFT.
    pub fn new(factors: Vec<Radix>) -> Self {
        assert!(!factors.is_empty(), "Factors vector must not be empty");

        let n = factors.iter().map(|f| f.radix()).product();
        assert_eq!(
            n % 2,
            0,
            "FFT length must be even for N/2 optimization, got {n}"
        );
        let n2 = n / 2;

        let factors = Self::compute_factors(&factors);

        // Check if we have pure radix-2 after expansion (for optimization)
        let is_pure_radix2 = factors.len() > 1 && factors.iter().all(|f| *f == Radix::Factor2);

        let twiddles = if factors.is_empty() || factors.len() == 1 {
            // N/2 = 1 or N/2 is a single radix: no twiddles needed.
            Vec::new()
        } else if is_pure_radix2 {
            Self::compute_cooley_tukey_radix_2_twiddles(n2)
        } else {
            Self::compute_cooley_tukey_radix_n_twiddles(n2, &factors)
        };

        let digit_reversal_permutation = if is_pure_radix2 || factors.is_empty() {
            // N/2 = 1, no permutation needed
            Vec::new()
        } else {
            // Compute permutation for N/2 data using N/2 factors
            Self::compute_digit_reversal_permutation(n2, &factors)
        };

        let real_complex_expansion_twiddles = Self::compute_real_complex_expansion_twiddles(n);
        let complex_real_reduction_twiddles = Self::compute_complex_real_reduction_twiddles(n);

        Self {
            n,
            n2,
            factors,
            twiddles,
            digit_reversal_permutation,
            real_complex_expansion_twiddles,
            complex_real_reduction_twiddles,
            is_pure_radix2,
            _direction: std::marker::PhantomData,
        }
    }

    /// Compute mixed-radix digit reversal permutation.
    fn digit_reverse_index(index: usize, radices: &[Radix]) -> usize {
        let mut digits = Vec::with_capacity(radices.len());
        let mut temp = index;

        // Step 1: Extract digits in forward order.
        radices.iter().for_each(|radix| {
            digits.push(temp % radix.radix());
            temp /= radix.radix();
        });

        // Step 2: Reconstruct index with reversed digits using reversed radices.
        let mut result = 0;
        let mut multiplier = 1;

        for i in (0..radices.len()).rev() {
            result += digits[i] * multiplier;
            // Use reversed radix for next multiplier.
            if i > 0 {
                multiplier *= radices[i].radix();
            }
        }

        result
    }

    /// Pre-compute the full digit reversal permutation
    fn compute_digit_reversal_permutation(n: usize, radices: &[Radix]) -> Vec<usize> {
        (0..n)
            .map(|i| Self::digit_reverse_index(i, radices))
            .collect()
    }

    /// Compute N/2 factors by removing one Factor2 or converting Factor4 to Factor2.
    fn compute_factors(factors: &[Radix]) -> Vec<Radix> {
        if factors.len() == 1 {
            let factor = factors[0];
            match factor {
                Radix::Factor2 => Vec::new(),
                Radix::Factor4 => vec![Radix::Factor2],
                _ => panic!("Unsupported single factor for N/2 optimization: {factor:?}"),
            }
        } else {
            let mut factors = factors.to_vec();
            if let Some(pos) = factors.iter().position(|&f| f == Radix::Factor2) {
                factors.remove(pos);
            } else if let Some(pos) = factors.iter().position(|&f| f == Radix::Factor4) {
                factors[pos] = Radix::Factor2;
            } else {
                panic!("Even-length FFT must have at least one Factor2 or Factor4");
            }
            factors
        }
    }

    /// Compute a twiddle factor with f64 precision, then convert to f32.
    /// This reduces accumulated floating-point error for large FFTs.
    #[inline(always)]
    fn compute_twiddle_f32(index: usize, fft_len: usize) -> Complex32 {
        let constant = -2.0 * std::f64::consts::PI / fft_len as f64;
        let angle = constant * index as f64;
        Complex32::new(angle.cos() as f32, angle.sin() as f32)
    }

    /// Compute twiddle factors for Cooley-Tukey radix-2 FFT.
    fn compute_cooley_tukey_radix_2_twiddles(n: usize) -> Vec<Complex32> {
        debug_assert!(n.is_power_of_two() && n > 0);

        let log2n = n.trailing_zeros() as usize;
        let mut twiddles = Vec::with_capacity(n - 1);

        for stage in 0..log2n {
            let num_twiddles = 1 << stage;
            let fft_size = 1 << (stage + 1);

            for k in 0..num_twiddles {
                twiddles.push(Self::compute_twiddle_f32(k, fft_size));
            }
        }

        twiddles
    }

    /// Compute twiddle factors for Cooley-Tukey mixed-radix FFT.
    ///
    /// This function computes twiddles stage-by-stage in the order they'll be accessed,
    /// enabling efficient sequential memory access during the FFT computation.
    ///
    /// For each stage s with radix r_s:
    /// - stage_size = product of radices from stage 0 to s (inclusive)
    /// - num_columns = stage_size / r_s (the butterfly width)
    /// - For each column c, we compute (r_s - 1) twiddles: W_N^(c * k) for k = 1..(r_s)
    fn compute_cooley_tukey_radix_n_twiddles(_n: usize, factors: &[Radix]) -> Vec<Complex32> {
        let mut twiddles = Vec::new();
        let mut stage_size = 1;

        for &radix in factors {
            let r = radix.radix();
            stage_size *= r;
            let num_columns = stage_size / r;

            // For each column, compute (r-1) twiddles
            for col in 0..num_columns {
                for k in 1..r {
                    twiddles.push(Self::compute_twiddle_f32(col * k, stage_size));
                }
            }
        }

        twiddles
    }

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
    ///
    /// The Cooley-Tukey algorithm (both pure radix-2 and mixed radix) is in-place,
    /// so we only need the main buffer of size N.
    pub fn scratchpad_size(&self) -> usize {
        self.n
    }

    /// For single-element FFTs, twiddles are all the identity value.
    fn apply_single_butterfly(data: &mut [Complex32], factor: Radix) {
        let radix = factor.radix();
        debug_assert_eq!(data.len(), radix);

        match factor {
            Radix::Factor2 => {
                let twiddles = [Complex32::new(1.0, 0.0)];
                butterfly_2(data, &twiddles, 1);
            }
            Radix::Factor4 => {
                let twiddles = [
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                ];
                butterfly_4(data, &twiddles, 1);
            }
            Radix::Factor3 => {
                let twiddles = [Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)];
                butterfly_3(data, &twiddles, 1);
            }
            Radix::Factor5 => {
                let twiddles = [
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(1.0, 0.0),
                ];
                butterfly_5(data, &twiddles, 1);
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
                butterfly_7(data, &twiddles, 1);
            }
        }
    }

    /// Perform N/2-point complex FFT for the N/2 optimization.
    ///
    /// # Arguments
    /// * `scratchpad` - Scratch buffer where first n2 elements contain the FFT data,
    ///   and remaining elements can be used for temporary storage
    fn process_forward_complex(&self, scratchpad: &mut [Complex32]) {
        debug_assert!(scratchpad.len() >= self.n);

        let (data, scratch_remainder) = scratchpad.split_at_mut(self.n2);

        // Use pre-computed N/2 factors
        if self.factors.is_empty() {
            // N/2 = 1, no FFT needed
        } else if self.factors.len() == 1 {
            // Single factor
            Self::apply_single_butterfly(data, self.factors[0]);
        } else if self.is_pure_radix2 {
            // Pure radix-2: use the optimized Cooley-Tukey algorithm.
            cooley_tukey_radix_2(data, &self.twiddles, &self.factors);
        } else {
            // Mixed-radix: use the standard Cooley-Tukey algorithm.
            cooley_tukey_radix_n(
                data,
                &self.twiddles,
                &self.factors,
                &self.digit_reversal_permutation,
                scratch_remainder,
            );
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

        let iter_count = output_left_middle
            .len()
            .min(output_right_middle.len())
            .min(self.real_complex_expansion_twiddles.len());

        let right_len = output_right_middle.len();
        for (i, out_left) in output_left_middle.iter_mut().enumerate().take(iter_count) {
            let twiddle = self.real_complex_expansion_twiddles[i];
            let out = *out_left;
            let out_rev_idx = right_len - 1 - i;
            let out_rev = output_right_middle[out_rev_idx];

            let sum = out.add(&out_rev);
            let diff = out.sub(&out_rev);

            // Apply factored twiddle multiplication.
            let twiddled_re_sum = Complex32::new(sum.re * twiddle.re, sum.im * twiddle.re);
            let twiddled_im_sum = Complex32::new(sum.re * twiddle.im, sum.im * twiddle.im);
            let twiddled_re_diff = Complex32::new(diff.re * twiddle.re, diff.im * twiddle.re);
            let twiddled_im_diff = Complex32::new(diff.re * twiddle.im, diff.im * twiddle.im);

            let half = 0.5;
            let half_sum_real = half * sum.re;
            let half_diff_imaginary = half * diff.im;

            // Combine components (exploiting conjugate symmetry).
            let real = twiddled_re_sum.im + twiddled_im_diff.re;
            let imaginary = twiddled_im_sum.im - twiddled_re_diff.re;

            *out_left = Complex32::new(half_sum_real + real, half_diff_imaginary + imaginary);
            output_right_middle[out_rev_idx] =
                Complex32::new(half_sum_real - real, imaginary - half_diff_imaginary);
        }

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
        debug_assert!(input.len() <= self.n);
        debug_assert!(output.len() > self.n2);
        debug_assert!(
            scratchpad.len() >= self.scratchpad_size(),
            "Scratchpad size must be at least {}",
            self.scratchpad_size()
        );

        // Reinterpret N real values as N/2 complex values by pairing consecutive samples.
        if input.len() >= self.n {
            let buf_in = unsafe {
                let ptr = input.as_ptr() as *const Complex32;
                std::slice::from_raw_parts(ptr, self.n2)
            };
            scratchpad[..self.n2].copy_from_slice(buf_in);
        } else {
            let available_pairs = input.len() / 2;
            if available_pairs > 0 {
                unsafe {
                    let ptr = input.as_ptr() as *const Complex32;
                    let buf_in = std::slice::from_raw_parts(ptr, available_pairs);
                    scratchpad[..available_pairs].copy_from_slice(buf_in);
                }
            }

            if input.len() % 2 == 1 {
                scratchpad[available_pairs] = Complex32::new(input[input.len() - 1], 0.0);
            }

            // Zero-pad the rest.
            let start = available_pairs + (input.len() % 2);
            scratchpad[start..self.n2].fill(Complex32::new(0.0, 0.0));
        }

        self.process_forward_complex(scratchpad);
        self.postprocess_fft(&scratchpad[..self.n2], output);
    }

    /// Internal inverse complex -> real FFT using N/2 optimization.
    fn process_inverse_internal(
        &self,
        input: &[Complex32],
        output: &mut [f32],
        scratchpad: &mut [Complex32],
    ) {
        debug_assert!(output.len() >= self.n);
        debug_assert!(input.len() > self.n2);
        debug_assert!(
            scratchpad.len() >= self.scratchpad_size(),
            "Scratchpad size must be at least {}",
            self.scratchpad_size()
        );

        // Copy input to scratch buffer (need N/2+1 elements including Nyquist bin).
        scratchpad[..input.len()].copy_from_slice(input);

        self.preprocess_ifft(&mut scratchpad[..self.n2 + 1]);
        self.process_inverse_complex(scratchpad);

        if output.len() >= self.n {
            let buf_out = unsafe {
                let ptr = output.as_mut_ptr() as *mut Complex32;
                std::slice::from_raw_parts_mut(ptr, self.n2)
            };
            buf_out.copy_from_slice(&scratchpad[..self.n2]);
        } else {
            for i in 0..self.n2 {
                if 2 * i < output.len() {
                    output[2 * i] = scratchpad[i].re;
                }
                if 2 * i + 1 < output.len() {
                    output[2 * i + 1] = scratchpad[i].im;
                }
            }
        }

        // Zero-pad remaining output if needed.
        output[self.n..].fill(0.0);
    }

    /// Pre-process real FFT input to prepare for N/2-point inverse complex FFT.
    fn preprocess_ifft(&self, buffer: &mut [Complex32]) {
        debug_assert!(buffer.len() > self.n2);

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

        let iter_count = input_left_middle
            .len()
            .min(input_right_middle.len())
            .min(self.complex_real_reduction_twiddles.len());

        for (i, &twiddle) in self.complex_real_reduction_twiddles[..iter_count]
            .iter()
            .enumerate()
        {
            let inp = input_left_middle[i];
            let inp_rev_idx = input_right_middle.len() - 1 - i;
            let inp_rev = input_right_middle[inp_rev_idx];

            let sum = inp.add(&inp_rev);
            let diff = inp.sub(&inp_rev);

            // Apply factored twiddle multiplication with conjugate twiddles.
            let twiddled_re_sum = Complex32::new(sum.re * twiddle.re, sum.im * twiddle.re);
            let twiddled_im_sum = Complex32::new(sum.re * twiddle.im, sum.im * twiddle.im);
            let twiddled_re_diff = Complex32::new(diff.re * twiddle.re, diff.im * twiddle.re);
            let twiddled_im_diff = Complex32::new(diff.re * twiddle.im, diff.im * twiddle.im);

            let real = twiddled_re_sum.im + twiddled_im_diff.re;
            let imaginary = twiddled_im_sum.im - twiddled_re_diff.re;

            input_left_middle[i] = Complex32::new(sum.re - real, diff.im - imaginary);
            input_right_middle[inp_rev_idx] = Complex32::new(sum.re + real, -imaginary - diff.im);
        }

        // Handle odd-length case: double and conjugate center element.
        if buffer.len() % 2 == 1 {
            let center_idx = buffer.len() / 2;
            let center_element = buffer[center_idx];
            let doubled = center_element.add(&center_element);
            buffer[center_idx] = doubled.conj();
        }
    }

    /// Perform N/2-point complex inverse FFT for the N/2 optimization.
    ///
    /// # Arguments
    /// * `scratchpad` - Scratch buffer where first n2 elements contain the FFT data,
    ///   and remaining elements can be used for temporary storage
    fn process_inverse_complex(&self, scratchpad: &mut [Complex32]) {
        debug_assert!(scratchpad.len() >= self.n);

        let (data, scratch_remainder) = scratchpad.split_at_mut(self.n2);

        // Conjugate input
        data.iter_mut().for_each(|x| {
            x.im = -x.im;
        });

        // Perform forward FFT (same as forward path) using pre-computed N/2 factors
        if self.factors.is_empty() {
            // N/2 = 1, no FFT needed
        } else if self.factors.len() == 1 {
            // Single factor
            Self::apply_single_butterfly(data, self.factors[0]);
        } else if self.is_pure_radix2 {
            // Pure radix-2: use optimized Cooley-Tukey algorithm
            cooley_tukey_radix_2(data, &self.twiddles, &self.factors);
        } else {
            // Mixed-radix: use standard Cooley-Tukey algorithm
            // Use scratch_remainder as temporary buffer for permutation
            cooley_tukey_radix_n(
                data,
                &self.twiddles,
                &self.factors,
                &self.digit_reversal_permutation,
                scratch_remainder,
            );
        }

        // Conjugate output
        data.iter_mut().for_each(|x| {
            x.im = -x.im;
        });
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
    /// * `input` - Real-valued input samples (length <= len)
    /// * `output` - Half-complex output (length >= len/2 + 1)
    /// * `scratchpad` - Workspace for intermediate calculations (length >= len)
    pub fn process(&self, input: &[f32], output: &mut [Complex32], scratchpad: &mut [Complex32]) {
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
    /// * `input` - Half-complex input (length >= len/2 + 1)
    /// * `output` - Real-valued output samples (length >= len)
    /// * `scratchpad` - Workspace for intermediate calculations (length >= len)
    pub fn process(&self, input: &[Complex32], output: &mut [f32], scratchpad: &mut [Complex32]) {
        self.process_inverse_internal(input, output, scratchpad);
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;

    const EPSILON: f32 = 1.0e-5;

    /// Single-stage radix factors to test (only even lengths for N/2 optimization).
    const SINGLE_STAGE_FACTORS: &[Radix] = &[Radix::Factor2, Radix::Factor4];

    /// Multi-stage radix factor combinations to test (only even lengths for N/2 optimization).
    ///
    /// Grouped by N/2 factors (result after N/2 optimization) to identify failure patterns.
    /// SUSPECTED FAILING GROUP at top: configs resulting in [2, 3] after N/2 optimization.
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
        debug_assert!(size.is_power_of_two(), "Size must be power of two");
        size.trailing_zeros() as usize
    }

    #[test]
    fn test_digit_reversal_permutation_validity() {
        // Test that digit reversal permutation produces valid indices (all < n).
        // This prevents index out of bounds errors that occur when the algorithm is incorrect.

        // Test various radix combinations
        let test_cases = vec![
            (vec![Radix::Factor2, Radix::Factor3], "2x3"),
            (vec![Radix::Factor3, Radix::Factor2], "3x2"),
            (vec![Radix::Factor2, Radix::Factor5], "2x5"),
            (
                vec![Radix::Factor2, Radix::Factor3, Radix::Factor5],
                "2x3x5",
            ),
            (
                vec![Radix::Factor5, Radix::Factor3, Radix::Factor2],
                "5x3x2",
            ),
            (
                vec![Radix::Factor2, Radix::Factor5, Radix::Factor7],
                "2x5x7",
            ),
            (
                vec![
                    Radix::Factor2,
                    Radix::Factor2,
                    Radix::Factor2,
                    Radix::Factor5,
                    Radix::Factor7,
                ],
                "2x2x2x5x7",
            ),
        ];

        for (radices, name) in test_cases {
            let n: usize = radices.iter().map(|r| r.radix()).product();
            let permutation = RadixFFT::<Forward>::compute_digit_reversal_permutation(n, &radices);

            // Verify permutation has correct length
            assert_eq!(
                permutation.len(),
                n,
                "{}: permutation length {} != n {}",
                name,
                permutation.len(),
                n
            );

            // Verify all indices are valid (< n)
            for (i, &perm_idx) in permutation.iter().enumerate() {
                assert!(
                    perm_idx < n,
                    "{}: permutation[{}] = {} >= n = {}",
                    name,
                    i,
                    perm_idx,
                    n
                );
            }

            // Verify permutation is actually a permutation (bijective)
            let mut seen = vec![false; n];
            for &perm_idx in &permutation {
                assert!(
                    !seen[perm_idx],
                    "{}: permutation has duplicate index {}",
                    name, perm_idx
                );
                seen[perm_idx] = true;
            }

            // Spot check: index 0 should map to 0 (all digits are 0)
            assert_eq!(permutation[0], 0, "{}: permutation[0] should be 0", name);

            // Spot check for 2x3 case: verify a few known mappings
            if radices == vec![Radix::Factor2, Radix::Factor3] {
                // n = 6, radices [2,3]
                // Index 1: digits = [1, 0], reversed reconstruction = 0*1 + 1*3 = 3
                assert_eq!(permutation[1], 3, "2x3: permutation[1] should be 3");
                // Index 2: digits = [0, 1], reversed reconstruction = 1*1 + 0*3 = 1
                assert_eq!(permutation[2], 1, "2x3: permutation[2] should be 1");
                // Index 4: digits = [0, 2], reversed reconstruction = 2*1 + 0*3 = 2
                assert_eq!(permutation[4], 2, "2x3: permutation[4] should be 2");
            }
        }
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
                "Factor {:?}: DC real failed",
                factor
            );
            assert!(
                approx_eq(output[0].im, 0.0, EPSILON),
                "Factor {:?}: DC imag failed",
                factor
            );

            // All other bins should be near zero.
            for i in 1..output.len() {
                assert!(
                    output[i].re.abs() < EPSILON,
                    "Factor {:?}: Bin {i} re: {}",
                    factor,
                    output[i].re
                );
                assert!(
                    output[i].im.abs() < EPSILON,
                    "Factor {:?}: Bin {i} im: {}",
                    factor,
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
                    "Factor {:?}: Bin {i} mag: {mag}",
                    factor
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
                "Factor {:?}: Parseval's theorem failed: time={time_energy}, freq={freq_energy}",
                factor
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
            if size % 2 != 0 {
                continue; // Skip odd sizes, they don't have a Nyquist bin
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
                "Factor {:?}: Nyquist should have high energy",
                factor
            );

            for i in 0..output.len() {
                if i != nyquist_idx {
                    let mag = (output[i].re * output[i].re + output[i].im * output[i].im).sqrt();
                    assert!(
                        mag < 1.0,
                        "Factor {:?}: Bin {i} should have low energy, got {mag}",
                        factor
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
                    "Factor {:?}: Random signal failed at index {i}: {} != {}",
                    factor,
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
                    "Factor {:?}: Bin {i} re should be 0",
                    factor
                );
                assert!(
                    approx_eq(output[i].im, 0.0, EPSILON),
                    "Factor {:?}: Bin {i} im should be 0",
                    factor
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
                "Factor {:?}: DC should be real",
                factor
            );

            // Nyquist bin (only exists for even sizes) should also be real.
            if size % 2 == 0 {
                assert!(
                    approx_eq(output[size / 2].im, 0.0, EPSILON),
                    "Factor {:?}: Nyquist should be real",
                    factor
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
                    "Factor {:?}: Sample {} = {}, expected 1.0",
                    factor,
                    i,
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
                    "Factor {:?}: Size {} bin {} failed: got ({}, {}), expected ({}, {})",
                    factor,
                    size,
                    i,
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
                    "Config {desc}: Size {} bin {} failed: got ({}, {}), expected ({}, {})",
                    size,
                    i,
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
        let complex = unsafe { std::slice::from_raw_parts(reals.as_ptr() as *const Complex32, 2) };

        assert_eq!(complex[0].re, 1.0);
        assert_eq!(complex[0].im, 2.0);
        assert_eq!(complex[1].re, 3.0);
        assert_eq!(complex[1].im, 4.0);
    }
}
