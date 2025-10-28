mod butterfly2;
mod butterfly3;
mod butterfly4;
mod butterfly5;
mod butterfly7;

pub(crate) use butterfly2::butterfly_2;
pub(crate) use butterfly3::butterfly_3;
pub(crate) use butterfly4::butterfly_4;
pub(crate) use butterfly5::butterfly_5;
pub(crate) use butterfly7::butterfly_7;

pub(crate) use crate::fft::{
    cooley_tukey_radix2::cooley_tukey_radix_2, cooley_tukey_radixn::cooley_tukey_radix_n,
};

#[cfg(test)]
mod test_helpers {
    use alloc::{format, vec};
    use core::f32;

    use crate::Complex32;

    /// Helper function to check if two complex numbers are approximately equal
    pub fn approx_eq_complex(a: &Complex32, b: &Complex32, epsilon: f32) -> bool {
        (a.re - b.re).abs() < epsilon && (a.im - b.im).abs() < epsilon
    }

    /// Helper function to compare two complex arrays with approximate equality
    pub fn assert_complex_arrays_approx_eq(
        actual: &[Complex32],
        expected: &[Complex32],
        epsilon: f32,
        context: &str,
    ) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{}: Array lengths differ",
            context
        );

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_complex(a, e, epsilon),
                "{}: Mismatch at index {}: actual = ({}, {}), expected = ({}, {}), diff = ({}, {})",
                context,
                i,
                a.re,
                a.im,
                e.re,
                e.im,
                (a.re - e.re).abs(),
                (a.im - e.im).abs()
            );
        }
    }

    /// Generic test helper for butterfly functions.
    /// Tests a SIMD implementation against a scalar reference implementation.
    ///
    /// # Arguments
    /// * `scalar_fn` - Reference scalar implementation
    /// * `simd_fn` - SIMD implementation to test (wrapped in closure for unsafe calls)
    /// * `radix` - The radix of the butterfly (2 for radix-2, 3 for radix-3, etc.)
    /// * `twiddles_per_column` - Number of twiddle factors per column
    /// * `test_name` - Name for error messages
    pub fn test_butterfly_against_scalar<F, G>(
        scalar_fn: F,
        simd_fn: G,
        radix: usize,
        twiddles_per_column: usize,
        test_name: &str,
    ) where
        F: Fn(&mut [Complex32], &[Complex32], usize),
        G: Fn(&mut [Complex32], &[Complex32], usize),
    {
        let test_sizes = vec![1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33];

        for num_columns in test_sizes {
            let mut scalar_data = vec![Complex32::zero(); radix * num_columns];
            let mut simd_data = vec![Complex32::zero(); radix * num_columns];

            for i in 0..scalar_data.len() {
                #[cfg(not(feature = "no_std"))]
                let val = Complex32::new((i as f32 * 0.5).sin(), (i as f32 * 0.3).cos());
                #[cfg(feature = "no_std")]
                let val = Complex32::new(libm::sinf(i as f32 * 0.5), libm::cosf(i as f32 * 0.3));
                scalar_data[i] = val;
                simd_data[i] = val;
            }

            let mut twiddles = vec![Complex32::zero(); num_columns * twiddles_per_column];
            for i in 0..twiddles.len() {
                let angle = 2.0 * f32::consts::PI * (i as f32) / (num_columns as f32);
                #[cfg(not(feature = "no_std"))]
                let tw = Complex32::new(angle.cos(), angle.sin());
                #[cfg(feature = "no_std")]
                let tw = Complex32::new(libm::cosf(angle), libm::sinf(angle));
                twiddles[i] = tw;
            }

            scalar_fn(&mut scalar_data, &twiddles, num_columns);
            simd_fn(&mut simd_data, &twiddles, num_columns);

            let context = format!("{} with {} columns", test_name, num_columns);
            assert_complex_arrays_approx_eq(&simd_data, &scalar_data, 1e-6, &context);
        }
    }
}
