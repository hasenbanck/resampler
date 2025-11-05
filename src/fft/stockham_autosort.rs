use crate::{Complex32, Radix};

macro_rules! define_stockham_autosort {
    (
        $fn_name:ident,
        $butterfly_2_p1:path,
        $butterfly_2_gen:path,
        $butterfly_3_p1:path,
        $butterfly_3_gen:path,
        $butterfly_4_p1:path,
        $butterfly_4_gen:path,
        $butterfly_5_p1:path,
        $butterfly_5_gen:path,
        $butterfly_7_p1:path,
        $butterfly_7_gen:path,
        cfg = $cfg:meta
    ) => {
        /// Performs an out-of-place mixed-radix FFT using the Stockham Autosort algorithm.
        ///
        /// The Stockham algorithm is an alternative to Cooley-Tukey that eliminates the need
        /// for explicit digit reversal permutation. It achieves this by using a ping-pong
        /// buffer approach, alternating reads and writes between the data and scratchpad
        /// buffers at each stage.
        ///
        /// # Scratchpad Requirements:
        ///
        /// The scratchpad buffer must be the same size as the data buffer.
        ///
        /// # Arguments
        ///
        /// * `data` - Input/output buffer (N/2 complex values for real FFT with N/2 optimization)
        /// * `twiddles` - Precomputed twiddle factors in sequential stage order
        /// * `factors` - Radix factors for each stage (N/2 factors after N/2 optimization)
        /// * `scratchpad` - Scratch buffer for ping-pong operations (size must equal data.len())
        ///
        /// # Returns
        ///
        /// `OutputLocation` indicating which buffer contains the final result
        #[cfg($cfg)]
        #[must_use]
        pub(crate) fn $fn_name(
            data: &mut [Complex32],
            twiddles: &[Complex32],
            factors: &[Radix],
            scratchpad: &mut [Complex32],
        ) -> OutputLocation {
            let n = data.len();
            debug_assert_eq!(n, scratchpad.len());

            if factors.is_empty() {
                return OutputLocation::Data;
            }

            let mut twiddle_offset = 0;
            let mut stride = 1;

            let mut input = data;
            let mut output = scratchpad;

            for factor in factors {
                let radix = factor.radix();
                let num_twiddles_per_column = radix - 1;

                let iterations = n / radix;
                let stage_twiddle_count = iterations * num_twiddles_per_column;
                let stage_twiddles =
                    &twiddles[twiddle_offset..twiddle_offset + stage_twiddle_count];

                // Use SIMD-specific butterfly functions with stride-based dispatch
                match (factor, stride) {
                    (Radix::Factor2, 1) => $butterfly_2_p1(input, output, stage_twiddles),
                    (Radix::Factor2, _) => $butterfly_2_gen(input, output, stage_twiddles, stride),
                    (Radix::Factor3, 1) => $butterfly_3_p1(input, output, stage_twiddles),
                    (Radix::Factor3, _) => $butterfly_3_gen(input, output, stage_twiddles, stride),
                    (Radix::Factor4, 1) => $butterfly_4_p1(input, output, stage_twiddles),
                    (Radix::Factor4, _) => $butterfly_4_gen(input, output, stage_twiddles, stride),
                    (Radix::Factor5, 1) => $butterfly_5_p1(input, output, stage_twiddles),
                    (Radix::Factor5, _) => $butterfly_5_gen(input, output, stage_twiddles, stride),
                    (Radix::Factor7, 1) => $butterfly_7_p1(input, output, stage_twiddles),
                    (Radix::Factor7, _) => $butterfly_7_gen(input, output, stage_twiddles, stride),
                }

                // Swap input and output buffers for next stage.
                core::mem::swap(&mut input, &mut output);
                twiddle_offset += stage_twiddle_count;
                stride *= radix;
            }

            // Return the location of the output data.
            if factors.len().is_multiple_of(2) {
                OutputLocation::Data
            } else {
                OutputLocation::Scratchpad
            }
        }
    };
}

define_stockham_autosort!(
    stockham_autosort_avx_fma,
    crate::fft::butterflies::butterfly_radix2_stride1_avx_fma_dispatch,
    crate::fft::butterflies::butterfly_radix2_generic_avx_fma_dispatch,
    crate::fft::butterflies::butterfly_radix3_stride1_avx_fma_dispatch,
    crate::fft::butterflies::butterfly_radix3_generic_avx_fma_dispatch,
    crate::fft::butterflies::butterfly_radix4_stride1_avx_fma_dispatch,
    crate::fft::butterflies::butterfly_radix4_generic_avx_fma_dispatch,
    crate::fft::butterflies::butterfly_radix5_stride1_avx_fma_dispatch,
    crate::fft::butterflies::butterfly_radix5_generic_avx_fma_dispatch,
    crate::fft::butterflies::butterfly_radix7_stride1_avx_fma_dispatch,
    crate::fft::butterflies::butterfly_radix7_generic_avx_fma_dispatch,
    cfg = all(target_arch = "x86_64", not(feature = "no_std"))
);

define_stockham_autosort!(
    stockham_autosort_sse2,
    crate::fft::butterflies::butterfly_radix2_stride1_sse2_dispatch,
    crate::fft::butterflies::butterfly_radix2_generic_sse2_dispatch,
    crate::fft::butterflies::butterfly_radix3_stride1_sse2_dispatch,
    crate::fft::butterflies::butterfly_radix3_generic_sse2_dispatch,
    crate::fft::butterflies::butterfly_radix4_stride1_sse2_dispatch,
    crate::fft::butterflies::butterfly_radix4_generic_sse2_dispatch,
    crate::fft::butterflies::butterfly_radix5_stride1_sse2_dispatch,
    crate::fft::butterflies::butterfly_radix5_generic_sse2_dispatch,
    crate::fft::butterflies::butterfly_radix7_stride1_sse2_dispatch,
    crate::fft::butterflies::butterfly_radix7_generic_sse2_dispatch,
    cfg = all(target_arch = "x86_64", not(feature = "no_std"))
);

define_stockham_autosort!(
    stockham_autosort_sse4_2,
    crate::fft::butterflies::butterfly_radix2_stride1_sse4_2_dispatch,
    crate::fft::butterflies::butterfly_radix2_generic_sse4_2_dispatch,
    crate::fft::butterflies::butterfly_radix3_stride1_sse4_2_dispatch,
    crate::fft::butterflies::butterfly_radix3_generic_sse4_2_dispatch,
    crate::fft::butterflies::butterfly_radix4_stride1_sse4_2_dispatch,
    crate::fft::butterflies::butterfly_radix4_generic_sse4_2_dispatch,
    crate::fft::butterflies::butterfly_radix5_stride1_sse4_2_dispatch,
    crate::fft::butterflies::butterfly_radix5_generic_sse4_2_dispatch,
    crate::fft::butterflies::butterfly_radix7_stride1_sse4_2_dispatch,
    crate::fft::butterflies::butterfly_radix7_generic_sse4_2_dispatch,
    cfg = all(target_arch = "x86_64", not(feature = "no_std"))
);

/// Indicates where the output data resides after the FFT computation.
///
/// The Stockham autosort algorithm uses ping-pong buffering, alternating between
/// the data and scratchpad buffers at each stage. The final output location depends
/// on whether there's an even or odd number of stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputLocation {
    /// Output is in the data buffer (first parameter of stockham_autosort).
    Data,
    /// Output is in the scratchpad buffer (fourth parameter of stockham_autosort).
    Scratchpad,
}

/// Performs an out-of-place mixed-radix FFT using the Stockham Autosort algorithm.
///
/// The Stockham algorithm is an alternative to Cooley-Tukey that eliminates the need
/// for explicit digit reversal permutation. It achieves this by using a ping-pong
/// buffer approach, alternating reads and writes between the data and scratchpad
/// buffers at each stage.
///
/// # Scratchpad Requirements:
///
/// The scratchpad buffer must be the same size as the data buffer.
///
/// # Arguments
///
/// * `data` - Input/output buffer (N/2 complex values for real FFT with N/2 optimization)
/// * `twiddles` - Precomputed twiddle factors in sequential stage order
/// * `factors` - Radix factors for each stage (N/2 factors after N/2 optimization)
/// * `scratchpad` - Scratch buffer for ping-pong operations (size must equal data.len())
///
/// # Returns
///
/// `OutputLocation` indicating which buffer contains the final result
#[cfg(any(not(target_arch = "x86_64"), feature = "no_std"))]
#[must_use]
pub(crate) fn stockham_autosort(
    data: &mut [Complex32],
    twiddles: &[Complex32],
    factors: &[Radix],
    scratchpad: &mut [Complex32],
) -> OutputLocation {
    use crate::fft::butterflies::{
        butterfly_radix2_dispatch, butterfly_radix3_dispatch, butterfly_radix4_dispatch,
        butterfly_radix5_dispatch, butterfly_radix7_dispatch,
    };

    let n = data.len();
    debug_assert_eq!(n, scratchpad.len());

    if factors.is_empty() {
        return OutputLocation::Data;
    }

    let mut twiddle_offset = 0;
    let mut stride = 1;

    let mut input = data;
    let mut output = scratchpad;

    for factor in factors {
        let radix = factor.radix();
        let num_twiddles_per_column = radix - 1;

        // With replicated twiddles: we have (n/r) iterations × (r-1) twiddles per iteration.
        let iterations = n / radix;
        let stage_twiddle_count = iterations * num_twiddles_per_column;
        let stage_twiddles = &twiddles[twiddle_offset..twiddle_offset + stage_twiddle_count];

        // Process entire array at once (Stockham's key property).
        match factor {
            Radix::Factor2 => butterfly_radix2_dispatch(input, output, stage_twiddles, stride),
            Radix::Factor3 => butterfly_radix3_dispatch(input, output, stage_twiddles, stride),
            Radix::Factor4 => butterfly_radix4_dispatch(input, output, stage_twiddles, stride),
            Radix::Factor5 => butterfly_radix5_dispatch(input, output, stage_twiddles, stride),
            Radix::Factor7 => butterfly_radix7_dispatch(input, output, stage_twiddles, stride),
        }

        // Swap input and output buffers for next stage.
        core::mem::swap(&mut input, &mut output);
        twiddle_offset += stage_twiddle_count;
        stride *= radix;
    }

    // Return the location of the output data.
    if factors.len().is_multiple_of(2) {
        OutputLocation::Data
    } else {
        OutputLocation::Scratchpad
    }
}
