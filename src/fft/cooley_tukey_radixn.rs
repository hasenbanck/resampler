use crate::{
    Complex32, Radix,
    fft::butterflies::{
        butterfly_2_dispatch, butterfly_3_dispatch, butterfly_4_dispatch, butterfly_5_dispatch,
        butterfly_7_dispatch,
    },
};

/// Applies digit reversal permutation using precomputed indices.
///
/// This is a generalization of bit reversal for mixed-radix FFTs.
/// The permutation is precomputed during FFT plan creation.
///
/// # Arguments
/// * `data` - Data to permute
/// * `permutation` - Permutation indices where `new[i] = old[permutation[i]]`
/// * `scratch` - Scratch buffer for temporary storage (size >= data.len())
fn apply_digit_reversal_permutation(
    data: &mut [Complex32],
    permutation: &[usize],
    scratch: &mut [Complex32],
) {
    let n = data.len();

    let temp = &mut scratch[..n];

    temp.iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[permutation[i]];
    });

    data.copy_from_slice(temp);
}

/// Performs an in-place mixed-radix FFT using the Cooley-Tukey DIT algorithm.
///
/// This is a generalization of the pure radix-2 Cooley-Tukey algorithm to handle
/// arbitrary mixed-radix factorizations. The algorithm proceeds in sequential stages,
/// enabling efficient SIMD vectorization.
///
/// # Arguments
/// * `data` - Input/output buffer (N/2 complex values for real FFT with N/2 optimization)
/// * `twiddles` - Precomputed twiddle factors in sequential stage order
/// * `factors` - Radix factors for each stage (N/2 factors after N/2 optimization)
/// * `digit_reversal_permutation` - Precomputed permutation indices
/// * `permutation_scratch` - Scratch buffer for permutation (size >= data.len())
pub(crate) fn cooley_tukey_radix_n(
    data: &mut [Complex32],
    twiddles: &[Complex32],
    factors: &[Radix],
    digit_reversal_permutation: &[usize],
    permutation_scratch: &mut [Complex32],
) {
    let n = data.len();
    debug_assert_eq!(n, digit_reversal_permutation.len());

    apply_digit_reversal_permutation(data, digit_reversal_permutation, permutation_scratch);

    let mut twiddle_offset = 0;
    let mut stage_size = 1;

    for &radix in factors {
        let r = radix.radix();
        stage_size *= r;
        let num_columns = stage_size / r;
        let num_twiddles_per_column = r - 1;

        for group_start in (0..n).step_by(stage_size) {
            let group = &mut data[group_start..group_start + stage_size];
            let stage_twiddles =
                &twiddles[twiddle_offset..twiddle_offset + num_columns * num_twiddles_per_column];

            match radix {
                Radix::Factor2 => butterfly_2_dispatch(group, stage_twiddles, num_columns),
                Radix::Factor3 => butterfly_3_dispatch(group, stage_twiddles, num_columns),
                Radix::Factor4 => butterfly_4_dispatch(group, stage_twiddles, num_columns),
                Radix::Factor5 => butterfly_5_dispatch(group, stage_twiddles, num_columns),
                Radix::Factor7 => butterfly_7_dispatch(group, stage_twiddles, num_columns),
            }
        }

        twiddle_offset += num_columns * num_twiddles_per_column;
    }
}

macro_rules! define_cooley_tukey_radixn {
    ($fn_name:ident, $dispatch_fn_2:ident, $dispatch_fn_3:ident, $dispatch_fn_4:ident, $dispatch_fn_5:ident, $dispatch_fn_7:ident, cfg = $cfg:meta) => {
        /// Performs an in-place mixed-radix FFT using the Cooley-Tukey DIT algorithm.
        #[cfg($cfg)]
        pub(crate) fn $fn_name(
            data: &mut [Complex32],
            twiddles: &[Complex32],
            factors: &[Radix],
            digit_reversal_permutation: &[usize],
            permutation_scratch: &mut [Complex32],
        ) {
            let n = data.len();
            debug_assert_eq!(n, digit_reversal_permutation.len());

            apply_digit_reversal_permutation(data, digit_reversal_permutation, permutation_scratch);

            let mut twiddle_offset = 0;
            let mut stage_size = 1;

            for &radix in factors {
                let r = radix.radix();
                stage_size *= r;
                let num_columns = stage_size / r;
                let num_twiddles_per_column = r - 1;

                for group_start in (0..n).step_by(stage_size) {
                    let group = &mut data[group_start..group_start + stage_size];
                    let stage_twiddles = &twiddles
                        [twiddle_offset..twiddle_offset + num_columns * num_twiddles_per_column];

                    match radix {
                        Radix::Factor2 => {
                            super::butterflies::$dispatch_fn_2(group, stage_twiddles, num_columns)
                        }
                        Radix::Factor3 => {
                            super::butterflies::$dispatch_fn_3(group, stage_twiddles, num_columns)
                        }
                        Radix::Factor4 => {
                            super::butterflies::$dispatch_fn_4(group, stage_twiddles, num_columns)
                        }
                        Radix::Factor5 => {
                            super::butterflies::$dispatch_fn_5(group, stage_twiddles, num_columns)
                        }
                        Radix::Factor7 => {
                            super::butterflies::$dispatch_fn_7(group, stage_twiddles, num_columns)
                        }
                    }
                }

                twiddle_offset += num_columns * num_twiddles_per_column;
            }
        }
    };
}

define_cooley_tukey_radixn!(
    cooley_tukey_radix_n_avx_fma,
    butterfly_2_dispatch_avx_fma,
    butterfly_3_dispatch_avx_fma,
    butterfly_4_dispatch_avx_fma,
    butterfly_5_dispatch_avx_fma,
    butterfly_7_dispatch_avx_fma,
    cfg = all(target_arch = "x86_64", not(feature = "no_std"))
);
define_cooley_tukey_radixn!(
    cooley_tukey_radix_n_avx,
    butterfly_2_dispatch_avx,
    butterfly_3_dispatch_avx,
    butterfly_4_dispatch_avx,
    butterfly_5_dispatch_avx,
    butterfly_7_dispatch_avx,
    cfg = all(target_arch = "x86_64", not(feature = "no_std"))
);
define_cooley_tukey_radixn!(
    cooley_tukey_radix_n_sse3,
    butterfly_2_dispatch_sse3,
    butterfly_3_dispatch_sse3,
    butterfly_4_dispatch_sse3,
    butterfly_5_dispatch_sse3,
    butterfly_7_dispatch_sse3,
    cfg = all(target_arch = "x86_64", not(feature = "no_std"))
);
define_cooley_tukey_radixn!(
    cooley_tukey_radix_n_sse,
    butterfly_2_dispatch_sse,
    butterfly_3_dispatch_sse,
    butterfly_4_dispatch_sse,
    butterfly_5_dispatch_sse,
    butterfly_7_dispatch_sse,
    cfg = all(target_arch = "x86_64", not(feature = "no_std"))
);
