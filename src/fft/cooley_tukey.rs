use crate::{Complex32, Radix, fft::butterflies::butterfly_2};

/// Reverses the bits of an integer value.
#[inline]
fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Performs bit-reversal permutation on the input data.
fn bit_reverse_radix2(data: &mut [Complex32], log2n: usize) {
    let n = data.len();

    for i in 0..n {
        let j = reverse_bits(i, log2n);
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Performs an in-place pure radix-2 FFT using the Cooley-Tukey DIT algorithm.
pub(crate) fn cooley_tukey_radix2(
    data: &mut [Complex32],
    twiddles: &[Complex32],
    factors: &[Radix],
) {
    let n = data.len();
    assert!(n.is_power_of_two() && n > 0);

    let log2n = n.trailing_zeros() as usize;

    bit_reverse_radix2(data, log2n);

    let mut twiddle_offset = 0;

    for (stage, _) in factors.iter().enumerate() {
        let n = data.len();
        let stage_size = 1 << (stage + 1);
        let half_stage = stage_size >> 1;

        let stage_twiddles = &twiddles[twiddle_offset..twiddle_offset + half_stage];

        for group_start in (0..n).step_by(stage_size) {
            let group = &mut data[group_start..group_start + stage_size];
            butterfly_2(group, stage_twiddles, half_stage);
        }

        twiddle_offset += half_stage;
    }
}
