use crate::Complex32;

/// Processes a single radix-2 butterfly stage across all columns using scalar operations.
#[inline(always)]
pub(crate) fn butterfly_2(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    for idx in 0..num_columns {
        let d = data[idx + num_columns];
        let t = stage_twiddles[idx].mul(&d);
        let u = data[idx];
        data[idx] = u.add(&t);
        data[idx + num_columns] = u.sub(&t);
    }
}
