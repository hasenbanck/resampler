use alloc::vec::Vec;

use super::radix_fft::Radix;

/// Optimize factors the merges power-of-two factors aggressively.
pub(crate) fn optimize_factors(mut factors: Vec<Radix>) -> Vec<Radix> {
    factors.sort_by_key(|f| core::cmp::Reverse(f.radix()));

    loop {
        if !apply_transformations(&mut factors) {
            break;
        }
        factors.sort_by_key(|f| core::cmp::Reverse(f.radix()));
    }

    // I could image this, but it seemed that this order performed better
    // by some percentage points than other orders.
    factors.sort_by_key(|f| f.radix());

    factors
}

/// Try to apply any available transformation. Returns true if a transformation was applied.
fn apply_transformations(factors: &mut Vec<Radix>) -> bool {
    const TRANSFORMATIONS: &[(&[Radix], &[Radix])] = &[
        (&[Radix::Factor8, Radix::Factor2], &[Radix::Factor16]),
        (&[Radix::Factor4, Radix::Factor4], &[Radix::Factor16]),
        (
            &[
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
            ],
            &[Radix::Factor16],
        ),
        (&[Radix::Factor4, Radix::Factor2], &[Radix::Factor8]),
        (
            &[Radix::Factor2, Radix::Factor2, Radix::Factor2],
            &[Radix::Factor8],
        ),
        (&[Radix::Factor2, Radix::Factor2], &[Radix::Factor4]),
    ];

    for &(remove, add) in TRANSFORMATIONS {
        if try_transform(factors, remove, add) {
            return true;
        }
    }

    false
}

fn try_transform(factors: &mut Vec<Radix>, remove: &[Radix], add: &[Radix]) -> bool {
    let mut temp = factors.clone();

    for &factor_to_remove in remove {
        match temp.iter().position(|&f| f == factor_to_remove) {
            Some(pos) => temp.remove(pos),
            None => return false, // Can't find required factor
        };
    }

    // Successfully removed all required factors; add replacements.
    *factors = temp;
    for &factor in add {
        factors.push(factor);
    }

    true
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn test_optimize_factors_basic() {
        let input = vec![Radix::Factor2, Radix::Factor2];
        let output = optimize_factors(input);
        assert_eq!(output, vec![Radix::Factor4]);
    }

    #[test]
    fn test_optimize_factors_multiple_pairs() {
        let input = vec![
            Radix::Factor2,
            Radix::Factor2,
            Radix::Factor4,
            Radix::Factor2,
            Radix::Factor2,
        ];
        let output = optimize_factors(input);
        // Now optimizes to Factor16 + Factor4 instead of Factor8 + Factor8
        assert_eq!(output, vec![Radix::Factor4, Radix::Factor16]);
    }

    #[test]
    fn test_optimize_factors_with_leading_factor2() {
        let input = vec![
            Radix::Factor2,
            Radix::Factor4,
            Radix::Factor4,
            Radix::Factor4,
            Radix::Factor4,
            Radix::Factor2,
        ];
        let output = optimize_factors(input);
        // Now optimizes to Factor16 + Factor16 + Factor4 instead of Factor2 + Factor8 + Factor8 + Factor8
        assert_eq!(
            output,
            vec![Radix::Factor4, Radix::Factor16, Radix::Factor16,]
        );
    }

    #[test]
    fn test_optimize_factors_four_factor4s() {
        let input = vec![
            Radix::Factor4,
            Radix::Factor4,
            Radix::Factor4,
            Radix::Factor4,
            Radix::Factor5,
        ];
        let output = optimize_factors(input);
        // Now optimizes to Factor16 + Factor16 + Factor5 instead of Factor4 + Factor5 + Factor8 + Factor8
        assert_eq!(
            output,
            vec![Radix::Factor5, Radix::Factor16, Radix::Factor16,]
        );
    }

    #[test]
    fn test_optimize_factors_with_factor4_pairs() {
        let input = vec![
            Radix::Factor2,
            Radix::Factor4,
            Radix::Factor4,
            Radix::Factor4,
        ];
        let output = optimize_factors(input);
        // Now optimizes to Factor16 + Factor8 instead of Factor2 + Factor8 + Factor8
        assert_eq!(output, vec![Radix::Factor8, Radix::Factor16]);
    }

    #[test]
    fn test_optimize_factors_factor4_pairs_with_factor8s() {
        let input = vec![
            Radix::Factor4,
            Radix::Factor4,
            Radix::Factor8,
            Radix::Factor8,
        ];
        let output = optimize_factors(input);
        assert_eq!(
            output,
            vec![Radix::Factor8, Radix::Factor8, Radix::Factor16,]
        );
    }

    #[test]
    fn test_optimize_factors_to_factor16_from_factor8_and_factor2() {
        let input = vec![Radix::Factor8, Radix::Factor2];
        let output = optimize_factors(input);
        assert_eq!(output, vec![Radix::Factor16]);
    }

    #[test]
    fn test_optimize_factors_to_factor16_from_two_factor4s() {
        let input = vec![Radix::Factor4, Radix::Factor4];
        let output = optimize_factors(input);
        assert_eq!(output, vec![Radix::Factor16]);
    }

    #[test]
    fn test_optimize_factors_to_factor16_from_four_factor2s() {
        let input = vec![
            Radix::Factor2,
            Radix::Factor2,
            Radix::Factor2,
            Radix::Factor2,
        ];
        let output = optimize_factors(input);
        assert_eq!(output, vec![Radix::Factor16]);
    }
}
