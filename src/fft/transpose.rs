const RECURSIVE_LIMIT: usize = 128;
const BLOCK_SIZE: usize = 16;
const SLICES_PER_BLOCK: usize = 2;

/// Transpose a matrix out-of-place using a recursive cache-aware algorithm.
///
/// This implementation combines recursive matrix partitioning with blocked tiling to
/// maximize cache locality. The algorithm recursively subdivides the matrix into smaller
/// regions until they fit within the cache hierarchy, then performs the transpose using
/// fixed-size tiles (16×16 blocks with further sub-slicing).
///
/// By processing data in cache-friendly blocks, this algorithm achieves significantly
/// better performance than naive row-by-row or column-by-column transpose implementations,
/// especially for large matrices where cache misses dominate execution time.
pub fn transpose<T: Copy>(input: &mut [T], output: &mut [T], width: usize, height: usize) {
    // # Note
    // Not using the specific type and using instead the copy trait somehow lets the compiler
    // generate faster code (up to 20% faster!).
    assert_eq!(input.len(), output.len());
    assert_eq!(width.checked_mul(height), Some(input.len()));
    recursive_transpose(input, output, width, height, 0, height, 0, width);
}

#[allow(clippy::too_many_arguments)]
fn recursive_transpose<T: Copy>(
    input: &[T],
    output: &mut [T],
    total_columns: usize,
    total_rows: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) {
    // Safety: verify region bounds.
    assert!(row_start <= row_end && row_end <= total_rows);
    assert!(col_start <= col_end && col_end <= total_columns);

    let column_count = col_end - col_start;
    let row_count = row_end - row_start;

    if row_count <= 2
        || column_count <= 2
        || (row_count <= RECURSIVE_LIMIT && column_count <= RECURSIVE_LIMIT)
    {
        let block_count_x = column_count / BLOCK_SIZE;
        let block_count_y = row_count / BLOCK_SIZE;

        let remainder_x = column_count - block_count_x * BLOCK_SIZE;
        let remainder_y = row_count - block_count_y * BLOCK_SIZE;

        for block_y in 0..block_count_y {
            for block_x in 0..block_count_x {
                unsafe {
                    sliced_block_transpose(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + block_x * BLOCK_SIZE,
                        row_start + block_y * BLOCK_SIZE,
                        BLOCK_SIZE,
                        BLOCK_SIZE,
                    );
                }
            }

            if remainder_x > 0 {
                unsafe {
                    block_transpose(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + block_count_x * BLOCK_SIZE,
                        row_start + block_y * BLOCK_SIZE,
                        remainder_x,
                        BLOCK_SIZE,
                    );
                }
            }
        }

        if remainder_y > 0 {
            for block_y in 0..block_count_x {
                unsafe {
                    block_transpose(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + block_y * BLOCK_SIZE,
                        row_start + block_count_y * BLOCK_SIZE,
                        BLOCK_SIZE,
                        remainder_y,
                    );
                }
            }

            if remainder_x > 0 {
                unsafe {
                    block_transpose(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + block_count_x * BLOCK_SIZE,
                        row_start + block_count_y * BLOCK_SIZE,
                        remainder_x,
                        remainder_y,
                    );
                }
            }
        }
    } else if row_count >= column_count {
        recursive_transpose(
            input,
            output,
            total_columns,
            total_rows,
            row_start,
            row_start + (row_count / 2),
            col_start,
            col_end,
        );
        recursive_transpose(
            input,
            output,
            total_columns,
            total_rows,
            row_start + (row_count / 2),
            row_end,
            col_start,
            col_end,
        );
    } else {
        recursive_transpose(
            input,
            output,
            total_columns,
            total_rows,
            row_start,
            row_end,
            col_start,
            col_start + (column_count / 2),
        );
        recursive_transpose(
            input,
            output,
            total_columns,
            total_rows,
            row_start,
            row_end,
            col_start + (column_count / 2),
            col_end,
        );
    }
}

/// # Safety
///
/// - `start_x + block_width <= width` and `start_y + block_height <= height`
/// - `input.len() == output.len() == width * height`
#[allow(clippy::too_many_arguments)]
unsafe fn block_transpose<T: Copy>(
    input: &[T],
    output: &mut [T],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    block_width: usize,
    block_height: usize,
) {
    for block_x in 0..block_width {
        for block_y in 0..block_height {
            let x = start_x + block_x;
            let y = start_y + block_y;
            let input_index = x + y * width;
            let output_index = x * height + y;
            unsafe { *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index) };
        }
    }
}

/// # Safety
///
/// - `start_x + block_width <= width` and `start_y + block_height <= height`
/// - `input.len() == output.len() == width * height`
/// - `block_height % SLICES_PER_BLOCK == 0`
#[allow(clippy::too_many_arguments)]
unsafe fn sliced_block_transpose<T: Copy>(
    input: &[T],
    output: &mut [T],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    block_width: usize,
    block_height: usize,
) {
    // Safety: BLOCK_SIZE must be divisible by SLICES_PER_BLOCK.
    const _: () = assert!(BLOCK_SIZE % SLICES_PER_BLOCK == 0);

    let slice_height = block_height / SLICES_PER_BLOCK;
    for slice_index in 0..SLICES_PER_BLOCK {
        for inner_x in 0..block_width {
            for inner_y in 0..slice_height {
                let x = start_x + inner_x;
                let y = start_y + inner_y + slice_index * slice_height;
                let input_index = x + y * width;
                let output_index = x * height + y;
                unsafe {
                    *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index)
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Complex32;

    const EPSILON: f32 = 1e-6;

    fn approx_eq_complex(a: Complex32, b: Complex32) -> bool {
        (a.re - b.re).abs() < EPSILON && (a.im - b.im).abs() < EPSILON
    }

    fn create_test_matrix(width: usize, height: usize) -> Vec<Complex32> {
        (0..width * height)
            .map(|i| Complex32::new(i as f32, (i * 10) as f32))
            .collect()
    }

    /// Verify transpose by checking that buffer[i][j] == original[j][i]
    fn verify_transpose(
        original: &[Complex32],
        transposed: &[Complex32],
        width: usize,
        height: usize,
    ) {
        for row in 0..height {
            for col in 0..width {
                let orig_idx = row * width + col;
                let trans_idx = col * height + row;
                assert!(
                    approx_eq_complex(original[orig_idx], transposed[trans_idx]),
                    "Mismatch at ({row}, {col}): original = ({}, {}), transposed = ({}, {})",
                    original[orig_idx].re,
                    original[orig_idx].im,
                    transposed[trans_idx].re,
                    transposed[trans_idx].im
                );
            }
        }
    }

    #[test]
    fn test_transpose_2x2() {
        let mut input = vec![
            Complex32::new(1.0, 10.0),
            Complex32::new(2.0, 20.0),
            Complex32::new(3.0, 30.0),
            Complex32::new(4.0, 40.0),
        ];
        let original = input.clone();
        let mut output = vec![Complex32::default(); 4];

        transpose(&mut input, &mut output, 2, 2);

        let expected = [
            Complex32::new(1.0, 10.0),
            Complex32::new(3.0, 30.0),
            Complex32::new(2.0, 20.0),
            Complex32::new(4.0, 40.0),
        ];

        for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq_complex(*got, *exp),
                "Index {i}: got ({}, {}), expected ({}, {})",
                got.re,
                got.im,
                exp.re,
                exp.im
            );
        }

        verify_transpose(&original, &output, 2, 2);
    }

    #[test]
    fn test_transpose_3x3() {
        // Test a 3x3 matrix
        let mut input = create_test_matrix(3, 3);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 9];

        transpose(&mut input, &mut output, 3, 3);

        verify_transpose(&original, &output, 3, 3);
    }

    #[test]
    fn test_transpose_4x4() {
        // Test a larger square matrix
        let mut input = create_test_matrix(4, 4);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 16];

        transpose(&mut input, &mut output, 4, 4);

        verify_transpose(&original, &output, 4, 4);
    }

    #[test]
    fn test_transpose_2x3() {
        let mut input = create_test_matrix(2, 3);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 6];

        transpose(&mut input, &mut output, 2, 3);

        verify_transpose(&original, &output, 2, 3);
    }

    #[test]
    fn test_transpose_3x2() {
        let mut input = create_test_matrix(3, 2);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 6];

        transpose(&mut input, &mut output, 3, 2);

        verify_transpose(&original, &output, 3, 2);
    }

    #[test]
    fn test_transpose_4x6() {
        let mut input = create_test_matrix(4, 6);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 24];

        transpose(&mut input, &mut output, 4, 6);

        verify_transpose(&original, &output, 4, 6);
    }

    #[test]
    fn test_transpose_6x4() {
        let mut input = create_test_matrix(6, 4);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 24];

        transpose(&mut input, &mut output, 6, 4);

        verify_transpose(&original, &output, 6, 4);
    }

    #[test]
    fn test_transpose_double_transpose_is_identity() {
        let width = 5;
        let height = 7;
        let original = create_test_matrix(width, height);
        let mut input = original.clone();
        let mut output = vec![Complex32::default(); width * height];

        // First transpose: width×height → height×width.
        transpose(&mut input, &mut output, width, height);

        // Second transpose: height×width → width×height (back to original).
        // Swap input and output for the second transpose.
        transpose(&mut output, &mut input, height, width);

        for (i, (got, exp)) in input.iter().zip(original.iter()).enumerate() {
            assert!(
                approx_eq_complex(*got, *exp),
                "Index {}: got ({}, {}), expected ({}, {})",
                i,
                got.re,
                got.im,
                exp.re,
                exp.im
            );
        }
    }

    #[test]
    fn test_transpose_prime_dimensions() {
        let mut input = create_test_matrix(5, 7);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 35];
        transpose(&mut input, &mut output, 5, 7);

        verify_transpose(&original, &output, 5, 7);
    }

    #[test]
    fn test_transpose_coprime_dimensions() {
        let mut input = create_test_matrix(4, 9);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 36];
        transpose(&mut input, &mut output, 4, 9);

        verify_transpose(&original, &output, 4, 9);
    }

    #[test]
    fn test_transpose_common_factor_dimensions() {
        let mut input = create_test_matrix(6, 9);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 54];
        transpose(&mut input, &mut output, 6, 9);

        verify_transpose(&original, &output, 6, 9);
    }

    #[test]
    fn test_transpose_power_of_two_dimensions() {
        let mut input = create_test_matrix(8, 16);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 128];
        transpose(&mut input, &mut output, 8, 16);

        verify_transpose(&original, &output, 8, 16);
    }

    #[test]
    fn test_transpose_1xn() {
        let mut input = create_test_matrix(1, 5);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 5];
        transpose(&mut input, &mut output, 1, 5);
        verify_transpose(&original, &output, 1, 5);
    }

    #[test]
    fn test_transpose_nx1() {
        let mut input = create_test_matrix(5, 1);
        let original = input.clone();
        let mut output = vec![Complex32::default(); 5];
        transpose(&mut input, &mut output, 5, 1);
        verify_transpose(&original, &output, 5, 1);
    }

    #[test]
    fn test_transpose_1x1() {
        let mut input = vec![Complex32::new(42.0, 24.0)];
        let original = input.clone();
        let mut output = vec![Complex32::default(); 1];
        transpose(&mut input, &mut output, 1, 1);
        assert!(approx_eq_complex(output[0], original[0]));
    }

    #[test]
    fn test_transpose_preserves_all_values() {
        let width = 7;
        let height = 5;
        let mut input = create_test_matrix(width, height);
        let original = input.clone();
        let mut output = vec![Complex32::default(); width * height];

        transpose(&mut input, &mut output, width, height);

        // Create sorted versions to compare
        let mut original_sorted = original.clone();
        let mut output_sorted = output.clone();

        original_sorted.sort_by(|a, b| {
            a.re.partial_cmp(&b.re)
                .unwrap()
                .then(a.im.partial_cmp(&b.im).unwrap())
        });
        output_sorted.sort_by(|a, b| {
            a.re.partial_cmp(&b.re)
                .unwrap()
                .then(a.im.partial_cmp(&b.im).unwrap())
        });

        for (got, exp) in output_sorted.iter().zip(original_sorted.iter()) {
            assert!(approx_eq_complex(*got, *exp));
        }
    }

    #[test]
    fn test_transpose_with_zero_values() {
        // Test with all zeros
        let mut input = vec![Complex32::default(); 12];
        let original = input.clone();
        let mut output = vec![Complex32::default(); 12];

        transpose(&mut input, &mut output, 3, 4);

        for val in &output {
            assert!(approx_eq_complex(*val, Complex32::default()));
        }

        verify_transpose(&original, &output, 3, 4);
    }

    #[test]
    fn test_transpose_with_negative_values() {
        // Test with negative values
        let mut input: Vec<Complex32> = (0..20)
            .map(|i| Complex32::new(-(i as f32), -(i as f32) * 0.5))
            .collect();
        let original = input.clone();
        let mut output = vec![Complex32::default(); 20];
        transpose(&mut input, &mut output, 4, 5);

        verify_transpose(&original, &output, 4, 5);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_transpose_panics_on_incorrect_input_buffer_size() {
        let mut input = vec![Complex32::default(); 10];
        let mut output = vec![Complex32::default(); 12];
        transpose(&mut input, &mut output, 3, 4);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_transpose_panics_on_incorrect_output_buffer_size() {
        let mut input = vec![Complex32::default(); 12];
        let mut output = vec![Complex32::default(); 10];
        transpose(&mut input, &mut output, 3, 4);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_transpose_panics_on_dimension_mismatch() {
        let mut input = vec![Complex32::default(); 12];
        let mut output = vec![Complex32::default(); 12];
        transpose(&mut input, &mut output, 5, 2);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_transpose_panics_on_overflow() {
        let mut input = vec![Complex32::default(); 1];
        let mut output = vec![Complex32::default(); 1];
        transpose(&mut input, &mut output, usize::MAX, 2);
    }
}
