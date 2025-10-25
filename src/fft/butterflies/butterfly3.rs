use crate::Complex32;

/// Processes a single radix-3 butterfly stage across all columns using scalar operations.
///
/// Implements the radix-3 DFT butterfly:
/// X[0] = x[0] + x[1]*W1 + x[2]*W2
/// X[1] = x[0] + x[1]*W1*W_3^1 + x[2]*W2*W_3^2
/// X[2] = x[0] + x[1]*W1*W_3^2 + x[2]*W2*W_3^1
///
/// Where W_3 = exp(-2πi/3) is the primitive 3rd root of unity, and
/// W1, W2 are the stage-specific twiddle factors.
pub(crate) fn butterfly_3(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    // Primitive 3rd roots of unity: W_3 = exp(-2πi/3)
    const W3_1_RE: f32 = -0.5;
    const W3_1_IM: f32 = -0.8660254; // -√3/2
    const W3_2_RE: f32 = -0.5;
    const W3_2_IM: f32 = 0.8660254; // √3/2

    for idx in 0..num_columns {
        let x0 = data[idx];
        let x1 = data[idx + num_columns];
        let x2 = data[idx + 2 * num_columns];

        // Apply stage twiddle factors.
        let w1 = stage_twiddles[idx * 2];
        let w2 = stage_twiddles[idx * 2 + 1];

        let t1 = x1.mul(&w1);
        let t2 = x2.mul(&w2);

        // X0 = x0 + t1 + t2
        data[idx] = x0.add(&t1).add(&t2);

        // X1 = x0 + t1*W_3^1 + t2*W_3^2
        // Multiply t1 by W_3^1 = -0.5 - i*√3/2
        let t1_w31 = Complex32::new(
            t1.re * W3_1_RE - t1.im * W3_1_IM,
            t1.re * W3_1_IM + t1.im * W3_1_RE,
        );
        // Multiply t2 by W_3^2 = -0.5 + i*√3/2
        let t2_w32 = Complex32::new(
            t2.re * W3_2_RE - t2.im * W3_2_IM,
            t2.re * W3_2_IM + t2.im * W3_2_RE,
        );
        data[idx + num_columns] = x0.add(&t1_w31).add(&t2_w32);

        // X2 = x0 + t1*W_3^2 + t2*W_3^1
        // Multiply t1 by W_3^2
        let t1_w32 = Complex32::new(
            t1.re * W3_2_RE - t1.im * W3_2_IM,
            t1.re * W3_2_IM + t1.im * W3_2_RE,
        );
        // Multiply t2 by W_3^1
        let t2_w31 = Complex32::new(
            t2.re * W3_1_RE - t2.im * W3_1_IM,
            t2.re * W3_1_IM + t2.im * W3_1_RE,
        );
        data[idx + 2 * num_columns] = x0.add(&t1_w32).add(&t2_w31);
    }
}
