use crate::Complex32;

/// Processes a single radix-4 butterfly stage across all columns using scalar operations.
///
/// Implements the standard radix-4 Decimation-In-Time (DIT) butterfly:
/// 1. Apply twiddles to inputs first: t0 = x0, t1 = x1*W1, t2 = x2*W2, t3 = x3*W3
/// 2. Compute butterfly: y[k] = sum_{j=0}^{3} t[j] * W_4^(k*j)
pub(crate) fn butterfly_4(
    data: &mut [Complex32],
    stage_twiddles: &[Complex32],
    num_columns: usize,
) {
    for idx in 0..num_columns {
        let i0 = idx;
        let i1 = idx + num_columns;
        let i2 = idx + 2 * num_columns;
        let i3 = idx + 3 * num_columns;

        let x0 = data[i0];
        let x1 = data[i1];
        let x2 = data[i2];
        let x3 = data[i3];

        // Apply stage twiddle factors.
        let w1 = stage_twiddles[idx * 3];
        let w2 = stage_twiddles[idx * 3 + 1];
        let w3 = stage_twiddles[idx * 3 + 2];

        let t0 = x0;
        let t1 = x1.mul(&w1);
        let t2 = x2.mul(&w2);
        let t3 = x3.mul(&w3);

        // Compute radix-4 butterfly using factored form:
        // y[0] = t[0] + t[1] + t[2] + t[3]
        // y[1] = t[0] - j*t[1] - t[2] + j*t[3]
        // y[2] = t[0] - t[1] + t[2] - t[3]
        // y[3] = t[0] + j*t[1] - t[2] - j*t[3]
        let u0 = t0.add(&t2); // t0 + t2
        let u1 = t0.sub(&t2); // t0 - t2
        let u2 = t1.add(&t3); // t1 + t3
        let u3 = t1.sub(&t3); // t1 - t3

        // Multiply u3 by -j for y1: -j * (a + bi) = b - ai
        let u3_neg_j = Complex32::new(u3.im, -u3.re);
        // Multiply u3 by +j for y3: +j * (a + bi) = -b + ai
        let u3_pos_j = Complex32::new(-u3.im, u3.re);

        // Combine to produce outputs.
        let y0 = u0.add(&u2); // u0 + u2
        let y1 = u1.add(&u3_neg_j); // u1 - j*u3
        let y2 = u0.sub(&u2); // u0 - u2
        let y3 = u1.add(&u3_pos_j); // u1 + j*u3

        // Store results back.
        data[i0] = y0;
        data[i1] = y1;
        data[i2] = y2;
        data[i3] = y3;
    }
}
