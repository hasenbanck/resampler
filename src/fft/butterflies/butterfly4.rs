use crate::Complex32;

/// Processes a single radix-4 butterfly stage across all columns using scalar operations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn butterfly_4(
    data: &mut [Complex32],
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
    w1: Complex32,
    w2: Complex32,
    w3: Complex32,
) {
    let x0 = data[i0];
    let x1 = data[i1];
    let x2 = data[i2];
    let x3 = data[i3];

    // First layer: 2 parallel radix-2 butterflies.
    let t0 = x0.add(&x2); // x0 + x2
    let t1 = x0.sub(&x2); // x0 - x2
    let t2 = x1.add(&x3); // x1 + x3
    let t3 = x1.sub(&x3); // x1 - x3

    // Multiply t3 by +j (rotation by +90°).
    // +j * (a + bi) = +j*a + j*j*b = +j*a - b = -b + j*a
    let t3_rot = Complex32::new(-t3.im, t3.re);

    // Second layer: combine results with twiddle factors.
    let y0 = t0.add(&t2);
    let y1 = t1.sub(&t3_rot).mul(&w1);
    let y2 = t0.sub(&t2).mul(&w2);
    let y3 = t1.add(&t3_rot).mul(&w3);

    // Store results back (in-place).
    data[i0] = y0;
    data[i1] = y1;
    data[i2] = y2;
    data[i3] = y3;
}
