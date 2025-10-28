use alloc::{vec, vec::Vec};
use core::f32::consts::PI;

/// Window type for Kaiser window generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowType {
    /// Periodic window (DFT-even) - used for FFT-based processing.
    /// The window is computed over N points but is periodic with period N,
    /// suitable for spectral analysis and overlap-add FFT methods.
    Periodic,
    /// Symmetric window (DFT-odd) - used for FIR filter design.
    /// The window is truly symmetric around its center point,
    /// suitable for time-domain FIR filter coefficients.
    Symmetric,
}

pub(crate) fn make_sincs_for_kaiser(
    sample_count: usize,
    factor: usize,
    f_cutoff: f32,
    beta: f64,
    window_type: WindowType,
) -> Vec<Vec<f32>> {
    let totpoints = sample_count * factor;
    let mut y = Vec::with_capacity(totpoints);
    let window = make_kaiser_window(totpoints, beta, window_type);
    let mut sum = 0.0;

    let sinc = |value: f32| -> f32 {
        match value == 0.0 {
            true => 1.0,
            #[cfg(not(feature = "no_std"))]
            false => (value * PI).sin() / (value * PI),
            #[cfg(feature = "no_std")]
            false => libm::sinf(value * PI) / (value * PI),
        }
    };

    for (x, w) in window.iter().enumerate().take(totpoints) {
        let val = *w * sinc((x as i32 - (totpoints / 2) as i32) as f32 * f_cutoff / factor as f32);
        sum += val;
        y.push(val);
    }
    sum /= factor as f32;

    let mut sincs = vec![vec![0.0; sample_count]; factor];

    (0..sample_count).for_each(|p| {
        (0..factor).for_each(|n| {
            sincs[factor - n - 1][p] = y[factor * p + n] / sum;
        });
    });

    sincs
}

/// Creates a Kaiser window for windowing sinc functions.
///
/// The Kaiser window is a near-optimal window function that provides a good trade-off
/// between main lobe width and side lobe attenuation. It is computed using the modified
/// Bessel function of the first kind, order zero (I₀).
///
/// # Window Types
/// - `WindowType::Periodic`: For FFT-based processing (overlap-add, spectral analysis)
/// - `WindowType::Symmetric`: For FIR filter design (time-domain filter coefficients)
fn make_kaiser_window(sample_count: usize, beta: f64, window_type: WindowType) -> Vec<f32> {
    let mut window = Vec::with_capacity(sample_count);

    let bessel_beta = bessel_i0(beta);

    for index in 0..sample_count {
        let x = index as f64;

        let normalized_x = match window_type {
            WindowType::Periodic => {
                // Periodic: x ∈ [0, N) mapped to [-1, 1) using N/2
                x / (sample_count as f64 / 2.0) - 1.0
            }
            WindowType::Symmetric => {
                // Symmetric: x ∈ [0, N) mapped to [-1, 1] using (N-1)/2
                2.0 * x / (sample_count - 1) as f64 - 1.0
            }
        };

        #[cfg(not(feature = "no_std"))]
        let value = bessel_i0(beta * f64::sqrt(1.0 - normalized_x.powi(2))) / bessel_beta;
        #[cfg(feature = "no_std")]
        let value = bessel_i0(beta * libm::sqrt(1.0 - libm::pow(normalized_x, 2.0))) / bessel_beta;

        window.push(value as f32);
    }

    window
}

fn bessel_i0(x: f64) -> f64 {
    let base = x * x / 4.0;

    let mut term = 1.0;
    let mut result = 1.0;

    for idx in 1..1500 {
        term = term * base / (idx * idx) as f64;
        let previous = result;
        result += term;
        if result == previous {
            break;
        }
    }

    result
}

pub(crate) fn calculate_cutoff_kaiser(sample_count: usize, beta: f64) -> f64 {
    let n = sample_count as f64;

    // Kaiser window transition bandwidth (from theory).
    // beta → stopband attenuation → transition width
    let a_db = beta / 0.1102 + 8.7; // Stopband attenuation (dB)
    let delta_f_nyquist = (a_db - 7.95) / (14.36 * n); // Transition width

    // Add small safety margin: widen transition band by ~0.5%
    // This provides headroom for numerical imperfections.
    const SAFETY_MARGIN: f64 = 1.005;

    // Cutoff: 1.0 (full sample rate) minus transition width.
    // This places the transition band edge just below Nyquist.
    let cutoff = 1.0 - (delta_f_nyquist * SAFETY_MARGIN);

    cutoff.clamp(0.7, 1.0)
}

#[cfg(test)]
mod tests {

    use super::*;

    fn assert_approx_f32(actual: f64, expected: f64) {
        assert!(
            (actual / expected - 1.0).abs() < 0.00001,
            "Expected {expected}, got {actual}"
        );
    }

    fn assert_approx_f64(actual: f64, expected: f64) {
        assert!(
            (actual / expected - 1.0).abs() < 0.000001,
            "Expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_bessel_i0_known_values() {
        // Test against scipy.special.i0 reference values
        assert_approx_f64(bessel_i0(0.0), 1.000000000000000);
        assert_approx_f64(bessel_i0(1.0), 1.266065877752008);
        assert_approx_f64(bessel_i0(2.0), 2.279585302336067);
        assert_approx_f64(bessel_i0(5.0), 27.239871823604442);
        assert_approx_f64(bessel_i0(10.0), 2815.716628466253951);
    }

    #[test]
    fn test_make_kaiser_window_small_beta_periodic() {
        // Test against scipy.signal.windows.kaiser(5, 0.5, sym=False)
        let window = make_kaiser_window(5, 0.5, WindowType::Periodic);
        let expected = vec![
            0.940306193319157,
            0.978296239370539,
            0.997576503537205,
            0.997576503537205,
            0.978296239370539,
        ];

        assert_eq!(window.len(), expected.len());
        for (&actual, &exp) in window.iter().zip(&expected) {
            assert_approx_f32(actual as f64, exp);
        }
    }

    #[test]
    fn test_make_kaiser_window_beta_5_periodic() {
        // Test against scipy.signal.windows.kaiser(15, 5.0, sym=False)
        let window = make_kaiser_window(15, 5.0, WindowType::Periodic);
        let expected = vec![
            0.036710892271287,
            0.120260370289032,
            0.248940523358684,
            0.414903639243367,
            0.599303856150336,
            0.775322104445407,
            0.913812483869200,
            0.990113103661532,
            0.990113103661532,
            0.913812483869200,
            0.775322104445407,
            0.599303856150336,
            0.414903639243367,
            0.248940523358684,
            0.120260370289032,
        ];

        assert_eq!(window.len(), expected.len());
        for (&actual, &exp) in window.iter().zip(&expected) {
            assert_approx_f32(actual as f64, exp);
        }
    }

    #[test]
    fn test_make_kaiser_window_beta_10_periodic() {
        // Test against scipy.signal.windows.kaiser(9, 10.0, sym=False)
        let window = make_kaiser_window(9, 10.0, WindowType::Periodic);
        let expected = vec![
            0.000355149374724,
            0.030999213508099,
            0.203914483842615,
            0.581810162428082,
            0.942963979134466,
            0.942963979134466,
            0.581810162428082,
            0.203914483842615,
            0.030999213508099,
        ];

        assert_eq!(window.len(), expected.len());
        for (&actual, &exp) in window.iter().zip(&expected) {
            assert_approx_f32(actual as f64, exp);
        }
    }

    #[test]
    fn test_calculate_cutoff_kaiser_various_sizes() {
        assert_approx_f64(calculate_cutoff_kaiser(64, 10.0), 0.8999482371370552);
        assert_approx_f64(calculate_cutoff_kaiser(128, 10.0), 0.9499741185685276);
        assert_approx_f64(calculate_cutoff_kaiser(256, 10.0), 0.9749870592842638);
        assert_approx_f64(calculate_cutoff_kaiser(512, 10.0), 0.9874935296421319);
        assert_approx_f64(calculate_cutoff_kaiser(1024, 10.0), 0.9937467648210659);
    }

    #[test]
    fn test_calculate_cutoff_kaiser_valid_range() {
        let test_sizes = vec![32, 64, 128, 256, 512, 1024, 2048];
        for size in test_sizes {
            let cutoff = calculate_cutoff_kaiser(size, 10.0);
            assert!(cutoff > 0.0, "Cutoff should be > 0, got {cutoff}");
            assert!(cutoff < 1.0, "Cutoff should be < 1, got {cutoff}");
        }
    }

    #[test]
    fn test_make_sincs_for_kaiser_dimensions() {
        let sample_count = 4;
        let factor = 2;
        let f_cutoff = 0.9;
        let beta = 10.0;

        let result =
            make_sincs_for_kaiser(sample_count, factor, f_cutoff, beta, WindowType::Periodic);

        assert_eq!(
            result.len(),
            factor,
            "Should have {factor} polyphase filters"
        );
        for (i, row) in result.iter().enumerate() {
            assert_eq!(
                row.len(),
                sample_count,
                "Polyphase filter {i} should have {sample_count} samples"
            );
        }
    }

    #[test]
    fn test_make_sincs_for_kaiser_reference_values_periodic() {
        // Test against numpy/scipy reference implementation (periodic window).
        let sample_count = 4;
        let factor = 2;
        let f_cutoff = 0.9;
        let beta = 10.0;

        let result =
            make_sincs_for_kaiser(sample_count, factor, f_cutoff, beta, WindowType::Periodic);

        let expected = vec![
            vec![-0.0084796025, 0.4976338439, 0.4976338439, -0.0084796025],
            vec![-0.0000355271, 0.0296676259, 0.9623917926, 0.0296676259],
        ];

        for (actual_row, expected_row) in result.iter().zip(&expected) {
            for (&actual, &exp) in actual_row.iter().zip(expected_row) {
                assert_approx_f32(actual as f64, exp);
            }
        }
    }

    #[test]
    fn test_make_kaiser_window_small_beta_symmetric() {
        // Test against scipy.signal.windows.kaiser(5, 0.5, sym=True)
        let window = make_kaiser_window(5, 0.5, WindowType::Symmetric);
        let expected = vec![
            0.940306193319157,
            0.984902269883833,
            1.000000000000000,
            0.984902269883833,
            0.940306193319157,
        ];

        assert_eq!(window.len(), expected.len());
        for (&actual, &exp) in window.iter().zip(&expected) {
            assert_approx_f32(actual as f64, exp);
        }
    }

    #[test]
    fn test_make_kaiser_window_beta_5_symmetric() {
        // Test against scipy.signal.windows.kaiser(15, 5.0, sym=True)
        let window = make_kaiser_window(15, 5.0, WindowType::Symmetric);
        let expected = vec![
            0.036710892271287,
            0.127982199301765,
            0.270694417889417,
            0.453689854203301,
            0.651738235245363,
            0.830535847455841,
            0.955247316456437,
            1.000000000000000,
            0.955247316456437,
            0.830535847455841,
            0.651738235245363,
            0.453689854203301,
            0.270694417889417,
            0.127982199301765,
            0.036710892271287,
        ];

        assert_eq!(window.len(), expected.len());
        for (&actual, &exp) in window.iter().zip(&expected) {
            assert_approx_f32(actual as f64, exp);
        }
    }

    #[test]
    fn test_make_kaiser_window_beta_10_symmetric() {
        // Test against scipy.signal.windows.kaiser(9, 10.0, sym=True)
        let window = make_kaiser_window(9, 10.0, WindowType::Symmetric);
        let expected = vec![
            0.000355149374724,
            0.041939800327748,
            0.282059620822733,
            0.740117133443384,
            1.000000000000000,
            0.740117133443384,
            0.282059620822733,
            0.041939800327748,
            0.000355149374724,
        ];

        assert_eq!(window.len(), expected.len());
        for (&actual, &exp) in window.iter().zip(&expected) {
            assert_approx_f32(actual as f64, exp);
        }
    }

    #[test]
    fn test_make_sincs_for_kaiser_reference_values_symmetric() {
        // Test against numpy/scipy reference implementation (symmetric window).
        let sample_count = 4;
        let factor = 2;
        let f_cutoff = 0.9;
        let beta = 10.0;

        let result =
            make_sincs_for_kaiser(sample_count, factor, f_cutoff, beta, WindowType::Symmetric);

        let expected = vec![
            vec![-0.0135119673, 0.6818196469, 0.3016755841, -0.0000802533],
            vec![-0.0000397065, 0.0471924586, 0.9759149497, 0.0070292878],
        ];

        for (actual_row, expected_row) in result.iter().zip(&expected) {
            for (&actual, &exp) in actual_row.iter().zip(expected_row) {
                assert_approx_f32(actual as f64, exp);
            }
        }
    }

    #[test]
    fn test_make_sincs_for_kaiser_normalization() {
        // Test that the sum of all polyphase filters is close to the number of filters
        // (each filter should sum to approximately 1).
        let sample_count = 8;
        let factor = 4;
        let f_cutoff = 0.95;
        let beta = 10.0;

        let result =
            make_sincs_for_kaiser(sample_count, factor, f_cutoff, beta, WindowType::Periodic);

        let mut total_sum = 0.0;
        for row in &result {
            let row_sum: f32 = row.iter().sum();
            total_sum += row_sum;
        }

        // Total sum should be close to the number of polyphase filters.
        assert!(
            (total_sum - factor as f32).abs() < 0.01,
            "Total sum {total_sum} should be close to {factor}"
        );
    }
}
