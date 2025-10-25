use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand_aes::tls::rand_f32;
use rsampler::{Complex32, Forward, Inverse, Radix, RadixFFT};

/// Configuration for a resampling pattern benchmark.
struct ResamplingConfig {
    /// Input FFT size.
    input_size: usize,
    /// Output FFT size.
    output_size: usize,
    /// Radix factorization for input FFT.
    input_factors: Vec<Radix>,
    /// Radix factorization for output FFT.
    output_factors: Vec<Radix>,
    /// Human-readable description of this configuration.
    description: &'static str,
}

fn generate_white_noise(size: usize) -> Vec<f32> {
    (0..size).map(|_| rand_f32()).collect()
}

fn bench_fft_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_cycle");

    let configs = vec![
        // Same family: 2→4 pattern, 512× multiplier = 1024→2048
        ResamplingConfig {
            input_size: 1024,
            output_size: 2048,
            input_factors: vec![Radix::Factor2; 10],
            output_factors: vec![Radix::Factor2; 11],
            description: "Same family 1024→2048 (2^10→2^11)",
        },
        // 22.05kHz→48kHz: 16→35 pattern, 50× multiplier = 800→1750
        ResamplingConfig {
            input_size: 800,
            output_size: 1750,
            input_factors: vec![
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor5,
                Radix::Factor5,
            ],
            output_factors: vec![
                Radix::Factor2,
                Radix::Factor5,
                Radix::Factor5,
                Radix::Factor5,
                Radix::Factor7,
            ],
            description: "22.05→48kHz 800→1750 (2^5×5^2→2×5^3×7)",
        },
        // 16kHz→48kHz: 64→192 pattern, 10× multiplier = 640→1920
        ResamplingConfig {
            input_size: 640,
            output_size: 1920,
            input_factors: vec![
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor5,
            ],
            output_factors: vec![
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor3,
                Radix::Factor5,
            ],
            description: "16→48kHz 640→1920 (2^7×5→2^7×3×5)",
        },
        // 16kHz→44.1kHz: 70→192 pattern, 10× multiplier = 700→1920
        ResamplingConfig {
            input_size: 700,
            output_size: 1920,
            input_factors: vec![
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor5,
                Radix::Factor5,
                Radix::Factor7,
            ],
            output_factors: vec![
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor2,
                Radix::Factor3,
                Radix::Factor5,
            ],
            description: "16→44.1kHz 700→1920 (2^2×5^2×7→2^7×3×5)",
        },
    ];

    for config in &configs {
        let bytes =
            (config.input_size * size_of::<f32>()) + (config.output_size * size_of::<f32>());
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("full_cycle", config.description),
            config,
            |b, config| {
                let fft_forward = RadixFFT::<Forward>::new(config.input_factors.clone());
                let fft_inverse = RadixFFT::<Inverse>::new(config.output_factors.clone());

                let input = generate_white_noise(config.input_size);

                let freq_size = (config.input_size / 2 + 1).max(config.output_size / 2 + 1);
                let mut freq_data = vec![Complex32::default(); freq_size];
                let mut output = vec![0.0f32; config.output_size];

                let scratchpad_size = fft_forward
                    .scratchpad_size()
                    .max(fft_inverse.scratchpad_size());
                let mut scratchpad = vec![Complex32::default(); scratchpad_size];

                b.iter(|| {
                    fft_forward.process(&input, &mut freq_data, &mut scratchpad);
                    black_box(&freq_data);
                    fft_inverse.process(&freq_data, &mut output, &mut scratchpad);
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_fft_cycle);
criterion_main!(benches);
