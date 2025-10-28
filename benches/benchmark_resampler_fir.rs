use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand_aes::tls::rand_f32;
use resampler::{Latency, ResamplerFir, SampleRate};

struct BenchmarkConfig {
    input_rate: SampleRate,
    output_rate: SampleRate,
    description: &'static str,
}

fn generate_white_noise(size: usize) -> Vec<f32> {
    (0..size).map(|_| rand_f32()).collect()
}

fn bench_resampler_fir(c: &mut Criterion) {
    let mut group = c.benchmark_group("fir");

    let configs = vec![
        BenchmarkConfig {
            input_rate: SampleRate::Hz48000,
            output_rate: SampleRate::Hz96000,
            description: "48kHz→96kHz (same family)",
        },
        BenchmarkConfig {
            input_rate: SampleRate::Hz22050,
            output_rate: SampleRate::Hz48000,
            description: "22.05kHz→48kHz",
        },
        BenchmarkConfig {
            input_rate: SampleRate::Hz44100,
            output_rate: SampleRate::Hz48000,
            description: "44.1kHz→48kHz",
        },
        BenchmarkConfig {
            input_rate: SampleRate::Hz48000,
            output_rate: SampleRate::Hz44100,
            description: "48kHz→44.1kHz",
        },
    ];

    for bench_config in &configs {
        const CHANNELS: usize = 2;
        const CHUNK_SIZE: usize = 256;

        // Calculate expected output size for throughput measurement
        let input_rate_hz = u32::from(bench_config.input_rate) as f64;
        let output_rate_hz = u32::from(bench_config.output_rate) as f64;
        let ratio = output_rate_hz / input_rate_hz;
        let expected_output = (CHUNK_SIZE as f64 * ratio) as usize;

        let bytes_per_iteration = expected_output * size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes_per_iteration as u64));

        group.bench_with_input(
            BenchmarkId::new("process", bench_config.description),
            bench_config,
            |b, bench_config| {
                let mut resampler = ResamplerFir::<CHANNELS>::new(
                    bench_config.input_rate,
                    bench_config.output_rate,
                    Latency::default(),
                );

                let input = generate_white_noise(CHUNK_SIZE);
                let mut output = vec![0.0f32; 512];

                b.iter(|| {
                    let (_, produced) = resampler
                        .resample(black_box(&input), black_box(&mut output))
                        .unwrap();
                    black_box(produced);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_resampler_fir);
criterion_main!(benches);
