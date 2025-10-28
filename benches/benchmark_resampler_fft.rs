use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand_aes::tls::rand_f32;
use resampler::{ResamplerFft, SampleRate};

struct BenchmarkConfig {
    input_rate: SampleRate,
    output_rate: SampleRate,
    description: &'static str,
}

fn generate_white_noise(size: usize) -> Vec<f32> {
    (0..size).map(|_| rand_f32()).collect()
}

fn bench_resampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("resampler");

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

        let temp_resampler =
            ResamplerFft::<CHANNELS>::new(bench_config.input_rate, bench_config.output_rate);

        let chunk_size_output = temp_resampler.chunk_size_output();

        let bytes_per_iteration = chunk_size_output * size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes_per_iteration as u64));

        group.bench_with_input(
            BenchmarkId::new("process", bench_config.description),
            bench_config,
            |b, bench_config| {
                let mut resampler = ResamplerFft::<CHANNELS>::new(
                    bench_config.input_rate,
                    bench_config.output_rate,
                );

                let chunk_size_input = resampler.chunk_size_input();
                let chunk_size_output = resampler.chunk_size_output();

                let input = generate_white_noise(chunk_size_input);
                let mut output = vec![0.0f32; chunk_size_output];

                b.iter(|| {
                    resampler
                        .resample(black_box(&input), black_box(&mut output))
                        .unwrap();
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_resampler);
criterion_main!(benches);
