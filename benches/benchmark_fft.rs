use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand_aes::tls::rand_f32;
use rsampler::planner::ConversionConfig;
use rsampler::{Complex32, Forward, Inverse, RadixFFT, SampleRate};

struct BenchmarkConfig {
    input_rate: SampleRate,
    output_rate: SampleRate,
    description: &'static str,
}

fn generate_white_noise(size: usize) -> Vec<f32> {
    (0..size).map(|_| rand_f32()).collect()
}

fn bench_fft_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_cycle");

    let configs = vec![
        BenchmarkConfig {
            input_rate: SampleRate::_48000,
            output_rate: SampleRate::_96000,
            description: "48kHz→96kHz (same family)",
        },
        BenchmarkConfig {
            input_rate: SampleRate::_22050,
            output_rate: SampleRate::_48000,
            description: "22.05kHz→48kHz",
        },
        BenchmarkConfig {
            input_rate: SampleRate::_44100,
            output_rate: SampleRate::_48000,
            description: "44.1kHz→48kHz",
        },
        BenchmarkConfig {
            input_rate: SampleRate::_48000,
            output_rate: SampleRate::_44100,
            description: "48kHz→44.1kHz",
        },
    ];

    for bench_config in &configs {
        let config =
            ConversionConfig::from_sample_rates(bench_config.input_rate, bench_config.output_rate);
        let (fft_size_in, factors_in, fft_size_out, factors_out) = config.scale_for_throughput();

        let bytes_per_iteration = (fft_size_in + fft_size_out) * size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes_per_iteration as u64));

        group.bench_with_input(
            BenchmarkId::new("full_cycle", bench_config.description),
            bench_config,
            |b, _| {
                let fft_forward = RadixFFT::<Forward>::new(factors_in.clone());
                let fft_inverse = RadixFFT::<Inverse>::new(factors_out.clone());

                let input_data = generate_white_noise(fft_size_in);

                let freq_size = (fft_size_in / 2 + 1).max(fft_size_out / 2 + 1);
                let mut freq_data = vec![Complex32::default(); freq_size];
                let mut output_data = vec![0.0f32; fft_size_out];

                let scratchpad_size = fft_forward
                    .scratchpad_size()
                    .max(fft_inverse.scratchpad_size());

                let mut scratchpad = vec![Complex32::default(); scratchpad_size];

                b.iter(|| {
                    fft_forward.process(&input_data, &mut freq_data, &mut scratchpad);
                    black_box(&freq_data);
                    fft_inverse.process(&freq_data, &mut output_data, &mut scratchpad);
                    black_box(&output_data);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_fft_cycle);
criterion_main!(benches);
