use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand_aes::tls::rand_f32;
use rsampler::{Complex32, transpose};

/// Generate random Complex32 data for benchmarking
fn generate_random_data(size: usize) -> Vec<Complex32> {
    (0..size)
        .map(|_| Complex32::new(rand_f32(), rand_f32()))
        .collect()
}

/// Benchmark transpose with non-square matrices (typical for Six-Step FFT algorithm)
fn bench_nonsquare_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_nonsquare");

    let configs = vec![
        // Small buffer region (~64) - 32x multiples
        (64, 128, "64x128 (8K elements, same family 2x, 32x base)"),
        (
            512,
            1120,
            "512x1120 (574K elements, 22.05→48 kHz, 32x base)",
        ),
        (64, 192, "64x192 (12K elements, 16→48 kHz, 1x base)"),
        (70, 192, "70x192 (13K elements, 16→44.1 kHz, 1x base)"),
        // Medium buffer region (~512) - 256x and 8x multiples
        (
            512,
            1024,
            "512x1024 (512K elements, same family 2x, 256x base)",
        ),
        (512, 3072, "512x3072 (1.5M elements, 16→48 kHz, 8x base)"),
        (560, 1536, "560x1536 (860K elements, 16→44.1 kHz, 8x base)"),
        // Large buffer region (~1024) - 512x and 16x multiples
        (
            1024,
            2048,
            "1024x2048 (2M elements, same family 2x, 512x base)",
        ),
        (1024, 3072, "1024x3072 (3M elements, 16→48 kHz, 16x base)"),
        (1120, 3072, "1120x3072 (3M elements, 16→44.1 kHz, 16x base)"),
    ];

    for (width, height, desc) in configs {
        let n = width * height;
        let bytes = n * std::mem::size_of::<Complex32>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("nonsquare", desc),
            &(width, height),
            |b, &(width, height)| {
                let mut input = generate_random_data(width * height);
                let mut output = vec![Complex32::default(); width * height];

                b.iter(|| {
                    transpose(&mut input, &mut output, width, height);
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_nonsquare_transpose);
criterion_main!(benches);
