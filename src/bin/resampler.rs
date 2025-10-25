use std::{env, time::Instant};

use hound::{WavReader, WavWriter};
use rsampler::{LatencyMode, Resampler, SampleRate};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!(
            "Usage: {} --sample-rate=<rate> <input.wav> <output.wav>",
            args[0]
        );
        std::process::exit(1);
    }

    let sample_rate_arg = &args[1];
    let target_sample_rate = if sample_rate_arg.starts_with("--sample-rate=") {
        sample_rate_arg
            .strip_prefix("--sample-rate=")
            .unwrap()
            .parse::<u32>()
            .unwrap()
    } else {
        eprintln!("Expected --sample-rate=<rate>, got: {sample_rate_arg}");
        std::process::exit(1);
    };

    let input_path = &args[2];
    let output_path = &args[3];

    // Read input WAV file.
    let mut reader = WavReader::open(input_path).unwrap();
    let spec = reader.spec();
    let input_sample_rate = spec.sample_rate;

    println!(
        "Input: {} Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );
    println!("Output: {} Hz", target_sample_rate);

    // Convert sample rates to SampleRate enum.
    let input_rate = match SampleRate::try_from(input_sample_rate as usize) {
        Ok(rate) => rate,
        Err(_) => {
            eprintln!(
                "Unsupported input sample rate: {}. Supported rates: 16000, 22050, 32000, 44100, 48000, 96000",
                input_sample_rate
            );
            std::process::exit(1);
        }
    };

    let output_rate = match SampleRate::try_from(target_sample_rate as usize) {
        Ok(rate) => rate,
        Err(_) => {
            eprintln!(
                "Unsupported output sample rate: {}. Supported rates: 16000, 22050, 32000, 44100, 48000, 96000",
                target_sample_rate
            );
            std::process::exit(1);
        }
    };

    // Read all samples and convert to stereo f32 (interleaved).
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            // Convert integer samples to f32 in range [-1.0, 1.0]
            let max_value = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_value)
                .collect()
        }
    };

    // Convert to stereo interleaved if mono.
    let mut stereo_samples = Vec::new();
    match spec.channels {
        1 => {
            // Mono: duplicate to both channels.
            for &sample in &samples {
                stereo_samples.push(sample); // left
                stereo_samples.push(sample); // right
            }
        }
        2 => {
            // Already stereo interleaved.
            stereo_samples = samples;
        }
        _ => {
            eprintln!("Unsupported channel count: {}", spec.channels);
            std::process::exit(1);
        }
    };

    let input_frames = stereo_samples.len() / 2;
    println!("Input frames: {}", input_frames);

    let input_size_mib = (stereo_samples.len() * size_of::<f32>()) as f64 / (1024.0 * 1024.0);

    let mut resampler =
        Resampler::<2>::new(input_rate, output_rate, LatencyMode::ThroughputOptimized);

    let start = Instant::now();
    let resampled_samples = resample_batch(&mut resampler, &stereo_samples);
    let elapsed = start.elapsed();

    let output_frames = resampled_samples.len() / 2;
    println!("Output frames: {}", output_frames);

    let elapsed_secs = elapsed.as_secs_f64();
    let throughput_mib_per_sec = input_size_mib / elapsed_secs;
    println!(
        "Resampling took {:.3} ms ({:.2} MiB/s)",
        elapsed.as_secs_f64() * 1000.0,
        throughput_mib_per_sec
    );

    let output_spec = hound::WavSpec {
        channels: 2,
        sample_rate: target_sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, output_spec).unwrap();

    for &sample in &resampled_samples {
        writer.write_sample(sample).unwrap();
    }

    writer.finalize().unwrap();

    println!("Done! Written to {output_path}");
}

/// Resample a batch of interleaved stereo samples using the chunk-based resampler API.
fn resample_batch(resampler: &mut Resampler<2>, input_samples: &[f32]) -> Vec<f32> {
    let chunk_size_input = resampler.chunk_size_input();
    let chunk_size_output = resampler.chunk_size_output();

    // Calculate how many complete chunks we can process and if there's a partial chunk.
    let input_frames = input_samples.len() / 2;
    let num_complete_chunks = input_frames / chunk_size_input;
    let remaining_frames = input_frames % chunk_size_input;
    let has_partial_chunk = remaining_frames > 0;

    // Total chunks includes the partial chunk if present.
    let total_chunks = if has_partial_chunk {
        num_complete_chunks + 1
    } else {
        num_complete_chunks
    };

    let total_output_samples = total_chunks * chunk_size_output * 2;
    let mut output_samples = vec![0.0f32; total_output_samples];

    // Process all complete chunks directly from input (no copying).
    for chunk_idx in 0..num_complete_chunks {
        let input_start = chunk_idx * chunk_size_input * 2;
        let input_end = input_start + chunk_size_input * 2;
        let output_start = chunk_idx * chunk_size_output * 2;
        let output_end = output_start + chunk_size_output * 2;

        let input_chunk = &input_samples[input_start..input_end];
        let output_chunk = &mut output_samples[output_start..output_end];

        resampler
            .process_into_buffer(input_chunk, output_chunk)
            .expect("Resampling failed");
    }

    // Process the last partial chunk if it exists (copy and pad with zeros).
    if has_partial_chunk {
        let input_start = num_complete_chunks * chunk_size_input * 2;
        let mut padded_chunk = vec![0.0f32; chunk_size_input * 2];
        padded_chunk[..input_samples.len() - input_start]
            .copy_from_slice(&input_samples[input_start..]);

        let output_start = num_complete_chunks * chunk_size_output * 2;
        let output_end = output_start + chunk_size_output * 2;
        let output_chunk = &mut output_samples[output_start..output_end];

        resampler
            .process_into_buffer(&padded_chunk, output_chunk)
            .expect("Resampling failed");
    }

    // Trim to expected output length based on original input length.
    let input_frames = input_samples.len() / 2;
    let expected_output_frames =
        (input_frames as f64 * chunk_size_output as f64 / chunk_size_input as f64).ceil() as usize;
    output_samples.truncate(expected_output_frames * 2);

    output_samples
}
