use std::time::Instant;

use clap::{Parser, ValueEnum};
use hound::{WavReader, WavWriter};
use resampler::{Attenuation, Latency, ResamplerFft, ResamplerFir, SampleRate};

mod interpolation_resampler;
use interpolation_resampler::{InterpolationMode, InterpolationResampler};

#[derive(Parser, Debug)]
#[command(name = "resample")]
#[command(about = "Resample WAV files to different sample rates", long_about = None)]
struct Cli {
    #[arg(long, value_enum)]
    filter: FilterType,
    #[arg(long, value_name = "RATE")]
    sample_rate: u32,
    #[arg(long, value_name = "SAMPLES")]
    latency: Option<u8>,
    #[arg(long, value_name = "DB")]
    attenuation: Option<u16>,
    input: String,
    output: String,
}

#[derive(Debug, Clone, ValueEnum)]
enum FilterType {
    Linear,
    Hermite,
    Fir,
    Fft,
}

fn parse_latency(value: u8) -> Result<Latency, String> {
    match value {
        8 => Ok(Latency::Sample8),
        16 => Ok(Latency::Sample16),
        32 => Ok(Latency::Sample32),
        64 => Ok(Latency::Sample64),
        _ => Err(format!(
            "Invalid latency value: {value}. Must be 8, 16, 32, or 64"
        )),
    }
}

fn parse_attenuation(value: u16) -> Result<Attenuation, String> {
    match value {
        60 => Ok(Attenuation::Db60),
        90 => Ok(Attenuation::Db90),
        120 => Ok(Attenuation::Db120),
        _ => Err(format!(
            "Invalid attenuation value: {value}. Must be 60, 90, or 120"
        )),
    }
}

fn main() {
    let cli = Cli::parse();

    let latency = match cli.latency {
        Some(value) => match parse_latency(value) {
            Ok(latency) => latency,
            Err(error) => {
                eprintln!("Error: {error}");
                std::process::exit(1);
            }
        },
        None => Latency::Sample64,
    };

    let attenuation = match cli.attenuation {
        Some(value) => match parse_attenuation(value) {
            Ok(attenuation) => attenuation,
            Err(error) => {
                eprintln!("Error: {error}");
                std::process::exit(1);
            }
        },
        None => Attenuation::Db90,
    };

    let input_path = &cli.input;
    let output_path = &cli.output;

    let mut reader = WavReader::open(input_path).unwrap();
    let spec = reader.spec();
    let input_sample_rate = spec.sample_rate;

    println!(
        "Input: {} Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );
    println!("Output: {} Hz", cli.sample_rate);
    println!(
        "Method: {}",
        match cli.filter {
            FilterType::Linear => "Linear interpolation".to_string(),
            FilterType::Hermite => "Hermite interpolation".to_string(),
            FilterType::Fir => format!(
                "FIR polyphase resampling (latency: {:?}, attenuation: {:?})",
                latency, attenuation
            ),
            FilterType::Fft => "FFT resampling".to_string(),
        }
    );

    let input_rate = match SampleRate::try_from(input_sample_rate) {
        Ok(rate) => rate,
        Err(_) => {
            eprintln!(
                "Unsupported input sample rate: {input_sample_rate}. Supported rates: 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000, 384000"
            );
            std::process::exit(1);
        }
    };

    let output_rate = match SampleRate::try_from(cli.sample_rate) {
        Ok(rate) => rate,
        Err(_) => {
            eprintln!(
                "Unsupported output sample rate: {}. Supported rates: 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000, 384000",
                cli.sample_rate
            );
            std::process::exit(1);
        }
    };

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max_value = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_value)
                .collect()
        }
    };

    let mut stereo_samples = Vec::new();
    match spec.channels {
        1 => {
            // Mono: duplicate to both channels.
            for &sample in &samples {
                stereo_samples.push(sample);
                stereo_samples.push(sample);
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
    println!("Input frames: {input_frames}");

    let start = Instant::now();
    let resampled_samples = match cli.filter {
        FilterType::Fir => {
            let mut resampler = ResamplerFir::new(2, input_rate, output_rate, latency, attenuation);
            resample_batch_fir(&mut resampler, &stereo_samples)
        }
        FilterType::Linear => resample_batch_interpolation(
            input_rate,
            output_rate,
            InterpolationMode::Linear,
            &stereo_samples,
        ),
        FilterType::Hermite => resample_batch_interpolation(
            input_rate,
            output_rate,
            InterpolationMode::Hermite,
            &stereo_samples,
        ),
        FilterType::Fft => {
            let mut resampler = ResamplerFft::new(2, input_rate, output_rate);
            resample_batch(&mut resampler, &stereo_samples)
        }
    };

    let elapsed = start.elapsed();
    let input_size_mib = (resampled_samples.len() * size_of::<f32>()) as f64 / (1024.0 * 1024.0);

    let output_frames = resampled_samples.len() / 2;
    println!("Output frames: {output_frames}");

    let elapsed_secs = elapsed.as_secs_f64();
    let throughput_mib_per_sec = input_size_mib / elapsed_secs;
    println!(
        "Resampling took {:.3} ms ({throughput_mib_per_sec:.2} MiB/s)",
        elapsed.as_secs_f64() * 1000.0
    );

    let output_spec = hound::WavSpec {
        channels: 2,
        sample_rate: cli.sample_rate,
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

fn resample_batch_interpolation(
    input_rate: SampleRate,
    output_rate: SampleRate,
    mode: InterpolationMode,
    input_samples: &[f32],
) -> Vec<f32> {
    let resampler = InterpolationResampler::<2>::new(input_rate, output_rate, mode);
    resampler.resample(input_samples)
}

fn resample_batch_fir(resampler: &mut ResamplerFir, input_samples: &[f32]) -> Vec<f32> {
    const CHUNK_SIZE: usize = 512;

    let mut output_samples = Vec::new();
    let mut input_offset = 0;

    let buffer_size_output = resampler.buffer_size_output();
    let mut output_buffer = vec![0.0f32; buffer_size_output];

    while input_offset < input_samples.len() {
        let remaining = input_samples.len() - input_offset;
        let chunk_size = remaining.min(CHUNK_SIZE);
        let input_chunk = &input_samples[input_offset..input_offset + chunk_size];

        let (consumed, produced) = resampler
            .resample(input_chunk, &mut output_buffer)
            .expect("FIR resampling failed");

        output_samples.extend_from_slice(&output_buffer[..produced]);

        input_offset += consumed;

        if consumed == 0 {
            break;
        }
    }

    output_samples
}

fn resample_batch(resampler: &mut ResamplerFft, input_samples: &[f32]) -> Vec<f32> {
    let chunk_size_input = resampler.chunk_size_input();
    let chunk_size_output = resampler.chunk_size_output();

    // Calculate how many complete chunks we can process and if there's a partial chunk.
    let num_complete_chunks = input_samples.len() / chunk_size_input;
    let remaining_samples = input_samples.len() % chunk_size_input;
    let has_partial_chunk = remaining_samples > 0;

    // Total chunks includes the partial chunk if present.
    let total_chunks = if has_partial_chunk {
        num_complete_chunks + 1
    } else {
        num_complete_chunks
    };

    let total_output_samples = total_chunks * chunk_size_output;
    let mut output_samples = vec![0.0f32; total_output_samples];

    // Process all complete chunks directly from input (no copying).
    for chunk_idx in 0..num_complete_chunks {
        let input_start = chunk_idx * chunk_size_input;
        let input_end = input_start + chunk_size_input;
        let output_start = chunk_idx * chunk_size_output;
        let output_end = output_start + chunk_size_output;

        let input_chunk = &input_samples[input_start..input_end];
        let output_chunk = &mut output_samples[output_start..output_end];

        resampler
            .resample(input_chunk, output_chunk)
            .expect("Resampling failed");
    }

    // Process the last partial chunk if it exists (copy and pad with zeros).
    if has_partial_chunk {
        let input_start = num_complete_chunks * chunk_size_input;
        let mut padded_chunk = vec![0.0f32; chunk_size_input];
        padded_chunk[..input_samples.len() - input_start]
            .copy_from_slice(&input_samples[input_start..]);

        let output_start = num_complete_chunks * chunk_size_output;
        let output_end = output_start + chunk_size_output;
        let output_chunk = &mut output_samples[output_start..output_end];

        resampler
            .resample(&padded_chunk, output_chunk)
            .expect("Resampling failed");
    }

    // Trim to expected output length based on original input length.
    let expected_output_samples = (input_samples.len() as f64 * chunk_size_output as f64
        / chunk_size_input as f64)
        .ceil() as usize;
    output_samples.truncate(expected_output_samples);

    output_samples
}
