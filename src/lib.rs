//! Resampler is a small, zero-dependency crate optimized for resampling audio data from one
//! sampling rate to another. Optimized for the most common audio sampling rates.
//!
//! ## Usage Example
//!
//! ```rust
//! use resampler::{Resampler, SampleRate};
//!
//! // Create a stereo resampler (2 channels) from 44.1 kHz to 48 kHz.
//! let mut resampler = Resampler::<2>::new(SampleRate::Hz44100, SampleRate::Hz48000);
//!
//! // Get required buffer sizes (already includes all channels).
//! let input_size = resampler.chunk_size_input();
//! let output_size = resampler.chunk_size_output();
//!
//! // Create input and output buffers (interleaved format: [L0, R0, L1, R1, ...]).
//! let input = vec![0.0f32; input_size];
//! let mut output = vec![0.0f32; output_size];
//!
//! // Process audio.
//! match resampler.resample(&input, &mut output) {
//!     Ok(()) => println!("Resample successful"),
//!     Err(error) => eprintln!("Resampling error: {error:?}"),
//! }
//! ```
//!
//! ## Implementation
//!
//! The resampler uses an FFT-based overlap-add algorithm with Kaiser windowing for high-quality
//! audio resampling. Key technical details:
//!
//! - Custom mixed-radix FFT with the standard Cooley-Tukey algorithm.
//! - SIMD optimizations: All butterflies have SSE, AVX, and ARM NEON implementations with compile
//!   time CPU feature detection.
//! - Real-valued FFT: Exploits conjugate symmetry for 2x performance.
//! - Kaiser window: Beta parameter of 10.0 provides excellent stopband attenuation of -100 dB while
//!   maintaining good time-domain localization.
//! - Optimal configurations: Pre-computed FFT sizes and factorizations for all supported sample
//!   rate pairs, with throughput scaling to ensure a latency around 256 samples.
//!
//! ## Performance
//!
//! SSE on x86_64 and NEON on aarch64 are enabled by default. But to get the best performance on
//! x86_64 AVX (+avx) and FMA (+fma) should be enabled at compile time as a target feature.
//!
//! ## no-std Compatibility
//!
//! The library supports `no-std` environments with `alloc`. To use the library in a `no-std` environment, enable the
//! `no_std` feature:
//!
//! ```toml
//! [dependencies]
//! resampler = { version = "0.1", features = ["no_std"] }
//! ```
//!
//! ### Behavior Differences
//!
//! When the `no_std` feature is enabled:
//!
//! - FFT Caching: The library will not cache FFT objects globally. Each `Resampler` instance will
//!   create its own FFT objects and filter spectra. This increases creation time and memory
//!   consumption for multiple `Resampler` for the same configuration.
//!
//! The default build (without `no_std` feature) has zero dependencies and uses the standard
//! library for optimal performance and memory efficiency through global FFT caching.
//!
//! ## License
//!
//! Licensed under either of
//!
//! - Apache License, Version 2.0
//! - MIT license
//!
//! at your option.
#![cfg_attr(feature = "no_std", no_std)]
#![forbid(missing_docs)]

extern crate alloc;

mod error;
mod fft;
mod planner;
mod resampler;
mod window;

pub use error::ResampleError;
pub(crate) use fft::*;
pub use resampler::*;

/// All sample rates the resampler can operate on.
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum SampleRate {
    /// 22.5 kHz
    Hz22050,
    /// 16 kHz
    Hz16000,
    /// 32 kHz
    Hz32000,
    /// 44.1 kHz
    Hz44100,
    /// 48 kHz
    Hz48000,
    /// 88.2 kHz
    Hz88200,
    /// 96 kHz
    Hz96000,
    /// 176.4 kHz
    Hz176400,
    /// 192 kHz
    Hz192000,
    /// 384 kHz
    Hz384000,
}

impl SampleRate {
    pub(crate) fn family(self) -> SampleRateFamily {
        match self {
            SampleRate::Hz22050 => SampleRateFamily::Hz22050,
            SampleRate::Hz16000 => SampleRateFamily::Hz16000,
            SampleRate::Hz32000 => SampleRateFamily::Hz16000,
            SampleRate::Hz44100 => SampleRateFamily::Hz22050,
            SampleRate::Hz48000 => SampleRateFamily::Hz48000,
            SampleRate::Hz88200 => SampleRateFamily::Hz22050,
            SampleRate::Hz96000 => SampleRateFamily::Hz48000,
            SampleRate::Hz176400 => SampleRateFamily::Hz22050,
            SampleRate::Hz192000 => SampleRateFamily::Hz48000,
            SampleRate::Hz384000 => SampleRateFamily::Hz48000,
        }
    }

    /// Returns the multiplier of the actual sample rate relative to its base family.
    ///
    /// For example:
    /// - 22050 is the base of its family, so it returns 1
    /// - 44100 is 2× the base (22050), so it returns 2
    /// - 96000 is 2× the base (48000), so it returns 2
    pub(crate) fn family_multiplier(self) -> usize {
        let actual_rate: usize = self.into();
        let family_rate: usize = self.family().into();
        actual_rate / family_rate
    }
}

impl From<SampleRate> for usize {
    fn from(value: SampleRate) -> Self {
        match value {
            SampleRate::Hz22050 => 22050,
            SampleRate::Hz16000 => 16000,
            SampleRate::Hz32000 => 32000,
            SampleRate::Hz44100 => 44100,
            SampleRate::Hz48000 => 48000,
            SampleRate::Hz88200 => 88200,
            SampleRate::Hz96000 => 96000,
            SampleRate::Hz176400 => 176400,
            SampleRate::Hz192000 => 192000,
            SampleRate::Hz384000 => 384000,
        }
    }
}

impl TryFrom<usize> for SampleRate {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            22050 => Ok(SampleRate::Hz22050),
            16000 => Ok(SampleRate::Hz16000),
            32000 => Ok(SampleRate::Hz32000),
            44100 => Ok(SampleRate::Hz44100),
            48000 => Ok(SampleRate::Hz48000),
            88200 => Ok(SampleRate::Hz88200),
            96000 => Ok(SampleRate::Hz96000),
            176400 => Ok(SampleRate::Hz176400),
            192000 => Ok(SampleRate::Hz192000),
            384000 => Ok(SampleRate::Hz384000),
            _ => Err(()),
        }
    }
}

/// The "family" of "lineage" that every sample rate must be a multiple of.
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
enum SampleRateFamily {
    /// 22.5 kHz Family
    Hz22050,
    /// 16.0 kHz Family
    Hz16000,
    /// 48 kHz Family
    Hz48000,
}

impl From<SampleRateFamily> for usize {
    fn from(value: SampleRateFamily) -> Self {
        match value {
            SampleRateFamily::Hz22050 => 22050,
            SampleRateFamily::Hz16000 => 16000,
            SampleRateFamily::Hz48000 => 48000,
        }
    }
}
