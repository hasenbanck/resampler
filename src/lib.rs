//! # Audio resampling library
//!
//! Resampler is a small, zero-dependency crate for high-quality audio resampling between common
//! sample rates. It provides both FFT-based and FIR-based resamplers optimized for different use
//! cases.
//!
//! ## Usage Examples
//!
//! ### FFT-Based Resampler (Highest Quality)
//!
//! ```rust
//! use resampler::{ResamplerFft, SampleRate};
//!
//! // Create a stereo resampler (2 channels) from 44.1 kHz to 48 kHz.
//! let mut resampler = ResamplerFft::new(2, SampleRate::Hz44100, SampleRate::Hz48000);
//!
//! // Get required buffer sizes (already includes all channels).
//! let input_size = resampler.chunk_size_input();
//! let output_size = resampler.chunk_size_output();
//!
//! // Create input and output buffers (interleaved format: [L0, R0, L1, R1, ...]).
//! let input = vec![0.0f32; input_size];
//! let mut output = vec![0.0f32; output_size];
//!
//! resampler.resample(&input, &mut output).unwrap();
//! ```
//!
//! ### FIR-Based Resampler (Low Latency, Streaming)
//!
//! ```rust
//! use resampler::{Attenuation, Latency, ResamplerFir, SampleRate};
//!
//! // Create a stereo resampler with configurable latency (16, 32, or 64 samples).
//! let mut resampler = ResamplerFir::new(
//!     2,
//!     SampleRate::Hz48000,
//!     SampleRate::Hz44100,
//!     Latency::Sample64,
//!     Attenuation::Db90,
//! );
//!
//! // Streaming API - accepts arbitrary input buffer sizes.
//! let input = vec![0.0f32; 512];
//! let mut output = vec![0.0f32; resampler.buffer_size_output()];
//!
//! let (consumed, produced) = resampler.resample(&input, &mut output).unwrap();
//! println!("Consumed {consumed} samples, produced {produced} samples");
//! ```
//!
//! ## Choosing a Resampler
//!
//! Both resamplers provide good quality, but are optimized for different use cases:
//!
//! | Feature     | [`ResamplerFft`]                 | [`ResamplerFir`]             |
//! |-------------|----------------------------------|------------------------------|
//! | Quality     | Very good (sharp rolloff)        | Good (slow rolloff)          |
//! | Performance | Very fast                        | Fast (configurable)          |
//! | Latency     | ~256 samples                     | 16-64 samples (configurable) |
//! | API         | Fixed chunk size                 | Flexible streaming           |
//! | Best for    | Non-latency sensitive processing | Low-latency processing       |
//!
//! Use [`ResamplerFft`] when:
//! - You need the absolute highest quality
//! - Latency is not a concern
//! - Processing pre-recorded audio files
//!
//! Use [`ResamplerFir`] when:
//! - You need low latency (real-time audio)
//! - You can live with a slower rolloff
//! - Working with streaming data
//!
//! ## FFT-Based Implementation
//!
//! The resampler uses an FFT-based overlap-add algorithm with Kaiser windowing for high-quality
//! audio resampling. Key technical details:
//!
//! - Custom mixed-radix FFT with the Stockham Autosort algorithm.
//! - SIMD optimizations: All butterflies have SSE2, SSE4.2, AVX+FMA, and ARM NEON implementations.
//! - Stopband attenuation of -100 dB using the Kaiser windows function.
//! - Latency around 256 samples.
//!
//! ## FIR-Based Implementation
//!
//! The FIR resampler uses a polyphase filter with linear interpolation for high-quality audio
//! resampling with low latency. Key technical details:
//!
//! - Polyphase decomposition: 1024 phases with linear interpolation between phases.
//! - SIMD optimizations: Convolution kernels optimized with SSE2, SSE4.2, AVX+FMA, AVX-512
//!   and ARM NEON.
//! - Configurable filter length: 32, 64, or 128 taps (16, 32, or 64 samples latency).
//! - Adjustable rolloff and stopband attenuation,
//! - Streaming API: Accepts arbitrary input buffer sizes for flexible real-time processing,
//!
//! ## Performance
//!
//! Both resamplers include SIMD optimizations with runtime CPU feature detection for maximum
//! performance and compatibility.
//!
//! But for up to 25% better performance on x86_64, compile with `target-cpu=x86-64-v3`
//! (enables AVX2, FMA, and other optimizations).
//!
//! Overall the SIMD for x86_64 have four levels implemented, targeting four possible CPU
//! generations that build up on each other:
//!
//!  * x86-64-v1: 128-bit SSE2 (around 2003-2004)
//!  * x86-64-v2: 128-bit SSE4.2 (around 2008-2011)
//!  * x86-64-v3: 256-bit AVX+FMA (around 2013-2015)
//!  * x86-64-v4: 512-bit AVX-512 (around 2017-2022)
//!
//! ## no-std Compatibility
//!
//! The library supports `no-std` environments with `alloc`. To use the library in a `no-std`
//! environment, enable the `no_std` feature:
//!
//! ```toml
//! [dependencies]
//! resampler = { version = "0.2", features = ["no_std"] }
//! ```
//!
//! ### Behavior Differences
//!
//! When the `no_std` feature is enabled:
//!
//! - Caching: The library will not cache FFT and FIR objects globally to shorten resampler creation
//!   time and lower overall memory consumption for multiple resamplers.
//!
//! - No runtime detection of SIMD functionality. You need to activate SIMD via compile time target
//!   features.
//!
//! The default build (without `no_std` feature) has zero dependencies and uses the standard
//! library for optimal performance and memory efficiency through global caching.
//!
//! ## Alternatives
//!
//! Other high-quality audio resampling libraries in Rust are:
//!
//! - [Rubato](https://github.com/HEnquist/rubato): The overlap-add resampling approach used in this
//!   library is based on Rubato's implementation.
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
mod fir;
mod resampler_fft;
mod resampler_fir;
mod window;

pub use error::ResampleError;
pub(crate) use fft::*;
pub use resampler_fft::*;
pub use resampler_fir::{Attenuation, Latency, ResamplerFir};

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
    pub(crate) fn family_multiplier(self) -> u32 {
        let actual_rate: u32 = self.into();
        let family_rate: u32 = self.family().into();
        actual_rate / family_rate
    }
}

impl From<SampleRate> for u32 {
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

impl TryFrom<u32> for SampleRate {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
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

impl From<SampleRateFamily> for u32 {
    fn from(value: SampleRateFamily) -> Self {
        match value {
            SampleRateFamily::Hz22050 => 22050,
            SampleRateFamily::Hz16000 => 16000,
            SampleRateFamily::Hz48000 => 48000,
        }
    }
}
