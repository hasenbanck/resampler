mod error;
mod fft;
pub mod planner;
mod resampler;
mod window;

pub use error::ResampleError;
pub use fft::*;
pub use resampler::*;

/// All sample rates the resampler can operate on.
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum SampleRate {
    /// 22.5 kHz
    _22050,
    /// 16 kHz
    _16000,
    /// 32 kHz
    _32000,
    /// 44.1 kHz
    _44100,
    /// 48 kHz
    _48000,
    /// 96 kHz
    _96000,
    /// 192 kHz
    _192000,
}

impl SampleRate {
    pub(crate) fn family(self) -> SampleRateFamily {
        match self {
            SampleRate::_22050 => SampleRateFamily::_22050,
            SampleRate::_16000 => SampleRateFamily::_16000,
            SampleRate::_32000 => SampleRateFamily::_16000,
            SampleRate::_44100 => SampleRateFamily::_22050,
            SampleRate::_48000 => SampleRateFamily::_48000,
            SampleRate::_96000 => SampleRateFamily::_48000,
            SampleRate::_192000 => SampleRateFamily::_48000,
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
            SampleRate::_22050 => 22050,
            SampleRate::_16000 => 16000,
            SampleRate::_32000 => 32000,
            SampleRate::_44100 => 44100,
            SampleRate::_48000 => 48000,
            SampleRate::_96000 => 96000,
            SampleRate::_192000 => 192000,
        }
    }
}

impl TryFrom<usize> for SampleRate {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            22050 => Ok(SampleRate::_22050),
            16000 => Ok(SampleRate::_16000),
            32000 => Ok(SampleRate::_32000),
            44100 => Ok(SampleRate::_44100),
            48000 => Ok(SampleRate::_48000),
            96000 => Ok(SampleRate::_96000),
            192000 => Ok(SampleRate::_192000),
            _ => Err(()),
        }
    }
}

/// The "family" of "lineage" that every sample rate must be a multiple of.
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
enum SampleRateFamily {
    /// 22.5 kHz Family
    _22050,
    /// 16.0 kHz Family
    _16000,
    /// 48 kHz Family
    _48000,
}

impl From<SampleRateFamily> for usize {
    fn from(value: SampleRateFamily) -> Self {
        match value {
            SampleRateFamily::_22050 => 22050,
            SampleRateFamily::_16000 => 16000,
            SampleRateFamily::_48000 => 48000,
        }
    }
}
