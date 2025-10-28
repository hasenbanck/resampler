/// Errors the process function can throw.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResampleError {
    /// Input buffer size is invalid.
    InvalidInputBufferSize,
    /// Output buffer size is invalid.
    InvalidOutputBufferSize,
}

impl core::fmt::Display for ResampleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInputBufferSize => "Input buffer size is invalid".fmt(f),
            Self::InvalidOutputBufferSize => "Output buffer size is invalid".fmt(f),
        }
    }
}

impl core::fmt::Debug for ResampleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self, f)
    }
}

#[cfg(not(feature = "no_std"))]
impl std::error::Error for ResampleError {}
