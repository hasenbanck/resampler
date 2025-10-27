/// Errors the process function can throw.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResampleError {
    /// Input buffer size is too small.
    InputBufferSize,
    /// Output buffer size is too small.
    OutputBufferSize,
}

impl std::fmt::Display for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputBufferSize => "Input buffer size is too small".fmt(f),
            Self::OutputBufferSize => "Output buffer size is too small".fmt(f),
        }
    }
}

impl std::fmt::Debug for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self, f)
    }
}

impl std::error::Error for ResampleError {}
