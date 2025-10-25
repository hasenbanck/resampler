pub enum ResampleError {
    InputBufferSizeSize,
    OutputBufferSizeSize,
}

impl std::fmt::Display for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputBufferSizeSize => "Input buffer size is too small".fmt(f),
            Self::OutputBufferSizeSize => "Output buffer size is too small".fmt(f),
        }
    }
}

impl std::fmt::Debug for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self, f)
    }
}

impl std::error::Error for ResampleError {}
