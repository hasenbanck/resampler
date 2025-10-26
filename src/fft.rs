mod butterflies;
pub(crate) mod cooley_tukey;
mod radix_fft;
mod transpose;

pub use radix_fft::{Forward, Inverse, Radix, RadixFFT};
pub use transpose::transpose;

/// Simple complex number struct
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    #[inline(always)]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    pub const fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    pub const fn add(&self, o: &Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }

    #[inline(always)]
    pub const fn sub(&self, o: &Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }

    #[inline(always)]
    pub const fn mul(&self, o: &Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    #[inline(always)]
    pub const fn scale(&self, f: f32) -> Self {
        Self {
            re: self.re * f,
            im: self.im * f,
        }
    }
}
