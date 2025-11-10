

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.7] - 2025-11-10

### Changed

- Improved performance for `ResamplerFft` by internal optimization by a small amount for x86_64 AVX.

## [0.3.6] - 2025-11-08

The maturity of the Stockham based FFT is reaching it's optimum.
On x86_64 AVX it is only around 10% slower than RustFFT.
On aarch64 NEON is around 16% faster than RustFFT for most cases,
with power-of-two cases being the exception and being 17% slower than RustFFT.

### Changed

- Improved performance for `ResamplerFft` by internal optimization.

## [0.3.5] - 2025-11-08

### Changed

- Improved performance for `ResamplerFft` by optimizing away a bottleneck caused by a modulo operation
  (15% to 25% improved throughout with AVX).

## [0.3.4] - 2025-11-07

### Changed

- Improved performance for `ResamplerFft` by internal optimizations (~10% improved throughout).

## [0.3.3] - 2025-11-05

This patch changes the internal FFT implementation to the Stockham Autosort algorithm which improves the performance
of the `ResamplerFft` greatly again. We also improved the `ResamplerFir` performance a bit:

- The `ResamplerFft` benchmarks are on my M4 Max 42% to 54% faster (~515 to ~826 MiB/s).
- The `ResamplerFft` benchmarks are on my AMD Ryzen 9 9950X3D 0% to 52% faster (~780 to ~1192 MiB/s).
- The `ResamplerFir` benchmarks are on my AMD Ryzen 9 9950X3D 27% to 34% faster (~503 to ~540 MiB/s).

### Changed

- Improved performance for `ResamplerFft` by changing the algorithm to the Stockham Autosort algorithm.
- Improved performance for `ResamplerFir` by merging two convolve functions into one improving the performance.
- Internally we now target 4 SIMD profiles: SSE2, SSE4.2, AVX+FMA and AVX-512. This lessens the maintenance burden and
  aligned to the x86-64-v1, x86-64-v2, x86-64-v3, x86-64-v4 microarchitectures. 

## [0.3.2] - 2025-11-02

For this patch we focused on improving the performance of both the FFT and FIR version:

- The `ResamplerFft` benchmarks are on my AMD Ryzen 9 9950X3D 3% to 50% faster (~550 to ~990 MiB/s).
- The `ResamplerFir` benchmarks are on my AMD Ryzen 9 9950X3D 3% to 15% faster (~380 to ~420 MiB/s).

### Changed

- Improved performance for `ResamplerFir` by using better memory layout and improved AVX-512 code.
- Improved performance for `ResamplerFft` when using pure factor 2 configurations.
- Improved performance for `ResamplerFft` by using SIMD for the real/complex pre- and post-processing.

### Fixed

- Internally butterflies were marked as x86_64 SSE even thought they used SSE2 functionality. This is now fixed. Since
  all x86_64 need to support both SSE and SSE2 this is a theoretical problem, but which should be fixed.

## [0.3.1] - 2025-10-31

### Changed

- `ResamplerFft` and `ResamplerFir` support runtime CPU feature detection for SIMD (no_std only support compile time
   CPU feature detection).
- `ResamplerFir` supports AVX-512. AVX-512 doesn't improve the performance for `ResamplerFft`, so is left out. 

## [0.3.0] - 2025-10-29

### Changed

- Renamed `ResamplerFir` now is configurable by latency and attenuation.
- Internal `ResamplerFir` optimizations.

## [0.2.0] - 2025-10-28

### Added

- Crate is now no_std compatible
- Added FIR based resampler `ResamplerFir`

### Changed

- Renamed `Resampler` to `ResamplerFft`

## [0.1.0] - 2025-10-27

### Added

- Initial release.
