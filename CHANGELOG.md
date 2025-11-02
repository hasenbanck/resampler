

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
