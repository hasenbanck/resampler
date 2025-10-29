

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2024-10-29

### Changed

- Renamed `ResamplerFir` now is configurable by latency and attenuation.
- Internal `ResamplerFir` optimizations.

## [0.2.0] - 2024-10-28

### Added

- Crate is now no_std compatible
- Added FIR based resampler `ResamplerFir`

### Changed

- Renamed `Resampler` to `ResamplerFft`

## [0.1.0] - 2025-10-27

### Added

- Initial release.
