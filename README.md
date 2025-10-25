# rsample

## Supported conversion patterns

Following "families" of sampling rates are supported:

- 16 kHz family (and all multiples)
- 22.05 kHz family (and all multiples)
- 48 kHz family (and all multiples)

| Conversion Type              | Input Size | Output Size | Ratio Error | Input FFT           | Output FFT        | Factorization      |
|------------------------------|------------|-------------|-------------|---------------------|-------------------|--------------------|
| Inside same family           | 2          | 4           | 0.0%        | Radix-2             | Radix-2           | 2 → 2²             |
| Between 22.05 kHz and 48 kHz | 16         | 35          | 0.4883%     | Radix-2             | Mixed-Radix (5,7) | 2⁴ → 5 × 7         |
| Between 16 kHz and 48 kHz    | 64         | 192         | 0.0%        | Radix-2             | Mixed-Radix (2,3) | 2⁶ → 2⁶ × 3        |
| Between 16 kHz and 44.1 kHz  | 70         | 192         | 0.4859%     | Mixed-Radix (2,5,7) | Mixed-Radix (2,3) | 2 × 5 × 7 → 2⁶ × 3 |

## License

Licensed under either of

- Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as
defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## transpose crate

```
transpose_nonsquare/nonsquare/64x128 (8K elements, same family 2x, 32x
base)                                                                             
    time:   [21.305 µs 21.480 µs 21.676 µs]
    thrpt:  [2.8158 GiB/s 2.8415 GiB/s 2.8648 GiB/s]
transpose_nonsquare/nonsquare/512x1120 (574K elements, 22.05→48 kHz, 32x
base)                                                                              
    time:   [10.397 ms 10.603 ms 10.889 ms]
    thrpt:  [401.79 MiB/s 412.63 MiB/s 420.81 MiB/s]
transpose_nonsquare/nonsquare/64x192 (12K elements, 16→48 kHz, 1x
base)                                                                               
    time:   [45.719 µs 46.348 µs 47.077 µs]
    thrpt:  [1.9448 GiB/s 1.9753 GiB/s 2.0025 GiB/s]
transpose_nonsquare/nonsquare/70x192 (13K elements, 16→44.1 kHz, 1x
base)                                                                               
    time:   [45.835 µs 45.924 µs 45.992 µs]
    thrpt:  [2.1772 GiB/s 2.1805 GiB/s 2.1847 GiB/s]
transpose_nonsquare/nonsquare/512x1024 (512K elements, same family 2x, 256x
base)                                                                            
    time:   [8.8936 ms 8.9508 ms 9.0095 ms]
    thrpt:  [443.97 MiB/s 446.89 MiB/s 449.76 MiB/s]
transpose_nonsquare/nonsquare/512x3072 (1.5M elements, 16→48 kHz, 8x
base)                                                                              
    time:   [33.025 ms 33.127 ms 33.238 ms]
    thrpt:  [361.03 MiB/s 362.24 MiB/s 363.36 MiB/s]
transpose_nonsquare/nonsquare/560x1536 (860K elements, 16→44.1 kHz, 8x
base)                                                                               
    time:   [3.2000 ms 3.2026 ms 3.2057 ms]
    thrpt:  [1.9992 GiB/s 2.0011 GiB/s 2.0027 GiB/s]
transpose_nonsquare/nonsquare/1024x2048 (2M elements, same family 2x, 512x
base)                                                                            
    time:   [41.671 ms 41.817 ms 41.976 ms]
    thrpt:  [381.17 MiB/s 382.62 MiB/s 383.96 MiB/s]
transpose_nonsquare/nonsquare/1024x3072 (3M elements, 16→48 kHz, 16x
base)                                                                              
    time:   [64.701 ms 65.080 ms 65.495 ms]
    thrpt:  [366.44 MiB/s 368.78 MiB/s 370.94 MiB/s]
transpose_nonsquare/nonsquare/1120x3072 (3M elements, 16→44.1 kHz, 16x
base)                                                                              
    time:   [13.818 ms 13.930 ms 14.052 ms]
    thrpt:  [1.8243 GiB/s 1.8403 GiB/s 1.8552 GiB/s
```