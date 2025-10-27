#!/bin/bash
for in_rate in 22050 44100 48000; do
    for out_rate in 44100 48000; do
        if [ $in_rate -ne $out_rate ]; then
            echo "Testing ${in_rate} -> ${out_rate}"
            python test_audio_resampler.py generate --input-rate $in_rate --output-rate $out_rate
            cargo run --release --package resample -- --sample-rate=$out_rate test_impulse.wav test_impulse_resampled.wav
            cargo run --release --package resample -- --sample-rate=$out_rate test_sweep.wav test_sweep_resampled.wav
            python test_audio_resampler.py analyze --input-rate $in_rate --output-rate $out_rate
            mv resampler_analysis.png analysis_${in_rate}_to_${out_rate}.png
        fi
    done
done

rm *.wav
