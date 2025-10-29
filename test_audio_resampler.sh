#!/bin/bash

IN_RATE=44100
OUT_RATE=48000
FILTER="fir"
LATENCY=64
ATTENUATION=120

if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    if python -c 'import sys; exit(0 if sys.version_info >= (3, 0) else 1)' 2>/dev/null; then
        PYTHON=python
    else
        echo "Error: Python 3 is required but not found"
        echo "Please install Python 3 or ensure it's in your PATH"
        exit 1
    fi
else
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3"
    exit 1
fi

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --filter <linear|hermite|fir|fft>  Filter type (default: fir)"
    echo "  --latency <8|16|32|64>             Latency for FIR filter (default: 64)"
    echo "  --attenuation <60|90|120>          Attenuation for FIR filter (default: 90)"
    echo "  --input-rate <rate>                Input sample rate (default: 22050)"
    echo "  --output-rate <rate>               Output sample rate (default: 48000)"
    echo "  --help                             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --filter fft --input-rate 44100 --output-rate 48000"
    echo "  $0 --filter fir --latency 16 --attenuation 90"
    echo "  $0 --filter hermite --input-rate 44100 --output-rate 48000"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --latency)
            LATENCY="$2"
            shift 2
            ;;
        --attenuation)
            ATTENUATION="$2"
            shift 2
            ;;
        --input-rate)
            IN_RATE="$2"
            shift 2
            ;;
        --output-rate)
            OUT_RATE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [[ "$FILTER" != "linear" && "$FILTER" != "hermite" && "$FILTER" != "fir" && "$FILTER" != "fft" ]]; then
    echo "Error: Invalid filter type '$FILTER'. Must be linear, hermite, fir, or fft"
    exit 1
fi

CLI_ARGS="--filter $FILTER --sample-rate=$OUT_RATE"

if [[ "$FILTER" == "fir" ]]; then
    CLI_ARGS="$CLI_ARGS --latency $LATENCY --attenuation $ATTENUATION"
fi

echo "Testing ${IN_RATE} Hz -> ${OUT_RATE} Hz using ${FILTER} filter"
if [[ "$FILTER" == "fir" ]]; then
    echo "FIR settings: latency=${LATENCY}, attenuation=${ATTENUATION}dB"
fi
echo ""

$PYTHON test_audio_resampler.py generate --input-rate $IN_RATE --output-rate $OUT_RATE

echo "Resampling impulse test..."
cargo run --release --package resample -- $CLI_ARGS test_impulse.wav test_impulse_resampled.wav

echo "Resampling sweep test..."
cargo run --release --package resample -- $CLI_ARGS test_sweep.wav test_sweep_resampled.wav

echo "Analyzing results..."
$PYTHON test_audio_resampler.py analyze --input-rate $IN_RATE --output-rate $OUT_RATE

OUTPUT_NAME="analysis_${IN_RATE}_to_${OUT_RATE}_${FILTER}"
if [[ "$FILTER" == "fir" ]]; then
    OUTPUT_NAME="${OUTPUT_NAME}_L${LATENCY}_A${ATTENUATION}"
fi
OUTPUT_NAME="${OUTPUT_NAME}.png"

mv resampler_analysis.png "$OUTPUT_NAME"
echo "Analysis saved to: $OUTPUT_NAME"


rm *.wav

echo "Done!"
