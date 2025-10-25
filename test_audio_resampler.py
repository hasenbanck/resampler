#!/usr/bin/env python3
"""
Audio Resampler Quality Testing Suite (Simplified)

This script generates test signals to evaluate audio resampler quality and
visualizes the results with filter frequency response and sweep spectrogram.

Usage:
    1. Generate test files:
       python test_audio_resampler.py generate --input-rate 22050 --output-rate 48000

    2. Run your resampler on the generated files:
       ./resampler test_impulse.wav test_impulse_resampled.wav
       ./resampler test_sweep.wav test_sweep_resampled.wav

    3. Analyze results:
       python test_audio_resampler.py analyze --input-rate 22050 --output-rate 48000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from pathlib import Path


class ResamplerTester:
    def __init__(self, input_rate, output_rate, duration=5.0, channels=2):
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.duration = duration
        self.channels = channels

    def generate_test_signals(self, output_dir="."):
        """Generate test signals as WAV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"Generating test signals at {self.input_rate} Hz...")
        print(f"Target output rate: {self.output_rate} Hz")
        print(f"Duration: {self.duration} seconds")
        print(f"Channels: {self.channels}")
        print()

        # 1. Impulse response test (for filter frequency response)
        impulse = self._generate_impulse()
        self._save_wav(output_dir / "test_impulse.wav", impulse)
        print(f"✓ Generated test_impulse.wav (for filter frequency response)")

        # 2. Sine sweep test (for spectrogram)
        sweep = self._generate_sweep()
        self._save_wav(output_dir / "test_sweep.wav", sweep)
        print(f"✓ Generated test_sweep.wav (for spectrogram)")

        print()
        print("Test signal generation complete!")
        print()
        print("Next steps:")
        print("1. Run your resampler on these files")
        print("2. Use 'analyze' command to visualize results")


    def _generate_impulse(self):
        """Generate impulse signal."""
        n_samples = int(self.duration * self.input_rate)
        impulse = np.zeros((n_samples, self.channels), dtype=np.float32)

        # Place impulse at 0.5 seconds (or middle if duration < 1 second)
        impulse_pos = min(int(0.5 * self.input_rate), n_samples - 1)
        impulse[impulse_pos, :] = 1.0

        return impulse

    def _generate_sweep(self):
        """Generate logarithmic sine sweep from 20 Hz to Nyquist."""
        n_samples = int(self.duration * self.input_rate)
        t = np.linspace(0, self.duration, n_samples)

        # Sweep from 20 Hz to 95% of Nyquist
        f0 = 20
        f1 = self.input_rate / 2 * 0.95

        # Logarithmic sweep
        sweep = signal.chirp(t, f0, self.duration, f1, method='logarithmic')

        # Apply fade in/out to reduce edge effects
        fade_samples = int(0.1 * self.input_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        sweep[:fade_samples] *= fade_in
        sweep[-fade_samples:] *= fade_out

        # Normalize and duplicate for channels
        sweep = sweep * 0.8  # Leave headroom
        return np.column_stack([sweep] * self.channels).astype(np.float32)


    def _save_wav(self, filename, data):
        """Save audio data as WAV file."""
        # Convert to int16 for WAV
        data_int16 = (data * 32767).astype(np.int16)
        wavfile.write(filename, self.input_rate, data_int16)

    def analyze_results(self, input_dir="."):
        """Analyze resampled output files and generate visualizations."""
        input_dir = Path(input_dir)

        print("Analyzing resampled outputs...")
        print()

        # Create figure with two plots stacked vertically
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'Audio Resampler Quality Analysis: {self.input_rate} Hz → {self.output_rate} Hz',
                     fontsize=16, fontweight='bold')

        # 1. Filter frequency response (from impulse) - top
        self._analyze_filter_frequency_response(fig, input_dir)

        # 2. Sweep spectrogram - bottom
        self._analyze_sweep_spectrogram(fig, input_dir)

        plt.tight_layout()
        output_file = input_dir / "resampler_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Analysis plot saved to: {output_file}")
        plt.show()

    def _analyze_filter_frequency_response(self, fig, input_dir):
        """Analyze impulse response to show filter frequency response."""
        impulse_file = input_dir / "test_impulse_resampled.wav"

        if not impulse_file.exists():
            print(f"⚠ Skipping filter frequency response: {impulse_file} not found")
            return

        rate, data = wavfile.read(impulse_file)

        # Convert to float and take first channel
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0

        if len(data.shape) > 1:
            data = data[:, 0]

        # Find the impulse peak
        peak_idx = np.argmax(np.abs(data))

        # Extract window around impulse
        window_size = int(0.1 * rate)  # 100ms window
        start = max(0, peak_idx - window_size // 2)
        end = min(len(data), start + window_size)
        impulse_response = data[start:end]

        # Plot frequency domain
        ax = plt.subplot(2, 1, 1)

        # FFT of impulse response
        n_fft = 8192
        freq_response = np.fft.rfft(impulse_response, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, 1/rate)

        # Convert to dB
        magnitude_db = 20 * np.log10(np.abs(freq_response) + 1e-10)

        ax.plot(freqs / 1000, magnitude_db, linewidth=0.8)
        ax.set_xlabel('Frequency (kHz)', fontsize=11)
        ax.set_ylabel('Magnitude (dB)', fontsize=11)
        ax.set_title('Filter Frequency Response', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-120, 10)

        # Mark passband and stopband
        nyquist_in = self.input_rate / 2
        ax.axvline(nyquist_in / 1000, color='r', linestyle='--',
                   alpha=0.5, linewidth=2, label=f'Input Nyquist ({nyquist_in:.0f} Hz)')

        # Add transition band shading
        cutoff = nyquist_in * 0.95  # Approximate passband edge
        ax.axvspan(0, cutoff / 1000, alpha=0.1, color='green', label='Passband')
        ax.axvspan(nyquist_in / 1000, rate / 2000, alpha=0.1, color='red', label='Stopband')

        ax.legend(loc='upper right', fontsize=9)

        # Measure key metrics
        passband_idx = np.argmax(freqs > cutoff * 0.5)
        cutoff_idx = np.argmax(freqs > cutoff)
        stopband_idx = np.argmax(freqs > nyquist_in)

        if stopband_idx < len(magnitude_db) and cutoff_idx > passband_idx:
            passband_ripple = np.ptp(magnitude_db[passband_idx:cutoff_idx])
            stopband_atten = np.max(magnitude_db[stopband_idx:])

            metrics_text = f'Passband ripple: ±{passband_ripple/2:.2f} dB\n'
            metrics_text += f'Stopband peak: {stopband_atten:.1f} dB'

            ax.text(0.02, 0.05, metrics_text, transform=ax.transAxes,
                    verticalalignment='bottom', fontsize=9, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        elif stopband_idx < len(magnitude_db):
            # Just show stopband if passband measurement fails
            stopband_atten = np.max(magnitude_db[stopband_idx:])

            metrics_text = f'Stopband peak: {stopband_atten:.1f} dB'

            ax.text(0.02, 0.05, metrics_text, transform=ax.transAxes,
                    verticalalignment='bottom', fontsize=9, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        print(f"✓ Analyzed filter frequency response")

    def _analyze_sweep_spectrogram(self, fig, input_dir):
        """Analyze frequency response from sweep."""
        sweep_file = input_dir / "test_sweep_resampled.wav"

        if not sweep_file.exists():
            print(f"⚠ Skipping frequency response: {sweep_file} not found")
            return

        rate, data = wavfile.read(sweep_file)

        # Convert to float
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0

        if len(data.shape) > 1:
            data = data[:, 0]

        # Compute spectrogram
        ax = plt.subplot(2, 1, 2)

        nperseg = 4096
        f, t, Sxx = signal.spectrogram(data, rate, nperseg=nperseg,
                                       noverlap=nperseg//2)

        # Plot spectrogram
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        im = ax.pcolormesh(t, f / 1000, Sxx_db, shading='gouraud',
                           cmap='viridis', vmin=-80, vmax=0)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Frequency (kHz)', fontsize=11)
        ax.set_title('Sweep Spectrogram', fontsize=13, fontweight='bold')

        # Mark input Nyquist
        nyquist_in = self.input_rate / 2
        ax.axhline(nyquist_in / 1000, color='r', linestyle='--',
                   alpha=0.7, linewidth=2, label=f'Input Nyquist ({nyquist_in:.0f} Hz)')

        # Add passband/stopband regions
        ax.axhspan(0, nyquist_in / 1000, alpha=0.05, color='green', zorder=0)
        ax.axhspan(nyquist_in / 1000, rate / 2000, alpha=0.1, color='red', zorder=0)

        ax.legend(loc='upper right', fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Magnitude (dB)')
        cbar.set_label('Magnitude (dB)', fontsize=10)

        print(f"✓ Analyzed sweep spectrogram")


def main():
    parser = argparse.ArgumentParser(
        description='Audio Resampler Quality Testing Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate test signals')
    gen_parser.add_argument('--input-rate', type=int, required=True,
                           help='Input sample rate (Hz)')
    gen_parser.add_argument('--output-rate', type=int, required=True,
                           help='Target output sample rate (Hz)')
    gen_parser.add_argument('--duration', type=float, default=5.0,
                           help='Duration of test signals (seconds, default: 5.0)')
    gen_parser.add_argument('--channels', type=int, default=2,
                           help='Number of channels (default: 2)')
    gen_parser.add_argument('--output-dir', type=str, default='.',
                           help='Output directory (default: current directory)')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze resampled outputs')
    analyze_parser.add_argument('--input-rate', type=int, required=True,
                               help='Original input sample rate (Hz)')
    analyze_parser.add_argument('--output-rate', type=int, required=True,
                               help='Target output sample rate (Hz)')
    analyze_parser.add_argument('--input-dir', type=str, default='.',
                               help='Directory with resampled files (default: current directory)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'generate':
        tester = ResamplerTester(
            input_rate=args.input_rate,
            output_rate=args.output_rate,
            duration=args.duration,
            channels=args.channels
        )
        tester.generate_test_signals(args.output_dir)

    elif args.command == 'analyze':
        tester = ResamplerTester(
            input_rate=args.input_rate,
            output_rate=args.output_rate
        )
        tester.analyze_results(args.input_dir)


if __name__ == '__main__':
    main()
