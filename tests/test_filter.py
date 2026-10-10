"""Tests for Butterworth high-pass filtering and streaming processor."""

import numpy as np
from scipy.signal import sosfilt

from acoustix.dsp.filter import StreamingHighPassFilter, offline_highpass_filter


def test_offline_wind_attenuation() -> None:
    """Verify offline filter attenuates 100 Hz >= 30 dB and preserves 1000 Hz (< 0.5 dB loss)."""
    fs = 48000
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs
    sig_wind = np.sin(2 * np.pi * 100 * t)
    sig_drone = np.sin(2 * np.pi * 1000 * t)
    composite = sig_wind + sig_drone

    filtered = offline_highpass_filter(composite, cutoff=350.0, fs=fs)

    fft_filtered = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)

    bin_100 = np.argmin(np.abs(freqs - 100))
    bin_1000 = np.argmin(np.abs(freqs - 1000))

    mag_100 = np.abs(fft_filtered[bin_100])
    mag_1000 = np.abs(fft_filtered[bin_1000])

    atten_db = 20.0 * np.log10(mag_1000 / mag_100)
    assert atten_db >= 30.0

    fft_orig_drone = np.fft.rfft(sig_drone)
    loss_1000 = 20.0 * np.log10(np.abs(fft_orig_drone[bin_1000]) / mag_1000)
    assert loss_1000 < 0.5


def test_streaming_wind_attenuation() -> None:
    """Verify StreamingHighPassFilter on chunks attenuates wind and preserves drone frequency."""
    fs = 48000
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs
    sig_wind = np.sin(2 * np.pi * 100 * t)
    sig_drone = np.sin(2 * np.pi * 1000 * t)
    composite = sig_wind + sig_drone

    flt = StreamingHighPassFilter(cutoff=350.0, fs=fs)
    chunk_size = 512
    streamed_chunks = []
    for i in range(0, len(composite), chunk_size):
        streamed_chunks.append(flt(composite[i : i + chunk_size]))
    filtered = np.concatenate(streamed_chunks)

    # Evaluate steady-state response (skip initial transient)
    steady = filtered[4800:]
    fft_steady = np.fft.rfft(steady * np.hanning(len(steady)))
    freqs = np.fft.rfftfreq(len(steady), d=1.0 / fs)

    bin_100 = np.argmin(np.abs(freqs - 100))
    bin_1000 = np.argmin(np.abs(freqs - 1000))

    atten_db = 20.0 * np.log10(np.abs(fft_steady[bin_1000]) / np.abs(fft_steady[bin_100]))
    assert atten_db >= 30.0

    orig_steady = sig_drone[4800:]
    fft_orig = np.fft.rfft(orig_steady * np.hanning(len(orig_steady)))
    loss_1000 = 20.0 * np.log10(np.abs(fft_orig[bin_1000]) / np.abs(fft_steady[bin_1000]))
    assert loss_1000 < 0.5


def test_streaming_filter_matches_continuous() -> None:
    """Verify output of StreamingHighPassFilter on chunks matches continuous filtering."""
    fs = 48000
    np.random.seed(42)
    signal = np.random.randn(48000)

    flt = StreamingHighPassFilter(cutoff=350.0, fs=fs)
    continuous_filtered = sosfilt(flt.sos, signal)

    chunk_size = 512
    flt.reset()
    streamed_chunks = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i : i + chunk_size]
        streamed_chunks.append(flt(chunk))

    streamed_filtered = np.concatenate(streamed_chunks)
    assert np.allclose(continuous_filtered, streamed_filtered)


def test_streaming_filter_reset() -> None:
    """Verify filter reset restores initial state."""
    flt = StreamingHighPassFilter(cutoff=350.0, fs=48000)
    chunk = np.random.randn(512)

    out1 = flt(chunk)
    flt.reset()
    out2 = flt(chunk)

    assert np.allclose(out1, out2)
