"""Tests for Short-Time Fourier Transform (STFT) and StreamingSTFT."""

import numpy as np
import pytest

from acoustix.dsp.stft import StreamingSTFT, compute_stft


def test_stft_frequency_resolution() -> None:
    """Verify STFT frequency resolution and bin count."""
    fs = 48000
    n_fft = 2048
    hop_length = 512
    expected_delta_f = fs / n_fft  # 23.4375 Hz

    signal = np.random.randn(fs)
    frequencies, _, _ = compute_stft(signal, n_fft=n_fft, hop_length=hop_length, fs=fs)

    assert len(frequencies) == n_fft // 2 + 1
    assert np.isclose(frequencies[0], 0.0)
    assert np.isclose(frequencies[-1], fs / 2.0)
    assert np.allclose(np.diff(frequencies), expected_delta_f)


def test_stft_bin_centering() -> None:
    """Verify that pure tone peaks at the exact frequency bin center and time bins are centered."""
    fs = 48000
    n_fft = 2048
    hop_length = 512
    delta_f = fs / n_fft

    target_bin = 40
    target_freq = target_bin * delta_f  # 937.5 Hz

    duration = 1.0
    t = np.arange(int(fs * duration)) / fs
    pure_tone = np.cos(2 * np.pi * target_freq * t)

    frequencies, times, magnitude_db = compute_stft(
        pure_tone, n_fft=n_fft, hop_length=hop_length, fs=fs
    )

    # Frequency bin centering: all frame peaks must align with target_bin
    peak_bins = np.argmax(magnitude_db, axis=0)
    assert np.all(peak_bins == target_bin)
    assert np.isclose(frequencies[target_bin], target_freq)

    # Time bin centering: verify window centers and frame hop spacing
    assert np.isclose(times[0], (n_fft / 2.0) / fs)
    assert np.allclose(np.diff(times), hop_length / fs)


def test_streaming_stft_matches_batch() -> None:
    """Verify StreamingSTFT rolling buffer frames match batch STFT frames."""
    fs = 48000
    n_fft = 2048
    hop_length = 512

    # Use length aligned with hop_length
    num_chunks = 96
    signal = np.random.randn(num_chunks * hop_length)

    _, _, batch_mag = compute_stft(signal, n_fft=n_fft, hop_length=hop_length, fs=fs)

    streamer = StreamingSTFT(n_fft=n_fft, hop_length=hop_length, fs=fs)
    chunks = [signal[i * hop_length : (i + 1) * hop_length] for i in range(num_chunks)]
    stream_frames = list(streamer.stream(chunks))

    # Rolling buffer of length N is primed after N // H chunks
    offset = n_fft // hop_length - 1
    num_batch_frames = batch_mag.shape[1]
    stream_aligned = np.array(stream_frames[offset : offset + num_batch_frames]).T

    assert np.allclose(batch_mag, stream_aligned)


def test_streaming_stft_validation_and_reset() -> None:
    """Verify chunk size validation and rolling buffer reset."""
    streamer = StreamingSTFT(n_fft=2048, hop_length=512)

    with pytest.raises(ValueError, match="Expected chunk size 512"):
        streamer.process(np.zeros(256))

    chunk = np.random.randn(512)
    frame1 = streamer.process(chunk)
    streamer.reset()
    frame2 = streamer.process(chunk)

    assert np.allclose(frame1, frame2)
