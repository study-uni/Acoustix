"""Short-Time Fourier Transform (STFT) processing for time-frequency analysis."""

from collections.abc import Iterable, Iterator

import numpy as np


class StreamingSTFT:
    """Streaming Short-Time Fourier Transform processor with a rolling buffer.

    Maintains a rolling buffer of length `n_fft` and computes the STFT frame
    upon receiving each incoming audio chunk of size `hop_length`.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        fs: int = 48000,
        eps: float = 1e-10,
    ) -> None:
        if not (0 < hop_length <= n_fft):
            raise ValueError(
                f"hop_length must satisfy 0 < hop_length <= n_fft, got {hop_length} and {n_fft}"
            )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fs = fs
        self.eps = eps
        self.window = np.hanning(n_fft)
        self.frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        self._buffer = np.zeros(n_fft, dtype=np.float64)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Accept incoming audio chunk of size `hop_length` and return STFT frame in dB."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if len(chunk) != self.hop_length:
            raise ValueError(f"Expected chunk size {self.hop_length}, got {len(chunk)}")

        # Shift rolling buffer and insert incoming chunk
        self._buffer[: -self.hop_length] = self._buffer[self.hop_length :]
        self._buffer[-self.hop_length :] = chunk

        windowed = self._buffer * self.window
        spec = np.fft.rfft(windowed, n=self.n_fft)
        return 20.0 * np.log10(np.abs(spec) + self.eps)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        return self.process(chunk)

    def stream(self, chunks: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        """Yield STFT frames for an iterable stream of audio chunks."""
        for chunk in chunks:
            yield self.process(chunk)

    def reset(self) -> None:
        """Reset internal rolling buffer to zeros."""
        self._buffer.fill(0.0)


def compute_stft(
    signal: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    fs: int = 48000,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform (STFT) magnitude in decibels.

    Parameters
    ----------
    signal : np.ndarray
        1D input audio signal.
    n_fft : int, default 2048
        FFT window length.
    hop_length : int, default 512
        Hop size between successive frames.
    fs : int, default 48000
        Sampling frequency in Hz.
    eps : float, default 1e-10
        Small epsilon preventing log(0).

    Returns
    -------
    frequencies : np.ndarray
        Bin center frequencies of shape (n_fft // 2 + 1,).
    times : np.ndarray
        Frame center timestamps of shape (num_frames,).
    magnitude_db : np.ndarray
        Spectrogram magnitude in dB of shape (n_fft // 2 + 1, num_frames).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < n_fft:
        raise ValueError(f"Signal length ({len(signal)}) must be >= n_fft ({n_fft})")

    num_frames = (len(signal) - n_fft) // hop_length + 1
    window = np.hanning(n_fft)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    times = (np.arange(num_frames) * hop_length + n_fft / 2.0) / fs

    frames = np.lib.stride_tricks.sliding_window_view(signal, window_shape=n_fft)[::hop_length]
    windowed = frames * window
    spec = np.fft.rfft(windowed, n=n_fft, axis=-1)
    magnitude_db = 20.0 * np.log10(np.abs(spec) + eps).T

    return frequencies, times, magnitude_db
