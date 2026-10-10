"""High-pass Butterworth filtering for low-frequency noise suppression."""

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt


class StreamingHighPassFilter:
    """Stateful 4th-order Butterworth high-pass filter using Second-Order Sections (SOS).

    Maintains internal filter state across chunks for block-by-block processing
    without boundary discontinuities.
    """

    def __init__(self, cutoff: float = 350.0, fs: int = 48000, order: int = 4) -> None:
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
        self._zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter an incoming chunk of audio samples and update internal state."""
        chunk = np.asarray(chunk, dtype=np.float64)
        filtered, self._zi = sosfilt(self.sos, chunk, zi=self._zi)
        return filtered

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        return self.process(chunk)

    def reset(self) -> None:
        """Reset internal filter state to zero."""
        self._zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)


def offline_highpass_filter(
    data: np.ndarray,
    cutoff: float = 350.0,
    fs: int = 48000,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase 4th-order Butterworth high-pass filter using sosfiltfilt."""
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(data, dtype=np.float64))
