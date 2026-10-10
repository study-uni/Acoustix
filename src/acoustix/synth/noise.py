"""Acoustic noise generation for drone environment simulation."""

import numpy as np
from scipy import signal


def generate_ambient_noise(
    num_samples: int,
    amplitude: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """Generate stationary ambient noise floor (white Gaussian noise)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, amplitude, size=num_samples).astype(np.float64)


def generate_wind_noise(
    num_samples: int,
    sample_rate: int = 48000,
    cutoff_hz: float = 300.0,
    low_cutoff_hz: float | None = 20.0,
    order: int = 2,
    seed: int | None = None,
) -> np.ndarray:
    """Generate low-frequency wind turbulence noise concentrated in 20-300 Hz.

    Uses a Butterworth filter (bandpass 20-300 Hz or lowpass at 300 Hz) applied to
    white Gaussian noise, normalized to unit RMS.
    """
    if num_samples <= 0:
        return np.empty(0, dtype=np.float64)

    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, size=num_samples)

    if low_cutoff_hz is not None and low_cutoff_hz > 0.0:
        sos = signal.butter(
            N=order,
            Wn=[low_cutoff_hz, cutoff_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
    else:
        sos = signal.butter(
            N=order,
            Wn=cutoff_hz,
            btype="lowpass",
            fs=sample_rate,
            output="sos",
        )

    wind = signal.sosfilt(sos, white)
    rms = np.sqrt(np.mean(wind**2))
    if rms > 0.0:
        wind = wind / rms
    return wind.astype(np.float64)
