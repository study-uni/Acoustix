"""Adaptive spectral subtraction for background noise rejection."""

import numpy as np


class SpectralSubtractor:
    """Adaptive background noise floor estimation and spectral subtraction.

    Estimates stationary background noise floor using recursive smoothing:
        N_hat(f, t) = alpha * N_hat(f, t-1) + (1 - alpha) * |Y(f, t)|

    Attenuates noise using over-subtraction with a spectral floor factor beta:
        |X_hat(f, t)| = max(|Y(f, t)| - oversubtraction * N_hat(f, t), beta * |Y(f, t)|)
    to prevent musical noise artifacts.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        oversubtraction: float = 1.0,
        beta: float = 0.02,
        noise_floor: np.ndarray | None = None,
    ) -> None:
        self.alpha = alpha
        self.oversubtraction = oversubtraction
        self.beta = beta
        self.noise_floor = np.copy(noise_floor) if noise_floor is not None else None

    def _process_frame(self, frame: np.ndarray, update_noise: bool = True) -> np.ndarray:
        """Process a single 1D spectral frame."""
        is_complex = np.iscomplexobj(frame)
        magnitude = np.abs(frame) if is_complex else np.asarray(frame, dtype=np.float64)

        if update_noise:
            if self.noise_floor is None:
                self.noise_floor = np.copy(magnitude)
            else:
                self.noise_floor = self.alpha * self.noise_floor + (1.0 - self.alpha) * magnitude

        if self.noise_floor is None:
            cleaned_mag = magnitude
        else:
            cleaned_mag = np.maximum(
                magnitude - self.oversubtraction * self.noise_floor,
                self.beta * magnitude,
            )

        if is_complex:
            return cleaned_mag * np.exp(1j * np.angle(frame))
        return cleaned_mag

    def process(self, spectrum: np.ndarray, update_noise: bool = True) -> np.ndarray:
        """Apply spectral subtraction to a 1D frame or 2D spectrogram.

        Parameters
        ----------
        spectrum : np.ndarray
            Magnitude or complex spectrum. Shape (n_bins,) for a single frame,
            or (n_bins, n_frames) for a 2D spectrogram across time.
        update_noise : bool, default True
            Whether to update recursive noise estimate.

        Returns
        -------
        cleaned : np.ndarray
            Noise-attenuated spectrum with the same shape as input.
        """
        arr = np.asarray(spectrum)
        if arr.ndim == 1:
            return self._process_frame(arr, update_noise=update_noise)
        elif arr.ndim == 2:
            n_bins, n_frames = arr.shape
            out = np.empty_like(arr)
            for t in range(n_frames):
                out[:, t] = self._process_frame(arr[:, t], update_noise=update_noise)
            return out
        else:
            raise ValueError(f"Expected 1D or 2D array, got ndim={arr.ndim}")

    def __call__(self, spectrum: np.ndarray, update_noise: bool = True) -> np.ndarray:
        return self.process(spectrum, update_noise=update_noise)

    def reset(self, noise_floor: np.ndarray | None = None) -> None:
        """Reset internal noise floor estimate."""
        self.noise_floor = np.copy(noise_floor) if noise_floor is not None else None
