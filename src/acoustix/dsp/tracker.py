"""Harmonic tracking and Harmonic Product Spectrum (HPS) for RPM estimation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackResult:
    """Harmonic tracking result for a single spectral frame."""

    bpf: float
    rpm: float
    confidence: float
    harmonics: list[float]
    snr: float


def compute_hps(
    magnitude_db: np.ndarray,
    freqs: np.ndarray,
    num_harmonics: int = 4,
    min_freq: float = 500.0,
    max_freq: float = 1500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Harmonic Product Spectrum (HPS) in decibels over a candidate band.

    Sums decibel spectrum across harmonic ranks k in 1..num_harmonics,
    equivalent to multiplication in linear magnitude.
    """
    mag = np.asarray(magnitude_db, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)

    if mag.ndim != 1 or f.ndim != 1:
        raise ValueError(f"Expected 1D arrays, got mag.ndim={mag.ndim}, f.ndim={f.ndim}")
    if len(mag) != len(f):
        raise ValueError(
            f"Array length mismatch: len(magnitude_db)={len(mag)} != len(freqs)={len(f)}"
        )
    if min_freq >= max_freq:
        raise ValueError(f"min_freq ({min_freq}) must be < max_freq ({max_freq})")
    if num_harmonics < 1:
        raise ValueError(f"num_harmonics must be >= 1, got {num_harmonics}")

    mask = (f >= min_freq) & (f <= max_freq)
    candidate_freqs = f[mask]
    if len(candidate_freqs) == 0:
        raise ValueError(f"No frequency bins found in range [{min_freq}, {max_freq}]")

    hps = np.zeros_like(candidate_freqs, dtype=np.float64)
    for k in range(1, num_harmonics + 1):
        hps += np.interp(k * candidate_freqs, f, mag)

    return hps, candidate_freqs


class HarmonicTracker:
    """Tracks Blade Passing Frequency (BPF) and RPM across audio frames."""

    def __init__(
        self,
        num_blades: int = 3,
        min_rpm: float = 10000.0,
        max_rpm: float = 30000.0,
        num_harmonics: int = 4,
        smoothing_factor: float = 0.3,
        max_rpm_jump_per_frame: float = 2500.0,
    ) -> None:
        if num_blades < 1:
            raise ValueError(f"num_blades must be >= 1, got {num_blades}")
        if min_rpm <= 0.0 or max_rpm <= min_rpm:
            raise ValueError(f"Invalid RPM range: min_rpm={min_rpm}, max_rpm={max_rpm}")
        if not (0.0 < smoothing_factor <= 1.0):
            raise ValueError(f"smoothing_factor must be in (0.0, 1.0], got {smoothing_factor}")
        if max_rpm_jump_per_frame <= 0.0:
            raise ValueError(f"max_rpm_jump_per_frame must be > 0.0, got {max_rpm_jump_per_frame}")

        self.num_blades = num_blades
        self.min_rpm = min_rpm
        self.max_rpm = max_rpm
        self.num_harmonics = num_harmonics
        self.smoothing_factor = smoothing_factor
        self.max_rpm_jump_per_frame = max_rpm_jump_per_frame

        self._smoothed_bpf: float | None = None
        self._smoothed_rpm: float | None = None

    def reset(self) -> None:
        """Reset internal tracking state."""
        self._smoothed_bpf = None
        self._smoothed_rpm = None

    def process_frame(self, magnitude_db: np.ndarray, freqs: np.ndarray) -> TrackResult:
        """Process a single spectral frame and estimate BPF, RPM, and confidence."""
        min_freq = self.min_rpm * self.num_blades / 60.0
        max_freq = self.max_rpm * self.num_blades / 60.0

        hps, candidate_freqs = compute_hps(
            magnitude_db=magnitude_db,
            freqs=freqs,
            num_harmonics=self.num_harmonics,
            min_freq=min_freq,
            max_freq=max_freq,
        )

        peak_idx = int(np.argmax(hps))

        # Parabolic interpolation for sub-bin frequency refinement
        if 0 < peak_idx < len(hps) - 1:
            alpha = hps[peak_idx - 1]
            beta = hps[peak_idx]
            gamma = hps[peak_idx + 1]
            denom = alpha - 2.0 * beta + gamma
            if denom < 0.0:
                delta = 0.5 * (alpha - gamma) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
                df = (candidate_freqs[peak_idx + 1] - candidate_freqs[peak_idx - 1]) / 2.0
                cand_bpf = float(candidate_freqs[peak_idx] + delta * df)
            else:
                cand_bpf = float(candidate_freqs[peak_idx])
        else:
            cand_bpf = float(candidate_freqs[peak_idx])

        cand_bpf = float(np.clip(cand_bpf, min_freq, max_freq))

        # Peak prominence over HPS median floor normalized across harmonic ranks
        hps_peak = float(hps[peak_idx])
        hps_floor = float(np.median(hps))
        norm_prom = (hps_peak - hps_floor) / self.num_harmonics
        confidence = float(np.clip((norm_prom - 7.0) / 15.0, 0.0, 1.0))

        raw_rpm = 60.0 * cand_bpf / self.num_blades

        # Continuity constraint against previous smoothed state
        if self._smoothed_bpf is None:
            self._smoothed_bpf = cand_bpf
            self._smoothed_rpm = raw_rpm
        else:
            rpm_diff = abs(raw_rpm - self._smoothed_rpm)
            if rpm_diff > self.max_rpm_jump_per_frame and confidence < 0.5:
                cand_bpf = self._smoothed_bpf

            self._smoothed_bpf = (
                self.smoothing_factor * cand_bpf
                + (1.0 - self.smoothing_factor) * self._smoothed_bpf
            )
            self._smoothed_rpm = 60.0 * self._smoothed_bpf / self.num_blades

        bpf = self._smoothed_bpf
        rpm = self._smoothed_rpm
        harmonics = [float((k + 1) * bpf) for k in range(self.num_harmonics)]

        # Estimate SNR of the fundamental peak against local spectrum
        f = np.asarray(freqs, dtype=np.float64)
        mag = np.asarray(magnitude_db, dtype=np.float64)
        peak_mask = np.abs(f - bpf) <= 30.0
        peak_db = (
            float(np.max(mag[peak_mask])) if np.any(peak_mask) else float(np.interp(bpf, f, mag))
        )

        local_mask = (f >= bpf - 200.0) & (f <= bpf + 200.0) & (np.abs(f - bpf) > 40.0)
        if np.sum(local_mask) < 3:
            local_mask = (f >= min_freq) & (f <= max_freq) & (np.abs(f - bpf) > 40.0)

        noise_floor_db = (
            float(np.median(mag[local_mask])) if np.any(local_mask) else float(np.median(mag))
        )
        snr = float(max(0.0, peak_db - noise_floor_db))

        return TrackResult(
            bpf=bpf,
            rpm=rpm,
            confidence=confidence,
            harmonics=harmonics,
            snr=snr,
        )

    def __call__(self, magnitude_db: np.ndarray, freqs: np.ndarray) -> TrackResult:
        return self.process_frame(magnitude_db, freqs)
