"""Tests for adaptive spectral subtraction and noise floor estimation."""

import numpy as np

from acoustix.dsp.spectral_sub import SpectralSubtractor


def test_spectral_sub_noise_floor_tracking() -> None:
    """Verify recursive noise floor estimation converges towards stationary noise."""
    n_bins = 1025
    subtractor = SpectralSubtractor(alpha=0.9, oversubtraction=1.0, beta=0.02)

    noise_profile = np.full(n_bins, 0.5)
    for _ in range(50):
        subtractor.process(noise_profile)

    assert subtractor.noise_floor is not None
    assert np.allclose(subtractor.noise_floor, noise_profile, atol=1e-2)


def test_spectral_sub_spectral_floor_bounds() -> None:
    """Verify spectral floor beta prevents negative values and excessive attenuation."""
    n_bins = 1025
    subtractor = SpectralSubtractor(alpha=0.95, oversubtraction=1.5, beta=0.05)

    noise_profile = np.full(n_bins, 1.0)
    for _ in range(30):
        cleaned = subtractor.process(noise_profile)

    # In stationary conditions where Y == noise_floor and oversubtraction > 1,
    # result should hit the beta * Y floor
    expected_floor = 0.05 * 1.0
    assert np.all(cleaned >= expected_floor - 1e-6)


def test_spectral_sub_tone_preservation() -> None:
    """Verify spectral subtraction preserves dominant harmonic while suppressing noise."""
    n_bins = 1025
    subtractor = SpectralSubtractor(alpha=0.95, oversubtraction=1.0, beta=0.02)

    noise_level = 0.1
    # Establish noise floor over 40 frames
    for _ in range(40):
        subtractor.process(np.full(n_bins, noise_level))

    # Frame with drone harmonic at bin 40
    frame = np.full(n_bins, noise_level)
    frame[40] = 2.0  # Dominant drone peak

    cleaned = subtractor.process(frame)

    # Noise bins attenuated significantly
    assert cleaned[10] < noise_level * 0.1
    # Tone preserved
    assert cleaned[40] > 1.8


def test_spectral_sub_2d_and_complex() -> None:
    """Verify 2D spectrogram processing and phase preservation on complex spectrum."""
    n_bins = 64
    n_frames = 20
    mag_spectrogram = np.full((n_bins, n_frames), 0.2)

    subtractor = SpectralSubtractor(alpha=0.9, oversubtraction=1.0, beta=0.02)
    cleaned_2d = subtractor.process(mag_spectrogram)
    assert cleaned_2d.shape == (n_bins, n_frames)

    # Complex frame with non-zero phase
    complex_frame = 0.2 * np.exp(1j * np.pi / 4) * np.ones(n_bins, dtype=np.complex128)
    subtractor.reset()
    cleaned_complex = subtractor.process(complex_frame)
    assert np.iscomplexobj(cleaned_complex)
    assert np.allclose(np.angle(cleaned_complex), np.pi / 4)
