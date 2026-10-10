"""Tests for Harmonic Product Spectrum (HPS) and HarmonicTracker."""

import numpy as np
import pytest

from acoustix.dsp.filter import StreamingHighPassFilter
from acoustix.dsp.stft import StreamingSTFT
from acoustix.dsp.tracker import HarmonicTracker, TrackResult, compute_hps
from acoustix.synth.generator import generate_scenario


def test_pure_synthetic_tone_with_harmonics() -> None:
    """Verify tracker detects 1200 Hz fundamental + harmonics at 24000 RPM (+/- 0.5%)."""
    fs = 48000
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs
    fundamental = 1200.0

    sig = (
        np.sin(2 * np.pi * fundamental * t)
        + 0.5 * np.sin(2 * np.pi * 2 * fundamental * t)
        + 0.25 * np.sin(2 * np.pi * 3 * fundamental * t)
        + 0.1 * np.sin(2 * np.pi * 4 * fundamental * t)
    )

    n_fft = 2048
    window = np.hanning(n_fft)
    spec = np.fft.rfft(sig[:n_fft] * window, n=n_fft)
    magnitude_db = 20.0 * np.log10(np.abs(spec) + 1e-10)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    tracker = HarmonicTracker(num_blades=3)
    result = tracker.process_frame(magnitude_db, freqs)

    assert isinstance(result, TrackResult)
    # Verify BPF within +/- 0.5% of 1200 Hz
    assert abs(result.bpf - 1200.0) / 1200.0 < 0.005
    # Verify RPM within +/- 0.5% of 24000 RPM
    assert abs(result.rpm - 24000.0) / 24000.0 < 0.005

    # Verify detected harmonics list
    expected_harmonics = [1200.0, 2400.0, 3600.0, 4800.0]
    for det_h, exp_h in zip(result.harmonics, expected_harmonics, strict=True):
        assert abs(det_h - exp_h) / exp_h < 0.005

    # Strong harmonics should yield high confidence and SNR
    assert result.confidence > 0.85
    assert result.snr > 30.0


def test_drone_scenario_streaming_pipeline() -> None:
    """Verify mean detected RPM error < 1.5% at 21000 RPM with wind noise at 0 dB SNR."""
    fs = 48000
    target_rpm = 21000.0

    audio, _, _ = generate_scenario(
        duration=2.0,
        sample_rate=fs,
        profile_type="constant",
        base_rpm=target_rpm,
        snr_db=0.0,
        blades=3,
        seed=42,
    )

    hp_filter = StreamingHighPassFilter(cutoff=350.0, fs=fs)
    stft = StreamingSTFT(n_fft=2048, hop_length=512, fs=fs)
    tracker = HarmonicTracker(num_blades=3)

    results = []
    chunk_size = 512
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i : i + chunk_size]
        filtered = hp_filter(chunk)
        mag_db = stft(filtered)
        track_res = tracker.process_frame(mag_db, stft.frequencies)
        results.append(track_res)

    # Discard initial transient priming frames (n_fft // hop_length)
    steady_results = results[4:]
    detected_rpms = [r.rpm for r in steady_results]
    mean_rpm = float(np.mean(detected_rpms))

    relative_error = abs(mean_rpm - target_rpm) / target_rpm
    assert relative_error < 0.015

    # Most steady frames should have high confidence and positive SNR
    confs = [r.confidence for r in steady_results]
    assert np.mean(confs) > 0.7
    snrs = [r.snr for r in steady_results]
    assert np.mean(snrs) > 15.0


def test_custom_blade_configuration_and_rpm_limits() -> None:
    """Verify 2-blade and 4-blade configurations and RPM search limits."""
    fs = 48000
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs
    n_fft = 2048
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # 1. 2-blade drone at 24000 RPM -> BPF = 2 * 24000 / 60 = 800 Hz
    sig_2blade = (
        np.sin(2 * np.pi * 800.0 * t)
        + 0.5 * np.sin(2 * np.pi * 1600.0 * t)
        + 0.25 * np.sin(2 * np.pi * 2400.0 * t)
    )
    spec_2blade = np.fft.rfft(sig_2blade[:n_fft] * window, n=n_fft)
    mag_2blade = 20.0 * np.log10(np.abs(spec_2blade) + 1e-10)

    tracker_2blade = HarmonicTracker(num_blades=2, min_rpm=10000.0, max_rpm=30000.0)
    res_2blade = tracker_2blade(mag_2blade, freqs)
    assert abs(res_2blade.bpf - 800.0) / 800.0 < 0.005
    assert abs(res_2blade.rpm - 24000.0) / 24000.0 < 0.005

    # 2. 4-blade drone at 24000 RPM -> BPF = 4 * 24000 / 60 = 1600 Hz
    sig_4blade = (
        np.sin(2 * np.pi * 1600.0 * t)
        + 0.5 * np.sin(2 * np.pi * 3200.0 * t)
        + 0.25 * np.sin(2 * np.pi * 4800.0 * t)
    )
    spec_4blade = np.fft.rfft(sig_4blade[:n_fft] * window, n=n_fft)
    mag_4blade = 20.0 * np.log10(np.abs(spec_4blade) + 1e-10)

    tracker_4blade = HarmonicTracker(num_blades=4, min_rpm=10000.0, max_rpm=30000.0)
    res_4blade = tracker_4blade(mag_4blade, freqs)
    assert abs(res_4blade.bpf - 1600.0) / 1600.0 < 0.005
    assert abs(res_4blade.rpm - 24000.0) / 24000.0 < 0.005

    # 3. RPM limits: signal at 800 Hz (24000 RPM for 2 blades) should not be tracked
    # if search band is set to [10000, 20000] RPM (BPF band [333, 666] Hz)
    tracker_limited = HarmonicTracker(num_blades=2, min_rpm=10000.0, max_rpm=20000.0)
    res_limited = tracker_limited(mag_2blade, freqs)
    # The peak at 800 Hz is outside [333, 666] Hz, so tracked BPF must remain <= 666.67 Hz
    assert res_limited.bpf <= 20000.0 * 2 / 60.0 + 1.0


def test_confidence_on_pure_noise() -> None:
    """Verify confidence metrics on pure white noise are low (< 0.2)."""
    fs = 48000
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    tracker = HarmonicTracker(num_blades=3)

    rng = np.random.default_rng(1234)
    confs = []
    for _ in range(30):
        noise = rng.normal(0.0, 1.0, size=n_fft) * np.hanning(n_fft)
        spec = np.fft.rfft(noise, n=n_fft)
        mag_db = 20.0 * np.log10(np.abs(spec) + 1e-10)
        res = tracker.process_frame(mag_db, freqs)
        confs.append(res.confidence)

    assert float(np.mean(confs)) < 0.15
    assert float(np.max(confs)) < 0.2


def test_continuity_constraint_and_outlier_rejection() -> None:
    """Verify tracker rejects sudden outlier jumps when frame confidence is low."""
    fs = 48000
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    window = np.hanning(n_fft)
    t = np.arange(n_fft) / fs

    tracker = HarmonicTracker(num_blades=3, max_rpm_jump_per_frame=2000.0)

    # Establish steady state at 24000 RPM (BPF 1200 Hz)
    sig_steady = (
        np.sin(2 * np.pi * 1200.0 * t)
        + 0.5 * np.sin(2 * np.pi * 2400.0 * t)
        + 0.25 * np.sin(2 * np.pi * 3600.0 * t)
        + 0.1 * np.sin(2 * np.pi * 4800.0 * t)
    )
    frame_steady = 20.0 * np.log10(np.abs(np.fft.rfft(sig_steady * window)) + 1e-10)

    res1 = tracker.process_frame(frame_steady, freqs)
    assert abs(res1.rpm - 24000.0) < 100.0

    # Inject pure noise with a small peak far away (e.g. at 600 Hz -> 12000 RPM, jump = 12000 RPM)
    noise = np.random.default_rng(99).normal(0.0, 1.0, size=n_fft) * window
    frame_noise = 20.0 * np.log10(np.abs(np.fft.rfft(noise)) + 1e-10)

    res_outlier = tracker.process_frame(frame_noise, freqs)
    # Confidence is low, jump exceeds max_rpm_jump_per_frame, outlier rejected
    assert res_outlier.confidence < 0.2
    assert abs(res_outlier.rpm - 24000.0) < 100.0

    # Test reset clears smoothed state
    tracker.reset()
    assert tracker._smoothed_bpf is None
    assert tracker._smoothed_rpm is None


def test_compute_hps_validation() -> None:
    """Verify input validation and error handling in compute_hps and HarmonicTracker."""
    freqs = np.linspace(0, 24000, 1025)
    mag = np.zeros(1025)

    with pytest.raises(ValueError, match="min_freq .* must be < max_freq"):
        compute_hps(mag, freqs, min_freq=1500.0, max_freq=500.0)

    with pytest.raises(ValueError, match="num_harmonics must be >= 1"):
        compute_hps(mag, freqs, num_harmonics=0)

    with pytest.raises(ValueError, match="Expected 1D arrays"):
        compute_hps(np.zeros((10, 10)), freqs)

    with pytest.raises(ValueError, match="Array length mismatch"):
        compute_hps(mag[:500], freqs)

    with pytest.raises(ValueError, match="num_blades must be >= 1"):
        HarmonicTracker(num_blades=0)

    with pytest.raises(ValueError, match="Invalid RPM range"):
        HarmonicTracker(min_rpm=20000.0, max_rpm=10000.0)
