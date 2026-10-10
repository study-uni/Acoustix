"""Tests for synthetic audio generation: motors, noise, and scenario generator."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from acoustix.synth.generator import export_wav, generate_scenario
from acoustix.synth.motors import (
    generate_constant_profile,
    generate_linear_sweep_profile,
    generate_rpm_profile,
    generate_step_burst_profile,
    rpm_to_bpf,
    rpm_to_shaft_frequency,
    synthesize_motor_signal,
    synthesize_quad_motors,
)
from acoustix.synth.noise import generate_ambient_noise, generate_wind_noise


def test_rpm_to_frequencies() -> None:
    """Verify shaft frequency and blade passing frequency calculations."""
    rpm = 24000.0
    f_shaft = rpm_to_shaft_frequency(rpm)
    assert np.isclose(f_shaft, 400.0)

    bpf_3blade = rpm_to_bpf(rpm, blades=3)
    assert np.isclose(bpf_3blade, 1200.0)

    # Test vectorized array inputs
    rpm_arr = np.array([12000.0, 24000.0, 30000.0])
    bpf_arr = rpm_to_bpf(rpm_arr, blades=3)
    expected_bpf = np.array([600.0, 1200.0, 1500.0])
    assert np.allclose(bpf_arr, expected_bpf)


def test_rpm_profiles() -> None:
    """Verify generation of constant, sweep, and step throttle burst RPM profiles."""
    duration = 1.0
    sample_rate = 1000
    expected_samples = 1000

    # Constant
    const = generate_constant_profile(duration, sample_rate, rpm=24000.0)
    assert len(const) == expected_samples
    assert np.all(const == 24000.0)

    # Linear sweep
    sweep = generate_linear_sweep_profile(duration, sample_rate, start_rpm=18000.0, end_rpm=27000.0)
    assert len(sweep) == expected_samples
    assert np.isclose(sweep[0], 18000.0)
    assert np.isclose(sweep[-1], 27000.0)
    assert np.all(np.diff(sweep) > 0)

    # Step burst
    burst = generate_step_burst_profile(
        duration,
        sample_rate,
        base_rpm=18000.0,
        burst_rpm=27000.0,
        burst_start=0.3,
        burst_duration=0.4,
    )
    assert len(burst) == expected_samples
    assert np.isclose(burst[100], 18000.0)
    assert np.isclose(burst[500], 27000.0)
    assert np.isclose(burst[800], 18000.0)

    # Dispatcher
    p_const = generate_rpm_profile(duration, sample_rate, profile_type="constant", rpm=22000.0)
    assert np.all(p_const == 22000.0)


def test_motor_bpf_generation_fft_peak() -> None:
    """Verify that for 24,000 RPM and 3 blades, BPF is 1200 Hz and FFT peak is at 1200 Hz."""
    sample_rate = 48000
    duration = 1.0
    num_samples = int(duration * sample_rate)
    rpm = 24000.0
    blades = 3

    rpm_profile = generate_constant_profile(duration, sample_rate, rpm=rpm)
    signal = synthesize_motor_signal(
        rpm_profile=rpm_profile,
        sample_rate=sample_rate,
        blades=blades,
        harmonic_weights=(1.0, 0.5, 0.25, 0.1),
    )

    assert len(signal) == num_samples

    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
    fft_mag = np.abs(np.fft.rfft(signal))

    peak_idx = int(np.argmax(fft_mag))
    peak_freq = freqs[peak_idx]
    bin_width = freqs[1] - freqs[0]

    assert abs(peak_freq - 1200.0) <= bin_width

    # Verify harmonics exist: 2x (2400 Hz), 3x (3600 Hz), 4x (4800 Hz)
    idx_1200 = int(round(1200.0 / bin_width))
    idx_2400 = int(round(2400.0 / bin_width))
    idx_3600 = int(round(3600.0 / bin_width))
    idx_4800 = int(round(4800.0 / bin_width))

    power_1 = fft_mag[idx_1200]
    power_2 = fft_mag[idx_2400]
    power_3 = fft_mag[idx_3600]
    power_4 = fft_mag[idx_4800]

    # Verify decaying harmonic progression
    assert power_1 > power_2 > power_3 > power_4


def test_quad_motors_phase_continuity_and_differential() -> None:
    """Verify quad motor signal continuity and differential handling."""
    sample_rate = 48000
    duration = 0.5
    rpm_sweep = generate_linear_sweep_profile(
        duration, sample_rate, start_rpm=18000.0, end_rpm=27000.0
    )

    # Test phase continuity: no extreme sample-to-sample derivative spikes
    sig = synthesize_motor_signal(rpm_sweep, sample_rate=sample_rate)
    max_step = np.max(np.abs(np.diff(sig)))
    max_bpf = 3.0 * (27000.0 / 60.0)
    # Theoretical maximum derivative for sum of harmonics
    theoretical_max = 2.0 * np.pi * max_bpf * (1.0 + 2 * 0.5 + 3 * 0.25 + 4 * 0.1) / sample_rate
    assert max_step < theoretical_max * 1.05

    # Quad motors with differential
    quad_sig = synthesize_quad_motors(
        rpm_profile=rpm_sweep,
        sample_rate=sample_rate,
        motor_offsets=(0.99, 1.01, 0.985, 1.015),
    )
    assert len(quad_sig) == len(rpm_sweep)
    assert np.all(np.isfinite(quad_sig))


def test_wind_noise_low_frequency_energy() -> None:
    """Verify that >80% energy of generated wind turbulence noise is below 350 Hz."""
    sample_rate = 48000
    duration = 2.0
    num_samples = int(duration * sample_rate)

    wind = generate_wind_noise(
        num_samples=num_samples,
        sample_rate=sample_rate,
        cutoff_hz=300.0,
        low_cutoff_hz=20.0,
        order=2,
        seed=42,
    )

    assert len(wind) == num_samples
    assert np.all(np.isfinite(wind))

    fft_power = np.abs(np.fft.rfft(wind)) ** 2
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)

    low_band_energy = np.sum(fft_power[freqs <= 350.0])
    total_energy = np.sum(fft_power)

    ratio_below_350 = low_band_energy / total_energy
    assert ratio_below_350 > 0.80, f"Expected >80% below 350 Hz, got {ratio_below_350 * 100:.2f}%"

    # Also verify lowpass-only mode (low_cutoff_hz=None)
    wind_lp = generate_wind_noise(
        num_samples=num_samples,
        sample_rate=sample_rate,
        cutoff_hz=300.0,
        low_cutoff_hz=None,
        order=2,
        seed=123,
    )
    fft_power_lp = np.abs(np.fft.rfft(wind_lp)) ** 2
    ratio_lp = np.sum(fft_power_lp[freqs <= 350.0]) / np.sum(fft_power_lp)
    assert ratio_lp > 0.80


def test_ambient_noise() -> None:
    """Verify stationary ambient noise floor properties."""
    num_samples = 48000
    ambient = generate_ambient_noise(num_samples, amplitude=0.02, seed=123)
    assert len(ambient) == num_samples
    assert np.abs(np.mean(ambient)) < 0.005
    assert np.isclose(np.std(ambient), 0.02, atol=0.002)


def test_generate_scenario_shapes_and_snr() -> None:
    """Verify scenario generator outputs correct shapes and meets target SNR."""
    duration = 1.0
    sample_rate = 48000
    target_snr = 15.0
    expected_samples = int(duration * sample_rate)

    audio, sr, gt_rpm = generate_scenario(
        duration=duration,
        sample_rate=sample_rate,
        profile_type="constant",
        base_rpm=24000.0,
        snr_db=target_snr,
        seed=42,
    )

    assert sr == sample_rate
    assert audio.shape == (expected_samples,)
    assert gt_rpm.shape == (expected_samples,)
    assert np.all(gt_rpm == 24000.0)

    # Reconstruct pure quad signal with same parameters to test SNR separation
    quad_clean = synthesize_quad_motors(
        rpm_profile=gt_rpm,
        sample_rate=sample_rate,
        seed=42,
    )
    noise_part = audio - quad_clean

    p_signal = float(np.mean(quad_clean**2))
    p_noise = float(np.mean(noise_part**2))
    measured_snr = 10.0 * np.log10(p_signal / p_noise)

    assert np.isclose(measured_snr, target_snr, atol=1e-3)


def test_scenario_clean_signal() -> None:
    """Verify scenario generator without noise (snr_db=None)."""
    audio, sr, gt_rpm = generate_scenario(
        duration=0.5,
        sample_rate=48000,
        snr_db=None,
    )
    assert len(audio) == int(0.5 * 48000)
    assert sr == 48000
    assert len(gt_rpm) == len(audio)
    assert np.all(np.isfinite(audio))


def test_export_wav(tmp_path: Path) -> None:
    """Verify WAV export and header/payload roundtrip."""
    wav_file = tmp_path / "test_synth_output.wav"
    sample_rate = 48000
    samples = np.sin(2.0 * np.pi * 440.0 * np.linspace(0, 0.1, int(0.1 * sample_rate)))

    export_wav(wav_file, samples, sample_rate=sample_rate)

    assert wav_file.exists()
    sr, data = wavfile.read(str(wav_file))
    assert sr == sample_rate
    assert len(data) == len(samples)
    assert data.dtype == np.int16
