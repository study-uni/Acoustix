"""High-level scenario synthesis combining quad motors and environmental noise."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from acoustix.synth.motors import generate_rpm_profile, synthesize_quad_motors
from acoustix.synth.noise import generate_ambient_noise, generate_wind_noise


def generate_scenario(
    duration: float = 2.0,
    sample_rate: int = 48000,
    profile_type: str = "constant",
    base_rpm: float = 24000.0,
    snr_db: float | None = 20.0,
    blades: int = 3,
    motor_offsets: tuple[float, float, float, float] | None = None,
    ambient_noise_level: float = 0.0,
    seed: int | None = None,
    **profile_kwargs,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Synthesize quadcopter acoustic scenario combining quad motors and environmental noise.

    Returns:
        tuple: (audio_samples, sample_rate, ground_truth_rpm)
    """
    ground_truth_rpm = generate_rpm_profile(
        duration=duration,
        sample_rate=sample_rate,
        profile_type=profile_type,
        base_rpm=base_rpm,
        **profile_kwargs,
    )

    num_samples = len(ground_truth_rpm)

    quad_signal = synthesize_quad_motors(
        rpm_profile=ground_truth_rpm,
        sample_rate=sample_rate,
        blades=blades,
        motor_offsets=motor_offsets,
        seed=seed,
    )

    if snr_db is None:
        return quad_signal, sample_rate, ground_truth_rpm

    # Synthesize noise components
    wind_noise = generate_wind_noise(
        num_samples=num_samples,
        sample_rate=sample_rate,
        seed=seed,
    )

    noise_total = wind_noise
    if ambient_noise_level > 0.0:
        ambient = generate_ambient_noise(
            num_samples=num_samples,
            amplitude=ambient_noise_level,
            seed=None if seed is None else seed + 1,
        )
        noise_total = noise_total + ambient

    # Scale noise to achieve target SNR (dB): SNR = 10 * log10(P_signal / P_noise)
    p_signal = np.mean(quad_signal**2)
    p_noise = np.mean(noise_total**2)

    if p_noise > 0.0 and p_signal > 0.0:
        target_p_noise = p_signal / (10.0 ** (snr_db / 10.0))
        noise_scale = np.sqrt(target_p_noise / p_noise)
        scaled_noise = noise_total * noise_scale
    else:
        scaled_noise = np.zeros_like(quad_signal)

    audio_samples = quad_signal + scaled_noise

    return audio_samples, sample_rate, ground_truth_rpm


def export_wav(
    filepath: str | Path,
    audio_samples: np.ndarray,
    sample_rate: int = 48000,
    normalize: bool = True,
) -> None:
    """Export audio samples to a 16-bit PCM WAV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(audio_samples, dtype=np.float64)
    if normalize and len(samples) > 0:
        max_val = np.max(np.abs(samples))
        if max_val > 0.0:
            samples = samples / max_val * 0.95
    elif not normalize:
        samples = np.clip(samples, -1.0, 1.0)

    int_samples = np.round(samples * 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, int_samples)
