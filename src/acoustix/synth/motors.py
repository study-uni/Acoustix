"""Multi-motor RPM simulation and harmonic acoustic signal synthesis."""

import numpy as np

DEFAULT_HARMONIC_WEIGHTS: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)
DEFAULT_MOTOR_OFFSETS: tuple[float, float, float, float] = (0.99, 1.01, 0.985, 1.015)
DEFAULT_INITIAL_PHASES: tuple[float, float, float, float] = (0.0, np.pi * 0.5, np.pi, np.pi * 1.5)


def rpm_to_shaft_frequency(rpm: float | np.ndarray) -> float | np.ndarray:
    """Calculate shaft frequency (f_shaft = RPM / 60)."""
    return rpm / 60.0


def rpm_to_bpf(rpm: float | np.ndarray, blades: int = 3) -> float | np.ndarray:
    """Calculate Blade Passing Frequency (BPF = blades * f_shaft)."""
    return blades * rpm_to_shaft_frequency(rpm)


def generate_constant_profile(
    duration: float,
    sample_rate: int = 48000,
    rpm: float = 24000.0,
) -> np.ndarray:
    """Generate a constant RPM profile."""
    num_samples = int(duration * sample_rate)
    return np.full(num_samples, float(rpm), dtype=np.float64)


def generate_linear_sweep_profile(
    duration: float,
    sample_rate: int = 48000,
    start_rpm: float = 18000.0,
    end_rpm: float = 27000.0,
) -> np.ndarray:
    """Generate a linear frequency sweep RPM profile."""
    num_samples = int(duration * sample_rate)
    return np.linspace(float(start_rpm), float(end_rpm), num_samples, dtype=np.float64)


def generate_step_burst_profile(
    duration: float,
    sample_rate: int = 48000,
    base_rpm: float = 18000.0,
    burst_rpm: float = 27000.0,
    burst_start: float = 0.3,
    burst_duration: float = 0.4,
) -> np.ndarray:
    """Generate an RPM profile with a step throttle burst."""
    num_samples = int(duration * sample_rate)
    profile = np.full(num_samples, float(base_rpm), dtype=np.float64)
    start_idx = max(0, min(int(burst_start * sample_rate), num_samples))
    end_idx = max(start_idx, min(int((burst_start + burst_duration) * sample_rate), num_samples))
    profile[start_idx:end_idx] = float(burst_rpm)
    return profile


def generate_rpm_profile(
    duration: float,
    sample_rate: int = 48000,
    profile_type: str = "constant",
    base_rpm: float = 24000.0,
    **kwargs,
) -> np.ndarray:
    """Generate an RPM profile based on profile type name."""
    profile_type = profile_type.lower()
    if profile_type == "constant":
        return generate_constant_profile(
            duration=duration,
            sample_rate=sample_rate,
            rpm=kwargs.get("rpm", base_rpm),
        )
    if profile_type in ("linear_sweep", "sweep"):
        return generate_linear_sweep_profile(
            duration=duration,
            sample_rate=sample_rate,
            start_rpm=kwargs.get("start_rpm", base_rpm),
            end_rpm=kwargs.get("end_rpm", 27000.0),
        )
    if profile_type in ("step_burst", "burst"):
        return generate_step_burst_profile(
            duration=duration,
            sample_rate=sample_rate,
            base_rpm=base_rpm,
            burst_rpm=kwargs.get("burst_rpm", 27000.0),
            burst_start=kwargs.get("burst_start", 0.3),
            burst_duration=kwargs.get("burst_duration", 0.4),
        )
    raise ValueError(f"Unknown RPM profile type: {profile_type}")


def synthesize_motor_signal(
    rpm_profile: np.ndarray,
    sample_rate: int = 48000,
    blades: int = 3,
    harmonic_weights: tuple[float, ...] = DEFAULT_HARMONIC_WEIGHTS,
    initial_phase: float = 0.0,
) -> np.ndarray:
    """Generate phase-continuous multi-harmonic acoustic signal for a single motor."""
    bpf = rpm_to_bpf(rpm_profile, blades=blades)

    # Discrete-time phase integration: Delta theta = 2 * pi * f(t) / Fs
    delta_phi = 2.0 * np.pi * bpf / sample_rate
    phase = (np.cumsum(delta_phi) + initial_phase) % (2.0 * np.pi)

    signal = np.zeros_like(rpm_profile, dtype=np.float64)
    for k, weight in enumerate(harmonic_weights, start=1):
        signal += weight * np.sin(k * phase)

    return signal


def synthesize_quad_motors(
    rpm_profile: np.ndarray,
    sample_rate: int = 48000,
    blades: int = 3,
    harmonic_weights: tuple[float, ...] = DEFAULT_HARMONIC_WEIGHTS,
    motor_offsets: tuple[float, float, float, float] | None = None,
    jitter_pct: float = 0.0,
    initial_phases: tuple[float, float, float, float] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate 4 quadcopter motors with inter-motor RPM differential and acoustic summation."""
    offsets = DEFAULT_MOTOR_OFFSETS if motor_offsets is None else motor_offsets
    phases = DEFAULT_INITIAL_PHASES if initial_phases is None else initial_phases

    rng = np.random.default_rng(seed) if (jitter_pct > 0.0 or seed is not None) else None

    quad_signal = np.zeros_like(rpm_profile, dtype=np.float64)
    for i in range(4):
        motor_rpm = rpm_profile * offsets[i]
        if jitter_pct > 0.0 and rng is not None:
            jitter = rng.normal(0.0, jitter_pct, size=len(rpm_profile))
            motor_rpm = np.maximum(motor_rpm * (1.0 + jitter), 0.0)

        motor_sig = synthesize_motor_signal(
            rpm_profile=motor_rpm,
            sample_rate=sample_rate,
            blades=blades,
            harmonic_weights=harmonic_weights,
            initial_phase=phases[i],
        )
        quad_signal += motor_sig

    return quad_signal / 4.0
