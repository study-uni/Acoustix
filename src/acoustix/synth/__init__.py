"""Synthetic drone audio and environmental noise simulation."""

from acoustix.synth.generator import export_wav, generate_scenario
from acoustix.synth.motors import (
    DEFAULT_HARMONIC_WEIGHTS,
    DEFAULT_MOTOR_OFFSETS,
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

__all__ = [
    "DEFAULT_HARMONIC_WEIGHTS",
    "DEFAULT_MOTOR_OFFSETS",
    "export_wav",
    "generate_ambient_noise",
    "generate_constant_profile",
    "generate_linear_sweep_profile",
    "generate_rpm_profile",
    "generate_scenario",
    "generate_step_burst_profile",
    "generate_wind_noise",
    "rpm_to_bpf",
    "rpm_to_shaft_frequency",
    "synthesize_motor_signal",
    "synthesize_quad_motors",
]
