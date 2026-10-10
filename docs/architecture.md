# Architecture & Technical Decisions

## Invariants & Design Principles
1. **Python-First Core**: Business logic, DSP orchestration, pipeline abstractions, audio synthetic benchmarks, and APIs are implemented in Python (>=3.12).
2. **Selective Rust Acceleration**: Rust is integrated via PyO3 / Maturin strictly when performance profiling indicates that Python/NumPy is a bottleneck (e.g., SIMD STFT in sub-millisecond audio loops, lock-free ring buffers).
3. **Zero Bloat & Zero Speculation**: Write only what is required by the current task. No unused helper functions or speculative abstractions.
4. **Minimalist English Comments**: Comments only for non-obvious mathematical or DSP logic.
5. **Enforced Verification**: All changes must pass `task pre-commit` (`ruff` check/format + `pytest`).

---

## ADR-001: Python-First Core with Selective Rust Acceleration

### Context
Acoustix requires real-time processing of acoustic signatures from high-speed FPV drones, involving Butterworth filtering, STFT computation, adaptive noise subtraction, and harmonic product spectrum tracking. While initially sketched as a pure Rust workspace, high-level flexibility, rapid algorithm tuning, and integration with scientific tooling (NumPy/SciPy/Matplotlib) make Python the primary development language.

### Decision
- Use Python 3.12+ managed by `uv` with `hatchling` as the build system.
- Utilize NumPy and SciPy for vectorized DSP routines (Butterworth filters, FFTs, STFT).
- Retain an isolated Rust extension interface (`rust/` crate via PyO3 / Maturin) ready to be invoked for hot DSP paths if profiling shows latency constraints (<0.3 ms/frame) cannot be met in pure vectorized Python.

### Consequences
- **Positive**: Rapid algorithm experimentation, seamless visualization, fast iteration cycle, zero compilation wait times during high-level pipeline development.
- **Negative**: Need to ensure vectorized NumPy operations avoid Python interpreter overhead in real-time streaming loops; fallback to Rust if GIL or loop latency becomes an issue.
