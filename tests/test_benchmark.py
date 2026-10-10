"""Benchmark test for the complete Acoustix real-time DSP pipeline."""

import time

import numpy as np

from acoustix.dsp.filter import StreamingHighPassFilter
from acoustix.dsp.stft import StreamingSTFT
from acoustix.dsp.tracker import HarmonicTracker


def test_pipeline_realtime_latency() -> None:
    sample_rate = 48000
    hop_size = 512
    n_fft = 2048

    # 1 frame duration is 512 / 48000 = 10.67 ms
    frame_duration_ms = (hop_size / sample_rate) * 1000.0

    # Initialize complete DSP chain
    hpf = StreamingHighPassFilter(cutoff=350.0, fs=sample_rate)
    stft = StreamingSTFT(n_fft=n_fft, hop_length=hop_size, fs=sample_rate)
    tracker = HarmonicTracker(num_blades=3)

    # Generate 1 second of test signal (approx 94 frames)
    rng = np.random.default_rng(42)
    test_audio = rng.standard_normal(sample_rate).astype(np.float64)

    # Warm-up pipeline
    for i in range(10):
        chunk = test_audio[i * hop_size : (i + 1) * hop_size]
        filtered = hpf.process(chunk)
        frame_db = stft.process(filtered)
        tracker.process_frame(frame_db, stft.frequencies)

    # Benchmark run
    n_frames = 50
    start_time = time.perf_counter()

    for i in range(10, 10 + n_frames):
        chunk = test_audio[i * hop_size : (i + 1) * hop_size]
        filtered = hpf.process(chunk)
        frame_db = stft.process(filtered)
        tracker.process_frame(frame_db, stft.frequencies)

    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / n_frames) * 1000.0

    print(f"\nFrame latency: {avg_latency_ms:.3f} ms (duration: {frame_duration_ms:.2f} ms)")

    # Processing a 10.7 ms audio frame must execute in < 1.0 ms (>10x real-time speedup)
    assert avg_latency_ms < 1.0, f"Latency {avg_latency_ms:.3f} ms exceeded 1.0 ms budget"
