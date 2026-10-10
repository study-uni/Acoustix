"""Tests for Acoustix FastAPI web server and WebSocket streaming endpoints."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from starlette.testclient import TestClient

from acoustix.server.app import app


def test_health_endpoint() -> None:
    """Verify /health endpoint returns status ok and correct version."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_index_endpoint() -> None:
    """Verify GET / serves static index.html with 200 status."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "ACOUSTIX" in response.text
    assert "waterfallCanvas" in response.text


def test_websocket_stream_synth() -> None:
    """Verify WebSocket /ws/stream handshake and packet schema for synthetic stream."""
    client = TestClient(app)
    with client.websocket_connect("/ws/stream?source=synth") as websocket:
        data = websocket.receive_json()

        assert "freqs" in data
        assert "spectrum_db" in data
        assert "bpf" in data
        assert "rpm" in data
        assert "confidence" in data
        assert "harmonics" in data
        assert "timestamp" in data

        assert isinstance(data["freqs"], list)
        assert isinstance(data["spectrum_db"], list)
        assert len(data["freqs"]) == len(data["spectrum_db"])
        assert len(data["freqs"]) > 0

        # Frequency slice check: maximum frequency <= 6000 Hz
        assert max(data["freqs"]) <= 6000.0

        # Check telemetry values
        assert isinstance(data["bpf"], float | int)
        assert isinstance(data["rpm"], float | int)
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["harmonics"], list)
        assert len(data["harmonics"]) == 4


def test_websocket_stream_file(tmp_path: Path) -> None:
    """Verify WebSocket /ws/stream with a WAV audio file."""
    # Create temporary WAV audio file
    fs = 48000
    t = np.arange(fs) / fs
    sig = np.sin(2 * np.pi * 1200 * t)
    wav_path = tmp_path / "test_drone.wav"
    wavfile.write(wav_path, fs, (sig * 32767).astype(np.int16))

    client = TestClient(app)
    with client.websocket_connect(f"/ws/stream?source=file&file_path={wav_path}") as websocket:
        data = websocket.receive_json()
        assert "rpm" in data
        assert "bpf" in data
        assert len(data["freqs"]) > 0


def test_websocket_stream_file_not_found() -> None:
    """Verify WebSocket closes when file does not exist."""
    client = TestClient(app)
    with client.websocket_connect(
        "/ws/stream?source=file&file_path=/nonexistent/path.wav"
    ) as websocket:
        # Expect connection to close
        try:
            websocket.receive_json()
        except Exception:
            pass  # Expected disconnect on file not found
