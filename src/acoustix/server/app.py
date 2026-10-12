"""FastAPI web server and real-time DSP streaming pipeline for Acoustix."""

import argparse
import asyncio
import tempfile
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.io import wavfile
from starlette.websockets import WebSocketDisconnect, WebSocketDisconnected

from acoustix.dsp.filter import StreamingHighPassFilter
from acoustix.dsp.stft import StreamingSTFT
from acoustix.dsp.tracker import HarmonicTracker
from acoustix.synth.generator import generate_scenario

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Acoustix",
    description="High-speed FPV drone acoustic signature detection and DSP engine",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    """Ensure static files and HTML pages are never cached stale by browsers."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/")
async def get_index() -> FileResponse:
    """Serve the real-time canvas waterfall dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def get_health() -> dict[str, str]:
    """Return server health and version status."""
    return {"status": "ok", "version": "0.1.0"}


def _create_synthetic_audio_buffer(
    duration: float = 10.0,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Generate a smooth back-and-forth RPM sweep scenario buffer for continuous looping."""
    half_dur = duration / 2.0
    # Sweep up: 18,000 -> 27,000 RPM
    audio_up, _, _ = generate_scenario(
        duration=half_dur,
        sample_rate=sample_rate,
        profile_type="sweep",
        start_rpm=18000.0,
        end_rpm=27000.0,
        snr_db=20.0,
        blades=3,
    )
    # Sweep down: 27,000 -> 18,000 RPM
    audio_down, _, _ = generate_scenario(
        duration=half_dur,
        sample_rate=sample_rate,
        profile_type="sweep",
        start_rpm=27000.0,
        end_rpm=18000.0,
        snr_db=20.0,
        blades=3,
    )
    return np.concatenate([audio_up, audio_down])


@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    source: str = Query("synth"),
    file_path: str | None = Query(None),
) -> None:
    """Stream audio chunks through the DSP pipeline and push spectral packets over WebSocket."""
    await websocket.accept()

    sample_rate = 48000
    hop_size = 512
    n_fft = 2048

    hp_filter = StreamingHighPassFilter(cutoff=350.0, fs=sample_rate)
    stft = StreamingSTFT(n_fft=n_fft, hop_length=hop_size, fs=sample_rate)
    tracker = HarmonicTracker(num_blades=3)

    freq_mask = stft.frequencies <= 6000.0
    sliced_freqs_list = np.round(stft.frequencies[freq_mask], 1).tolist()

    # Live client-side microphone streaming (browser Web Audio API sends Float32 PCM)
    if source == "mic":
        mic_accumulator = np.empty(0, dtype=np.float64)
        frame_idx = 0
        try:
            while True:
                msg = await websocket.receive()
                if "bytes" in msg and msg["bytes"]:
                    chunk = np.frombuffer(msg["bytes"], dtype=np.float32).astype(np.float64)
                    mic_accumulator = np.concatenate([mic_accumulator, chunk])

                    while len(mic_accumulator) >= hop_size:
                        sub_chunk = mic_accumulator[:hop_size]
                        mic_accumulator = mic_accumulator[hop_size:]

                        filtered_chunk = hp_filter.process(sub_chunk)
                        spectrum_db = stft.process(filtered_chunk)
                        track_result = tracker.process_frame(spectrum_db, stft.frequencies)
                        sliced_spectrum = spectrum_db[freq_mask]

                        timestamp = round(frame_idx * (hop_size / sample_rate), 3)
                        packet = {
                            "freqs": sliced_freqs_list,
                            "spectrum_db": np.round(sliced_spectrum, 1).tolist(),
                            "bpf": round(float(track_result.bpf), 1),
                            "rpm": round(float(track_result.rpm), 1),
                            "confidence": round(float(track_result.confidence), 3),
                            "harmonics": [round(float(h), 1) for h in track_result.harmonics],
                            "timestamp": timestamp,
                        }
                        await websocket.send_json(packet)
                        frame_idx += 1
        except (WebSocketDisconnect, WebSocketDisconnected, asyncio.CancelledError):
            pass
        return

    # Server-side hardware microphone streaming via sounddevice
    if source == "server_mic":
        try:
            import sounddevice as sd
        except ImportError:
            await websocket.close(code=1011, reason="sounddevice library is not installed")
            return

        queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=100)
        loop = asyncio.get_running_loop()

        def audio_callback(indata, frames, time_info, status):
            chunk = np.copy(indata[:, 0]).astype(np.float64)
            loop.call_soon_threadsafe(queue.put_nowait, chunk)

        stream = sd.InputStream(
            samplerate=sample_rate,
            blocksize=hop_size,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        )
        try:
            with stream:
                frame_idx = 0
                while True:
                    chunk = await queue.get()
                    filtered_chunk = hp_filter.process(chunk)
                    spectrum_db = stft.process(filtered_chunk)
                    track_result = tracker.process_frame(spectrum_db, stft.frequencies)
                    sliced_spectrum = spectrum_db[freq_mask]

                    timestamp = round(frame_idx * (hop_size / sample_rate), 3)
                    packet = {
                        "freqs": sliced_freqs_list,
                        "spectrum_db": np.round(sliced_spectrum, 1).tolist(),
                        "bpf": round(float(track_result.bpf), 1),
                        "rpm": round(float(track_result.rpm), 1),
                        "confidence": round(float(track_result.confidence), 3),
                        "harmonics": [round(float(h), 1) for h in track_result.harmonics],
                        "timestamp": timestamp,
                    }
                    await websocket.send_json(packet)
                    frame_idx += 1
        except (WebSocketDisconnect, WebSocketDisconnected, asyncio.CancelledError):
            pass
        return

    # Ingest audio from file
    if source == "file":
        if not file_path:
            await websocket.close(
                code=1008, reason="file_path parameter is required when source='file'"
            )
            return
        path = Path(file_path).resolve()
        allowed_bases = (Path.cwd().resolve(), Path(tempfile.gettempdir()).resolve())
        is_safe = any(path.is_relative_to(base) for base in allowed_bases)
        if not path.is_file() or not is_safe or path.suffix.lower() not in {".wav", ".wave"}:
            await websocket.close(
                code=1008, reason=f"Audio file not found or unauthorized: {file_path}"
            )
            return

        try:
            fs, data = wavfile.read(path)
            sample_rate = fs
            if data.ndim > 1:
                data = data.mean(axis=-1)
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                audio_buffer = data.astype(np.float64) / max_val
            else:
                audio_buffer = data.astype(np.float64)
        except Exception as exc:
            await websocket.close(code=1011, reason=f"Failed to load audio file: {exc}")
            return
    else:
        # Default synthetic drone audio stream
        audio_buffer = _create_synthetic_audio_buffer(duration=10.0, sample_rate=sample_rate)

    if len(audio_buffer) < hop_size:
        await websocket.close(code=1011, reason="Audio buffer too short")
        return

    buffer_len = len(audio_buffer)
    offset = 0
    frame_idx = 0
    sleep_interval = 0.016  # ~60 fps streaming rate

    try:
        while True:
            # Extract audio chunk with seamless wrap-around looping
            if offset + hop_size > buffer_len:
                offset = 0
            chunk = audio_buffer[offset : offset + hop_size]
            offset += hop_size

            # Process chunk: HighPass -> STFT -> HarmonicTracker
            filtered_chunk = hp_filter.process(chunk)
            spectrum_db = stft.process(filtered_chunk)

            track_result = tracker.process_frame(spectrum_db, stft.frequencies)
            sliced_spectrum = spectrum_db[freq_mask]

            timestamp = round(frame_idx * (hop_size / sample_rate), 3)
            packet = {
                "freqs": sliced_freqs_list,
                "spectrum_db": np.round(sliced_spectrum, 1).tolist(),
                "bpf": round(float(track_result.bpf), 1),
                "rpm": round(float(track_result.rpm), 1),
                "confidence": round(float(track_result.confidence), 3),
                "harmonics": [round(float(h), 1) for h in track_result.harmonics],
                "timestamp": timestamp,
            }

            await websocket.send_json(packet)
            await asyncio.sleep(sleep_interval)
            frame_idx += 1

    except (WebSocketDisconnect, WebSocketDisconnected, asyncio.CancelledError):
        pass


def main() -> None:
    """CLI entrypoint for Acoustix server."""
    parser = argparse.ArgumentParser(description="Acoustix real-time stream server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    uvicorn.run("acoustix.server.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
