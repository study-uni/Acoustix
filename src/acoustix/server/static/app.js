/**
 * Acoustix - Real-Time Acoustic Waterfall Spectrogram & Telemetry Dashboard
 * Minimalist, monochromatic brutalist interface.
 */

(() => {
  // DOM Elements
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const rpmValue = document.getElementById("rpmValue");
  const bpfValue = document.getElementById("bpfValue");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceFill = document.getElementById("confidenceFill");
  const harmonicsList = document.getElementById("harmonicsList");
  const fpsInfo = document.getElementById("fpsInfo");
  const canvas = document.getElementById("waterfallCanvas");
  const canvasWrapper = document.getElementById("canvasWrapper");

  const ctx = canvas.getContext("2d", { alpha: false });

  // Offscreen canvas for scrolling waterfall spectrogram buffer
  let offscreenCanvas = document.createElement("canvas");
  let offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

  // Dimensions
  let width = 1000;
  let height = 520;

  function resizeCanvas() {
    const rect = canvasWrapper.getBoundingClientRect();
    width = Math.floor(rect.width) || 1000;
    height = Math.floor(rect.height) || 520;

    canvas.width = width;
    canvas.height = height;

    // Preserve existing offscreen buffer content during resize
    const oldBuffer = document.createElement("canvas");
    oldBuffer.width = offscreenCanvas.width;
    oldBuffer.height = offscreenCanvas.height;
    if (oldBuffer.width > 0 && oldBuffer.height > 0) {
      oldBuffer.getContext("2d").drawImage(offscreenCanvas, 0, 0);
    }

    offscreenCanvas.width = width;
    offscreenCanvas.height = height;
    offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });
    offscreenCtx.fillStyle = "#040405";
    offscreenCtx.fillRect(0, 0, width, height);

    if (oldBuffer.width > 0 && oldBuffer.height > 0) {
      offscreenCtx.drawImage(oldBuffer, 0, 0, width, height);
    }
  }

  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  // Precompute 256-color Grayscale colormap LUT
  const COLOR_LUT = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255.0;
    // High dynamic range monochrome curve: near-black noise floor, bright peaks
    let v = 0;
    if (t > 0.02) {
      v = Math.round(255 * Math.pow((t - 0.02) / 0.98, 1.25));
    }
    v = Math.max(0, Math.min(255, v));

    const idx = i * 4;
    COLOR_LUT[idx] = v;
    COLOR_LUT[idx + 1] = v;
    COLOR_LUT[idx + 2] = v;
    COLOR_LUT[idx + 3] = 255;
  }

  // State
  let lastPacket = null;
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let rowImageData = offscreenCtx.createImageData(width, 1);

  // Push new spectral slice to waterfall buffer
  function processSpectralFrame(packet) {
    const spectrum = packet.spectrum_db;
    const numBins = spectrum.length;
    if (numBins === 0) return;

    if (rowImageData.width !== width) {
      rowImageData = offscreenCtx.createImageData(width, 1);
    }

    const data = rowImageData.data;
    const minDb = -90.0;
    const maxDb = 0.0;
    const dbRange = maxDb - minDb;

    // Map each horizontal pixel x to frequency bin and grayscale LUT
    for (let x = 0; x < width; x++) {
      const binIdx = Math.min(numBins - 1, Math.floor((x / width) * numBins));
      const db = spectrum[binIdx];
      const norm = Math.max(0.0, Math.min(1.0, (db - minDb) / dbRange));
      const lutIdx = Math.floor(norm * 255) * 4;

      const px = x * 4;
      data[px] = COLOR_LUT[lutIdx];
      data[px + 1] = COLOR_LUT[lutIdx + 1];
      data[px + 2] = COLOR_LUT[lutIdx + 2];
      data[px + 3] = 255;
    }

    // Scroll waterfall down by 1 pixel and blit new row at top
    offscreenCtx.drawImage(offscreenCanvas, 0, 0, width, height - 1, 0, 1, width, height - 1);
    offscreenCtx.putImageData(rowImageData, 0, 0);

    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
      fpsInfo.textContent = `STREAM: ${fps} FPS`;
      frameCount = 0;
      lastFpsTime = now;
    }
  }

  // Render loop using requestAnimationFrame
  function render() {
    // 1. Draw scrolling waterfall background from offscreen buffer
    ctx.drawImage(offscreenCanvas, 0, 0, width, height);

    const maxFreq = lastPacket && lastPacket.freqs && lastPacket.freqs.length > 0
      ? lastPacket.freqs[lastPacket.freqs.length - 1]
      : 6000.0;

    // 2. Frequency Grid Overlay (1 kHz intervals)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillStyle = "rgba(255, 255, 255, 0.35)";

    for (let f = 1000; f < maxFreq; f += 1000) {
      const x = (f / maxFreq) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      ctx.fillText(`${f / 1000}K`, x + 3, height - 8);
    }

    // 3. Harmonic Tracking Overlays (monochrome precision markers)
    if (lastPacket && lastPacket.bpf > 0 && lastPacket.confidence > 0.1) {
      const bpf = lastPacket.bpf;
      const harmonics = lastPacket.harmonics || [bpf, bpf * 2, bpf * 3, bpf * 4];

      harmonics.forEach((hFreq, i) => {
        if (hFreq <= 0 || hFreq > maxFreq) return;
        const x = (hFreq / maxFreq) * width;

        ctx.beginPath();
        if (i === 0) {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 2]);
        } else {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 3]);
        }
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        ctx.fillStyle = i === 0 ? "#ffffff" : "rgba(255, 255, 255, 0.6)";
        ctx.font = i === 0 ? "bold 10px ui-monospace, monospace" : "10px ui-monospace, monospace";
        const tag = `${i + 1}×`;
        ctx.fillText(tag, x + 3, 14 + i * 14);
      });
    }

    ctx.setLineDash([]);
    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);

  // Update HUD Metrics
  function updateHUD(packet) {
    const rpm = packet.rpm || 0;
    const bpf = packet.bpf || 0;
    const conf = Math.max(0, Math.min(1, packet.confidence || 0));

    rpmValue.textContent = `${Math.round(rpm).toLocaleString()} RPM`;
    bpfValue.textContent = `${bpf.toFixed(1)} HZ`;

    const confPct = (conf * 100).toFixed(1);
    confidenceValue.textContent = `${confPct}%`;
    confidenceFill.style.width = `${confPct}%`;

    const harmonics = packet.harmonics || [];
    if (harmonics.length > 0) {
      harmonicsList.innerHTML = harmonics
        .map((h, i) => `<span class="harmonic-tag">${i + 1}&times; ${Math.round(h)} HZ</span>`)
        .join("");
    }
  }

  // Source Switcher & Live Microphone Capture
  const btnSynth = document.getElementById("btnSynth");
  const btnMic = document.getElementById("btnMic");

  let currentSource = "synth";
  let micStream = null;
  let audioCtx = null;
  let micProcessor = null;
  let currentWs = null;
  let reconnectTimer = null;

  async function startMic() {
    // If browser supports getUserMedia (secure context: localhost, 127.0.0.1, or https)
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
        });

        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
        const micSource = audioCtx.createMediaStreamSource(micStream);
        micProcessor = audioCtx.createScriptProcessor(1024, 1, 1);

        micProcessor.onaudioprocess = (e) => {
          if (currentSource === "mic" && currentWs && currentWs.readyState === WebSocket.OPEN) {
            const channelData = e.inputBuffer.getChannelData(0);
            currentWs.send(channelData.buffer);
          }
        };

        micSource.connect(micProcessor);
        micProcessor.connect(audioCtx.destination);

        currentSource = "mic";
        if (btnMic) btnMic.classList.add("active");
        if (btnSynth) btnSynth.classList.remove("active");
        connectWebSocket("?source=mic");
        return;
      } catch (err) {
        console.warn("Browser mic access failed, falling back to server_mic:", err);
      }
    }

    // Fallback: direct hardware mic capture on server via sounddevice
    currentSource = "server_mic";
    if (btnMic) btnMic.classList.add("active");
    if (btnSynth) btnSynth.classList.remove("active");
    connectWebSocket("?source=server_mic");
  }

  function stopMic() {
    if (micStream) {
      micStream.getTracks().forEach((track) => track.stop());
      micStream = null;
    }
    if (audioCtx) {
      audioCtx.close().catch(() => {});
      audioCtx = null;
    }
    micProcessor = null;
    currentSource = "synth";
    if (btnSynth) btnSynth.classList.add("active");
    if (btnMic) btnMic.classList.remove("active");
  }

  if (btnSynth) {
    btnSynth.addEventListener("click", () => {
      if (currentSource === "synth") return;
      stopMic();
      connectWebSocket("?source=synth");
    });
  }

  if (btnMic) {
    btnMic.addEventListener("click", () => {
      if (currentSource === "mic" || currentSource === "server_mic") return;
      startMic();
    });
  }

  // WebSocket Connection Manager
  function connectWebSocket(customQuery) {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }

    if (currentWs) {
      currentWs.onclose = null;
      currentWs.close();
      currentWs = null;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    let query = customQuery;
    if (query === undefined) {
      query = window.location.search || `?source=${currentSource}`;
    }

    const wsUrl = `${protocol}//${window.location.host}/ws/stream${query}`;
    const ws = new WebSocket(wsUrl);
    currentWs = ws;

    ws.onopen = () => {
      statusDot.classList.add("connected");
      statusText.textContent = "CONNECTED";
    };

    ws.onmessage = (event) => {
      try {
        const packet = JSON.parse(event.data);
        lastPacket = packet;
        processSpectralFrame(packet);
        updateHUD(packet);
      } catch (err) {
        console.error("Failed to parse packet:", err);
      }
    };

    ws.onclose = () => {
      statusDot.classList.remove("connected");
      statusText.textContent = "DISCONNECTED";
      reconnectTimer = setTimeout(() => connectWebSocket(query), 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  connectWebSocket();
})();
