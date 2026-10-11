/**
 * Acoustix - Real-Time Acoustic Waterfall Spectrogram & Telemetry Dashboard
 */

(() => {
  // DOM Elements
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const rpmValue = document.getElementById("rpmValue");
  const rpmSub = document.getElementById("rpmSub");
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
    offscreenCtx.fillStyle = "#000004";
    offscreenCtx.fillRect(0, 0, width, height);

    if (oldBuffer.width > 0 && oldBuffer.height > 0) {
      offscreenCtx.drawImage(oldBuffer, 0, 0, width, height);
    }
  }

  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  // Precompute 256-color Turbo colormap LUT
  const COLOR_LUT = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255.0;
    // Turbo colormap polynomial fit
    const r = Math.max(0, Math.min(255, Math.round(255 * (0.1357 + t * (4.61539 - t * (42.6603 - t * (132.131 - t * (161.076 - 63.666 * t))))))));
    const g = Math.max(0, Math.min(255, Math.round(255 * (0.0914 + t * (2.19418 + t * (16.4498 - t * (53.5458 + t * (58.3797 - 21.6629 * t))))))));
    const b = Math.max(0, Math.min(255, Math.round(255 * (0.1067 + t * (12.5833 - t * (90.2838 - t * (242.062 - t * (282.883 - 120.306 * t))))))));
    const idx = i * 4;
    COLOR_LUT[idx] = r;
    COLOR_LUT[idx + 1] = g;
    COLOR_LUT[idx + 2] = b;
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

    // Map each horizontal pixel x to frequency bin and colormap color
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
      fpsInfo.textContent = `Stream Rate: ${fps} FPS`;
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
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";

    for (let f = 1000; f < maxFreq; f += 1000) {
      const x = (f / maxFreq) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      ctx.fillText(`${f / 1000}k`, x + 3, height - 8);
    }

    // 3. Harmonic Tracking Overlays
    if (lastPacket && lastPacket.bpf > 0 && lastPacket.confidence > 0.1) {
      const bpf = lastPacket.bpf;
      const harmonics = lastPacket.harmonics || [bpf, bpf * 2, bpf * 3, bpf * 4];

      harmonics.forEach((hFreq, i) => {
        if (hFreq <= 0 || hFreq > maxFreq) return;
        const x = (hFreq / maxFreq) * width;

        ctx.beginPath();
        if (i === 0) {
          // Fundamental BPF line (solid / distinct cyan glow)
          ctx.strokeStyle = "rgba(6, 182, 212, 0.9)";
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 2]);
        } else {
          // Harmonics (softer cyan)
          ctx.strokeStyle = "rgba(6, 182, 212, 0.45)";
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 3]);
        }
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        // Harmonic label header tag
        ctx.fillStyle = i === 0 ? "#06b6d4" : "rgba(6, 182, 212, 0.75)";
        ctx.font = i === 0 ? "bold 11px ui-monospace, monospace" : "10px ui-monospace, monospace";
        const tag = `${i + 1}×`;
        ctx.fillText(tag, x + 3, 14 + i * 12);
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

    // Dynamic color coding based on tracking confidence
    let color = "var(--text-muted)";
    let statusTextMsg = "Tracking Lost";
    if (conf >= 0.7) {
      color = "var(--accent-green)";
      statusTextMsg = "Locked (Nominal)";
    } else if (conf >= 0.4) {
      color = "var(--accent-amber)";
      statusTextMsg = "Acquiring Lock";
    } else {
      color = "var(--accent-red)";
      statusTextMsg = "Low Confidence";
    }

    rpmValue.textContent = `${Math.round(rpm).toLocaleString()} RPM`;
    rpmValue.style.color = color;
    rpmSub.textContent = statusTextMsg;

    bpfValue.textContent = `${bpf.toFixed(1)} Hz`;

    const confPct = (conf * 100).toFixed(1);
    confidenceValue.textContent = `${confPct}%`;
    confidenceValue.style.color = color;
    confidenceFill.style.width = `${confPct}%`;
    confidenceFill.style.background = color;

    // Tracked Harmonics Tags
    const harmonics = packet.harmonics || [];
    if (harmonics.length > 0) {
      harmonicsList.innerHTML = harmonics
        .map((h, i) => `<span class="harmonic-tag">${i + 1}&times; ${Math.round(h)} Hz</span>`)
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
    } catch (err) {
      console.error("Microphone access error:", err);
      alert("Не удалось получить доступ к микрофону: " + err.message);
      stopMic();
    }
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
      if (currentSource === "mic") return;
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
      statusText.textContent = `Connected (${currentSource === "mic" ? "Mic" : "Synth"})`;
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
      statusText.textContent = "Disconnected (Reconnecting...)";
      reconnectTimer = setTimeout(() => connectWebSocket(query), 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  connectWebSocket();
})();
