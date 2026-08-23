const ui = Object.fromEntries([
  "connect", "interrupt", "status", "transcript", "send", "state", "decision",
  "latency", "frames", "delivered", "unheard", "events", "clear"
].map((id) => [id, document.getElementById(id)]));

let socket;
let audioContext;
let microphone;
let processor;
let frameCount = 0;
let nextAudioMeta = null;
const activeAudio = new Map();

function send(payload) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
}

function setConnected(connected) {
  ui.status.textContent = connected ? "online" : "offline";
  ui.status.className = `pill ${connected ? "online" : "offline"}`;
  ui.connect.disabled = connected;
  ui.interrupt.disabled = !connected;
  ui.send.disabled = !connected;
  document.querySelectorAll(".quick button").forEach((button) => { button.disabled = !connected; });
}

function downsample(input, sourceRate, targetRate = 16000) {
  if (sourceRate === targetRate) return input;
  const ratio = sourceRate / targetRate;
  const output = new Float32Array(Math.round(input.length / ratio));
  for (let i = 0; i < output.length; i += 1) {
    const start = Math.round(i * ratio);
    const end = Math.min(input.length, Math.round((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += input[j];
    output[i] = sum / Math.max(1, end - start);
  }
  return output;
}

function pcm16(floatSamples) {
  const result = new Int16Array(floatSamples.length);
  floatSamples.forEach((sample, index) => {
    const clamped = Math.max(-1, Math.min(1, sample));
    result[index] = clamped < 0 ? clamped * 32768 : clamped * 32767;
  });
  return result.buffer;
}

async function startMicrophone() {
  audioContext ||= new AudioContext();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
  });
  microphone = audioContext.createMediaStreamSource(stream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    if (socket?.readyState !== WebSocket.OPEN) return;
    const downsampled = downsample(event.inputBuffer.getChannelData(0), audioContext.sampleRate);
    socket.send(pcm16(downsampled));
    frameCount += 1;
    ui.frames.textContent = `${frameCount} Frames`;
  };
  microphone.connect(processor);
  processor.connect(audioContext.destination);
}

function startBrowserStt() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return;
  const recognition = new Recognition();
  recognition.lang = "de-DE";
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.onresult = (event) => {
    const result = event.results[event.resultIndex];
    const text = result[0].transcript.trim();
    send({
      type: "transcript", text, final: result.isFinal,
      signals: {
        speech_active: !result.isFinal,
        silence_duration_ms: result.isFinal ? 350 : 0,
        utterance_duration_ms: 600,
        semantic_complete: result.isFinal,
        acoustic_completion: result.isFinal ? 0.8 : 0.1,
        interruption_probability: /^(moment|stopp|warte|nein stopp)/i.test(text) ? 0.98 : 0.0
      }
    });
  };
  recognition.onend = () => { if (socket?.readyState === WebSocket.OPEN) recognition.start(); };
  recognition.start();
}

function playPcm(buffer, meta) {
  audioContext ||= new AudioContext();
  const samples = new Int16Array(buffer);
  const floats = Float32Array.from(samples, (sample) => sample / 32768);
  const audioBuffer = audioContext.createBuffer(1, floats.length, meta.sample_rate_hz);
  audioBuffer.copyToChannel(floats, 0);
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);
  const playback = { source, startedAt: audioContext.currentTime, cancelled: false, samples: meta.samples, rate: meta.sample_rate_hz };
  activeAudio.set(meta.chunk_id, playback);
  source.onended = () => {
    if (!playback.cancelled) send({ type: "playback_completed", chunk_id: meta.chunk_id });
    activeAudio.delete(meta.chunk_id);
  };
  source.start();
  send({ type: "playback_started", chunk_id: meta.chunk_id });
}

function cancelAudio(chunkId) {
  const playback = activeAudio.get(chunkId);
  if (!playback) {
    send({ type: "playback_stopped", chunk_id: chunkId, played_samples: 0 });
    return;
  }
  playback.cancelled = true;
  const elapsed = Math.max(0, audioContext.currentTime - playback.startedAt);
  const played = Math.min(playback.samples, Math.floor(elapsed * playback.rate));
  try { playback.source.stop(); } catch (_) { /* already stopped */ }
  activeAudio.delete(chunkId);
  send({ type: "playback_stopped", chunk_id: chunkId, played_samples: played });
}

function addEvent(event) {
  const item = document.createElement("li");
  item.textContent = `${event.sequence.toString().padStart(3, "0")} ${event.event_type} · ${event.reason_code}`;
  ui.events.prepend(item);
  while (ui.events.children.length > 120) ui.events.lastChild.remove();
  if (event.event_type === "STATE_TRANSITIONED") ui.state.textContent = event.payload.to_state;
  if (event.event_type === "AGENT_AUDIO_CANCELLED") {
    const value = event.payload.audible_barge_in_latency_ms;
    ui.latency.textContent = value == null ? "—" : `${value.toFixed(2)} ms`;
    ui.delivered.textContent = event.payload.delivered_text || "—";
    ui.unheard.textContent = event.payload.unheard_text || "—";
  }
  if (event.event_type === "AGENT_AUDIO_COMPLETED") ui.delivered.textContent = event.payload.delivered_text || "—";
}

ui.connect.addEventListener("click", async () => {
  try {
    await startMicrophone();
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws`);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => { setConnected(true); startBrowserStt(); };
    socket.onclose = () => setConnected(false);
    socket.onmessage = (message) => {
      if (message.data instanceof ArrayBuffer) {
        if (nextAudioMeta) playPcm(message.data, nextAudioMeta);
        nextAudioMeta = null;
        return;
      }
      const payload = JSON.parse(message.data);
      if (payload.type === "audio_chunk") nextAudioMeta = payload;
      else if (payload.type === "cancel_audio") cancelAudio(payload.chunk_id);
      else if (payload.type === "telemetry") addEvent(payload.event);
      else if (payload.type === "turn_decision") ui.decision.textContent = `${payload.decision} · ${payload.confidence.toFixed(2)}`;
    };
  } catch (error) {
    ui.status.textContent = `Fehler: ${error.name}`;
  }
});

ui.interrupt.addEventListener("click", () => send({ type: "interrupt" }));
ui.send.addEventListener("click", () => send({
  type: "transcript", text: ui.transcript.value, final: true,
  signals: { speech_active: false, silence_duration_ms: 350, utterance_duration_ms: 900, semantic_complete: true, acoustic_completion: 0.9 }
}));
document.querySelectorAll(".quick button").forEach((button) => button.addEventListener("click", () => send({
  type: "transcript", text: button.dataset.phrase, final: true,
  signals: {
    speech_active: true, silence_duration_ms: 0, utterance_duration_ms: 250,
    semantic_complete: false, acoustic_completion: 0,
    interruption_probability: button.dataset.phrase.startsWith("Moment") ? 0.98 : 0
  }
})));
ui.clear.addEventListener("click", () => ui.events.replaceChildren());
setConnected(false);
