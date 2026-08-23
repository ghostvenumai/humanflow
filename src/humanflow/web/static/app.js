const ui = Object.fromEntries([
  "connect", "interrupt", "status", "transcript", "send", "state", "decision",
  "latency", "frames", "delivered", "unheard", "events", "clear", "conversation",
  "provider-stt", "provider-reasoning", "provider-tts"
].map((id) => [id, document.getElementById(id)]));

let socket;
let audioContext;
let microphone;
let processor;
let frameCount = 0;
let nextAudioMeta = null;
let partialTranscriptItem = null;
let speechStartedAt = null;
let speechEndedAt = null;
const activeAudio = new Map();
const assistantItems = new Map();

function browserCapabilities() {
  return {
    stt: Boolean(window.SpeechRecognition || window.webkitSpeechRecognition),
    tts: Boolean(window.speechSynthesis && window.SpeechSynthesisUtterance)
  };
}

function setProvider(role, text, state = "") {
  const target = ui[`provider-${role}`];
  target.textContent = text;
  target.className = state;
}

function renderProviders(providers) {
  for (const item of providers || []) {
    if (!ui[`provider-${item.role}`]) continue;
    const available = item.availability === "CONFIGURED" || item.availability === "AVAILABLE";
    const state = item.mode === "REAL" && available ? "real" : "unavailable";
    setProvider(item.role, `${item.provider} · ${item.model} · ${item.mode} · ${item.availability}`, state);
  }
  const capabilities = browserCapabilities();
  setProvider("stt", `Browser Web Speech · de-DE · REAL · ${capabilities.stt ? "AVAILABLE" : "UNAVAILABLE"}`, capabilities.stt ? "real" : "unavailable");
  setProvider("tts", `Browser Web Speech · Systemstimme · REAL · ${capabilities.tts ? "AVAILABLE" : "UNAVAILABLE"}`, capabilities.tts ? "real" : "unavailable");
}

function addConversation(role, text, extraClass = "") {
  const item = document.createElement("li");
  item.className = `${role} ${extraClass}`.trim();
  const label = document.createElement("small");
  label.textContent = role === "user" ? "Sie" : "HumanFlow";
  const body = document.createElement("span");
  body.textContent = text;
  item.append(label, body);
  ui.conversation.append(item);
  ui.conversation.scrollTop = ui.conversation.scrollHeight;
  return item;
}

function showTranscript(text, isFinal) {
  if (isFinal) {
    partialTranscriptItem?.remove();
    partialTranscriptItem = null;
    addConversation("user", text);
    return;
  }
  if (!partialTranscriptItem) partialTranscriptItem = addConversation("user", text, "partial");
  else partialTranscriptItem.querySelector("span").textContent = text;
}

function showAssistantChunk(meta) {
  let item = assistantItems.get(meta.response_id);
  if (!item) {
    item = addConversation("assistant", meta.text_boundary);
    assistantItems.set(meta.response_id, item);
    return;
  }
  const body = item.querySelector("span");
  body.textContent = `${body.textContent} ${meta.text_boundary}`;
}

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
  if (!Recognition) throw new Error("Browser-STT ist nicht verfügbar");
  const recognition = new Recognition();
  recognition.lang = "de-DE";
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.onspeechstart = () => {
    speechStartedAt = performance.now();
    speechEndedAt = null;
  };
  recognition.onspeechend = () => {
    speechEndedAt = performance.now();
  };
  recognition.onresult = (event) => {
    const observedAt = performance.now();
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      const text = result[0].transcript.trim();
      if (!text) continue;
      const utteranceDuration = speechStartedAt == null ? 0 : Math.max(0, (speechEndedAt || observedAt) - speechStartedAt);
      const silenceDuration = speechEndedAt == null ? 0 : Math.max(0, observedAt - speechEndedAt);
      const explicitInterruption = /^(moment|stopp|warte|nein stopp)(\s|$)/i.test(text);
      showTranscript(text, result.isFinal);
      send({
        type: "transcript", source: "browser_stt", text, final: result.isFinal,
        signals: {
          speech_active: !result.isFinal,
          silence_duration_ms: Math.round(silenceDuration),
          utterance_duration_ms: Math.round(utteranceDuration),
          semantic_complete: result.isFinal,
          provider_endpointed: result.isFinal,
          acoustic_completion: result.isFinal ? 1.0 : 0.0,
          interruption_probability: explicitInterruption ? 1.0 : 0.0
        }
      });
      if (result.isFinal) {
        speechStartedAt = null;
        speechEndedAt = null;
      }
    }
  };
  recognition.onerror = (event) => {
    if (event.error === "no-speech") return;
    ui.status.textContent = `STT-Fehler: ${event.error}`;
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      setProvider("stt", `Browser Web Speech · de-DE · REAL · ${event.error.toUpperCase()}`, "unavailable");
    }
  };
  recognition.onend = () => { if (socket?.readyState === WebSocket.OPEN) recognition.start(); };
  recognition.start();
}

function playSpeech(meta) {
  const utterance = new SpeechSynthesisUtterance(meta.text_boundary);
  utterance.lang = "de-DE";
  utterance.rate = 1.02;
  const playback = { utterance, cancelled: false, samples: meta.samples, mode: "speech" };
  activeAudio.set(meta.chunk_id, playback);
  utterance.onstart = () => send({ type: "playback_started", chunk_id: meta.chunk_id });
  utterance.onend = () => {
    if (!playback.cancelled) send({ type: "playback_completed", chunk_id: meta.chunk_id });
    activeAudio.delete(meta.chunk_id);
  };
  utterance.onerror = () => {
    if (!playback.cancelled) send({ type: "playback_stopped", chunk_id: meta.chunk_id, played_samples: 0 });
    activeAudio.delete(meta.chunk_id);
  };
  window.speechSynthesis.speak(utterance);
}

function playOutput(buffer, meta) {
  void buffer;
  if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {
    send({ type: "playback_stopped", chunk_id: meta.chunk_id, played_samples: 0 });
    ui.status.textContent = "Fehler: echtes Browser-TTS fehlt";
    return;
  }
  playSpeech(meta);
}

function cancelAudio(chunkId) {
  const playback = activeAudio.get(chunkId);
  if (!playback) {
    send({ type: "playback_stopped", chunk_id: chunkId, played_samples: 0 });
    return;
  }
  playback.cancelled = true;
  window.speechSynthesis.cancel();
  activeAudio.delete(chunkId);
  send({ type: "playback_stopped", chunk_id: chunkId, played_samples: 0 });
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
  if (event.event_type === "RECOVERY_STARTED") ui.status.textContent = "Providerfehler · weiter im Hörmodus";
}

ui.connect.addEventListener("click", async () => {
  try {
    const capabilities = browserCapabilities();
    renderProviders([]);
    if (!capabilities.stt || !capabilities.tts) throw new Error("Echter Browser-STT/TTS-Provider fehlt");
    await startMicrophone();
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws`);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      setConnected(true);
      startBrowserStt();
      send({ type: "provider_capabilities", stt_available: capabilities.stt, tts_available: capabilities.tts });
    };
    socket.onclose = () => setConnected(false);
    socket.onmessage = (message) => {
      if (message.data instanceof ArrayBuffer) {
        if (nextAudioMeta) playOutput(message.data, nextAudioMeta);
        nextAudioMeta = null;
        return;
      }
      const payload = JSON.parse(message.data);
      if (payload.type === "ready") renderProviders(payload.providers);
      else if (payload.type === "audio_chunk") {
        nextAudioMeta = payload;
        showAssistantChunk(payload);
      }
      else if (payload.type === "cancel_audio") cancelAudio(payload.chunk_id);
      else if (payload.type === "telemetry") addEvent(payload.event);
      else if (payload.type === "turn_decision") ui.decision.textContent = `${payload.decision} · ${payload.confidence.toFixed(2)}`;
    };
  } catch (error) {
    ui.status.textContent = `Fehler: ${error.message || error.name}`;
  }
});

ui.interrupt.addEventListener("click", () => send({ type: "interrupt" }));
ui.send.addEventListener("click", () => send({
  type: "transcript", source: "manual_diagnostic", text: ui.transcript.value, final: true,
  signals: { speech_active: false, silence_duration_ms: 350, utterance_duration_ms: 900, semantic_complete: true, acoustic_completion: 0.9 }
}));
ui.send.addEventListener("click", () => showTranscript(ui.transcript.value, true));
document.querySelectorAll(".quick button").forEach((button) => button.addEventListener("click", () => send({
  type: "transcript", source: "manual_diagnostic", text: button.dataset.phrase, final: true,
  signals: {
    speech_active: true, silence_duration_ms: 0, utterance_duration_ms: 250,
    semantic_complete: false, acoustic_completion: 0,
    interruption_probability: button.dataset.phrase.startsWith("Moment") ? 0.98 : 0
  }
})));
ui.clear.addEventListener("click", () => ui.events.replaceChildren());
renderProviders([]);
setConnected(false);
