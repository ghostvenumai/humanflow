const ui = Object.fromEntries([
  "connect", "interrupt", "status", "transcript", "send", "state", "decision",
  "latency", "frames", "delivered", "unheard", "events", "clear", "conversation",
  "provider-stt", "provider-reasoning", "provider-tts", "provider-tts-fallback",
  "voice-candidate", "voice-sample-count", "voice-ratings", "voice-notes",
  "save-voice-rating", "voice-rating-status"
].map((id) => [id, document.getElementById(id)]));

let socket;
let audioContext;
let microphone;
let processor;
let recognition;
let recognitionRun = 0;
let frameCount = 0;
let nextAudioMeta = null;
let partialTranscriptItem = null;
let speechStartedAt = null;
let speechEndedAt = null;
const activeAudio = new Map();
const assistantItems = new Map();
const receivedChunkIds = new Set();
const scheduledChunkSequences = new Set();
const responsePlaybackProviders = new Map();
const cancelledResponseIds = new Set();
const sentFinalRecognitionIds = new Set();
const browserSessionId = crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`;
const inputOnlyMode = new URLSearchParams(location.search).get("mode") === "input-only";
let activePlaybackProvider = null;
let activePlaybackOwner = null;
let sourceNodeSequence = 0;
let activeTtsCandidate = "noch nicht gehört";
let declaredProviders = [];
const voiceRatingFields = {
  naturalness: "Natürlichkeit", prosody: "Prosodie", pacing: "Sprechtempo",
  voice_pleasantness: "Stimmklang", turn_timing: "Turn-Timing",
  interruption_feel: "Unterbrechungsgefühl",
  non_mechanical_impression: "Nicht-mechanischer Eindruck",
  overall_conversational_realism: "Gesprächsrealismus"
};

function initializeVoiceRatings() {
  for (const [field, labelText] of Object.entries(voiceRatingFields)) {
    const wrapper = document.createElement("label");
    wrapper.textContent = labelText;
    const select = document.createElement("select");
    select.dataset.voiceRating = field;
    select.innerHTML = '<option value="">—</option>' + [1, 2, 3, 4, 5]
      .map((value) => `<option value="${value}">${value}</option>`).join("");
    wrapper.append(select);
    ui["voice-ratings"].append(wrapper);
  }
}

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
  if (providers?.length) declaredProviders = providers;
  for (const item of declaredProviders) {
    if (!ui[`provider-${item.role}`]) continue;
    const available = item.availability === "CONFIGURED" || item.availability === "AVAILABLE";
    const state = item.mode === "REAL" && available ? "real" : "unavailable";
    setProvider(item.role, `${item.provider} · ${item.model} · ${item.mode} · ${item.availability}`, state);
  }
  const capabilities = browserCapabilities();
  setProvider("stt", `Browser Web Speech · de-DE · REAL · ${capabilities.stt ? "AVAILABLE" : "UNAVAILABLE"}`, capabilities.stt ? "real" : "unavailable");
  setProvider("tts-fallback", `Browser Web Speech · Systemstimme · REAL · ${capabilities.tts ? "AVAILABLE" : "UNAVAILABLE"}`, capabilities.tts ? "real" : "unavailable");
}

function activateTtsProvider(provider) {
  renderProviders(declaredProviders);
  const fallbackActive = provider.provider === "browser-web-speech-api";
  const role = fallbackActive ? "tts-fallback" : "tts";
  setProvider(role, `${provider.provider} · ${provider.model} · ${provider.mode} · ACTIVE`, provider.mode === "REAL" ? "real" : "unavailable");
  activeTtsCandidate = `${provider.provider} · ${provider.model}`;
  ui["voice-candidate"].textContent = activeTtsCandidate;
  ui["save-voice-rating"].disabled = false;
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
  if (!meta.text_boundary) return;
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
  recognition = new Recognition();
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
      const recognitionResultId = `${browserSessionId}:${recognitionRun}:${index}`;
      if (result.isFinal && sentFinalRecognitionIds.has(recognitionResultId)) continue;
      const utteranceDuration = speechStartedAt == null ? 0 : Math.max(0, (speechEndedAt || observedAt) - speechStartedAt);
      const silenceDuration = speechEndedAt == null ? 0 : Math.max(0, observedAt - speechEndedAt);
      const explicitInterruption = /^(moment|stopp|warte|nein stopp)(\b|[,.!?])/i.test(text);
      showTranscript(text, result.isFinal);
      send({
        type: "transcript", source: "browser_stt", text, final: result.isFinal,
        recognition_result_id: recognitionResultId,
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
        sentFinalRecognitionIds.add(recognitionResultId);
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
  recognition.onend = () => {
    if (socket?.readyState === WebSocket.OPEN) {
      recognitionRun += 1;
      recognition.start();
    }
  };
  recognitionRun += 1;
  recognition.start();
}

function playbackProviderKey(meta) {
  const provider = meta.tts_provider || {};
  return `${provider.provider || "unknown"}|${provider.model || "unknown"}|${meta.playback_mode}`;
}

function rejectPlayback(meta, code) {
  send({ type: "playback_stopped", chunk_id: meta.chunk_id, played_samples: 0 });
  ui.status.textContent = `Playback blockiert: ${code}`;
}

function acceptAudioMeta(meta) {
  const sequenceKey = `${meta.stream_id}:${meta.sequence}`;
  const providerKey = playbackProviderKey(meta);
  const pinnedProvider = responsePlaybackProviders.get(meta.response_id);
  if (cancelledResponseIds.has(meta.response_id)) {
    rejectPlayback(meta, "stale_cancelled_response");
    return false;
  }
  if (receivedChunkIds.has(meta.chunk_id) || scheduledChunkSequences.has(sequenceKey)) {
    rejectPlayback(meta, "duplicate_chunk_or_sequence");
    return false;
  }
  if (pinnedProvider && pinnedProvider !== providerKey) {
    rejectPlayback(meta, "tts_provider_changed_within_response");
    return false;
  }
  if (nextAudioMeta || activeAudio.size > 0) {
    rejectPlayback(meta, "multiple_playback_consumers");
    return false;
  }
  receivedChunkIds.add(meta.chunk_id);
  responsePlaybackProviders.set(meta.response_id, providerKey);
  return true;
}

function acquirePlayback(meta) {
  const providerKey = playbackProviderKey(meta);
  if (activePlaybackProvider && activePlaybackProvider !== providerKey) {
    rejectPlayback(meta, "active_tts_playback_providers_exceeded");
    return false;
  }
  if (activePlaybackOwner && activePlaybackOwner !== meta.response_id) {
    rejectPlayback(meta, "playback_owner_conflict");
    return false;
  }
  const sequenceKey = `${meta.stream_id}:${meta.sequence}`;
  if (scheduledChunkSequences.has(sequenceKey)) {
    rejectPlayback(meta, "sequence_scheduled_twice");
    return false;
  }
  scheduledChunkSequences.add(sequenceKey);
  activePlaybackProvider = providerKey;
  activePlaybackOwner = meta.response_id;
  if (meta.playback_mode === "pcm" && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  return true;
}

function playSpeech(meta) {
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(meta.text_boundary);
  utterance.lang = "de-DE";
  utterance.rate = meta.speaking_rate || 1.0;
  const playback = {
    utterance, cancelled: false, finished: false, samples: meta.samples,
    mode: "speech", pauseTimer: null, responseId: meta.response_id,
    providerKey: playbackProviderKey(meta), sourceNodeId: `speech-${meta.response_id}-${++sourceNodeSequence}`
  };
  activeAudio.set(meta.chunk_id, playback);
  utterance.onstart = () => send({
    type: "playback_started", chunk_id: meta.chunk_id,
    source_node_id: playback.sourceNodeId,
    browser_scheduled_start_ms: performance.now(),
    browser_actual_playback_start_ms: performance.now()
  });
  utterance.onend = () => {
    if (playback.cancelled) return;
    const complete = () => finishPlayback(meta.chunk_id, false, meta.samples);
    playback.pauseTimer = window.setTimeout(complete, meta.pause_after_ms || 0);
  };
  utterance.onerror = () => {
    if (!playback.cancelled) finishPlayback(meta.chunk_id, true, 0);
  };
  window.speechSynthesis.speak(utterance);
}

function finishPlayback(chunkId, cancelled, playedSamples, stopCallbackLatencyMs = null) {
  const playback = activeAudio.get(chunkId);
  if (!playback || playback.finished) return;
  playback.finished = true;
  if (playback.pauseTimer != null) window.clearTimeout(playback.pauseTimer);
  if (playback.stopFallbackTimer != null) window.clearTimeout(playback.stopFallbackTimer);
  activeAudio.delete(chunkId);
  if (activeAudio.size === 0) {
    activePlaybackProvider = null;
    activePlaybackOwner = null;
  }
  if (cancelled) {
    send({
      type: "playback_stopped", chunk_id: chunkId, played_samples: playedSamples,
      player_stop_callback_latency_ms: stopCallbackLatencyMs
    });
  } else {
    send({ type: "playback_completed", chunk_id: chunkId });
  }
}

function playPcm(buffer, meta) {
  audioContext ||= new AudioContext({ latencyHint: "interactive" });
  const input = new Int16Array(buffer);
  const channels = Math.max(1, meta.channels || 1);
  const frames = Math.floor(input.length / channels);
  const decoded = audioContext.createBuffer(channels, frames, meta.sample_rate_hz);
  for (let channel = 0; channel < channels; channel += 1) {
    const output = decoded.getChannelData(channel);
    for (let frame = 0; frame < frames; frame += 1) {
      output[frame] = input[frame * channels + channel] / 32768;
    }
  }
  const source = audioContext.createBufferSource();
  source.buffer = decoded;
  source.connect(audioContext.destination);
  const sourceNodeId = `pcm-${meta.response_id}-${meta.sequence}-${++sourceNodeSequence}`;
  const scheduledStartMs = audioContext.currentTime * 1000;
  const playback = {
    source, cancelled: false, finished: false, samples: meta.samples,
    sampleRate: meta.sample_rate_hz, startedAt: audioContext.currentTime,
    mode: "pcm", pauseTimer: null, sourceEnded: false,
    stopRequestedAt: null, stopFallbackTimer: null, playedSamplesAtStop: 0,
    responseId: meta.response_id, providerKey: playbackProviderKey(meta), sourceNodeId
  };
  activeAudio.set(meta.chunk_id, playback);
  source.onended = () => {
    playback.sourceEnded = true;
    if (playback.cancelled) {
      const latency = playback.stopRequestedAt == null
        ? null : Math.max(0, performance.now() - playback.stopRequestedAt);
      finishPlayback(meta.chunk_id, true, playback.playedSamplesAtStop, latency);
      return;
    }
    const complete = () => finishPlayback(meta.chunk_id, false, meta.samples);
    playback.pauseTimer = window.setTimeout(complete, meta.pause_after_ms || 0);
  };
  source.start(0);
  const actualPlaybackStartMs = audioContext.currentTime * 1000;
  send({
    type: "playback_started",
    chunk_id: meta.chunk_id,
    browser_audio_context_base_latency_ms: (audioContext.baseLatency || 0) * 1000,
    browser_audio_context_output_latency_ms: (audioContext.outputLatency || 0) * 1000,
    source_node_id: sourceNodeId,
    browser_scheduled_start_ms: scheduledStartMs,
    browser_actual_playback_start_ms: actualPlaybackStartMs
  });
}

function playOutput(buffer, meta) {
  if (!acquirePlayback(meta)) return;
  if (meta.playback_mode === "pcm") {
    playPcm(buffer, meta);
    return;
  }
  if (meta.playback_mode !== "browser_speech") {
    send({ type: "playback_stopped", chunk_id: meta.chunk_id, played_samples: 0 });
    ui.status.textContent = `Fehler: unbekannter Audio-Modus ${meta.playback_mode}`;
    return;
  }
  if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {
    send({ type: "playback_stopped", chunk_id: meta.chunk_id, played_samples: 0 });
    ui.status.textContent = "Fehler: Browser-TTS-Fallback fehlt";
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
  cancelledResponseIds.add(playback.responseId);
  playback.cancelled = true;
  if (playback.pauseTimer != null) window.clearTimeout(playback.pauseTimer);
  let playedSamples = 0;
  if (playback.mode === "pcm") {
    const elapsed = Math.max(0, audioContext.currentTime - playback.startedAt);
    playedSamples = Math.min(playback.samples, Math.floor(elapsed * playback.sampleRate));
    playback.playedSamplesAtStop = playedSamples;
    playback.stopRequestedAt = performance.now();
    if (playback.sourceEnded) {
      finishPlayback(chunkId, true, playedSamples, 0);
      return;
    }
    try { playback.source.stop(); } catch (_) { /* source already ended */ }
    playback.stopFallbackTimer = window.setTimeout(() => {
      const latency = Math.max(0, performance.now() - playback.stopRequestedAt);
      finishPlayback(chunkId, true, playedSamples, latency);
    }, 250);
    return;
  } else {
    window.speechSynthesis.cancel();
  }
  window.setTimeout(() => finishPlayback(chunkId, true, playedSamples), 0);
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
  if (event.event_type === "TTS_PROVIDER_ACTIVATED" && event.payload.provider) {
    activateTtsProvider(event.payload.provider);
  }
  if (event.event_type === "TTS_PROVIDER_DEACTIVATED") renderProviders(declaredProviders);
}

ui.connect.addEventListener("click", async () => {
  try {
    const capabilities = browserCapabilities();
    renderProviders([]);
    if (!capabilities.stt) throw new Error("Echter Browser-STT-Provider fehlt");
    await startMicrophone();
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws${inputOnlyMode ? "?mode=input-only" : ""}`);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      setConnected(true);
      startBrowserStt();
      send({ type: "provider_capabilities", stt_available: capabilities.stt, tts_available: capabilities.tts });
    };
    socket.onclose = () => {
      setConnected(false);
      recognition?.stop();
      for (const chunkId of [...activeAudio.keys()]) cancelAudio(chunkId);
    };
    socket.onmessage = (message) => {
      if (message.data instanceof ArrayBuffer) {
        if (nextAudioMeta) playOutput(message.data, nextAudioMeta);
        nextAudioMeta = null;
        return;
      }
      const payload = JSON.parse(message.data);
      if (payload.type === "ready") {
        renderProviders(payload.providers);
        if (inputOnlyMode) {
          setProvider("reasoning", "DEAKTIVIERT · INPUT-ONLY-ISOLATION", "unavailable");
          setProvider("tts", "DEAKTIVIERT · INPUT-ONLY-ISOLATION", "unavailable");
          setProvider("tts-fallback", "DEAKTIVIERT · INPUT-ONLY-ISOLATION", "unavailable");
          ui.status.textContent = "online · INPUT-ONLY";
        }
      }
      else if (payload.type === "audio_chunk") {
        if (!acceptAudioMeta(payload)) {
          nextAudioMeta = null;
          return;
        }
        nextAudioMeta = payload;
        showAssistantChunk(payload);
        if (payload.tts_provider?.provider) {
          activateTtsProvider(payload.tts_provider);
        }
      }
      else if (payload.type === "cancel_audio") cancelAudio(payload.chunk_id);
      else if (payload.type === "telemetry") addEvent(payload.event);
      else if (payload.type === "turn_decision") ui.decision.textContent = `${payload.decision} · ${payload.confidence.toFixed(2)}`;
      else if (payload.type === "error") ui.status.textContent = `Fehler: ${payload.code}`;
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
ui["save-voice-rating"].addEventListener("click", async () => {
  const ratings = {};
  for (const select of ui["voice-ratings"].querySelectorAll("select")) {
    if (!select.value) {
      ui["voice-rating-status"].textContent = "Bitte alle acht Felder bewerten.";
      return;
    }
    ratings[select.dataset.voiceRating] = Number(select.value);
  }
  try {
    const response = await fetch("/api/voice-quality", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        candidate: activeTtsCandidate, ratings, notes: ui["voice-notes"].value
      })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const summary = await response.json();
    ui["voice-sample-count"].textContent = summary.sample_count;
    ui["voice-rating-status"].textContent = "Bewertung gespeichert.";
  } catch (error) {
    ui["voice-rating-status"].textContent = `Speichern fehlgeschlagen: ${error.message}`;
  }
});
initializeVoiceRatings();
renderProviders([]);
setConnected(false);
