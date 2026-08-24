const ui = Object.fromEntries([
  "connect", "interrupt", "status", "transcript", "send", "state", "decision",
  "latency", "frames", "delivered", "unheard", "events", "clear", "conversation",
  "provider-stt", "provider-reasoning", "provider-tts", "provider-tts-fallback",
  "voice-candidate", "voice-sample-count", "voice-ratings", "voice-notes",
  "save-voice-rating", "voice-rating-status", "debug-stt-raw",
  "debug-user-accepted", "debug-suppressed", "debug-assistant",
  "debug-history-roles", "debug-stt-partial", "debug-stt-final",
  "mic-source", "pcm-source", "browser-stt-status", "metric-acoustic-onset",
  "metric-soft-yield", "metric-confirmed-interruption", "metric-audible-stop",
  "metric-backchannel-recovery", "metric-false-interruptions",
  "metric-first-partial", "metric-final-stt"
  , "tts-ab-selection"
].map((id) => [id, document.getElementById(id)]));

let socket;
let audioContext;
let playbackGain;
let microphone;
let processor;
let recognition;
let recognitionRun = 0;
let recognitionSessionId = null;
let audioCaptureId = null;
let microphoneStreamId = null;
let transcriptSequence = 0;
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
const cancelledChunkIds = new Set();
const responsePlaybackEpochs = new Map();
const sentFinalRecognitionIds = new Set();
const browserSessionId = crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`;
const inputOnlyMode = new URLSearchParams(location.search).get("mode") === "input-only";
const browserSttDiagnosticMode = new URLSearchParams(location.search).get("browser-stt-diagnostic") === "1";
let activePlaybackProvider = null;
let activePlaybackOwner = null;
let softYieldResponseId = null;
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
    microphone: Boolean(navigator.mediaDevices?.getUserMedia),
    browserSttDiagnostic: Boolean(window.SpeechRecognition || window.webkitSpeechRecognition),
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
  ui["browser-stt-status"].textContent = browserSttDiagnosticMode
    ? `MOCK/DIAGNOSTIC · ${capabilities.browserSttDiagnostic ? "AVAILABLE" : "UNAVAILABLE"}`
    : "OFF · nicht im Produktionspfad";
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

function addConversation(role, text, badge, extraClass = "") {
  const item = document.createElement("li");
  item.className = `${role} ${extraClass}`.trim();
  const label = document.createElement("small");
  label.textContent = badge;
  const body = document.createElement("span");
  body.textContent = text;
  item.append(label, body);
  ui.conversation.append(item);
  ui.conversation.scrollTop = ui.conversation.scrollHeight;
  return item;
}

function showPartialTranscript(text, provider = "ElevenLabs Scribe") {
  if (!partialTranscriptItem) partialTranscriptItem = addConversation("partial", text, `PARTIAL / ${provider}`, "partial");
  else partialTranscriptItem.querySelector("span").textContent = text;
}

function showAcceptedUser(text, provenance) {
  partialTranscriptItem?.remove();
  partialTranscriptItem = null;
  const provider = provenance?.origin === "STREAMING_STT_PROVIDER"
    ? "ElevenLabs Scribe" : "Diagnostic-Text";
  addConversation("user", text, `USER / ${provider}`);
}

function showAssistantChunk(meta) {
  if (!meta.text_boundary) return;
  let item = assistantItems.get(meta.response_id);
  if (!item) {
    const provider = meta.tts_provider?.provider === "elevenlabs-text-to-speech-stream"
      ? "ElevenLabs" : (meta.tts_provider?.provider || "TTS");
    item = addConversation("assistant", meta.text_boundary, `ASSISTANT / Claude · TTS / ${provider}`);
    assistantItems.set(meta.response_id, item);
    return;
  }
  const body = item.querySelector("span");
  body.textContent = `${body.textContent} ${meta.text_boundary}`;
}

function appendDebug(target, text) {
  const item = document.createElement("li");
  item.textContent = text;
  target.prepend(item);
  while (target.children.length > 40) target.lastChild.remove();
}

function diagnosticProvenance(source, final = true) {
  return {
    transcript_id: `${browserSessionId}:diagnostic:${++transcriptSequence}`,
    event_kind: final ? "USER_TRANSCRIPT_FINAL" : "USER_TRANSCRIPT_PARTIAL",
    source,
    origin: "DIAGNOSTIC_TEXT_INPUT",
    stream_id: "diagnostic-text-input",
    browser_timestamp_ms: performance.timeOrigin + performance.now(),
    response_id: activePlaybackOwner
  };
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
  ui["tts-ab-selection"].disabled = connected;
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
  if (audioContext.state === "suspended") await audioContext.resume();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
  });
  const track = stream.getAudioTracks()[0];
  audioCaptureId = crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  microphoneStreamId = track?.id || `get-user-media-${audioCaptureId}`;
  microphone = audioContext.createMediaStreamSource(stream);
  processor = audioContext.createScriptProcessor(1024, 1, 1);
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

function stopMicrophone() {
  try { processor?.disconnect(); } catch (_) { /* already disconnected */ }
  try { microphone?.disconnect(); } catch (_) { /* already disconnected */ }
  for (const track of microphone?.mediaStream?.getTracks?.() || []) track.stop();
  processor = null;
  microphone = null;
  audioCaptureId = null;
  microphoneStreamId = null;
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
  recognition.onstart = () => {
    recognitionSessionId = `${browserSessionId}:${recognitionRun}`;
    send({
      type: "recognition_session_started",
      browser_recognition_session_id: recognitionSessionId,
      audio_capture_id: audioCaptureId,
      recognition_input_binding: "UNVERIFIED_INDEPENDENT_BROWSER_CAPTURE"
    });
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
      const transcriptId = `${recognitionResultId}:${result.isFinal ? "final" : `partial-${++transcriptSequence}`}`;
      const provenance = {
        transcript_id: transcriptId,
        event_kind: result.isFinal ? "USER_TRANSCRIPT_FINAL" : "USER_TRANSCRIPT_PARTIAL",
        source: "browser_stt",
        origin: "BROWSER_SPEECH_RECOGNITION",
        stream_id: `browser-recognition:${recognitionSessionId}`,
        browser_recognition_session_id: recognitionSessionId,
        audio_capture_id: audioCaptureId,
        response_id: activePlaybackOwner,
        browser_timestamp_ms: performance.timeOrigin + observedAt,
        recognition_input_binding: "UNVERIFIED_INDEPENDENT_BROWSER_CAPTURE"
      };
      appendDebug(ui["debug-stt-raw"], `${result.isFinal ? "FINAL" : "PARTIAL"} · ${text} · ${transcriptId}`);
      send({
        type: "transcript", source: "browser_stt", text, final: result.isFinal,
        recognition_result_id: recognitionResultId,
        provenance,
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
      ui["browser-stt-status"].textContent = `DIAGNOSTIC MOCK/FALLBACK · ${event.error.toUpperCase()}`;
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
  const incomingEpoch = Number.isInteger(meta.playback_epoch) ? meta.playback_epoch : 0;
  const currentEpoch = responsePlaybackEpochs.get(meta.response_id) ?? incomingEpoch;
  if (cancelledResponseIds.has(meta.response_id)) {
    rejectPlayback(meta, "stale_cancelled_response");
    return false;
  }
  if (incomingEpoch < currentEpoch) {
    rejectPlayback(meta, "stale_playback_epoch");
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
  responsePlaybackEpochs.set(meta.response_id, incomingEpoch);
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

function ensurePlaybackGain() {
  audioContext ||= new AudioContext({ latencyHint: "interactive" });
  if (!playbackGain) {
    playbackGain = audioContext.createGain();
    playbackGain.gain.value = 1;
    playbackGain.connect(audioContext.destination);
  }
  return playbackGain;
}

function setPlaybackGain(value, timeConstant = 0.012) {
  const gain = ensurePlaybackGain().gain;
  const now = audioContext.currentTime;
  gain.cancelScheduledValues(now);
  gain.setTargetAtTime(value, now, timeConstant);
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
  source.connect(ensurePlaybackGain());
  if (softYieldResponseId !== meta.response_id) setPlaybackGain(1, 0.006);
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
  if (cancelledChunkIds.has(chunkId)) return;
  cancelledChunkIds.add(chunkId);
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

function duckPlayback(message) {
  const responseId = message.response_id;
  if (!responseId || (activePlaybackOwner !== responseId && nextAudioMeta?.response_id !== responseId)) return;
  const epoch = responsePlaybackEpochs.get(responseId) ?? message.playback_epoch ?? 0;
  if ((message.playback_epoch ?? epoch) < epoch || cancelledResponseIds.has(responseId)) return;
  softYieldResponseId = responseId;
  const targetGain = Number.isFinite(message.target_gain) ? message.target_gain : 0.08;
  const modes = [...activeAudio.values()].filter((item) => item.responseId === responseId).map((item) => item.mode);
  if (modes.includes("speech")) window.speechSynthesis.pause();
  setPlaybackGain(targetGain, 0.010);
  send({
    type: "playback_ducked", response_id: responseId,
    playback_epoch: epoch, target_gain: targetGain,
    browser_applied_ms: performance.now()
  });
}

function resumePlayback(message) {
  const responseId = message.response_id;
  if (!responseId || cancelledResponseIds.has(responseId) || softYieldResponseId !== responseId) return;
  const epoch = responsePlaybackEpochs.get(responseId) ?? message.playback_epoch ?? 0;
  if ((message.playback_epoch ?? epoch) < epoch) return;
  setPlaybackGain(1, 0.025);
  window.speechSynthesis.resume?.();
  softYieldResponseId = null;
  send({
    type: "playback_resumed", response_id: responseId,
    playback_epoch: epoch, browser_applied_ms: performance.now()
  });
}

function invalidatePlayback(message) {
  const responseId = message.response_id;
  if (!responseId) return;
  const epoch = Number.isInteger(message.playback_epoch) ? message.playback_epoch : 0;
  responsePlaybackEpochs.set(responseId, Math.max(epoch, responsePlaybackEpochs.get(responseId) ?? 0));
  cancelledResponseIds.add(responseId);
  softYieldResponseId = null;
  setPlaybackGain(0, 0.004);
  if (nextAudioMeta?.response_id === responseId) nextAudioMeta = null;
  for (const [chunkId, playback] of [...activeAudio.entries()]) {
    if (playback.responseId === responseId) cancelAudio(chunkId);
  }
  window.speechSynthesis.cancel();
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
  if (event.event_type === "USER_AUDIO_STARTED" && event.payload.acoustic_speech_onset_latency_ms != null) {
    ui["metric-acoustic-onset"].textContent = `${event.payload.acoustic_speech_onset_latency_ms.toFixed(1)} ms`;
  }
  if (event.event_type === "PLAYBACK_DUCK_STARTED") {
    ui["metric-soft-yield"].textContent = `${event.payload.speech_onset_to_soft_duck_ms.toFixed(1)} ms`;
  }
  if (event.event_type === "INTERRUPTION_CONFIRMED") {
    ui["metric-confirmed-interruption"].textContent = `${event.payload.speech_onset_to_hard_cancel_ms.toFixed(1)} ms`;
  }
  if (event.event_type === "AUDIBLE_STOP_ACK" && event.payload.speech_onset_to_audible_stop_ms != null) {
    ui["metric-audible-stop"].textContent = `${event.payload.speech_onset_to_audible_stop_ms.toFixed(1)} ms`;
  }
  if (event.event_type === "BACKCHANNEL_RECOVERY") {
    ui["metric-backchannel-recovery"].textContent = `${event.payload.backchannel_recovery_latency_ms.toFixed(1)} ms`;
  }
  if (event.event_type === "FALSE_INTERRUPTION_DETECTED") {
    const count = Number(ui["metric-false-interruptions"].dataset.count || 0) + 1;
    ui["metric-false-interruptions"].dataset.count = String(count);
    ui["metric-false-interruptions"].textContent = String(count);
  }
  if (event.event_type === "PARTIAL_TRANSCRIPT" && event.payload.first_stt_partial_ms != null) {
    ui["metric-first-partial"].textContent = `${event.payload.first_stt_partial_ms.toFixed(1)} ms`;
  }
  if (event.event_type === "FINAL_TRANSCRIPT" && event.payload.final_stt_ms != null) {
    ui["metric-final-stt"].textContent = `${event.payload.final_stt_ms.toFixed(1)} ms`;
  }
  if (event.event_type === "AGENT_AUDIO_COMPLETED") ui.delivered.textContent = event.payload.delivered_text || "—";
  if (event.event_type === "RECOVERY_STARTED") ui.status.textContent = "Providerfehler · weiter im Hörmodus";
  if (event.event_type === "STT_PROVIDER_FAILED") {
    ui.status.textContent = "Streaming-STT ausgefallen · kein Browser-Fallback";
    setProvider("stt", "ElevenLabs Scribe · REAL · FAILED", "unavailable");
  }
  if (
    event.event_type === "TRANSCRIPT_PROVENANCE_RECORDED"
    && event.payload.source === "streaming_stt"
  ) {
    const transcriptId = event.payload.transcript_id || "unknown";
    const text = event.payload.raw_text || "";
    if (event.payload.is_partial) {
      showPartialTranscript(text);
      appendDebug(ui["debug-stt-partial"], `${transcriptId} · ${text}`);
    } else {
      appendDebug(ui["debug-stt-final"], `${transcriptId} · ${text}`);
      partialTranscriptItem?.remove();
      partialTranscriptItem = null;
      if (event.payload.accepted_as_user_turn) {
        showAcceptedUser(text, event.payload);
        appendDebug(ui["debug-user-accepted"], `${transcriptId} · ${text}`);
      } else if (event.payload.rejection_reason) {
        appendDebug(ui["debug-suppressed"], `${event.payload.rejection_reason} · ${text}`);
        addConversation("suppressed", text, "SUPPRESSED / Self-Echo or Invalid");
      }
    }
  }
  if (event.event_type === "TTS_PROVIDER_ACTIVATED" && event.payload.provider) {
    activateTtsProvider(event.payload.provider);
  }
  if (event.event_type === "TTS_PROVIDER_DEACTIVATED") renderProviders(declaredProviders);
  if (event.event_type === "AGENT_GENERATION_COMPLETED") {
    const roles = event.payload.conversation_history_roles || [];
    ui["debug-history-roles"].textContent = JSON.stringify(roles);
  }
}

ui.connect.addEventListener("click", async () => {
  try {
    const capabilities = browserCapabilities();
    renderProviders([]);
    if (!capabilities.microphone) throw new Error("getUserMedia-Mikrofon fehlt");
    await startMicrophone();
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const query = new URLSearchParams();
    if (inputOnlyMode) query.set("mode", "input-only");
    query.set("tts", ui["tts-ab-selection"].value);
    socket = new WebSocket(`${scheme}://${location.host}/ws?${query.toString()}`);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      setConnected(true);
      send({
        type: "pcm_stream_started",
        audio_capture_id: audioCaptureId,
        microphone_stream_id: microphoneStreamId,
        format: "pcm_s16le_16000_mono"
      });
      if (browserSttDiagnosticMode) startBrowserStt();
      send({
        type: "provider_capabilities",
        microphone_available: capabilities.microphone,
        tts_available: capabilities.tts,
        audio_capture_id: audioCaptureId,
        microphone_stream_id: microphoneStreamId,
        recognition_input_binding: "EXACT_GETUSERMEDIA_PCM16"
      });
    };
    socket.onclose = () => {
      setConnected(false);
      recognition?.stop();
      stopMicrophone();
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
        ui.conversation.replaceChildren();
        assistantItems.clear();
        partialTranscriptItem = null;
        ui["debug-history-roles"].textContent = "[] · CLEAN_NEW_SESSION";
        addConversation("system", `Neue saubere Session ${payload.conversation_id}`, "SYSTEM");
        activeTtsCandidate = payload.tts_ab_selection === "candidate"
          ? "B · Rebecca / eleven_v3_conversational"
          : "A · Rebecca / eleven_flash_v2_5";
        ui["voice-candidate"].textContent = activeTtsCandidate;
        renderProviders(payload.providers);
        ui["mic-source"].textContent = payload.input_topology?.microphone_source || "getUserMedia";
        ui["pcm-source"].textContent = payload.input_topology?.pcm_source || "pcm_s16le 16000 Hz mono";
        ui["browser-stt-status"].textContent = payload.input_topology?.browser_speech_recognition_production_status || "OFF";
        if (inputOnlyMode) {
          setProvider("reasoning", "DEAKTIVIERT · INPUT-ONLY-ISOLATION", "unavailable");
          setProvider("tts", "DEAKTIVIERT · INPUT-ONLY-ISOLATION", "unavailable");
          setProvider("tts-fallback", "DEAKTIVIERT · INPUT-ONLY-ISOLATION", "unavailable");
          ui.status.textContent = "online · INPUT-ONLY-DIAGNOSE";
        }
      }
      else if (payload.type === "pcm_stream_result") {
        ui["mic-source"].textContent = `getUserMedia · ${payload.audio_capture_id}`;
        ui["pcm-source"].textContent = `${payload.microphone_stream_id} · pcm_s16le 16000 Hz mono`;
        ui["browser-stt-status"].textContent = payload.browser_speech_recognition_production_status;
      }
      else if (payload.type === "audio_chunk") {
        if (!acceptAudioMeta(payload)) {
          nextAudioMeta = null;
          return;
        }
        nextAudioMeta = payload;
        showAssistantChunk(payload);
        appendDebug(ui["debug-assistant"], `${payload.response_id} · ${payload.text_boundary}`);
        if (payload.tts_provider?.provider) {
          activateTtsProvider(payload.tts_provider);
        }
      }
      else if (payload.type === "cancel_audio") cancelAudio(payload.chunk_id);
      else if (payload.type === "playback_duck") duckPlayback(payload);
      else if (payload.type === "playback_resume") resumePlayback(payload);
      else if (payload.type === "invalidate_playback") invalidatePlayback(payload);
      else if (payload.type === "transcript_result") {
        const final = payload.provenance?.event_kind === "USER_TRANSCRIPT_FINAL";
        if (payload.accepted && final) {
          showAcceptedUser(payload.raw_text, payload.provenance);
          appendDebug(ui["debug-user-accepted"], `${payload.provenance?.transcript_id || "unknown"} · ${payload.raw_text}`);
        } else {
          partialTranscriptItem?.remove();
          partialTranscriptItem = null;
          appendDebug(ui["debug-suppressed"], `${payload.rejection_reason} · ${payload.raw_text}`);
          if (payload.rejection_reason === "probable_assistant_self_speech") {
            addConversation("suppressed", payload.raw_text, "SUPPRESSED / Self-Echo");
          }
        }
      }
      else if (payload.type === "input_probe_transcript") {
        if (payload.final) showAcceptedUser(payload.text, payload.provenance);
        else showPartialTranscript(payload.text, "Diagnostic");
        appendDebug(ui["debug-user-accepted"], `${payload.provenance?.transcript_id || "input-only"} · ${payload.text}`);
      }
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
  provenance: diagnosticProvenance("manual_diagnostic"),
  signals: { speech_active: false, silence_duration_ms: 350, utterance_duration_ms: 900, semantic_complete: true, acoustic_completion: 0.9 }
}));
document.querySelectorAll(".quick button").forEach((button) => button.addEventListener("click", () => send({
  type: "transcript", source: "manual_diagnostic", text: button.dataset.phrase, final: true,
  provenance: diagnosticProvenance("manual_diagnostic"),
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
