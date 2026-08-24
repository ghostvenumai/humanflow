# Live browser conversation path

Status after the failed manual validation and architecture correction on 2026-08-24. This document labels
execution, not intended architecture. `REAL` means the component performs its
declared live function; it does not imply a production SLA or a passed manual
quality gate.

| Stage | Runtime component | Status | Evidence / limitation |
| --- | --- | --- | --- |
| Browser microphone | `getUserMedia` + Web Audio capture | REAL | Captures the selected microphone with browser echo cancellation, noise suppression and automatic gain control. |
| Audio transport | binary PCM16 over `/ws` | REAL | Downsampled 16 kHz mono frames reach `RealtimeVoiceSession.receive_audio` while playback is active. |
| Production STT from PCM | `ElevenLabsRealtimeSTTProvider` / Scribe v2 Realtime | IMPLEMENTED, LIVE AUTH BLOCKED | Binds immutably to the exact `getUserMedia` capture and stream IDs. The sanitized live handshake result is currently `auth_error`; there is no fallback to Browser SpeechRecognition. |
| Browser SpeechRecognition | Web Speech `SpeechRecognition`, `de-DE` | OFF PRODUCTION / MOCK DIAGNOSTIC | Only starts with the explicit diagnostic query flag. Its events are rejected from the production conversation path. |
| Partial/final transcript path | internal Scribe events | IMPLEMENTED | Partial and provider-final hypotheses remain ephemeral. Only `committed_transcript` becomes FINAL and can enter the authoritative user gate. |
| Turn detector | `HybridTurnPolicy` | REAL | Uses Scribe transcript state, server VAD endpointing, backchannel vocabulary and explicit German interruption phrases. This is not a neural end-of-turn model. |
| Conversation controller | `RealtimeVoiceSession` + `ConversationStateMachine` | REAL | Owns listening/thinking/speaking/recovery state, operation epochs and cancellation. |
| Reasoning / LLM | `AnthropicReasoner` via Messages API | REAL | Uses `claude-haiku-4-5-20251001` by default, streams German semantic chunks and retains up to twelve user/assistant turns per WebSocket session. No API key means startup failure; there is no echo fallback. |
| Response text | Anthropic streamed text | REAL | System instructions require relevant German answers and honest Web/weather/calendar limitations. The former canned acknowledgement is not in the demo path. |
| Prosody planner | `ProsodyPlanner` | REAL, DETERMINISTIC | Preserves model wording while mapping stable phrases to intent, rate, stability, style and short semantic pauses. It never inserts random fillers. |
| Primary TTS | `ElevenLabsStreamingTTSProvider` | REAL, LIVE SMOKE PASS | Uses `eleven_flash_v2_5` and streams 16 kHz PCM. The latest 36-character smoke produced eight chunks and 2.461 seconds of audio; measured first PCM latency was 1277.948 ms. Voice quality remains a human-only judgment. |
| TTS fallback | `BrowserSpeechSynthesisAdapter` | REAL, BROWSER-DEPENDENT FALLBACK | May handle transient primary failures and is always shown in provider telemetry. Credential and voice-configuration errors fail closed and may not hide behind this fallback. |
| Streaming playback | Web Audio PCM player | REAL | PCM begins before a complete response is synthesized. Browser start/end/stop callbacks acknowledge every ordered audio chunk; only final semantic boundaries advance delivered text. |
| Interruption / cancellation | turn policy or stop button → provider stream close → queued chunk invalidation → `AudioBufferSourceNode.stop()` | REAL WITH MANUAL AUDIBLE CHECK REQUIRED | The browser records player stop-callback latency and AudioContext latency. These are not treated as proof of the human-perceived audible stop; that remains a manual rating. |
| Telemetry | state-machine events mirrored over WebSocket | REAL | Includes STT and reasoning identity, first stable LLM segment, TTS request, first PCM, browser playback start, actual active TTS provider/model/mode, playback receipts, sink latency, turn decisions and recovery events. |

## Root cause and architecture correction

`src/humanflow/web/app.py` previously instantiated three deterministic demo
adapters directly:

- `NullTranscriber()` for server PCM ingestion;
- `GermanDemoReasoner()`, whose only output was the hard-coded sentence
  `Ich habe Sie verstanden. Sie sagten: ...`;
- `ToneSpeechSynthesizer()` as a timing envelope.

The browser supplied genuine microphone input, genuine Web Speech transcripts
and genuine browser speech playback, but the semantic response itself was always
the local acknowledgement stub. Automated controller, ledger and timing results
therefore could not establish conversational quality. The manual validation
failure was correct. Later human evidence proved that Browser SpeechRecognition's
independent browser-managed audio path could still transcribe assistant playback and
label it as user speech, even after provenance and similarity guards were added.

The production path therefore no longer accepts Browser SpeechRecognition. One
`getUserMedia` PCM stream now owns VAD, turn signals, Scribe transcription and
interruption provenance. The similarity guard remains a secondary acoustic-leakage
defense, not the user/assistant trust boundary.

## Fail-closed contract

`make demo` constructs only configured real Scribe, reasoning and ElevenLabs TTS
providers. Scribe authentication failures stop the PCM input path and emit
`STT_PROVIDER_FAILED`; they never activate Browser SpeechRecognition. Web Speech
synthesis is a visible TTS fallback, not the primary quality provider. Invalid
credentials or voice configuration fail closed. Deterministic reasoners and tone
generators remain available to unit, golden and torture tests, but are not
selectable by the live demo runtime. The visible manual text control and automated
WebSocket smoke input are explicitly tagged `MOCK` diagnostic transcript sources
in telemetry and cannot be mistaken for browser STT evidence.
