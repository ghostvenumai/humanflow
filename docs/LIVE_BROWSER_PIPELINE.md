# Live browser conversation path

Status after the failed manual validation on 2026-08-23. This document labels
execution, not intended architecture. `REAL` means the component performs its
declared live function; it does not imply a production SLA or a passed manual
quality gate.

| Stage | Runtime component | Status | Evidence / limitation |
| --- | --- | --- | --- |
| Browser microphone | `getUserMedia` + Web Audio capture | REAL | Captures the selected microphone with browser echo cancellation, noise suppression and automatic gain control. |
| Audio transport | binary PCM16 over `/ws` | REAL | Downsampled 16 kHz mono frames reach `RealtimeVoiceSession.receive_audio` while playback is active. |
| Server STT from PCM | `NullTranscriber` | MOCK / NOT USED FOR TRANSCRIPT | It deliberately consumes the PCM frames without producing text. The live demo does not claim server-side PCM transcription. |
| Live STT | browser Web Speech `SpeechRecognition`, `de-DE` | REAL, BROWSER-DEPENDENT | Emits actual interim and final hypotheses. The demo now fails visibly when this provider is unavailable. |
| Partial/final transcript transport | JSON `transcript` messages | REAL | Each hypothesis is validated and recorded as `PARTIAL_TRANSCRIPT` or `FINAL_TRANSCRIPT` with provider identity. |
| Turn detector | `HybridTurnPolicy` | REAL | Uses transcript, finality, measured browser `speechstart`/`speechend` durations, backchannel vocabulary and explicit German interruption phrases. A final Web Speech hypothesis is an explicit provider endpoint and no longer relies on fabricated fixed silence/duration values. This is not a neural end-of-turn model. |
| Conversation controller | `RealtimeVoiceSession` + `ConversationStateMachine` | REAL | Owns listening/thinking/speaking/recovery state, operation epochs and cancellation. |
| Reasoning / LLM | `AnthropicReasoner` via Messages API | REAL | Uses `claude-haiku-4-5-20251001` by default, streams German semantic chunks and retains up to twelve user/assistant turns per WebSocket session. No API key means startup failure; there is no echo fallback. |
| Response text | Anthropic streamed text | REAL | System instructions require relevant German answers and honest Web/weather/calendar limitations. The former canned acknowledgement is not in the demo path. |
| TTS | browser Web Speech `speechSynthesis`, `de-DE` | REAL, BROWSER-DEPENDENT | Speaks each semantic response boundary using the selected system voice. Missing browser TTS is a visible hard failure. |
| Server TTS envelope | `BrowserSpeechSynthesisAdapter` | TRANSPORT ADAPTER | Produces silent duration/sample envelopes for the transport-neutral ledger. These bytes are never rendered and are not represented as synthesized speech. |
| Streaming playback | ordered `SpeechSynthesisUtterance` chunks | REAL | Browser start/end/error callbacks acknowledge each chunk. Chunks are ordered by the server output transport. |
| Interruption / cancellation | turn policy or stop button → `speechSynthesis.cancel()` | REAL WITH CONSERVATIVE BOUNDARY | The browser explicitly acknowledges audible stop. Because Web Speech exposes no exact rendered sample offset, interrupted speech reports zero played samples instead of inventing a partial delivery metric. |
| Telemetry | state-machine events mirrored over WebSocket | REAL | Includes STT identity, reasoning provider/model/mode, first-output latency, generation duration, provider token usage, TTS identity, playback receipts, turn decisions and recovery events. |

## Root cause of the failed validation

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
failure was correct and remains unresolved until a human validates the new path.

## Fail-closed contract

`make demo` may start only with a configured real reasoning provider. Browser
connection may proceed only when both Web Speech recognition and synthesis are
available. Deterministic reasoners and tone generators remain available to unit,
golden and torture tests, but are not selectable by the live demo runtime. The
visible manual text control and automated WebSocket smoke input are explicitly
tagged `MOCK` diagnostic transcript sources in telemetry and cannot be mistaken
for browser STT evidence.
