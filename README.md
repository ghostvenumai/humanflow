# HumanFlow

HumanFlow is a measurable German realtime conversation engine focused on turn
detection, interruption control, backchannel tolerance, played-audio truth,
failure recovery, and evidence-driven self-improvement.

The repository is developed under an explicitly authorized 72-hour sprint. The
authoritative start evidence is written by `./go` and preserved under `sprint/`.

## Sprint controls

```bash
./go          # one-time start; refuses a second start
make status   # elapsed time and remaining sprint window
make test     # automated tests
make realtime-benchmark  # measured local controller/queue/cancellation timings
make torture-test        # tests plus executable T01-T20 contract scenarios
make replay              # capture and strictly replay raw JSONL telemetry
make scorecard           # build evidence-linked local quality gates
```

No GitHub push or deployment is performed automatically.

## Release validation and freeze

Automated readiness is available through `make release-readiness`. The final
72-hour tag is guarded by a human browser/audio attestation; see
`docs/MANUAL_VALIDATION.md`. Automation must not create that attestation.

```bash
./freeze-72h --confirm-freeze
```

The command re-runs the complete evidence suite and creates only local freeze
evidence/tagging. It does not push or deploy.

## Realtime core

`RealtimeVoiceSession` accepts validated PCM16 frames without waiting for agent
playback, consumes transcript/turn signals, streams reasoner fragments through
synthesis, and drives a cancel-aware audio sink. `PlayedAudioLedger` records what
was generated, queued, fully played, partially played, and cancelled. Its
delivered-text view includes only complete played semantic chunks.

The dependency-free local adapters are intended for deterministic runtime and
cancellation testing. Reports label these measurements as local event-loop data;
they are not presented as browser, sound-device, telephony, provider, or real-call
quality measurements.

The generated scorecard deliberately distinguishes `PASS_LOCAL_EVIDENCE` from a
production claim. Without a real-call dataset, the production release claim remains
`NOT_ESTABLISHED_NO_REAL_CALL_DATA`, even when all local engineering gates pass.

## Browser demo

Install the local demo extra and start the loopback-only server:

```bash
python3 -m pip install -e '.[demo]'
make demo
```

Set `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` in
`$HOME/.config/humanflow/runtime.env`, then open `http://127.0.0.1:8765`. `make
demo` loads that file without printing its contents. The ElevenLabs key must have
Speech-to-Text access; alternatively set a separately scoped
`ELEVENLABS_STT_API_KEY`. The browser sends one live PCM16 microphone stream to
ElevenLabs Scribe Realtime, which emits German partial and committed transcripts.
Browser SpeechRecognition is OFF in production and exists only behind the explicit
diagnostic query mode. A session-scoped Anthropic reasoner streams context-aware German
answers into a modular ElevenLabs Flash v2.5 PCM stream. Browser Web Speech remains
a visibly reported TTS fallback for transient provider failures; invalid credentials
or voice configuration fail closed. Playback start, completion and cancellation
are acknowledged by the browser and define the demo ledger's played boundary. The
live demo never substitutes deterministic echo/tone adapters used by tests. This
remains local browser evidence until the separate human validation passes.

See `docs/LIVE_BROWSER_PIPELINE.md` for the real/mock boundary at every stage.
