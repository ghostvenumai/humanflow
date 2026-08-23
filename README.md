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

Then open `http://127.0.0.1:8765`. The browser sends live PCM16 microphone frames
while the agent output is active. Playback start/completion/cancellation is
acknowledged by the browser; those acknowledgements define the demo ledger's
played boundary. The included synthesizer emits timed tones and is clearly labeled
as a mock provider rather than production speech.
