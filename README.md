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
