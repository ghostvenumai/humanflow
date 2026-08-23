# HumanFlow Architecture

## Design boundary

HumanFlow separates the realtime conversation runtime from the development
optimization system. Live calls never depend on Codex or Claude. Provider APIs
are adapters; the HumanFlow controller owns turn-taking, cancellation,
recovery, delivered-audio truth, and conversational state.

## Runtime flow

```text
browser / caller
  -> audio transport and frame timing
  -> acoustic/VAD and transcript signals
  -> turn decision policy
  -> explicit conversation state machine
  -> reasoning and tool adapters
  -> streaming synthesis
  -> playback buffer and cancellation

all transitions and meaningful timings
  -> telemetry envelope
  -> deterministic replay
  -> metrics and scorecard
```

## Current foundation

- `humanflow.turns`: deterministic fixed-silence baseline and first hybrid
  policy with separate decisions for completion, interruption, backchannel, and
  uncertainty.
- `humanflow.controller`: explicit lifecycle graph. Illegal state transitions
  fail closed and every accepted transition emits reason-coded telemetry.
- `humanflow.telemetry`: timestamped, correlated and sequenced event envelope
  with key-based secret redaction plus in-memory and JSONL sinks.
- Operation epochs reject late results after interruption, cancellation, or
  recovery invalidates outstanding async work.
- `humanflow.runtime`: one asynchronous session owns independent PCM input and
  streaming-response tasks. Input remains accepted while output is paced, and
  intentional interruption is complete only after the sink returns an actual
  playback-stop timestamp.
- `humanflow.audio`: validated PCM16 frames plus an append-only Played Audio
  Ledger. Generated, queued, partially played, completely played, and cancelled
  chunks remain distinct; conversation memory receives only complete semantic
  chunks.
- Provider protocols isolate STT, reasoning, TTS, and audio transport. The local
  tone/output adapters exercise timing and cancellation but explicitly make no
  speech-quality or production-provider claim.
- `humanflow.tools`: bounded, timeout-enforced provider execution with explicit
  `TOOL_WAIT` and `RECOVERING` transitions. Injected latency, failure, timeout,
  malformed and duplicate responses cannot write stale state; exhausted attempts
  use an explicit safe fallback.
- `humanflow.web`: a loopback FastAPI/WebSocket demo streams browser microphone
  PCM into the core. Browser playback acknowledgements establish actual start,
  completion, partial-play and cancellation boundaries; missing stop acknowledgements
  enter safe handoff and never manufacture an audible-stop metric.

## Invariants

1. VAD activity is never equated with conversational turn completion.
2. An internal cancellation is not an audible-stop measurement.
3. Generated content is not assumed to have been heard.
4. Late async results cannot mutate current state after epoch invalidation.
5. Exceptions enter explicit recovery/failure states; they do not implicitly
   end a call.
6. Every metric must be derivable from raw events with sample counts.

## Next runtime components

1. Deterministic event replay, metrics aggregation, and baseline comparison.
