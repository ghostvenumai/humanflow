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

## Invariants

1. VAD activity is never equated with conversational turn completion.
2. An internal cancellation is not an audible-stop measurement.
3. Generated content is not assumed to have been heard.
4. Late async results cannot mutate current state after epoch invalidation.
5. Exceptions enter explicit recovery/failure states; they do not implicitly
   end a call.
6. Every metric must be derivable from raw events with sample counts.

## Next runtime components

1. Played Audio Ledger with queued/played/cancelled semantic boundaries.
2. Deterministic replay runner over the protected German corpus.
3. Tool router with latency, timeout, malformed, duplicate and failure injection.
4. Metrics aggregation and baseline comparison.
5. Browser audio demo through replaceable provider adapters.

