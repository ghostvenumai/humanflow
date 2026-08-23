# HumanFlow Metric Definitions

This document is a protected evaluation artifact. Metric names, formulas,
labels, gates, and sample requirements must not be weakened to manufacture an
improvement.

## Timing rules

All durations use monotonic timestamps from runtime events. Wall-clock time is
retained for provenance only. Latency reports include `n`, P50, P90, P95 and
P99; a percentile without a sample count is incomplete.

## Metrics

### Time to first audio (`ttfa_ms`)

```text
AGENT_AUDIO_STARTED.monotonic_ns - TURN_CONFIRMED.monotonic_ns
```

The start event must represent audio becoming audible, not merely generation or
queue insertion.

### Intentional audible barge-in (`audible_barge_in_latency_ms`)

```text
timestamp(agent audio actually stopped)
- timestamp(intentional user speech onset detected)
```

The end is verified from the playback ledger/buffer, never from `cancel()`.

### False interruption rate

```text
false intentional-interruption predictions / labeled non-interruption cases
```

Backchannels, background speech and transient noise are non-interruption cases.

### Premature endpoint rate

```text
turns ended before labeled conversational completion / labeled user turns
```

### Call completion rate

```text
calls ending in an expected termination reason / all started calls
```

### Tool failure recovery rate

```text
injected tool failures reaching RECOVERY_COMPLETED / injected tool failures
```

### Unexpected hangups

Count of `CALL_ENDED` events with an unexpected termination reason. The gate is
zero, so this metric is not reported as a rounded percentage.

### German interruption accuracy

Accuracy across labeled German intentional-interruption and non-interruption
examples. A confusion matrix is required alongside the scalar value.

