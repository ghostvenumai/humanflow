# Voice Failure Taxonomy

| Code | Class | Meaning |
|---|---|---|
| `TURN_PREMATURE_ENDPOINT` | Turn detection | Caller was cut off before conversational completion. |
| `TURN_MISSED_ENDPOINT` | Turn detection | Complete turn was not accepted within its latency gate. |
| `INTERRUPTION_FALSE_POSITIVE` | Interruption | Backchannel, noise or background speech cancelled output. |
| `INTERRUPTION_FALSE_NEGATIVE` | Interruption | Intentional takeover failed to stop output. |
| `BARGE_IN_AUDIBLE_LATE` | Playback | Internal cancellation occurred but audible output stopped late. |
| `LEDGER_DELIVERY_MISMATCH` | Context | Memory included assistant content not actually heard. |
| `OVERLAP_STATE_RACE` | Concurrency | Async timing, rather than policy, determined overlap behavior. |
| `STALE_RESULT_APPLIED` | Concurrency | Cancelled/obsolete async work changed current state. |
| `TOOL_LOOP_BLOCKED` | Tooling | Tool latency froze the realtime audio/event loop. |
| `TOOL_RECOVERY_FAILED` | Recovery | Injected failure did not reach a controlled recovery outcome. |
| `UNEXPECTED_HANGUP` | Lifecycle | Call ended without an expected termination reason. |
| `TELEMETRY_INCOMPLETE` | Evidence | Timeline cannot reconstruct the measured behavior. |
| `SENSITIVE_TRACE_DATA` | Security | Trace contains secret or unapproved personal information. |

Validated failures should become sanitized regression fixtures with stable IDs,
labels, expected decisions and provenance.

