# HumanFlow — Final Report (Everlast Frozen Candidate)

Generated: 2026-08-26 · Human-validated browser/audio release candidate

## System
- Canonical repo: `/home/serverserver/HumanFlow-PreStart/humanflow/`
- Working branch: `claude/humanflow-finish`
- **Frozen runtime base: `ea3677e`** ("fix: retry transient pre-playback TTS failures")
- Runtime: Python 3.12, `anthropic` 0.84.0, FastAPI; reasoning model default `claude-haiku-4-5-20251001`
- Providers: ElevenLabs Scribe Realtime (STT), Anthropic Messages API (reasoning),
  ElevenLabs Flash v2.5 (streaming TTS), local SQLite appointment tools
- Historical freeze tag `everlast-72h-build` @ `798256e` is unchanged (not moved).

## Final status: **FROZEN — human-validated browser/audio release candidate**
Not telephony-production-certified. See claim boundaries below.

## Validated runtime lineage (on `claude/humanflow-finish`)
```
ea3677e  fix: retry transient pre-playback TTS failures        <- frozen runtime base
0d81b56  fix: resolve German relative-week dates
3b098ff  fix: recover stalled pre-playback responses
1692093  fix: ground appointment availability follow-ups
```

## Automated validation — 367 passed (UNIT + INTEGRATION + SIMULATED)
Full suite green on `ea3677e`. Deterministic regression coverage for:
- grounded appointment availability (`search_availability` grounding)
- German relative-week date resolution (`in N Wochen` = N full weeks)
- "Mittwoch in zwei Wochen" distinct from "nächsten Mittwoch"
- pre-playback TTS stall recovery (no permanent THINKING)
- COMPLETE Turn-B takeover during a stalled pre-playback response
- stale/late audio protection (cancelled audio never re-enters state)
- transient pre-first-audio TTS retry (same grounded text, once)
- no duplicate reasoner/tool execution during a speech retry
- internal non-TTS exceptions not misclassified as a TTS retry
- cancellation / barge-in bypassing speech retry
- deterministic LISTENING recovery on persistent provider failure

## Real browser / real audio validation (REAL_BROWSER · REAL_AUDIO)
Runtime `ea3677e` was exercised end-to-end through the real browser, real microphone,
ElevenLabs Scribe STT, Claude reasoner and ElevenLabs streaming TTS.

- **Session `9a1ad0a6-f106-4f0b-b0ca-d8d2e53b0834`** — clean browser session; German
  voice input; Orthopädie appointment intent; "Mittwoch in einer Woche" correctly
  resolved to Wednesday, 2 September 2026; grounded demo availability; appointment
  booked for 11:30; follow-up to check/change available times; "für den gleichen Tag"
  correctly resolved against existing appointment state; availability queried for the
  same date; existing booking retained; several further natural German turns; clean
  conversational termination through "Ciao".
- **Observed:** no permanent THINKING state; no lost user turn; no stale/late assistant
  audio; no unexpected hangup. Operator assessment: "lief perfekt".
- **Temporal live confirmation (separate session):** "Mittwoch in zwei Wochen" from
  Wednesday 2026-08-26 resolved to Wednesday **2026-09-09** (the previous incorrect
  2026-09-02 is fixed). A TTS failure in that earlier session exposed the transient
  pre-playback TTS path, subsequently addressed by `ea3677e`.

This is REAL_BROWSER / REAL_AUDIO evidence, not a synthetic benchmark.

## Metrics (SIMULATED / local scope — NOT real-call latency)
Latency figures are local event-loop / PCM-sink timings, explicitly not
browser/telephony latency; only p50 is published by the scorecard.

| Metric | Observed (p50) | Target | Scope |
|---|---|---|---|
| Time to first audio | 11.4 ms | < 700 ms | local_event_loop_transport (SIMULATED) |
| Audible barge-in latency | 1.85 ms | < 250 ms | local_timed_pcm_sink (SIMULATED) |
| False interruption rate | 0.0 | < 5% | protected German fixture corpus (UNIT) |
| Premature endpoint rate | 0.0 | < 5% | protected German fixture corpus (UNIT) |
| German interruption accuracy | 1.0 | > 0.95 | protected German fixture corpus (UNIT) |
| Call completion rate | 1.0 | > 98% | synthetic torture termination (SIMULATED) |
| Tool failure recovery | 1.0 | > 95% | local deterministic fault injection (SIMULATED) |
| Unexpected hangups | 0 | 0 | synthetic torture termination (SIMULATED) |

p95/p99 for real-call latency are **not established** (no real-call dataset).

## Claim boundaries (explicit)
- **Real browser / real audio multi-turn conversation: VALIDATED** (session above).
- **Hidden acceptance suite: NOT RUN.** No operator-owned hidden acceptance suite has
  executed (`config/engineering-harness.yaml`: `hidden_acceptance.configured: false`,
  `external_agent_execution: false`). No hidden-acceptance PASS is claimed.
- **Production telephony / PSTN: NOT ESTABLISHED.** No production telephony validation,
  no PSTN validation, no production SLA compliance claim. Production release claim
  remains `TELEPHONY_PRODUCTION_NOT_ESTABLISHED`.
- Synthetic/local latency metrics are labeled SIMULATED and are not real-call metrics.

## Known risks / limitations
- Real-call (telephony) latency is unmeasured; only local proxies exist.
- Hidden acceptance suite is operator-owned and has not run.
- Live path depends on external providers (Scribe / Anthropic / ElevenLabs); a
  persistent provider outage degrades to deterministic LISTENING recovery.

## Final status
**FROZEN — human-validated browser/audio release candidate on `ea3677e`.**
Automated: 367 passed. Real browser/audio multi-turn: validated. Hidden acceptance:
not run. Production telephony: not established.
