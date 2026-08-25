# HumanFlow Engineering Harness — Repository Audit

Audit date: 2026-08-25  
Audited branch: `humanflow/appointment-tools`  
Audited HEAD: `843b90a949d488640482a88a50475ad5b12b7f7f`  
Immutable 72h baseline: `798256e447a7d7cb116d2186035f4a85b29744e2`
(`everlast-72h-build`)

## Architecture summary

HumanFlow already separates the live voice runtime from offline development
evaluation. The live path owns authoritative browser PCM, Scribe STT, turn and
interruption control, Claude reasoning, transactional SQLite tools, ElevenLabs
TTS, playback ownership, Played Audio Ledger, recovery and runtime/cost
telemetry. It does not depend on coding agents.

The offline side already contains useful primitives: protected-artifact hashes,
full worktree evaluation, KEEP/REVERT comparison, a bounded finish-loop state,
an explainable static development router, non-spending Codex/Claude adapters and
a same-baseline tournament evaluator. Several historical Git worktrees prove
that isolated development has been used manually.

What does not yet exist is the system joining those primitives into an
autonomous engineering organization: there is no authoritative HF task
registry, improvement discovery engine, worktree manager, conflict-aware queue,
independent reviewer orchestration, hidden acceptance boundary, circuit breaker,
merge-candidate gate or post-release measurement loop.

## Capability matrix

| Capability | Status | Evidence / existing files | Risk | Recommended action |
|---|---|---|---|---|
| Live STT → controller → LLM/tools → TTS → playback | IMPLEMENTED | `runtime/session.py`, `web/app.py`, `docs/LIVE_BROWSER_PIPELINE.md`; 303-test full regression | High if coupled to harness | Keep live runtime outside engineering execution dependencies |
| PCM provenance, Final Admission and self-speech protection | IMPLEMENTED | `runtime/final_admission.py`, `runtime/self_speech.py`, transcript-integrity fixtures | High | Treat as protected behavior and require relevant regression |
| Barge-in, backchannel and stale-audio safety | IMPLEMENTED | `runtime/acoustic_barge_in.py`, Played Audio Ledger, realtime tests | High | Never optimize from engineering-loop metrics without protected voice gates |
| Transactional appointment engine | IMPLEMENTED | `appointment_state.py`, `appointment_coordinator.py`, `sqlite_appointments.py` | High | Use as protected application acceptance domain |
| Runtime telemetry envelope and redaction | IMPLEMENTED | `telemetry/events.py`, `telemetry/sinks.py`, schema and tests | Medium | Reuse event/evidence conventions; keep engineering data separate |
| Runtime Cost Ledger | IMPLEMENTED | `cost/*`, SQLite persistence, reporting and failure-isolation tests | Low | Reuse evidence semantics; do not put accounting on live critical path |
| Deterministic replay, Golden Turns and T01–T20 | IMPLEMENTED | `evaluation/*`, `tests/golden/*`, `scripts/run_torture.py` | Low | Make them protected verification inputs |
| Offline Quality KEEP/REVERT | IMPLEMENTED | `quality/loop.py`, worktree evaluator and comparison scripts | Medium | Reuse as verifier primitive, not as scheduler |
| Protected artifact hashing | PARTIALLY_IMPLEMENTED | `config/loop.yaml`, `_protected_hashes()` | High | Add candidate write-scope/integrity comparison against trusted baseline |
| Hidden acceptance tests | MISSING | No supervisor-only hidden suite or invocation boundary found | High | Add opaque command interface and result contract; do not expose assertions |
| Independent reviewer | MISSING | Manual one-off audits exist, no orchestrator/state/evidence model | High | Add fresh read-only review job and required verdict |
| Engineering task model | PARTIALLY_IMPLEMENTED | `development/models.py` has routing-only `EngineeringTask` | Medium | Extend via separate registry record; do not overload routing DTO |
| Machine-readable HF task registry | MISSING | No `.engineering/feature_list.json` or state-transition authority | High | Add versioned JSON registry and validated transitions |
| ProblemCandidate / discovery engine | MISSING | Runtime evidence exists, no detectors or candidate model | High | Build deterministic detector layer over sanitized fixtures |
| Duplicate problem/task suppression | MISSING | No fingerprint registry | Medium | Normalize evidence fingerprints and link repeated evidence |
| Worktree manager | MISSING | Four manually created Git worktrees; no lifecycle manager | High | Add safe explicit-path worktree create/inspect/cleanup service |
| Path/dependency conflict detector | MISSING | No allowed-path overlap analysis | High | Serialize overlaps before permitting parallel execution |
| Parallel scheduler | MISSING | No queue or worker-cap enforcement | High | Start with configurable maximum 2; fail closed on conflict |
| CLI development adapters | PARTIALLY_IMPLEMENTED | Codex and Claude adapters build commands and default-deny execution | Medium | Preserve authorization gate; attach task/worktree/evidence contracts |
| Static development router | PARTIALLY_IMPLEMENTED | Risk/category router with budget cap | Medium | Use initially; add historical evidence only after sufficient samples |
| Tournament evaluator | PARTIALLY_IMPLEMENTED | Same-baseline scoring and no-winner behavior | Medium | Do not execute until worktree, integrity and reviewer gates exist |
| Stuck detection / circuit breaker | MISSING | Only fixed finish-loop iteration limits | High | Add normalized failure fingerprints, repeat/no-progress/oscillation rules |
| Per-task/agent harness metrics | MISSING | Runtime/cost metrics exist; no engineering task history | Medium | Persist nullable duration/iteration/test/review/cost evidence |
| Merge candidate gate | MISSING | Candidate comparison exists, no latest-main revalidation orchestration | High | Require integrity, tests, hidden acceptance, review and updated-baseline proof |
| Release readiness | PARTIALLY_IMPLEMENTED | `release_readiness.py`, freeze evidence, manual validation gate | High | Keep production manual; add post-freeze candidate evidence bundle only |
| Rollback triggers/post-release measurement | MISSING | No release-to-task measurement linkage | High | Generate evidence/alerts only; no autonomous production rollback initially |
| Unified `hf-loop` operator CLI | MISSING | Several focused scripts exist | Low | Wrap existing primitives only after core domain/state is stable |
| Deployment/staging separation | PARTIALLY_IMPLEMENTED | Live runtime cannot self-modify; no automated deployment mechanism found | High | Preserve manual deployment boundary; do not add deployment in initial phases |

## Existing components to reuse

1. `TelemetryEvent`, JSONL and in-memory sinks for reason-coded evidence.
2. `evaluate_worktree()` and `compare_candidates()` for protected evaluation.
3. `FinishLoopState` and the frozen-tag/ancestor guard.
4. `DevelopmentModelRouter`, `CodexCliAdapter` and `ClaudeCliAdapter`.
5. `TournamentEvaluator` for later same-baseline candidate comparison.
6. Golden Turn, T01–T20, appointment, Final Admission, realtime, recovery,
   Played Audio and Cost Ledger suites as protected acceptance domains.
7. Existing report and dashboard conventions for truthful evidence scopes.

## Highest-risk gaps

1. A worker can currently be pointed at a worktree, but no coordinator enforces
   task path scope, protected paths, branch identity or trusted-baseline hashes.
2. There is no authoritative state machine deciding who may change task status
   or declare verified completion.
3. There is no hidden acceptance boundary or required independent-review
   verdict, so visible green tests alone could be over-trusted.
4. There is no failure fingerprint/circuit breaker; an external worker runner
   could repeat the same failure until budget exhaustion.
5. Telemetry can describe runtime behavior, but nothing conservatively converts
   repeated evidence into deduplicated problem candidates and proposed tasks.
6. Existing reports in the main worktree are runtime-generated and currently
   modified; harness implementation must not overwrite or misclassify them as
   clean candidate evidence.

## Minimal phase plan and proof

| Phase | Minimal deliverable | Proof |
|---|---|---|
| 1 | This factual audit and frozen-boundary confirmation | Audit file, tag/ancestor checks, existing full regression evidence |
| 2 | Engineering evidence domain with privacy-safe references | Schema/model tests, redaction and null-metric tests |
| 3 | Versioned task registry and coordinator-owned transitions | Schema validation, invalid-transition, dedupe and atomic persistence tests |
| 4 | Safe worktree lifecycle using explicit paths/branches | Temporary-repository create/isolate/inspect/cleanup tests |
| 5 | Allowed/protected-path conflict detector and max-2 scheduler | Overlap serialization, queue and worker-limit tests |
| 6 | Fresh read-only reviewer job and verification record | Reviewer separation and fail-closed verdict tests |
| 7 | Protected/hidden command boundary and integrity gate | Unauthorized test edit, skip/xfail and hidden-command result tests |
| 8 | Failure fingerprint, no-progress and circuit breaker | Third-repeat stop, retry cap, oscillation and diagnostic-package tests |
| 9 | Nullable per-task/per-agent execution metrics | Aggregation, missing-cost and verified-success denominator tests |
| 10 | Tournament orchestration using existing evaluator | Shared-baseline, no-winner and protected-change rejection tests |
| 11 | Conservative deterministic problem detectors | Healthy/noisy/regression/drift/critical synthetic fixtures |
| 12 | Deduplicated task proposals with autonomy policy | Threshold, duplicate link and high-risk approval tests |
| 13 | Evidence-informed routing with minimum sample policy | Cold-start fallback, exploration and quality-gate precedence tests |
| 14 | Merge-candidate bundle and post-release evidence state | Latest-main revalidation, release reject and rollback-trigger tests |

## Audit conclusion

Do not rewrite the live runtime or replace the existing quality/router modules.
The smallest coherent next increment is Phase 2 plus Phase 3: define the
engineering evidence/task contracts and their authoritative state transitions.
No worker, worktree scheduler or autonomous agent execution should be enabled
until those contracts and integrity rules are tested.
