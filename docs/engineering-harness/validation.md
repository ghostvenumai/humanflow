# Engineering Harness Validation — 2026-08-25

Validated post-freeze commit: `7be3f0ef41d5f2c46145ae65ec3d9007c1bf05b9`  
Immutable baseline: `798256e447a7d7cb116d2186035f4a85b29744e2`  
Freeze tag: `everlast-72h-build` (dereferences to the immutable baseline)

## Independent audit

One independent read-only adversarial audit reproduced eight concrete harness
issues. The fixes were independently rechecked: seven were fixed immediately;
the remaining partial merge-evidence finding was then closed by requiring
`record_verified_merge()` to validate a full Git object ID, containment in
current main and verifier evidence. The auditor made no repository changes and
made no provider calls.

## V3 definition-of-done evidence

| Scenario | Evidence | Result |
|---|---|---|
| A: independent parallel tasks | Two barrier-synchronized workers, distinct worktrees and candidate commits | PASS |
| B: path conflict serialization | Conservative scheduler overlap fixture | PASS |
| C: protected-test mutation | Deep recursive path and weakening-pattern fixtures | PASS |
| D: third repeated failure | Circuit opens at three, allows one bounded retry, then blocks | PASS |
| E: tournament | Same-baseline candidates; failed/unverified candidates disqualified; no-winner supported | PASS |
| F: latest-main revalidation | Merge gate rejects stale main and requires candidate ancestry | PASS |
| G: evidence discovery | Healthy/noisy/regression/drift/critical deterministic fixtures | PASS |
| H: candidate-to-task | Threshold, dedupe, measurable target and approval-policy fixtures | PASS |
| I: autonomous candidate preparation | Injected worker + separate reviewer + hidden command -> merge candidate, no merge | PASS (LOCAL_SYNTHETIC) |
| J: post-release measured success | Target-improvement fixture | PASS (LOCAL_SYNTHETIC) |
| K: release regression | Release reject and rollback recommendation without automatic rollback | PASS (LOCAL_SYNTHETIC) |
| L: existing regressions | 342 pytest cases plus T01–T20 | PASS |

## Test results

- Targeted harness security/supervisor suite: `39 passed`.
- Broad harness + existing router/tournament/quality/finish integration:
  `40 passed`.
- One final complete normal run: `342 passed in 14.72s`.
- T01–T20 torture run: `20 passed`, `0 failed`.
- Paid provider calls made by this program: `0`.

## Deliberate operating limits

- `external_agent_execution` remains `false`.
- Hidden acceptance has an enforced opaque command boundary but no
  repository-owned hidden corpus is claimed; `hidden_acceptance.configured`
  remains `false`.
- Therefore mutation-oriented `hf-loop` commands fail closed. Read-only status,
  task, problem, metric and release-candidate inspection remains available.
- The underlying supervisor is exercised with deterministic injected runners;
  enabling real Codex/Claude workers requires explicit operator authorization,
  a supervisor-owned hidden command and a sandbox policy protecting shared Git
  refs.
- Production deployment and rollback execution remain human-only.
- Browser/microphone/audio quality was not self-attested by this harness run.

These limits are safety properties, not green release claims. The live
HumanFlow runtime does not depend on the engineering harness.
