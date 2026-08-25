# HumanFlow Offline Engineering Harness

The harness turns sanitized HumanFlow evidence into bounded engineering work
without becoming a dependency of the live voice runtime. The immutable 72-hour
core remains at `everlast-72h-build`; this harness exists only on the
post-freeze branch.

## Safety boundary

- Workers receive a task-scoped Git worktree and allowed paths.
- Protected fixtures, metric definitions and quality gates are supervisor-owned.
- Hidden acceptance is an opaque supervisor command; its assertions are never
  included in worker context.
- A fresh reviewer session is mandatory. A worker cannot set PASS.
- Merge eligibility requires relevant, protected and hidden tests, integrity,
  independent review and latest-main revalidation.
- The harness creates merge-candidate evidence only. Production deployment and
  rollback execution remain human operations.
- Real browser/audio claims remain manual-validation-gated.

## Operator surface

Read-only commands are available now:

```text
./hf-loop status
./hf-loop problems
./hf-loop tasks
./hf-loop metrics
./hf-loop release-candidates
```

Mutation-oriented commands fail closed while
`external_agent_execution: false` or hidden acceptance is unconfigured:

```text
./hf-loop run HF-241
./hf-loop run-ready
./hf-loop parallel --workers 2
./hf-loop tournament HF-300
./hf-loop verify HF-241
./hf-loop cleanup
```

This is intentional. The existing manual development workflow remains the
fallback. Enabling external agent execution requires an operator-owned runner,
an explicit budget/authorization decision and a supervisor-only hidden command.
The underlying supervisor is operational with injected authorized worker and
reviewer runners: it connects the conflict-aware scheduler, bounded retries,
circuit breaker, metrics, independent verification and release-candidate
evidence. The shipped CLI deliberately does not invent an operator runner or
hidden test corpus.

## Evidence flow

```text
sanitized telemetry / test evidence
  -> ProblemCandidate (thresholded and deduplicated)
  -> HF task registry (policy and approval gates)
  -> conflict-aware scheduler
  -> isolated worker worktree
  -> supervisor verification + hidden acceptance
  -> independent reviewer
  -> merge-candidate evidence
  -> human deployment decision
  -> post-release measurement / rollback recommendation
```

Detailed logs and diagnostics stay in `.engineering/`; the CLI keeps terminal
output compact. Raw audio, raw transcripts, credentials and secrets are not
valid engineering-evidence metadata.

## Known initial limitations

- External coding-agent execution is disabled by default and has not consumed
  provider credits during harness verification.
- No repository-owned hidden acceptance corpus is claimed. An operator must
  configure the opaque invocation boundary before autonomous mutation can run.
- The harness does not deploy or automatically roll back production.
- Historical development worktrees are retained as evidence and are not
  deleted automatically.
