# HumanFlow Engineering Contract

## Mission

Build and measure a natural German realtime conversation engine during the
authorized 72-hour sprint. Runtime claims require reproducible event evidence.

## Non-negotiable safeguards

- Never commit, print, or inspect secret values.
- Never weaken golden tests, quality gates, metric formulas, labels, or score
  thresholds to make a candidate pass.
- Treat `tests/golden/`, `eval/golden/`, `config/quality-gates.yaml`,
  `docs/METRICS.md`, `schemas/metric-definitions.*`, and `sprint/` evidence as
  protected artifacts.
- Never self-modify or deploy from a live customer call.
- Perform optimization only in isolated worktrees, followed by the same
  immutable evaluation and explicit KEEP/REVERT evidence.
- Do not push, deploy, publish, spend Fable credit, or change account/billing
  settings without explicit human authorization.
- Preserve failures as sanitized telemetry or regression fixtures where useful.

## Engineering priorities

1. Turn detection and intentional barge-in.
2. Backchannel and overlap handling.
3. Played Audio Ledger and explicit conversation state.
4. Tool-failure recovery and deterministic replay.
5. Metrics, scorecard, and controlled quality loop.
6. Router/tournament and dashboard polish only after the core is strong.

Run targeted tests after each change and the complete immutable benchmark before
accepting any candidate improvement.

