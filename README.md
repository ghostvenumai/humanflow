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
```

No GitHub push or deployment is performed automatically.

