# HumanFlow Manual Browser/Audio Validation

This step must be performed by a human with a real microphone and audible output.
Automation must not create or approve this evidence.

1. Start the loopback demo with `make demo`.
2. Open `http://127.0.0.1:8765` in a supported Chrome browser.
3. Allow microphone access and complete at least three representative calls.
4. Verify audible German browser speech, `mhm` without cancellation, audible
   stop after `Moment, stopp`, and tool-failure recovery without a hangup.
5. Inspect `/dashboard`, including event timestamps and evidence scopes.
6. Record the human attestation:

```bash
python3 scripts/record_manual_validation.py \
  --browser "Google Chrome <version>" \
  --calls 3 \
  --confirm-microphone \
  --confirm-german-speech \
  --confirm-backchannel \
  --confirm-audible-barge-in \
  --confirm-tool-recovery
```

Review `sprint/manual-validation.json`, commit it, and then run:

```bash
./freeze-72h --confirm-freeze
```

The freeze command refuses to run without this attestation, after the 72-hour
deadline, with a dirty worktree, with failed tests/gates, or when already frozen.
It performs no remote push.
