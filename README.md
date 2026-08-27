# HumanFlow

**A German real-time voice agent focused on the hard part of speech-to-speech: turn
detection, barge-in, playback ownership, conversation state, tool grounding and
failure recovery.**

HumanFlow runs over a real browser + microphone path: streaming STT, a Claude
reasoner, SQLite-grounded appointment tools, and streaming TTS — with an emphasis on
*reliability* rather than a happy-path demo.

![HumanFlow live conversation](docs/images/live-conversation.jpg)

*Live browser session: streaming STT, a grounded German appointment dialog, the
active TTS provider, and per-turn provenance/telemetry.*

---

## What it does

- Natural **German multi-turn** voice conversation.
- **Turn detection** over Scribe transcript state, server VAD, backchannel
  vocabulary and explicit German interruption phrases.
- **Barge-in** as a real state transition (detect → validate → stop playback →
  cancel TTS → invalidate obsolete response → accept the new turn), not a bare
  `stop()`.
- **Backchannel tolerance** — "mhm", "okay", "aha" don't cancel the agent.
- **Played-audio truth**: a ledger distinguishes generated / queued / played /
  cancelled audio, so conversation state never assumes the user heard content that
  was cancelled before playback.
- **Tool-grounded appointments** against an authoritative local **SQLite** source —
  availability, booking, rescheduling, cancellation — never hallucinated slots.
- **Deterministic German temporal resolution** (e.g. *"Mittwoch in zwei Wochen"* is
  two full weeks and is distinct from *"nächsten Mittwoch"*).
- **Failure recovery** and an evidence-driven **engineering improvement loop**.

## Architecture

```
Microphone
  → ElevenLabs Scribe (streaming STT)
  → Turn Detection / Final Admission
  → Conversation Controller (state machine, operation epochs, cancellation)
  → Claude Reasoner (Messages API)
  → Grounded Tools / SQLite (appointments)
  → ElevenLabs Streaming TTS
  → Playback Controller (single-owner, cancel-aware)
  → Browser Audio
```

## Reliability highlights

These are reproduced-and-tested behaviours, each backed by deterministic regression
tests:

- A permanent **THINKING deadlock** was reproduced and fixed — a response that never
  reaches audio can no longer strand the session.
- A **stalled pre-playback generation** recovers deterministically to `LISTENING`.
- A **legitimate new user turn during a stall** is no longer lost — it takes over.
- **Late/stale audio protection**: after a cancellation, old audio can never become
  audible or mutate state.
- A **transient pre-first-audio TTS failure** is retried **once** with the *same*
  already-generated text — the tool and reasoner are **not** re-run (no duplicate
  side effects, no second LLM call).
- The speech-retry boundary is narrowed to genuine stream/provider failures, so an
  internal non-TTS exception is never misclassified as a transient TTS retry.
- **Barge-in / cancellation** always bypass the speech retry.
- Appointment availability follow-ups stay **type-preserving and grounded**
  (a generic "what else is free?" broadens to real alternatives instead of dumping
  the whole database or leaking unrelated appointment types).

## The engineering improvement loop

Beyond runtime self-recovery, HumanFlow includes a controlled **engineering
improvement loop**: problems become tasks and isolated fix candidates, verified by
targeted tests, adversarial review, and protected/golden checks, with persistent
findings/metrics and explicit KEEP / REJECT gates before a controlled freeze.

While fixing the original TTS/THINKING issue, an adversarial review pass surfaced an
*additional* race condition that was **not** part of the original report — a
legitimate new user turn could be lost during a pre-playback TTS stall. It was
reproduced, tested and fixed. (HumanFlow does **not** rewrite its own production code
during a live customer call; runtime anomalies are turned into engineering tasks
under human-gated review.)

## Tech stack

Python 3.12 · FastAPI · Anthropic Messages API (Claude) · ElevenLabs Scribe (STT) ·
ElevenLabs Flash v2.5 (streaming TTS) · SQLite · dependency-free deterministic test
adapters.

## Tests

- **Frozen Everlast candidate** (`humanflow-everlast-frozen-20260826`, validated
  runtime `ea3677e`): **367** automated tests passed; a real browser/audio multi-turn
  session was human-validated.
- **Post-freeze demo candidate** (additional, adversarially verified appointment /
  temporal fixes): **385** local tests passed.

These two states are kept separate and are not mixed.

## Scope & honesty

- Validated on a **real browser / real-audio** multi-turn path.
- **No** production-telephony / PSTN certification and **no** production-SLA claim.
- Latency figures in the repo are labelled local/simulated and are not real-call
  latency metrics.
- Voice naturalness and per-answer intelligibility remain a human judgement.

## Running the demo

```bash
python3 -m pip install -e '.[demo]'
# provide ANTHROPIC_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID
#   in $HOME/.config/humanflow/runtime.env  (never commit secrets)
make demo   # serves http://127.0.0.1:8765
make test   # run the automated suite
```

The server is **fail-closed**: without real provider credentials it refuses to start
rather than falling back to a fake conversation.
