# HumanFlow Manual Browser/Audio Validation

This step must be performed by a human with a real microphone and audible output.
Automation must not create or approve this evidence.

Prerequisite: `reports/live-tts-smoke.json` must show `PASS` for the configured
ElevenLabs voice. A failed smoke must not be replaced by Browser Web Speech.

1. Start the loopback demo with `make demo`.
2. Open `http://127.0.0.1:8765` in a supported Chrome browser.
3. Confirm that the provider panel shows ElevenLabs Flash v2.5 as `ACTIVE` and
   `REAL`, not the browser fallback.
4. Test normal informal German conversation.
5. Request a longer explanation and then a short question/answer.
6. Correct an appointment from one date or time to another.
7. Say `mhm` while HumanFlow speaks; playback should continue.
8. Say `Moment, stopp` while HumanFlow speaks; judge the actually heard stop,
   not merely the telemetry acknowledgement.
9. Test an empathetic sentence plus numbers, dates and times.
10. Continue a contextual conversation for several minutes.
11. Rate all eight voice-quality dimensions in the browser form. Only this
    human submission creates a sample; automation leaves the sample count at zero.
12. Inspect `/dashboard`, event timing and provider identity.

For the current naturalness milestone, do not create the release attestation and
do not freeze the sprint. Human feedback determines the next KEEP/REVERT decision.
