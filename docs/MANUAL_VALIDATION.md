# HumanFlow Manual Browser/Audio Validation

This step must be performed by a human with a real microphone and audible output.
Automation must not create or approve this evidence.

Prerequisites: `reports/live-tts-smoke.json` and `reports/live-stt-smoke.json`
must show `PASS` for the configured ElevenLabs providers. A failed STT smoke must
not be replaced by Browser SpeechRecognition; a failed TTS smoke must not be hidden
by Browser Web Speech synthesis.

1. Start the loopback demo with `make demo`.
2. Open `http://127.0.0.1:8765` in a supported Chrome browser.
3. Confirm that the panel shows microphone `getUserMedia`, PCM 16 kHz mono,
   ElevenLabs Scribe Realtime as `REAL`, and Browser SpeechRecognition as `OFF`.
4. Confirm that ElevenLabs Flash v2.5 becomes `ACTIVE` and `REAL`, not the TTS fallback.
5. Say `Was ist 25 mal 17?`, then `Und zieh davon 25 ab.`, then ask for the first result.
6. Correct an appointment from Thursday to Friday and then to Monday.
7. Request a longer explanation and say `mhm`; playback should continue.
8. During another long answer say `Moment, stopp`; judge the actually heard stop,
   not merely the telemetry acknowledgement.
9. Test free conversation, an empathetic sentence, numbers, dates and times.
10. Continue a contextual conversation for three to five minutes.
11. Verify that no assistant sentence appears as an accepted USER history item.
12. Rate all eight voice-quality dimensions in the browser form. Only this
    human submission creates a sample; automation leaves the sample count at zero.
13. Inspect `/dashboard`, event timing, transcript provenance and provider identity.

For the current naturalness milestone, do not create the release attestation and
do not freeze the sprint. Human feedback determines the next KEEP/REVERT decision.
