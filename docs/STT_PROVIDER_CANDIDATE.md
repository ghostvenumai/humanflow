# Source-bound production streaming STT contract

The production browser demo has one authoritative user-audio chain:

- `getUserMedia` captures one selected microphone track.
- The browser sends that track as mono PCM16 at 16 kHz over the HumanFlow websocket.
- `ElevenLabsRealtimeSTTProvider` binds immutably to its `audio_capture_id` and
  `stream_id` and sends exactly those frames to Scribe v2 Realtime.
- Only Scribe `committed_transcript` events become FINAL user transcripts.

The provider implements `StreamingSTTProvider` and returns immutable
`TranscriptUpdate` objects with
`TranscriptOrigin.STREAMING_STT_PROVIDER`, partial/final provenance, provider
endpointing, and stable transcript IDs.

PARTIAL and provider `final_transcript` hypotheses are ephemeral. They may support
turn detection, preview and barge-in, but cannot write history or invoke reasoning.
Only the provider's immutable committed event can pass `accept_user_transcript()`.

Browser SpeechRecognition is not allowlisted for production history. It remains only
as an explicit `?browser-stt-diagnostic=1` comparison and is labelled MOCK/FALLBACK.
It never replaces a failed Scribe provider.

The live provider handshake currently returns the sanitized code `auth_error` for
the configured key. The key works for TTS, so production microphone validation is
blocked until the key is granted Speech-to-Text access or a separately scoped
`ELEVENLABS_STT_API_KEY` is configured. No manual STT quality claim is made before
that blocker is resolved.
