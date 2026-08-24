# Source-bound streaming STT candidate contract

The current browser demo uses two independent capture mechanisms:

- `getUserMedia` PCM is sent over the websocket and consumed by `NullTranscriber`.
- Browser SpeechRecognition produces the actual transcripts from a separate,
  browser-managed input that cannot be bound to the selected PCM stream.

A production STT candidate must implement `StreamingSTTProvider` and consume only
the exact `AudioFrame` stream identified by `audio_capture_id` and `stream_id`.
It must return immutable `TranscriptUpdate` objects with
`TranscriptOrigin.STREAMING_STT_PROVIDER`, partial/final provenance, provider
endpointing, and stable transcript IDs.

Required benchmark dimensions are German semantic preservation, numbers, dates,
times, compounds, corrections, hesitations, names, short utterances, partial/final
stability, source binding, and speech during assistant playback. No provider may be
selected until the same recorded German inputs are compared and actual human
transcripts are stored. Browser SpeechRecognition remains explicitly marked as an
unverified independent capture path until then.
