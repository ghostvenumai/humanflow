from __future__ import annotations

import struct
from dataclasses import dataclass

import pytest

from humanflow.audio.models import AudioFrame
from humanflow.runtime.final_admission import (
    FinalAdmissionReason,
    FinalTranscriptAdmissionGate,
)
from humanflow.runtime.providers import TranscriptUpdate
from humanflow.runtime.self_speech import SelfSpeechGuard
from humanflow.runtime.transcript_events import (
    ConversationEventKind,
    TranscriptOrigin,
    TranscriptProvenance,
)
from humanflow.turns.models import TurnDecisionType, TurnSignals
from humanflow.turns.policies import HybridTurnPolicy


FRAME_NS = 20_000_000
BASE_NS = 10_000_000_000


@dataclass(frozen=True, slots=True)
class MatrixCase:
    scenario_id: str
    text: str
    voiced_frames: int
    expected: bool
    reason: FinalAdmissionReason
    fragmented: bool = False
    assistant_playback_active: bool = False


MATRIX_CASES = (
    MatrixCase(
        "long_real_sentence",
        "Ich brauche einen Orthopädentermin am Mittwoch um elf Uhr.",
        32,
        True,
        FinalAdmissionReason.ACCEPTED,
    ),
    MatrixCase("short_ja", "Ja.", 3, True, FinalAdmissionReason.ACCEPTED),
    MatrixCase("short_nein", "Nein.", 3, True, FinalAdmissionReason.ACCEPTED),
    MatrixCase("short_okay", "Okay.", 3, True, FinalAdmissionReason.ACCEPTED),
    MatrixCase("short_stopp", "Stopp.", 3, True, FinalAdmissionReason.ACCEPTED),
    MatrixCase(
        "moment_stopp",
        "Moment, stopp.",
        5,
        True,
        FinalAdmissionReason.ACCEPTED,
        assistant_playback_active=True,
    ),
    MatrixCase(
        "mhm_backchannel",
        "mhm",
        3,
        True,
        FinalAdmissionReason.ACCEPTED,
        assistant_playback_active=True,
    ),
    MatrixCase(
        "cough_unrelated_final",
        "Ja okay dann viel Erfolg.",
        4,
        False,
        FinalAdmissionReason.INSUFFICIENT_ACOUSTIC_EVIDENCE,
    ),
    MatrixCase(
        "throat_clear_unrelated_final",
        "Das ist ein echter Befehl.",
        3,
        False,
        FinalAdmissionReason.INSUFFICIENT_ACOUSTIC_EVIDENCE,
    ),
    MatrixCase(
        "chair_noise_transient",
        "Stopp.",
        1,
        False,
        FinalAdmissionReason.PCM_EPISODE_TOO_SHORT,
    ),
    MatrixCase(
        "real_overlap",
        "Nein, mach den Termin lieber am Montag.",
        24,
        True,
        FinalAdmissionReason.ACCEPTED,
        assistant_playback_active=True,
    ),
    MatrixCase(
        "very_fast_speech",
        "Bitte nimm den späteren Termin.",
        11,
        True,
        FinalAdmissionReason.ACCEPTED,
    ),
    MatrixCase(
        "slow_speech",
        "Ich möchte den Termin auf Freitag verschieben.",
        45,
        True,
        FinalAdmissionReason.ACCEPTED,
    ),
    MatrixCase(
        "filler_heavy_speech",
        "Äh, also, ich meine doch lieber Montag.",
        30,
        True,
        FinalAdmissionReason.ACCEPTED,
    ),
    MatrixCase(
        "fragmented_genuine_utterance",
        "Ich brauche, äh, einen Termin am Donnerstag.",
        24,
        True,
        FinalAdmissionReason.ACCEPTED,
        fragmented=True,
    ),
)


def _no_self_speech(text: str, *, playback_active: bool):
    return SelfSpeechGuard().assess(
        text=text,
        observed_ns=BASE_NS + 2_000_000_000,
        origin=TranscriptOrigin.STREAMING_STT_PROVIDER,
        playback_active=playback_active,
    )


def _update(
    text: str,
    *,
    transcript_id: str,
    timestamp_ns: int,
    audio_frame_sequence: int | None,
    stream_id: str = "browser-pcm-track",
) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        signals=TurnSignals(
            speech_active=False,
            silence_duration_ms=350,
            utterance_duration_ms=900,
            semantic_complete=True,
            acoustic_completion=1.0,
            provider_endpointed=True,
        ),
        provenance=TranscriptProvenance(
            transcript_id=transcript_id,
            event_kind=ConversationEventKind.USER_TRANSCRIPT_FINAL,
            source="streaming_stt",
            origin=TranscriptOrigin.STREAMING_STT_PROVIDER,
            stream_id=stream_id,
            stt_session_id="scribe-matrix",
            audio_capture_id="capture-matrix",
            timestamp_ns=timestamp_ns,
            recognition_input_binding="EXACT_GETUSERMEDIA_PCM16",
            audio_frame_sequence=audio_frame_sequence,
        ),
    )


def _feed_episode(
    gate: FinalTranscriptAdmissionGate,
    *,
    voiced_frames: int,
    fragmented: bool = False,
    start_sequence: int = 0,
    start_ns: int = BASE_NS,
    release: bool = True,
) -> tuple[int, int, int]:
    sequence = start_sequence
    captured_ns = start_ns
    split = voiced_frames // 2 if fragmented else voiced_frames
    for index in range(voiced_frames):
        if fragmented and index == split:
            for _ in range(3):
                gate.observe(
                    AudioFrame(
                        stream_id="browser-pcm-track",
                        sequence=sequence,
                        pcm16=struct.pack("<h", 0) * 320,
                        captured_ns=captured_ns,
                    )
                )
                sequence += 1
                captured_ns += FRAME_NS
        gate.observe(
            AudioFrame(
                stream_id="browser-pcm-track",
                sequence=sequence,
                pcm16=struct.pack("<h", 5_000) * 320,
                captured_ns=captured_ns,
            )
        )
        sequence += 1
        captured_ns += FRAME_NS
    speech_end_ns = captured_ns
    if release:
        for _ in range(10):
            gate.observe(
                AudioFrame(
                    stream_id="browser-pcm-track",
                    sequence=sequence,
                    pcm16=struct.pack("<h", 0) * 320,
                    captured_ns=captured_ns,
                )
            )
            sequence += 1
            captured_ns += FRAME_NS
    return speech_end_ns, captured_ns, sequence - 1


@pytest.mark.parametrize("case", MATRIX_CASES, ids=lambda case: case.scenario_id)
def test_input_integrity_admission_matrix(case: MatrixCase) -> None:
    gate = FinalTranscriptAdmissionGate()
    _, final_ns, final_sequence = _feed_episode(
        gate,
        voiced_frames=case.voiced_frames,
        fragmented=case.fragmented,
    )
    assessment = gate.assess_final(
        _update(
            case.text,
            transcript_id=case.scenario_id,
            timestamp_ns=final_ns,
            audio_frame_sequence=final_sequence,
        ),
        assistant_playback_active=case.assistant_playback_active,
        self_speech=_no_self_speech(
            case.text, playback_active=case.assistant_playback_active
        ),
    )

    assert assessment.accepted is case.expected
    assert assessment.reason_code == case.reason.value


@pytest.mark.parametrize(
    ("scenario_id", "text", "expected_reason"),
    (
        ("silence", "Ein erfundener Satz.", FinalAdmissionReason.NO_PCM_EPISODE),
        ("scribe_ghost", "Tschüss.", FinalAdmissionReason.NO_PCM_EPISODE),
    ),
)
def test_final_without_pcm_episode_is_rejected(
    scenario_id: str, text: str, expected_reason: FinalAdmissionReason
) -> None:
    gate = FinalTranscriptAdmissionGate()
    assessment = gate.assess_final(
        _update(
            text,
            transcript_id=scenario_id,
            timestamp_ns=BASE_NS,
            audio_frame_sequence=None,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech(text, playback_active=False),
    )

    assert assessment.accepted is False
    assert assessment.reason_code == expected_reason.value


def test_borderline_real_orthopaedie_final_uses_combined_pcm_evidence() -> None:
    gate = FinalTranscriptAdmissionGate()
    _, final_ns, final_sequence = _feed_episode(
        gate,
        voiced_frames=8,
        fragmented=True,
    )
    assessment = gate.assess_final(
        _update(
            "Ich brauch 'n Termin für 'n Orthopäden.",
            transcript_id="real-borderline-orthopaedie-final",
            timestamp_ns=final_ns,
            audio_frame_sequence=final_sequence,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech(
            "Ich brauch 'n Termin für 'n Orthopäden.", playback_active=False
        ),
    )

    assert assessment.accepted is True
    assert (
        assessment.reason_code
        == FinalAdmissionReason.ACCEPTED_RECOVERED_ACOUSTIC.value
    )
    assert assessment.voiced_duration_ms == 160.0
    assert assessment.required_voiced_ms == 245.0
    assert assessment.acoustic_span_ms == 220.0
    assert assessment.evidence_class == "RECOVERED_ACOUSTIC"


def test_low_volume_meaningful_final_uses_persistent_pcm_evidence() -> None:
    gate = FinalTranscriptAdmissionGate()
    sequence = 0
    captured_ns = BASE_NS
    for amplitude in (1_100, 1_100, 0, 0, 0, 1_100, 1_100):
        gate.observe(
            AudioFrame(
                stream_id="browser-pcm-track",
                sequence=sequence,
                pcm16=struct.pack("<h", amplitude) * 320,
                captured_ns=captured_ns,
            )
        )
        sequence += 1
        captured_ns += FRAME_NS

    assessment = gate.assess_final(
        _update(
            "Ich brauche einen Orthopädentermin.",
            transcript_id="low-volume-meaningful-final",
            timestamp_ns=captured_ns,
            audio_frame_sequence=sequence - 1,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech(
            "Ich brauche einen Orthopädentermin.", playback_active=False
        ),
    )

    assert assessment.accepted is True
    assert (
        assessment.reason_code
        == FinalAdmissionReason.ACCEPTED_RECOVERED_ACOUSTIC.value
    )
    assert assessment.voiced_duration_ms == 80.0
    assert assessment.acoustic_span_ms == 140.0


def test_borderline_evidence_never_recovers_during_assistant_playback() -> None:
    gate = FinalTranscriptAdmissionGate()
    text = "Eine längere fremde Hintergrundstimme spricht im Zimmer."
    _, final_ns, final_sequence = _feed_episode(
        gate,
        voiced_frames=8,
        fragmented=True,
    )
    assessment = gate.assess_final(
        _update(
            text,
            transcript_id="borderline-during-playback",
            timestamp_ns=final_ns,
            audio_frame_sequence=final_sequence,
        ),
        assistant_playback_active=True,
        self_speech=_no_self_speech(
            text, playback_active=True
        ),
    )

    assert assessment.accepted is False
    assert (
        assessment.reason_code
        == FinalAdmissionReason.INSUFFICIENT_ACOUSTIC_EVIDENCE.value
    )


def test_delayed_valid_final_remains_eligible_during_bounded_grace_window() -> None:
    gate = FinalTranscriptAdmissionGate()
    speech_end_ns, _, final_sequence = _feed_episode(gate, voiced_frames=25)
    assessment = gate.assess_final(
        _update(
            "Ich möchte einen Termin am Donnerstag.",
            transcript_id="delayed-valid-final",
            timestamp_ns=speech_end_ns + 4_500_000_000,
            audio_frame_sequence=final_sequence,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech(
            "Ich möchte einen Termin am Donnerstag.", playback_active=False
        ),
    )

    assert assessment.accepted is True
    assert assessment.reason_code == FinalAdmissionReason.ACCEPTED.value


def test_provider_final_may_lag_local_frame_observation_without_false_reject() -> None:
    gate = FinalTranscriptAdmissionGate()
    _, _, last_sequence = _feed_episode(
        gate, voiced_frames=28, release=False, start_sequence=100
    )
    assessment = gate.assess_final(
        _update(
            "Ich brauche einen langen Orthopädentermin am Mittwoch um elf Uhr.",
            transcript_id="provider-frame-lag",
            timestamp_ns=BASE_NS + 18 * FRAME_NS,
            audio_frame_sequence=110,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech(
            "Ich brauche einen langen Orthopädentermin am Mittwoch um elf Uhr.",
            playback_active=False,
        ),
    )

    assert last_sequence > 110
    assert assessment.accepted is True
    assert assessment.alignment_ms is not None and assessment.alignment_ms < 0


def test_consumed_episode_cannot_authorize_second_final() -> None:
    gate = FinalTranscriptAdmissionGate()
    _, final_ns, final_sequence = _feed_episode(gate, voiced_frames=12)
    first = gate.assess_final(
        _update(
            "Ja.",
            transcript_id="first-final",
            timestamp_ns=final_ns,
            audio_frame_sequence=final_sequence,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech("Ja.", playback_active=False),
    )
    second = gate.assess_final(
        _update(
            "Ja.",
            transcript_id="second-final",
            timestamp_ns=final_ns + 1_000_000,
            audio_frame_sequence=final_sequence,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech("Ja.", playback_active=False),
    )

    assert first.accepted is True
    assert second.accepted is False
    assert second.reason_code == FinalAdmissionReason.EPISODE_ALREADY_CONSUMED.value


def test_delayed_finals_claim_multiple_completed_pcm_episodes_in_fifo_order() -> None:
    gate = FinalTranscriptAdmissionGate()
    _, first_observed_end, first_last_sequence = _feed_episode(
        gate, voiced_frames=6, start_sequence=0
    )
    _, second_observed_end, second_last_sequence = _feed_episode(
        gate,
        voiced_frames=6,
        start_sequence=first_last_sequence + 1,
        start_ns=first_observed_end + FRAME_NS,
    )

    first = gate.assess_final(
        _update(
            "Ja.",
            transcript_id="ordered-final-1",
            timestamp_ns=second_observed_end,
            audio_frame_sequence=second_last_sequence,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech("Ja.", playback_active=False),
    )
    second = gate.assess_final(
        _update(
            "Nein.",
            transcript_id="ordered-final-2",
            timestamp_ns=second_observed_end + 1_000_000,
            audio_frame_sequence=second_last_sequence,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech("Nein.", playback_active=False),
    )

    assert first.accepted is True
    assert first.speech_episode_id == "pcm-speech-1"
    assert second.accepted is True
    assert second.speech_episode_id == "pcm-speech-2"


def test_early_final_detaches_consumed_active_episode_before_continued_speech() -> None:
    gate = FinalTranscriptAdmissionGate()
    _, first_end, first_last_sequence = _feed_episode(
        gate, voiced_frames=3, release=False
    )
    first = gate.assess_final(
        _update(
            "Ja.",
            transcript_id="early-active-final",
            timestamp_ns=first_end,
            audio_frame_sequence=first_last_sequence,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech("Ja.", playback_active=False),
    )
    _, second_observed_end, second_last_sequence = _feed_episode(
        gate,
        voiced_frames=3,
        start_sequence=first_last_sequence + 1,
        start_ns=first_end,
    )
    second = gate.assess_final(
        _update(
            "Nein.",
            transcript_id="continued-speech-final",
            timestamp_ns=second_observed_end,
            audio_frame_sequence=second_last_sequence,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech("Nein.", playback_active=False),
    )

    assert first.accepted is True
    assert first.speech_episode_id == "pcm-speech-1"
    assert second.accepted is True
    assert second.speech_episode_id == "pcm-speech-2"


@pytest.mark.parametrize(
    ("scenario_id", "stream_id", "timestamp_offset_ns", "final_sequence", "reason"),
    (
        (
            "stream_mismatch",
            "other-browser-track",
            0,
            110,
            FinalAdmissionReason.STREAM_ID_MISMATCH,
        ),
        (
            "final_too_early",
            "browser-pcm-track",
            -2_000_000_000,
            110,
            FinalAdmissionReason.FINAL_TOO_EARLY,
        ),
        (
            "final_too_late",
            "browser-pcm-track",
            6_000_000_000,
            110,
            FinalAdmissionReason.FINAL_TOO_LATE,
        ),
        (
            "frame_range_mismatch",
            "browser-pcm-track",
            0,
            99,
            FinalAdmissionReason.FRAME_RANGE_MISMATCH,
        ),
    ),
)
def test_precise_episode_association_rejection_reasons(
    scenario_id: str,
    stream_id: str,
    timestamp_offset_ns: int,
    final_sequence: int,
    reason: FinalAdmissionReason,
) -> None:
    gate = FinalTranscriptAdmissionGate()
    _, observed_end_ns, _ = _feed_episode(
        gate, voiced_frames=12, start_sequence=100
    )
    timestamp_ns = (
        BASE_NS + timestamp_offset_ns
        if timestamp_offset_ns < 0
        else observed_end_ns + timestamp_offset_ns
    )
    assessment = gate.assess_final(
        _update(
            "Ja.",
            transcript_id=scenario_id,
            timestamp_ns=timestamp_ns,
            audio_frame_sequence=final_sequence,
            stream_id=stream_id,
        ),
        assistant_playback_active=False,
        self_speech=_no_self_speech("Ja.", playback_active=False),
    )

    assert assessment.accepted is False
    assert assessment.reason_code == reason.value


def test_required_rejection_reason_vocabulary_is_stable() -> None:
    assert {
        reason.value for reason in FinalAdmissionReason
    } >= {
        "NO_PCM_EPISODE",
        "PCM_EPISODE_TOO_SHORT",
        "FINAL_TOO_EARLY",
        "FINAL_TOO_LATE",
        "STREAM_ID_MISMATCH",
        "GENERATION_MISMATCH",
        "FRAME_RANGE_MISMATCH",
        "EPISODE_ALREADY_CONSUMED",
        "SELF_SPEECH_MATCH",
        "INSUFFICIENT_ACOUSTIC_EVIDENCE",
        "STALE_FINAL",
        "DUPLICATE_FINAL",
        "SESSION_MISMATCH",
        "UNKNOWN",
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("mhm", TurnDecisionType.BACKCHANNEL),
        ("Moment, stopp.", TurnDecisionType.INTERRUPTION),
    ),
)
def test_final_admission_remains_separate_from_turn_classification(
    text: str, expected: TurnDecisionType
) -> None:
    decision = HybridTurnPolicy().decide(
        TurnSignals(
            speech_active=False,
            silence_duration_ms=350,
            utterance_duration_ms=200,
            final_transcript=text,
            semantic_complete=True,
            acoustic_completion=1.0,
            interruption_probability=0.98 if "stopp" in text.casefold() else 0.0,
            agent_speaking=True,
            provider_endpointed=True,
        )
    )

    assert decision.decision is expected
