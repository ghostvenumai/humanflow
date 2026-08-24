#!/usr/bin/env python3
"""Capture reproducible static evidence for the self-transcription routing defect."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "d08cbeef075159f63d7e56498be2680a5d74dcf6"
OUTPUT = ROOT / "reports" / "self-transcription-root-cause.json"


def git_source(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{BASELINE}:{path}"],
        cwd=ROOT,
        text=True,
    )


def run() -> dict[str, object]:
    baseline_js = git_source("src/humanflow/web/static/app.js")
    baseline_app = git_source("src/humanflow/web/app.py")
    baseline_session = git_source("src/humanflow/runtime/session.py")
    baseline_reasoner = git_source("src/humanflow/runtime/anthropic_provider.py")
    current_js = (ROOT / "src/humanflow/web/static/app.js").read_text(
        encoding="utf-8"
    )
    current_session = (ROOT / "src/humanflow/runtime/session.py").read_text(
        encoding="utf-8"
    )

    baseline_handler = baseline_js.split("recognition.onresult =", 1)[1].split(
        "recognition.onerror =", 1
    )[0]
    current_handler = current_js.split("recognition.onresult =", 1)[1].split(
        "recognition.onerror =", 1
    )[0]
    checks = {
        "baseline_browser_rendered_recognition_as_user_before_server_acceptance": (
            "showTranscript(text, result.isFinal)" in baseline_handler
        ),
        "baseline_browser_sent_recognition_as_browser_stt": (
            'type: "transcript", source: "browser_stt"' in baseline_handler
        ),
        "baseline_server_forwarded_update_to_session": (
            "decision = await session.submit_transcript(update)" in baseline_app
        ),
        "baseline_session_had_no_authoritative_accept_user_gate": (
            "def accept_user_transcript" not in baseline_session
        ),
        "baseline_reasoner_appended_every_forwarded_text_as_user": (
            '{"role": "user", "content": user_text}' in baseline_reasoner
            and "self._history.extend" in baseline_reasoner
        ),
        "baseline_used_independent_get_user_media_and_speech_recognition": (
            "createMediaStreamSource(stream)" in baseline_js
            and "recognition = new Recognition()" in baseline_js
        ),
        "baseline_websocket_onmessage_handler_count": baseline_js.count(
            "socket.onmessage ="
        ),
        "baseline_recognition_onresult_handler_count": baseline_js.count(
            "recognition.onresult ="
        ),
        "current_browser_waits_for_server_before_user_render": (
            "showTranscript(" not in current_handler
            and 'payload.type === "transcript_result"' in current_js
        ),
        "current_server_has_authoritative_user_gate": (
            "def accept_user_transcript" in current_session
        ),
        "current_assistant_origin_history_assertion": (
            "assistant_origin_event_to_user_history" in current_session
            and '"FORBIDDEN"' in current_session
        ),
    }
    if not all(bool(value) for value in checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError("routing evidence check failed: " + ", ".join(failed))
    return {
        "status": "ROOT_CAUSE_CONFIRMED_AND_GUARD_IMPLEMENTED",
        "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "baseline_commit": BASELINE,
        "human_evidence": {
            "headphones_changed_failure": False,
            "assistant_sentence_visible_as_user_transcript": True,
            "manual_validation": "FAILED",
        },
        "classification": {
            "A_ui_only_duplication": False,
            "B_browser_stt_self_recognition": True,
            "C_server_event_routing_contamination": True,
            "D_conversation_history_contamination": True,
            "E_multiple_causes": True,
        },
        "root_cause_chain": [
            "Browser SpeechRecognition runs independently from getUserMedia PCM.",
            "Its onresult callback rendered every result as USER before server acceptance.",
            "Every browser_stt final was allowlisted solely by a mutable source string.",
            "The session forwarded completed text to the reasoner without self-speech classification.",
            "AnthropicReasoner appended every forwarded transcript with role=user.",
        ],
        "causes_not_supported_by_static_evidence": {
            "duplicate_websocket_onmessage_handlers": "NOT_FOUND",
            "duplicate_recognition_onresult_handlers": "NOT_FOUND",
            "assistant_text_directly_invokes_recognition_callback": "NOT_FOUND",
            "shared_role_object_mutation": "NOT_FOUND",
        },
        "audio_source_observation": {
            "server": "PulseAudio on PipeWire 1.0.5",
            "default_source": "alsa_input.pci-0000_00_1f.3.analog-stereo",
            "monitor_source_exists": True,
            "monitor_source_is_default": False,
            "browser_recognition_source_binding": "UNVERIFIED_BROWSER_MANAGED",
        },
        "checks": checks,
        "agent_manual_validation_attestation": False,
    }


def main() -> None:
    report = run()
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": report["classification"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
