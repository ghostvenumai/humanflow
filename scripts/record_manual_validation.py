#!/usr/bin/env python3
"""Record an explicit human attestation after real browser/audio demo calls."""

from __future__ import annotations

import argparse
import getpass
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "sprint" / "manual-validation.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human-only attestation; run this only after listening to real browser demo calls."
    )
    parser.add_argument("--browser", required=True)
    parser.add_argument("--calls", required=True, type=int)
    parser.add_argument("--confirm-microphone", action="store_true")
    parser.add_argument("--confirm-german-speech", action="store_true")
    parser.add_argument("--confirm-backchannel", action="store_true")
    parser.add_argument("--confirm-audible-barge-in", action="store_true")
    parser.add_argument("--confirm-tool-recovery", action="store_true")
    args = parser.parse_args()
    confirmations = {
        "microphone_pcm_streamed": args.confirm_microphone,
        "german_speech_audible": args.confirm_german_speech,
        "backchannel_did_not_cancel": args.confirm_backchannel,
        "intentional_barge_in_stopped_audible_output": args.confirm_audible_barge_in,
        "tool_failure_recovered_without_hangup": args.confirm_tool_recovery,
    }
    if args.calls < 3:
        parser.error("--calls must be at least 3 representative demo calls")
    if not all(confirmations.values()):
        parser.error("every --confirm-* flag is required; partial validation is not a release attestation")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "approved": True,
        "attested_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "attested_by_local_user": getpass.getuser(),
        "browser": args.browser,
        "representative_calls": args.calls,
        "validated_commit": commit,
        "confirmations": confirmations,
        "statement": "I personally ran and heard the HumanFlow browser demo; these confirmations are not agent-generated.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("HUMANFLOW_MANUAL_VALIDATION_RECORDED")
    print(f"path={OUTPUT}")
    print("Review and commit this attestation before freezing.")


if __name__ == "__main__":
    main()
