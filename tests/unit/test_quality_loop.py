from __future__ import annotations

from pathlib import Path

import humanflow.quality.loop as quality_loop
from humanflow.quality.loop import compare_candidates


def _snapshot(score: float, *, commands_passed: bool = True) -> dict[str, object]:
    return {
        "git_commit": "commit",
        "protected_hashes": {"tests/golden/x": "abc"},
        "commands_passed": commands_passed,
        "runtime_quality": {"score": score, "immutable_eval_sha256": "fixture"},
    }


def test_quality_loop_keeps_only_strict_improvement_with_immutable_evidence() -> None:
    decision = compare_candidates(_snapshot(0.0), _snapshot(1.0))
    assert decision["decision"] == "KEEP"


def test_quality_loop_reverts_unchanged_score_or_modified_protected_artifact() -> None:
    unchanged = compare_candidates(_snapshot(1.0), _snapshot(1.0))
    modified = _snapshot(1.0)
    modified["protected_hashes"] = {"tests/golden/x": "changed"}
    protected = compare_candidates(_snapshot(0.0), modified)
    assert unchanged["decision"] == "REVERT"
    assert protected["decision"] == "REVERT"


def test_protected_hashes_ignore_only_runtime_bytecode_cache(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "tests" / "golden" / "__pycache__").mkdir(parents=True)
    (tmp_path / "eval" / "quality").mkdir(parents=True)
    (tmp_path / "config" / "loop.yaml").write_text(
        "protected_paths: [tests/golden/]\n"
        "immutable_iteration_eval: [eval/quality/runtime.json]\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "golden" / "fixture.json").write_text("fixture", encoding="utf-8")
    (tmp_path / "tests" / "golden" / "__pycache__" / "test.pyc").write_bytes(b"cache")
    (tmp_path / "eval" / "quality" / "runtime.json").write_text("eval", encoding="utf-8")
    hashes = quality_loop._protected_hashes(tmp_path)
    assert "tests/golden/fixture.json" in hashes
    assert not any("__pycache__" in path for path in hashes)
