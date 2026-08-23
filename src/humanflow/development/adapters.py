"""Non-spending CLI adapter boundaries for Codex and Claude Code."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AdapterStatus:
    agent: str
    executable: str
    installed: bool
    version: str | None
    authentication: str
    external_execution_authorized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "executable": self.executable,
            "installed": self.installed,
            "version": self.version,
            "authentication": self.authentication,
            "external_execution_authorized": self.external_execution_authorized,
        }


@dataclass(frozen=True, slots=True)
class CliDevelopmentAdapter:
    agent: str
    executable: str
    external_execution_authorized: bool = False

    def status(self) -> AdapterStatus:
        resolved = shutil.which(self.executable)
        version: str | None = None
        if resolved is not None:
            completed = subprocess.run(
                [resolved, "--version"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )
            if completed.returncode == 0:
                lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
                version = lines[-1] if lines else None
        return AdapterStatus(
            agent=self.agent,
            executable=resolved or self.executable,
            installed=resolved is not None,
            version=version,
            authentication="UNKNOWN_NOT_PROBED",
            external_execution_authorized=self.external_execution_authorized,
        )

    def command(self, *, prompt_file: Path, worktree: Path) -> tuple[str, ...]:
        if not prompt_file.is_file() or not worktree.is_dir():
            raise FileNotFoundError("prompt file and worktree must exist")
        if self.agent == "codex":
            return (self.executable, "exec", "--cd", str(worktree), "-")
        return (self.executable, "--print", "-")

    def execute(self, *, prompt_file: Path, worktree: Path) -> subprocess.CompletedProcess[str]:
        if not self.external_execution_authorized:
            raise PermissionError("external development-agent execution is not authorized")
        return subprocess.run(
            list(self.command(prompt_file=prompt_file, worktree=worktree)),
            check=False,
            text=True,
            input=prompt_file.read_text(encoding="utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1_800,
            cwd=worktree,
        )


class CodexCliAdapter(CliDevelopmentAdapter):
    def __init__(self, *, external_execution_authorized: bool = False) -> None:
        super().__init__("codex", "codex", external_execution_authorized)


class ClaudeCliAdapter(CliDevelopmentAdapter):
    def __init__(self, *, external_execution_authorized: bool = False) -> None:
        super().__init__("claude", "claude", external_execution_authorized)
