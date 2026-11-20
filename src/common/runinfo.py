"""Run provenance: record the exact code + config behind every result.

Reproducibility in this project comes from version control, not code duplication:
each run writes the git commit it was produced from and the resolved config into
its results, so any number traces back to the exact code and settings.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git(args: list[str], repo_dir: str | Path = ".") -> str | None:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(repo_dir),
                                      stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:  # git missing, not a repo, etc. — provenance is best-effort
        return None


def provenance(config: dict, repo_dir: str | Path = ".") -> dict:
    """Everything needed to reproduce a run: commit, dirty flag, time, config."""
    commit = _git(["rev-parse", "HEAD"], repo_dir)
    status = _git(["status", "--porcelain"], repo_dir)
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
    }
