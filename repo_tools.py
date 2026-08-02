"""Repo tools shared by both lanes.

Every tool here is deliberately chatty — real agent tools are. Dragstrip's
whole point is measuring what compression does to realistic tool output,
so we don't pre-trim anything the way a hand-optimized agent might.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

CACHE_DIR = Path(os.environ.get("DRAGSTRIP_CACHE", "/tmp/dragstrip-repos"))
CLONE_TIMEOUT_S = 90
MAX_REPO_MB = 200
MAX_FILE_BYTES = 120_000      # read_file cap per call
MAX_GREP_LINES = 200
_clone_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist",
              "build", ".next", "target", ".tox", "vendor"}


class RepoError(Exception):
    pass


def _slug(repo_url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", repo_url.rstrip("/").split("github.com/")[-1])


def clone_repo(repo_url: str) -> Path:
    """Shallow-clone a public GitHub repo into the cache, once."""
    m = re.match(r"^https://github\.com/[\w.-]+/[\w.-]+/?$", repo_url.rstrip("/"))
    if not m:
        raise RepoError("Only public github.com/<owner>/<repo> URLs are supported.")
    dest = CACHE_DIR / _slug(repo_url)
    with _locks_guard:
        lock = _clone_locks.setdefault(str(dest), threading.Lock())
    with lock:
        if (dest / ".git").exists():
            return dest
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--single-branch", repo_url, str(dest)],
                check=True, capture_output=True, timeout=CLONE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise RepoError("Clone timed out — repo may be too large.")
        except subprocess.CalledProcessError as e:
            raise RepoError(f"Clone failed: {e.stderr.decode(errors='replace')[-300:]}")
        size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
        if size_mb > MAX_REPO_MB:
            shutil.rmtree(dest)
            raise RepoError(f"Repo is {size_mb:.0f}MB — over the {MAX_REPO_MB}MB demo cap.")
        return dest


def _safe(root: Path, rel: str) -> Path:
    p = (root / rel.lstrip("/")).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise RepoError("Path escapes the repository.")
    return p


# ── the four tools ──

def list_files(root: Path, path: str = ".", max_entries: int = 400) -> str:
    """Tree listing with file sizes."""
    base = _safe(root, path)
    if not base.exists():
        return f"No such path: {path}"
    lines, count = [], 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        rel_dir = os.path.relpath(dirpath, root)
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        if depth > 4:
            dirnames[:] = []
            continue
        indent = "  " * depth
        if rel_dir != ".":
            lines.append(f"{indent}{os.path.basename(dirpath)}/")
        for name in sorted(filenames):
            try:
                sz = os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                sz = 0
            lines.append(f"{indent}  {name} ({sz:,}B)")
            count += 1
            if count >= max_entries:
                lines.append(f"... truncated at {max_entries} entries")
                return "\n".join(lines)
    return "\n".join(lines) or "(empty)"


def read_file(root: Path, path: str, start_line: int = 1, end_line: int | None = None) -> str:
    p = _safe(root, path)
    if not p.is_file():
        return f"No such file: {path}"
    try:
        raw = p.read_bytes()[: MAX_FILE_BYTES * 2]
        text = raw.decode("utf-8", errors="replace")
    except OSError as e:
        return f"Read error: {e}"
    all_lines = text.splitlines()
    end = end_line or len(all_lines)
    picked = all_lines[max(0, start_line - 1): end]
    body = "\n".join(f"{i:>5}\t{l}" for i, l in enumerate(picked, start=start_line))
    if len(body) > MAX_FILE_BYTES:
        body = body[:MAX_FILE_BYTES] + f"\n... truncated ({len(all_lines)} lines total)"
    return f"== {path} (lines {start_line}-{min(end, len(all_lines))} of {len(all_lines)}) ==\n{body}"


def search_code(root: Path, pattern: str, glob: str | None = None) -> str:
    cmd = ["grep", "-rn", "-I", "--max-count=8", "-E", pattern, "."]
    for d in _SKIP_DIRS:
        cmd.insert(1, f"--exclude-dir={d}")
    if glob:
        cmd.insert(1, f"--include={glob}")
    try:
        out = subprocess.run(cmd, cwd=root, capture_output=True, timeout=30, text=True)
    except subprocess.TimeoutExpired:
        return "Search timed out."
    if out.returncode == 2:
        return f"Bad pattern: {out.stderr[-200:]}"
    lines = out.stdout.splitlines()
    if not lines:
        return f"No matches for /{pattern}/"
    if len(lines) > MAX_GREP_LINES:
        lines = lines[:MAX_GREP_LINES] + [f"... {len(lines) - MAX_GREP_LINES} more matches truncated"]
    return "\n".join(lines)


def file_outline(root: Path, path: str) -> str:
    """Rough symbol outline: def/class/function/export lines with line numbers."""
    p = _safe(root, path)
    if not p.is_file():
        return f"No such file: {path}"
    pat = re.compile(
        r"^\s*(def |class |function |export |const [A-Z_a-z]\w* *= *(async )?\(|"
        r"public |private |fn |func |impl |interface |type [A-Z])"
    )
    out = []
    try:
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if pat.match(line):
                out.append(f"{i:>5}: {line.strip()[:160]}")
    except OSError as e:
        return f"Read error: {e}"
    return "\n".join(out[:250]) or "(no obvious symbols found)"


# Anthropic-style schemas — Paritok's engine (tool discovery + virtual tool
# injection) speaks this format natively; agent.py converts to OpenAI wire
# format at the API boundary.
TOOL_SCHEMAS = [
    {
        "name": "list_files",
        "description": "List the repository tree (dirs and files with sizes). Use path to scope to a subdirectory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list, relative to repo root. Default '.'"}
            },
        },
    },
    {
        "name": "read_file",
        "description": "Read a file (line-numbered). Optionally pass start_line/end_line to read a slice.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Regex search across the repo (grep -E). Returns file:line:match rows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "glob": {"type": "string", "description": "Optional filename filter like *.py"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "file_outline",
        "description": "Symbol outline of one file: def/class/function/export lines with line numbers. Cheaper than reading the whole file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]

TOOL_FNS = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "file_outline": file_outline,
}
