"""Repo tools shared by both lanes.

Every tool here is deliberately chatty — real agent tools are. Dragstrip's
whole point is measuring what compression does to realistic tool output,
so we don't pre-trim anything the way a hand-optimized agent might.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tarfile
import threading
from pathlib import Path

import httpx

CACHE_DIR = Path(os.environ.get("DRAGSTRIP_CACHE", "/tmp/dragstrip-repos"))
CLONE_TIMEOUT_S = 90
MAX_REPO_MB = 200
MAX_TARBALL_MB = 80
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
    """Fetch a public GitHub repo's default branch into the cache, once.

    Uses the tarball endpoint rather than `git clone` so it works anywhere
    Python does — serverless included, where there is no git binary.
    """
    m = re.match(r"^https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$",
                 repo_url.rstrip("/"))
    if not m:
        raise RepoError("Only public github.com/<owner>/<repo> URLs are supported.")
    owner, name = m.group(1), m.group(2)
    dest = CACHE_DIR / _slug(repo_url)
    marker = dest / ".dragstrip-ready"
    with _locks_guard:
        lock = _clone_locks.setdefault(str(dest), threading.Lock())
    with lock:
        if marker.exists():
            return dest
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Prefer git when the host has it (dev boxes); fall back to the
        # tarball endpoint where it doesn't (serverless).
        if shutil.which("git"):
            import subprocess
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--single-branch",
                     f"https://github.com/{owner}/{name}", str(dest)],
                    check=True, capture_output=True, timeout=CLONE_TIMEOUT_S)
                size_mb = sum(f.stat().st_size for f in dest.rglob("*")
                              if f.is_file()) / 1e6
                if size_mb > MAX_REPO_MB:
                    shutil.rmtree(dest)
                    raise RepoError(f"Repo is {size_mb:.0f}MB — over the "
                                    f"{MAX_REPO_MB}MB demo cap.")
                marker.touch()
                return dest
            except subprocess.TimeoutExpired:
                raise RepoError("Clone timed out — repo may be too large.")
            except subprocess.CalledProcessError as e:
                raise RepoError("Clone failed: "
                                + e.stderr.decode(errors="replace")[-300:])

        url = f"https://codeload.github.com/{owner}/{name}/tar.gz/HEAD"
        try:
            with httpx.stream("GET", url, timeout=CLONE_TIMEOUT_S,
                              follow_redirects=True) as resp:
                if resp.status_code == 404:
                    raise RepoError(f"{owner}/{name} not found (private repos aren't supported).")
                resp.raise_for_status()
                buf, total = io.BytesIO(), 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > MAX_TARBALL_MB * 1e6:
                        raise RepoError(f"Repo tarball is over the {MAX_TARBALL_MB}MB demo cap.")
                    buf.write(chunk)
        except httpx.HTTPError as e:
            raise RepoError(f"Fetch failed: {e}")
        buf.seek(0)
        tmp = dest.parent / (dest.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                members = []
                for mem in tar.getmembers():
                    # strip the "<repo>-<sha>/" top folder; refuse path escapes
                    parts = mem.name.split("/", 1)
                    if len(parts) < 2 or not parts[1]:
                        continue
                    rel = parts[1]
                    if rel.startswith("/") or ".." in rel.split("/"):
                        continue
                    if not (mem.isfile() or mem.isdir()):
                        continue  # no symlinks/devices from untrusted archives
                    mem.name = rel
                    members.append(mem)
                tar.extractall(tmp, members=members)
        except tarfile.TarError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RepoError(f"Bad tarball: {e}")
        size_mb = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file()) / 1e6
        if size_mb > MAX_REPO_MB:
            shutil.rmtree(tmp)
            raise RepoError(f"Repo is {size_mb:.0f}MB — over the {MAX_REPO_MB}MB demo cap.")
        tmp.rename(dest)
        marker.touch()
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
    """Pure-Python grep -rn: portable to environments with no grep binary."""
    import fnmatch

    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"Bad pattern: {e}"
    hits, truncated = [], False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in sorted(filenames):
            if glob and not fnmatch.fnmatch(name, glob):
                continue
            p = Path(dirpath) / name
            rel = p.relative_to(root)
            try:
                if p.stat().st_size > 2_000_000:
                    continue
                text = p.read_text(errors="strict")
            except (OSError, UnicodeDecodeError):
                continue  # binary or unreadable
            per_file = 0
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"./{rel}:{i}:{line.strip()[:200]}")
                    per_file += 1
                    if per_file >= 8:
                        break
                    if len(hits) > MAX_GREP_LINES:
                        truncated = True
                        break
            if truncated:
                break
        if truncated:
            break
    if not hits:
        return f"No matches for /{pattern}/"
    if truncated or len(hits) > MAX_GREP_LINES:
        hits = hits[:MAX_GREP_LINES] + ["... more matches truncated"]
    return "\n".join(hits)


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
