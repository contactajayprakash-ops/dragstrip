"""Dragstrip — race the same agent with and without Paritok compression.

FastAPI app: kicks off two identical agent lanes against a repo question,
streams live telemetry over SSE, and finishes with a savings receipt and a
blind quality verdict. Finished races are saved and replayable, so the UI
works (and demos) even with no API keys configured.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import repo_tools
from agent import Lane, judge_answers, price_per_mtok

ROOT = Path(__file__).parent
_default_races = "/tmp/dragstrip-races" if os.environ.get("VERCEL") else ROOT / "races"
RACES_DIR = Path(os.environ.get("DRAGSTRIP_RACES", _default_races))
RACES_DIR.mkdir(parents=True, exist_ok=True)

# Seed demo replays into the live races dir (fresh deploys boot with something
# on the shelf even before anyone runs a race). Local seeds win; otherwise pull
# them from the public repo — covers hosts where the seed files aren't bundled.
def _seed_replays():
    seeded = False
    for demo in (ROOT / "races_seed").glob("*.json"):
        target = RACES_DIR / demo.name
        if not target.exists():
            try:
                target.write_text(demo.read_text())
            except OSError:
                return
        seeded = True
    if seeded or any(RACES_DIR.glob("*.json")):
        return
    repo = os.environ.get("DRAGSTRIP_SEED_REPO", "contactajayprakash-ops/dragstrip")
    try:
        import httpx
        listing = httpx.get(
            f"https://api.github.com/repos/{repo}/contents/races_seed",
            timeout=10, headers={"User-Agent": "dragstrip"}).json()
        for item in listing if isinstance(listing, list) else []:
            if item.get("name", "").endswith(".json") and item.get("download_url"):
                body = httpx.get(item["download_url"], timeout=10,
                                 headers={"User-Agent": "dragstrip"}).text
                (RACES_DIR / item["name"]).write_text(body)
    except Exception:
        pass  # no seeds is a degraded shelf, not a broken app


_seed_replays()

app = FastAPI(title="Dragstrip")


class RaceRequest(BaseModel):
    repo_url: str
    question: str


class RaceState:
    def __init__(self, race_id: str, repo_url: str, question: str):
        self.race_id = race_id
        self.repo_url = repo_url
        self.question = question
        self.events: list[dict] = []
        self.done = False
        self.cond = threading.Condition()
        self.stop_flag = threading.Event()
        self.t0 = time.time()

    def emit(self, etype: str, payload: dict):
        evt = {"t": round(time.time() - self.t0, 2), "type": etype, **payload}
        with self.cond:
            self.events.append(evt)
            if etype == "race_done":
                self.done = True
            self.cond.notify_all()


RACES: dict[str, RaceState] = {}


def _warm_gpu():
    """Nudge Paritok's hosted GPU awake so the first compression isn't a cold start."""
    key = os.environ.get("PARITOK_API_KEY")
    if not key:
        return

    def ping():
        try:
            import httpx
            httpx.get("https://www.paritok.com/api/test", timeout=60,
                      headers={"Authorization": f"Bearer {key}"})
        except Exception:
            pass  # warmup is best-effort; the strategy degrades gracefully anyway

    threading.Thread(target=ping, daemon=True).start()


def _persist(state: RaceState, receipt: dict | None):
    doc = {
        "race_id": state.race_id,
        "repo_url": state.repo_url,
        "question": state.question,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "receipt": receipt,
        "events": state.events,
    }
    try:
        (RACES_DIR / f"{state.race_id}.json").write_text(json.dumps(doc, indent=1))
    except OSError:
        pass  # read-only host — the race still streamed fine


def _run_race(state: RaceState):
    emit = state.emit
    try:
        _warm_gpu()
        emit("status", {"message": f"Cloning {state.repo_url} …"})
        repo_root = repo_tools.clone_repo(state.repo_url)
        emit("status", {"message": "Repo ready. Lights out — both lanes running."})

        lanes = {
            "raw": Lane("raw", repo_root, state.question, emit,
                        use_paritok=False, stop_flag=state.stop_flag),
            "paritok": Lane("paritok", repo_root, state.question, emit,
                            use_paritok=True, stop_flag=state.stop_flag),
        }
        threads = [threading.Thread(target=l.run, daemon=True) for l in lanes.values()]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=900)

        raw, par = lanes["raw"], lanes["paritok"]
        verdict = None
        if raw.answer and par.answer and not (raw.error or par.error):
            emit("status", {"message": "Photo finish — blind judge comparing answers."})
            try:
                # A/B order fixed: A = raw, B = paritok. The judge never knows which.
                verdict = judge_answers(state.question, raw.answer, par.answer)
            except Exception as e:
                verdict = {"score_a": None, "score_b": None,
                           "note": f"judge unavailable: {e}"}
            emit("verdict", {"raw_score": verdict.get("score_a"),
                             "paritok_score": verdict.get("score_b"),
                             "note": verdict.get("note", "")})

        model = raw.model
        rate = price_per_mtok(model)
        saved = raw.meter.input_tokens - par.meter.input_tokens
        receipt = {
            "model": model,
            "usd_per_mtok": rate,
            "raw": raw.meter.as_dict() | {"wall_seconds": raw.wall_seconds,
                                          "error": raw.error},
            "paritok": par.meter.as_dict() | {"wall_seconds": par.wall_seconds,
                                              "error": par.error},
            "input_tokens_saved": saved,
            "input_pct_saved": round(100 * saved / raw.meter.input_tokens, 1)
                               if raw.meter.input_tokens else 0.0,
            "cost_saved_usd": round(saved / 1e6 * rate, 6),
            # what the same token gap costs at Claude Sonnet list price —
            # the bill most coding agents actually pay
            "cost_saved_at_sonnet_usd": round(saved / 1e6 * 3.0, 6),
            "verdict": verdict,
        }
        emit("race_done", {"receipt": receipt})
        _persist(state, receipt)
    except repo_tools.RepoError as e:
        emit("race_error", {"message": str(e)})
        emit("race_done", {"receipt": None})
    except Exception as e:
        emit("race_error", {"message": f"{type(e).__name__}: {e}"})
        emit("race_done", {"receipt": None})


@app.post("/api/race")
def start_race(req: RaceRequest):
    if not (os.environ.get("DRAGSTRIP_LLM_API_KEY") or os.environ.get("GROQ_API_KEY")):
        raise HTTPException(503, "No LLM API key configured — try a saved replay below.")
    if not req.question.strip():
        raise HTTPException(400, "Ask a question about the repo.")
    race_id = uuid.uuid4().hex[:12]
    state = RaceState(race_id, req.repo_url.strip(), req.question.strip())
    RACES[race_id] = state
    threading.Thread(target=_run_race, args=(state,), daemon=True).start()
    return {"race_id": race_id}


@app.get("/api/race/{race_id}/events")
def race_events(race_id: str):
    state = RACES.get(race_id)
    if state is None:
        raise HTTPException(404, "No such race.")

    def gen():
        i = 0
        while True:
            with state.cond:
                while i >= len(state.events) and not state.done:
                    state.cond.wait(timeout=15)
                batch = state.events[i:]
                i = len(state.events)
                finished = state.done
            for evt in batch:
                yield f"data: {json.dumps(evt)}\n\n"
            if finished and i >= len(state.events):
                return
            if not batch:
                yield ": keepalive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/race/stream")
def race_stream(repo_url: str, question: str):
    """Single-request race: start it and stream its SSE until the flag drops.

    This is the whole race in one HTTP response, which is what lets Dragstrip
    run on serverless hosts (no cross-request state needed). The regular
    POST /api/race + GET /events pair still works for long-lived servers.
    """
    if not (os.environ.get("DRAGSTRIP_LLM_API_KEY") or os.environ.get("GROQ_API_KEY")):
        raise HTTPException(503, "No LLM API key configured — try a saved replay below.")
    if not question.strip():
        raise HTTPException(400, "Ask a question about the repo.")
    race_id = uuid.uuid4().hex[:12]
    state = RaceState(race_id, repo_url.strip(), question.strip())
    RACES[race_id] = state
    threading.Thread(target=_run_race, args=(state,), daemon=True).start()

    def gen():
        yield f"data: {json.dumps({'t': 0, 'type': 'race_id', 'race_id': race_id})}\n\n"
        i = 0
        while True:
            with state.cond:
                state.cond.wait(timeout=10)
                batch = state.events[i:]
                i = len(state.events)
                finished = state.done
            for evt in batch:
                yield f"data: {json.dumps(evt)}\n\n"
            if finished:
                return
            if not batch:
                yield ": keepalive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/race/{race_id}/stop")
def stop_race(race_id: str):
    state = RACES.get(race_id)
    if state is None:
        raise HTTPException(404, "No such race.")
    state.stop_flag.set()
    return {"ok": True}


@app.get("/api/replays")
def list_replays():
    out = []
    for f in sorted(RACES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime,
                    reverse=True)[:30]:
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        r = doc.get("receipt")
        # only clean finishes make the replay shelf
        if not r or r["raw"].get("error") or r["paritok"].get("error"):
            continue
        out.append({
            "race_id": doc["race_id"],
            "repo_url": doc["repo_url"],
            "question": doc["question"],
            "saved_at": doc.get("saved_at"),
            "pct_saved": r.get("input_pct_saved"),
            "tokens_saved": r.get("input_tokens_saved"),
        })
    return out


@app.get("/api/replay/{race_id}")
def get_replay(race_id: str):
    if not re.fullmatch(r"[0-9a-f]{12}", race_id):
        raise HTTPException(400, "Bad race id.")
    f = RACES_DIR / f"{race_id}.json"
    if not f.exists():
        raise HTTPException(404, "No such replay.")
    return json.loads(f.read_text())


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "llm_key": bool(os.environ.get("DRAGSTRIP_LLM_API_KEY")
                        or os.environ.get("GROQ_API_KEY")),
        "paritok_hosted": bool(os.environ.get("PARITOK_API_KEY")),
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
