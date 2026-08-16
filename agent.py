"""The two lanes.

Both lanes run the exact same agent: same system prompt, same tools, same
model, same turn budget. The only difference is that the paritok lane pushes
every request through Paritok's compression engine first. Whatever gap shows
up on the meters is compression's doing, nothing else.

Internal message format is Anthropic-style blocks (what Paritok's engine
speaks natively); we translate to OpenAI wire format at the API boundary so
any OpenAI-compatible provider works (Groq by default).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from openai import OpenAI

from paritok.config import ParitokConfig
from paritok.middleware.wrapper import ParitokEngine
from paritok.proxy.pricing import DEFAULT_USD_PER_MTOK, INPUT_USD_PER_MTOK

import repo_tools

MAX_TURNS = int(os.environ.get("DRAGSTRIP_MAX_TURNS", "10"))
MAX_TOOL_RESULT_CHARS = 60_000

SYSTEM_PROMPT = (
    "You are a code-analysis agent answering one question about a repository. "
    "Investigate properly before answering: map the layout, search for the "
    "relevant symbols, then READ the actual implementations — the real "
    "function bodies, not just search hits or file names. Follow the call "
    "chain across files when the question spans one. Do not answer from "
    "assumptions or from README prose alone, and do not re-read content you "
    "already have. When you can back every claim with a file path and line "
    "number, stop calling tools and write the final answer in plain prose "
    "with those citations."
)


# List prices (per 1M input tokens) for models Paritok's table doesn't know.
EXTRA_USD_PER_MTOK = {
    "llama-3.3-70b-versatile": 0.59,
    "openai/gpt-oss-120b": 0.15,
    "openai/gpt-oss-20b": 0.075,
    "gemini-3.5-flash": 0.75,
    "gemini-3.5-flash-lite": 0.30,
}


def _guard_empty_compression(engine: ParitokEngine):
    """Belt-and-braces guard against an empty compression being treated as a
    valid summary, which silently destroys the tool result.

    This was a live hosted-GPU bug when Dragstrip was built: /compress returned
    {"compressed": "", "gpu_available": true} and the client accepted "" as a
    summary, so the agent went blind on that tool output and burned MORE tokens
    re-exploring. Reported as paritok-4b-v1 issue #20 and fixed upstream in
    1.3.5 / 1.3.6 — the endpoint no longer returns an empty body, and a
    query-conditioned drop now comes back as a recoverable stub carrying
    dropped: true (distinguishable from a GPU outage, which sets
    gpu_available: false).

    The guard is inert against a current hosted backend — the stub is longer
    than the threshold below, so it passes through and the intended behaviour
    wins. Kept for self-hosted and pinned-older setups, where the client-side
    check is still the only thing standing between an empty body and a blinded
    agent."""
    strategy = engine.pipeline._model
    inner = strategy.compress

    def compress(content: str, **kw) -> str:
        out = inner(content, **kw)
        if len(out.strip()) < 20 and len(content.strip()) >= 200:
            return content
        return out

    strategy.compress = compress


def price_per_mtok(model: str) -> float:
    """Longest-prefix match against Paritok's pricing table + Groq additions."""
    table = INPUT_USD_PER_MTOK | EXTRA_USD_PER_MTOK
    best, best_len = DEFAULT_USD_PER_MTOK, 0
    for prefix, usd in table.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best, best_len = usd, len(prefix)
    return best


@dataclass
class LaneMeter:
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    tool_calls: int = 0
    compressed_from: int = 0     # original tokens fed to compressor
    compressed_to: int = 0       # what came back
    turn_log: list = field(default_factory=list)

    def as_dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "api_calls": self.api_calls,
            "tool_calls": self.tool_calls,
            "compressed_from": self.compressed_from,
            "compressed_to": self.compressed_to,
        }


class Lane:
    """One racer: an agent loop over a repo, with metering."""

    def __init__(self, lane_id: str, repo_root, question: str, emit,
                 use_paritok: bool, stop_flag: threading.Event):
        self.lane_id = lane_id
        self.repo_root = repo_root
        self.question = question
        self.emit = emit                      # emit(event_type, payload)
        self.use_paritok = use_paritok
        self.stop_flag = stop_flag
        self.meter = LaneMeter()
        self.answer: str | None = None
        self.error: str | None = None
        self.wall_seconds: float = 0.0

        base_url = os.environ.get("DRAGSTRIP_LLM_BASE_URL", "https://api.groq.com/openai/v1")
        api_key = os.environ.get("DRAGSTRIP_LLM_API_KEY") or os.environ.get("GROQ_API_KEY", "")
        self.model = os.environ.get("DRAGSTRIP_LLM_MODEL", "llama-3.3-70b-versatile")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)

        self.engine = None
        if use_paritok:
            cfg_path = os.environ.get("PARITOK_CONFIG", "paritok.yaml")
            cfg = ParitokConfig.load(cfg_path) if os.path.exists(cfg_path) else ParitokConfig()
            # 4 tools < top_k, so embedding-based tool discovery never fires;
            # keep the lane honest and the deploy slim.
            cfg.tool_discovery.strategy = "passthrough"
            self.engine = ParitokEngine(cfg)
            _guard_empty_compression(self.engine)

    # ── wire-format translation (canonical Anthropic blocks ↔ OpenAI) ──

    @staticmethod
    def _to_openai(messages: list[dict], tools: list[dict] | None):
        oai_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in messages:
            role, content = m["role"], m["content"]
            if isinstance(content, str):
                oai_msgs.append({"role": role, "content": content})
                continue
            if role == "assistant":
                text_parts, tool_calls = [], []
                for b in content:
                    if b["type"] == "text":
                        text_parts.append(b["text"])
                    elif b["type"] == "tool_use":
                        tc = {
                            "id": b["id"], "type": "function",
                            "function": {"name": b["name"],
                                         "arguments": json.dumps(b["input"])},
                        }
                        # Gemini 3.x requires its thought_signature round-tripped
                        # on every historical tool call.
                        if b.get("extra_content"):
                            tc["extra_content"] = b["extra_content"]
                        tool_calls.append(tc)
                msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                oai_msgs.append(msg)
            else:  # user turn: plain text and/or tool results
                for b in content:
                    if b["type"] == "text":
                        oai_msgs.append({"role": "user", "content": b["text"]})
                    elif b["type"] == "tool_result":
                        body = b["content"]
                        if isinstance(body, list):
                            body = "\n".join(x.get("text", "") for x in body
                                             if isinstance(x, dict))
                        oai_msgs.append({"role": "tool",
                                         "tool_call_id": b["tool_use_id"],
                                         "content": body})
        oai_tools = None
        if tools:
            oai_tools = [{"type": "function",
                          "function": {"name": t["name"],
                                       "description": t.get("description", ""),
                                       "parameters": t.get("input_schema", {})}}
                         for t in tools]
        return oai_msgs, oai_tools

    @staticmethod
    def _from_openai(msg) -> list[dict]:
        blocks = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            block = {"type": "tool_use", "id": tc.id,
                     "name": tc.function.name, "input": args}
            extra = getattr(tc, "extra_content", None)
            if extra is None and getattr(tc, "model_extra", None):
                extra = tc.model_extra.get("extra_content")
            if extra:
                block["extra_content"] = extra
            blocks.append(block)
        return blocks

    # ── the loop ──

    def run(self):
        t0 = time.time()
        try:
            self._run_inner()
        except Exception as e:  # surface, don't crash the race
            self.error = f"{type(e).__name__}: {e}"
            self.emit("lane_error", {"lane": self.lane_id, "message": self.error})
        finally:
            self.wall_seconds = round(time.time() - t0, 1)
            self.emit("lane_done", {
                "lane": self.lane_id, "answer": self.answer, "error": self.error,
                "wall_seconds": self.wall_seconds, "meter": self.meter.as_dict(),
            })

    def _run_inner(self):
        messages = [{"role": "user", "content": self.question}]
        stubbed: list = []
        nudges = 0

        for turn in range(1, MAX_TURNS + 1):
            if self.stop_flag.is_set():
                self.error = "stopped"
                return

            send_messages, send_tools = messages, list(repo_tools.TOOL_SCHEMAS)

            has_tool_results = any(
                isinstance(m.get("content"), list)
                and any(isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in m["content"])
                for m in messages)
            if self.engine and has_tool_results:
                self.emit("compressing", {"lane": self.lane_id, "turn": turn})
                send_messages, send_tools, stats, stubbed = self.engine.process_request(
                    [dict(m) for m in messages], send_tools, upstream_model=self.model)
                # snapshot, not a running sum: each request re-compresses the
                # whole context (cached), so the latest stats already cover
                # every tool result so far
                self.meter.compressed_from = stats.original_tokens
                self.meter.compressed_to = stats.compressed_tokens
                if stats.original_tokens:
                    self.emit("compression", {
                        "lane": self.lane_id, "turn": turn,
                        "original": stats.original_tokens,
                        "compressed": stats.compressed_tokens,
                        "ratio": stats.ratio,
                        "items": stats.items_compressed,
                        "cache_hits": stats.cache_hits,
                    })

            oai_msgs, oai_tools = self._to_openai(send_messages, send_tools)
            resp = self._call_llm(oai_msgs, oai_tools, turn)
            usage = resp.usage
            self.meter.api_calls += 1
            self.meter.input_tokens += usage.prompt_tokens
            self.meter.output_tokens += usage.completion_tokens
            self.emit("llm_usage", {
                "lane": self.lane_id, "turn": turn,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_input": self.meter.input_tokens,
                "cost_usd": round(self.meter.input_tokens / 1e6
                                  * price_per_mtok(self.model), 6),
            })

            blocks = self._from_openai(resp.choices[0].message)
            tool_uses = [b for b in blocks if b["type"] == "tool_use"]

            if not tool_uses:
                # An answer with no repo evidence behind it isn't an answer —
                # send the agent back in. Same nudge in both lanes; two max.
                if self.meter.tool_calls == 0 and nudges < 2:
                    nudges += 1
                    messages.append({"role": "assistant", "content": blocks})
                    messages.append({"role": "user", "content": (
                        "You have not opened a single file. Do not answer from "
                        "prior knowledge of this library — investigate this "
                        "actual checkout with the tools, then answer with file "
                        "and line citations.")})
                    self.emit("status_lane", {"lane": self.lane_id,
                                              "message": "answered without looking — sent back in"})
                    continue
                self.answer = "\n".join(b["text"] for b in blocks
                                        if b["type"] == "text").strip()
                messages.append({"role": "assistant", "content": blocks})
                return

            messages.append({"role": "assistant", "content": blocks})
            results = []
            for tu in tool_uses:
                out = self._run_tool(tu, stubbed, turn)
                results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                "content": out})
            messages.append({"role": "user", "content": results})

        # Out of turns: no more tools, answer with what's in hand.
        send_messages = messages
        if self.engine:
            send_messages, _, _, _ = self.engine.process_request(
                [dict(m) for m in messages], None, upstream_model=self.model)
        send_messages = send_messages + [{
            "role": "user",
            "content": "Turn budget exhausted — answer the question now, as "
                       "well as you can, from what you have already seen.",
        }]
        oai_msgs, _ = self._to_openai(send_messages, None)
        resp = self._call_llm(oai_msgs, None, MAX_TURNS + 1)
        self.meter.api_calls += 1
        self.meter.input_tokens += resp.usage.prompt_tokens
        self.meter.output_tokens += resp.usage.completion_tokens
        self.emit("llm_usage", {
            "lane": self.lane_id, "turn": MAX_TURNS + 1,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_input": self.meter.input_tokens,
            "cost_usd": round(self.meter.input_tokens / 1e6
                              * price_per_mtok(self.model), 6),
        })
        self.answer = (resp.choices[0].message.content or "").strip() or \
            "(ran out of turns before finishing)"

    def _call_llm(self, oai_msgs, oai_tools, turn):
        """One chat completion, riding out provider rate limits.

        Free-tier TPM limits are part of the story this app tells — the raw
        lane slams into them much harder than the compressed lane — so instead
        of failing the race we wait out the window and say so on the feed.
        """
        import openai as openai_mod

        last_err = None
        bad_tool_strikes = 0
        for attempt in range(8):
            if self.stop_flag.is_set():
                raise RuntimeError("stopped")
            try:
                kwargs = dict(model=self.model, messages=oai_msgs,
                              temperature=0.1, max_tokens=1800)
                if oai_tools and bad_tool_strikes < 3:
                    kwargs["tools"] = oai_tools
                    kwargs["tool_choice"] = "auto"
                return self.client.chat.completions.create(**kwargs)
            except openai_mod.RateLimitError as e:
                last_err = e
                m = re.search(r"(?:try again|retry) in ([0-9.]+)\s*s", str(e))
                wait = min(float(m.group(1)) + 1.0 if m else 12.0, 45.0)
                self.emit("rate_limited", {"lane": self.lane_id, "turn": turn,
                                           "wait_s": round(wait, 1)})
                time.sleep(wait)
            except openai_mod.BadRequestError as e:
                # Some providers hard-fail the request when the model emits a
                # malformed tool call. Regenerate; after 3 strikes ask for
                # prose so the lane finishes instead of crashing.
                if "tool" not in str(e).lower():
                    raise
                last_err = e
                bad_tool_strikes += 1
                self.emit("status_lane", {"lane": self.lane_id,
                                          "message": "model fumbled a tool call — regenerating"})
        raise last_err

    def _run_tool(self, tu: dict, stubbed: list, turn: int) -> str:
        name, args = tu["name"], tu["input"]

        # Paritok's virtual tools (expand_context) resolve locally — free.
        if self.engine:
            virt = self.engine.resolve_virtual_call(name, args, stubbed_tools=stubbed)
            if virt is not None:
                self.emit("tool_call", {"lane": self.lane_id, "turn": turn,
                                        "tool": name, "args": args,
                                        "virtual": True, "chars": len(str(virt))})
                self.meter.tool_calls += 1
                return virt.get("content", json.dumps(virt))

        fn = repo_tools.TOOL_FNS.get(name)
        if fn is None:
            return f"Unknown tool: {name}"
        try:
            out = fn(self.repo_root, **args)
        except TypeError as e:
            out = f"Bad arguments for {name}: {e}"
        except repo_tools.RepoError as e:
            out = f"Tool error: {e}"
        out = out[:MAX_TOOL_RESULT_CHARS]
        self.meter.tool_calls += 1
        self.emit("tool_call", {"lane": self.lane_id, "turn": turn, "tool": name,
                                "args": args, "virtual": False, "chars": len(out)})
        return out


JUDGE_PROMPT = """You are grading two answers to the same question about a code repository. \
You do not know how either was produced. Score each 0-10 for correctness, specificity \
(file paths / line numbers), and completeness. Respond ONLY with JSON:
{{"score_a": <0-10>, "score_b": <0-10>, "note": "<one sentence comparing them>"}}

QUESTION: {question}

ANSWER A:
{a}

ANSWER B:
{b}"""


def judge_answers(question: str, answer_a: str, answer_b: str) -> dict:
    """Blind quality check, run outside both lanes."""
    base_url = os.environ.get("DRAGSTRIP_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.environ.get("DRAGSTRIP_LLM_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    model = os.environ.get("DRAGSTRIP_JUDGE_MODEL",
                           os.environ.get("DRAGSTRIP_LLM_MODEL", "llama-3.3-70b-versatile"))
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, a=answer_a[:6000], b=answer_b[:6000])}],
        # generous: thinking models spend part of this budget on reasoning
        max_tokens=4000,
    )
    text = resp.choices[0].message.content or "{}"
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"score_a": None, "score_b": None, "note": text[:200]}
