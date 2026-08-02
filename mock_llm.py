"""Offline stand-in for the LLM API (dev only, not part of the product).

Speaks just enough of the OpenAI Chat Completions dialect to exercise the
whole dragstrip pipeline without a key: plays a scripted code-exploring
agent, reports usage as ~chars/4 so message size differences (i.e. Paritok
compression) show up in prompt_tokens exactly like a real bill would.
"""

import json
import time
import uuid

from fastapi import FastAPI, Request

app = FastAPI()


def _est(o) -> int:
    return max(1, len(json.dumps(o)) // 4)


SCRIPT = [
    ("list_files", {"path": "."}),
    ("read_file", {"path": "README.md"}),
    ("search_code", {"pattern": "def main|class .*App|routing"}),
    ("read_file", {"path": "pyproject.toml"}),
]


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    has_tools = bool(body.get("tools"))
    prompt_tokens = _est(messages) + _est(body.get("tools") or [])

    if not has_tools:  # judge call
        content = json.dumps({"score_a": 8, "score_b": 8,
                              "note": "Both answers identify the same mechanism; B cites one extra file."})
    else:
        n_tool_rounds = sum(1 for m in messages if m.get("role") == "tool")
        if n_tool_rounds < len(SCRIPT):
            name, args = SCRIPT[n_tool_rounds]
            resp = {
                "id": "mock", "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model", "mock"),
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)}}],
                }}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 24,
                          "total_tokens": prompt_tokens + 24},
            }
            return resp
        content = ("The project wires its entry point through the CLI module: see "
                   "pyproject.toml [project.scripts] and the README quickstart. "
                   "(scripted mock answer for pipeline testing)")

    return {
        "id": "mock", "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": _est(content),
                  "total_tokens": prompt_tokens + _est(content)},
    }
