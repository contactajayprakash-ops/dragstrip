# Dragstrip

Race the same coding agent twice — once talking straight to the model, once with
[Paritok](https://github.com/Paritok-official/paritok-4b-v1) compressing its
context — and watch the two token bills split in real time.

Context compression tools all make the same promise: smaller bills, same answers.
Before you route your production agent through one, you want to see that claim
tested on *your* repo with *your* kind of questions, not on a benchmark table.
Dragstrip is that test, as a drag race.

## What a race is

You give it a public GitHub repo and a question that takes real digging
("how does request routing work, end to end?"). Dragstrip runs two identical
agents in parallel — same model, same tools (`list_files`, `read_file`,
`search_code`, `file_outline`), same turn budget, same prompt:

- **Raw lane** sends every request to the LLM as-is.
- **Paritok lane** pushes every request through Paritok's compression engine
  first: big tool outputs get rewritten by the paritok-4b model into compact
  summaries with `[REF:id]` tags, and the agent gets an `expand_context` tool
  to pull back any original, locally and for free, if the summary isn't enough.

Every API call's *billed* input tokens (straight from the provider's `usage`
field, not an estimate) feed two meters. At the finish you get a receipt —
tokens saved, dollars saved at the actual model's rate and at Claude Sonnet
rates — plus a blind judge's quality scores: a third model grades both answers
without knowing which lane produced which.

If compression hurt the answer, the judge will say so. That's the point.

## Running it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...           # or any OpenAI-compatible provider, see below
uvicorn server:app --port 8000
```

Open http://localhost:8000, paste a repo, ask a question, hit START RACE.

Compression needs the paritok-4b model, one of two ways:

**Paritok hosted GPU (recommended)** — create a free API key at
[paritok.com](https://www.paritok.com), then:

```bash
export PARITOK_API_KEY=...
```

and set `use_gpu_server: true` in `paritok.yaml`. Your compression traffic also
shows up on your paritok.com dashboard.

**Self-hosted** — run the open model locally:

```bash
ollama pull paritok/paritok-4b-v1 && ollama cp paritok/paritok-4b-v1 paritok-4b-v1
```

with `use_gpu_server: false`. Works fine on CPU, just slower — the Paritok lane
waits on each compression.

### Configuration

| Env var | Default | What it does |
|---|---|---|
| `GROQ_API_KEY` / `DRAGSTRIP_LLM_API_KEY` | — | key for the agent's LLM |
| `DRAGSTRIP_LLM_BASE_URL` | Groq's endpoint | any OpenAI-compatible base URL |
| `DRAGSTRIP_LLM_MODEL` | `llama-3.3-70b-versatile` | model both lanes use |
| `DRAGSTRIP_JUDGE_MODEL` | same as above | model for the blind judge |
| `PARITOK_API_KEY` | — | Paritok hosted GPU key |
| `DRAGSTRIP_MAX_TURNS` | `10` | per-lane turn budget |

Finished races are saved under `races/` and replayable from the UI, so the app
demos fine even with no keys configured.

## Reading the results honestly

- The savings number is real billed usage, but it's one race, not a benchmark.
  Savings swing a lot with repo size and how chatty the tools get — boring
  questions on tiny repos can save nothing (compression skips outputs under
  ~512 tokens).
- The Paritok lane is slower on wall clock: compression is an extra model call
  before every API request. The race measures tokens, not latency, and the
  lane feeds timestamp everything so you can see the trade.
- The judge is a language model with the usual caveats; treat scores as a
  smoke test, not a verdict. Both full answers are shown so you can judge too.

## How it's put together

```
server.py       FastAPI: race orchestration, SSE telemetry, replay store
agent.py        the two lanes + the blind judge (OpenAI-compatible client)
repo_tools.py   clone + the four repo tools both lanes share
static/         the UI, one file
```

The Paritok integration uses `paritok.middleware.ParitokEngine` directly
(SDK mode): each turn's messages go through `process_request` before the API
call, and `expand_context` calls are resolved locally through
`resolve_virtual_call`. Per-turn compression stats stream to the UI from the
same `CompressionStats` the engine returns, and dollar figures use Paritok's
own pricing table.

## Credit

Compression by [Paritok](https://github.com/Paritok-official/paritok-4b-v1)
(paritok-4b-v1, Apache 2.0) — built for the Build with Paritok hackathon,
August 2026.

## License

Apache 2.0 — see [LICENSE](LICENSE).
