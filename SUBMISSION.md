# Dragstrip — Devpost story (draft; voice-match against past submission before filing)

## Inspiration

My Claude Code bill is mostly input tokens. Every tool call an agent makes —
a file read, a grep, a test run — comes back into context and gets re-billed
on every turn after it. So when Paritok showed up claiming 74% input savings
from compressing exactly that stuff, my first reaction wasn't "great, I'll
install it." It was "prove it on my repo, on my questions." I couldn't find a
tool that would show me that. Benchmarks are someone else's repo and someone
else's questions.

So I built the proof as a spectacle: put the raw agent and the compressed
agent on a drag strip and let the meters settle it.

## What it does

Dragstrip races two identical coding agents on any public GitHub repo. You ask
one hard question ("how does request routing work, end to end?"). Both lanes
get the same model, the same four repo tools, the same turn budget. The only
difference: the Paritok lane pushes every request through Paritok's compression
engine before it hits the API.

The UI is a live timing tower — cumulative billed input tokens per lane (read
straight from the provider's usage field, not estimated), cost tickers, every
tool call and compression event timestamped in each lane's feed. At the finish
you get a receipt: tokens saved, dollars saved at the actual model's rate and
projected at Claude Sonnet rates, and a blind judge's verdict — a third model
scores both answers without knowing which lane wrote which. Every race is
saved and replayable.

## How I built it

FastAPI backend, one-file frontend, no framework. The Paritok integration is
SDK-mode, not the proxy: each turn's messages go through
`paritok.middleware.ParitokEngine.process_request()` before the API call, and
when the model calls Paritok's `expand_context` virtual tool to pull back an
original file, `resolve_virtual_call()` answers it locally — the lane feed
marks those "resolved locally, zero re-read." Per-turn savings stream to the
UI from the engine's own CompressionStats; dollar figures use Paritok's
pricing table. Compression runs on Paritok's hosted GPU (or a local
paritok-4b-v1 via Ollama — both paths work). The agent side is any
OpenAI-compatible provider; I ran Groq's llama-3.3-70b.

## Challenges I ran into

Getting the comparison honest was most of the work. Billed usage instead of
token estimates; identical prompts and budgets in both lanes; a judge that's
blind to lane identity; and a receipt that shows compression's cost too — the
Paritok lane is slower on wall clock, and the feeds don't hide it.

## Accomplishments I'm proud of

The race format turns a benchmark table into something you can watch. And the
numbers are real: on my test races the Paritok lane's input bill came in
50–70% under the raw lane with the judge scoring the answers even.

## What I learned

Compression is query-aware summarization, not truncation — watching
paritok-4b rewrite a 4,700-token grep dump into 340 tokens that still name the
right files changed how I think about agent context. Also: expand_context is
the clever part. Compression is safe because it's reversible.

## What's next

CI mode — run your agent's eval suite through both lanes on every Paritok
release and get the savings/quality delta as a PR comment. The race becomes a
regression test.
