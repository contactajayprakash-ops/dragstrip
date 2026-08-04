# Dragstrip — Devpost submission text

## Elevator pitch (≤200 chars)

Dragstrip — race the same coding agent with and without Paritok compression on any GitHub repo, and get a receipt: real billed tokens, real dollars, and a blind judge scoring both answers.

## Inspiration

My Claude Code bill is mostly input tokens. Every file an agent reads gets
re-sent, and re-billed, on every turn after it. So when Paritok showed up
claiming context compression could cut most of that, my first reaction wasn't
"great, I'll install it" — it was "prove it on my repo, on my questions."
No tool existed to show me that. Benchmarks are someone else's repo and
someone else's questions. So I built the proof as a spectacle: put the raw
agent and the compressed agent on a drag strip and let the meters settle it.

## What it does

Dragstrip races two identical coding agents against any public GitHub repo.
You ask one hard question ("how does request routing work, end to end?").
Both lanes get the same model, the same four repo tools, the same turn
budget. The only difference: the Paritok lane pushes every request through
Paritok's compression engine before it hits the API.

The UI is a live timing tower — cumulative billed input tokens per lane
(read straight from the provider's usage field, not estimated), cost
tickers, every tool call and compression event timestamped in each lane's
feed. At the finish you get a receipt: tokens saved, dollars saved at the
actual model's rate and projected at Claude Sonnet rates, plus a blind
judge's verdict — a third model scores both answers without knowing which
lane wrote which. Every race is saved and replayable.

On my recorded races the Paritok lane came in 18–71% under the raw lane on
billed input tokens, with the judge scoring the answers even (it preferred
the compressed lane's answer once).

## How I built it

FastAPI backend, one-file frontend, no framework, deployed on Vercel
serverless. The Paritok integration is SDK-mode, not the proxy: each turn's
messages go through `ParitokEngine.process_request()` before the API call,
and when the model calls Paritok's `expand_context` virtual tool to pull
back an original file, `resolve_virtual_call()` answers it from local shadow
storage — the lane feed marks those "resolved locally, zero re-read."
Per-turn savings stream to the UI from the engine's own CompressionStats,
and dollar figures use Paritok's pricing table. Compression runs on
Paritok's hosted GPU, so every request also lands on my Paritok dashboard —
the savings are verifiable, not self-reported. The agent side is any
OpenAI-compatible provider; the deployed app runs Gemini 3.5 Flash Lite.

## Challenges I ran into

Keeping the comparison honest was most of the work: billed usage instead of
token estimates, identical prompts and budgets in both lanes, a judge that's
blind to lane identity, and a guard that refuses to accept an answer from an
agent that never opened a file. Serverless was its own fight — the whole
race now runs inside a single streaming request so it works without shared
state. And I hit a real Paritok bug along the way: the hosted GPU sometimes
returns an empty string as a "successful" compression, which silently
destroys the tool result. I filed it upstream with a repro and a proposed
fix (paritok-4b-v1 issue #20) and guarded against it in the app.

## Accomplishments that I'm proud of

The race format turns a benchmark table into something you can watch. The
numbers are real billed usage. And the tool is honest enough to show
compression losing ground — small explorations save 18%, deep ones save
70%, and the feeds show the wall-clock cost compression adds. That honesty
is what makes the wins believable.

## What I learned

Compression is query-aware summarization, not truncation — watching
paritok-4b rewrite a 5,300-token file read into 583 tokens that still name
the right classes changed how I think about agent context. The clever part
is `expand_context`: compression is safe because it's reversible. Also:
tool ecosystems are young — the difference between "it demos" and "it
holds up" is finding the empty-compression bugs before your users do.

## What's next for Dragstrip

CI mode. Run your agent's eval suite through both lanes on every Paritok
release and get the savings/quality delta as a PR comment. The race becomes
a regression test — for Paritok itself and for anyone deciding whether
compression is safe for their stack.

## Scope note (also goes in questionnaire + YouTube description)

To be clear about scope: race results are single runs, not benchmarks —
savings swing with repo size and how deep the agent digs, and the two lanes
can take different exploration paths since compression changes what the
model sees. The blind judge is an LLM and should be read as a smoke test;
both full answers are shown so you can judge yourself. The live demo runs
on free-tier API quotas, so lanes occasionally pause to ride out rate
limits — the feeds display those stalls rather than hiding them.

---

# YouTube metadata

**Title:** Dragstrip — race your AI agent's token bill (Build with Paritok hackathon)

**Description:**

Two identical coding agents answer the same question about a GitHub repo.
One talks straight to the model. One runs every request through Paritok's
context compression first. Dragstrip shows what that does to the bill —
live billed tokens, a savings receipt, and a blind judge scoring both
answers.

Try it: https://dragstrip.vercel.app
Code (Apache 2.0): https://github.com/contactajayprakash-ops/dragstrip
Compression: Paritok (paritok-4b-v1) hosted GPU — https://github.com/Paritok-official/paritok-4b-v1

All numbers in this video are from real recorded races. Scope note: results
are single runs, not benchmarks — savings vary with how deep the agent digs,
and the compressed lane pays a wall-clock cost for compression calls. The
judge is an LLM; treat its scores as a smoke test.

Built solo for the Build with Paritok hackathon, August 2026.

Music: Funkorama — Kevin MacLeod (incompetech.com), licensed under CC BY 4.0

---

# Devpost gallery captions

1. gal_title.png — Dragstrip: two identical agents, one difference. The blue lane's requests pass through Paritok's compression engine before every API call.
2. gal_race.png — A real race on pallets/flask. Feeds show tool calls, per-segment compression ratios from Paritok's hosted GPU, and expand_context retrievals resolved locally for free.
3. gal_receipt.png — The receipt is billed usage from the provider's own usage field, not an estimate. The blind judge scores both answers without knowing which lane produced which.
4. gal_honest.png — Four recorded races, four different savings. That variance is the reason the tool exists: test compression on your repo before adopting it.
