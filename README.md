---
title: Cyber Risk Assistant
<<<<<<< HEAD
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# TawasolPay AI Cyber Risk Assistant

A working system that takes TawasolPay's asset, vulnerability, threat intel, and business context, and produces a top-5 prioritized risk list with remediation guidance retrieved live from the real NIST SP 800-53 control catalog.

## A note on the LLM provider

The LLM client (`src/synthesis.py`) is provider-agnostic by design — it talks to any OpenAI-compatible chat completions endpoint. That's exactly why the env vars are named `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` rather than after whichever provider happens to be configured: a vendor-named variable pointing at a different vendor is a mismatch this project has deliberately avoided elsewhere (see Supporting Question 2), so it isn't introduced here either. The project currently runs on **Gemini 3.1 Flash-Lite**, chosen for its free-tier rate limit (15 RPM vs. 5 RPM on non-Lite Flash) and its `MINIMAL`-thinking default, which avoids the reasoning-token-exhaustion failure mode a previous provider hit (see Supporting Question 2). Switching providers — Groq, OpenRouter, or anything else OpenAI-compatible — only needs those three `.env` values changed, no code changes.

## Architecture

Two lanes, plus a thin synthesis step:

- **Structured lane** (`src/data_loader.py`, `src/scoring.py`): plain pandas joins across the five CSVs, and a deterministic point-based score with zero LLM involvement. This is what makes the ranking reproducible: the same data always produces the same top 5.
- **Retrieval lane** (`src/nist_ingest.py`, `src/retrieval.py`): the NIST SP 800-53 control catalog (1189 controls) embedded locally with `sentence-transformers` and searched via ChromaDB. Nothing here is hardcoded; the best-matching control is found by semantic search against the real control text every time.
- **Synthesis** (`src/synthesis.py`): the only place an LLM is called. It's handed pre-computed facts and pre-retrieved control text, and its only job is to write two readable paragraphs around them. If the call fails, a plain template takes over instead of crashing the app.

```
CSVs + threat report  --pandas joins-->  scored findings (top 5)
                                                |
NIST 800-53 CSV --embed--> ChromaDB  <--query---+
                                                |
                                          retrieved controls
                                                |
                                     LLM (facts + controls in, prose out)
                                                |
                                          Streamlit UI
```

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd tawasolpay-risk-assistant

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# then edit .env and paste in your Gemini key from https://aistudio.google.com/apikey

# 5. (Optional but recommended) Pre-build the NIST vector store so the app's
#    first load isn't slow - this embeds ~1,000 controls once and caches it
python scripts/build_vector_store.py

# 6. Run the app
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Running the tests

```bash
pytest tests/
```

This runs three kinds of tests, and it's worth knowing which is which:

| File | What it checks | Needs a key or network? |
|---|---|---|
| `tests/test_scoring.py` | The scoring formula in isolation, on hand-built example findings — including the exact CVSS-isn't-everything scenario the assignment brief describes | No |
| `tests/test_ranking_ground_truth.py` | The **entire structured lane end to end** (load → join → score → rank) against the real `data/*.csv` files, checked against a known-correct top 5 | No |
| `tests/test_synthesis_grounding.py` | Whether the LLM's generated explanation stays anchored to the facts it was given (right CVE, right control, doesn't invent a campaign match that wasn't there) | Yes — skips automatically without `LLM_API_KEY` |

`test_ranking_ground_truth.py` is the one to run first after any change to `data_loader.py` or `scoring.py` — it's a full-pipeline regression test, not just a formula check. If it fails, run `python tests/test_ranking_ground_truth.py` directly to print the current top 5 and compare it against the `EXPECTED_TOP_5` list in that file.

### Evaluating the retrieval lane

The NIST retrieval quality can't be checked with a fast unit test — it needs the real
embedding model, which downloads on first use, so it lives separately as a benchmark
script rather than in `pytest`:

```bash
python eval/nist_retrieval_eval.py
```

This runs 10 hand-labeled cases (each grounded in a real finding pattern from this data
pack, with an expected NIST control chosen by reading the actual control text) through the
live vector store and reports two pairs of metrics, kept clearly separate:

- **hit@1 / hit@3** (strict) — was the retrieved control_id an *exact* match for the
  expected (or an acceptable) control, at rank 1 or anywhere in the top 3?
- **family hit@1 / hit@3** — does the retrieved control's *base* id (its own `(n)`
  enhancement suffix stripped, if any) match the expected control's base id? The metric
  exists because retrieving, say, `IR-4(12)` when `IR-4` was expected is a
  human-reviewable correct answer — the right control family, findable by name — but it's
  still a looser match than an exact ID, so it's reported as its own labeled number rather
  than folded into the strict one. (In the current run below, every hit happens to be an
  exact match, so the two numbers coincide — the metric exists for when they don't, not
  because they currently do.)

A companion script is worth running alongside it whenever chunking or the embedding model
changes:

```bash
python eval/check_chunk_lengths.py
```

It counts every real chunk `nist_ingest.build_control_chunks()` produces with the embedding
model's own tokenizer and reports any that exceed the model's limit. This isn't a
theoretical concern — this project truncated long controls silently twice (see Supporting
Question 3 below for the full story), so it's cheap insurance against the same bug coming
back. It counts tokens rather than words deliberately: a word-count version of this same
check is what let the second occurrence through.

Real output from a run of `eval/nist_retrieval_eval.py`, on the current configuration
(chunking + cross-encoder re-ranking + generalized promotion + backfill, described in
Supporting Question 3, `all-MiniLM-L6-v2` for the embedding step):

```
Strict (exact control_id match):
  hit@1: 7/10 (70%)
  hit@3: 8/10 (80%)
Family (base control_id match, enhancements credited to their base):
  family hit@1: 7/10 (70%)
  family hit@3: 8/10 (80%)
```

A concrete case where cross-encoder re-ranking visibly changes the order —
`secrets-in-build-logs` (expects `IA-5`):

```
Top 3 by raw embedding distance (before re-ranking):
  SA-17      dist=0.5871
  CA-6       dist=0.5974
  SA-15(12)  dist=0.5994          <- IA-5 not in the top 3 at all here

Final top 3 (after cross-encoder re-rank + promotion + backfill):
  IA-5       dist=0.6085          <- now ranked first
  SA-17      dist=0.6456
  SA-17(7)   dist=0.6218
```

`IA-5` was a worse embedding-distance match than three other controls, but the
cross-encoder — which scores the query against each candidate's actual text directly,
rather than comparing two independently-embedded vectors — judged it the best answer once
it was in the running.

(Run it yourself to confirm — embedding-model behavior can shift slightly between
versions, so treat the numbers above as the measured result as of the fix described above,
not a permanent guarantee.) hit@3 not reaching 100% is a real, unresolved gap, not
something papered over — see Supporting Question 3 for exactly which cases still miss
and why.

## Deploying (public URL)

The easiest path is **Streamlit Community Cloud**, since it deploys directly from a
public GitHub repo with no separate hosting step:

1. Push this repo to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and pick this repo
   and `app.py` as the entrypoint.
3. In the app's settings, add your `LLM_API_KEY` (and `LLM_MODEL` / `LLM_BASE_URL` if
   you changed them) under **Secrets**, in `.env`-like TOML format:
   ```toml
   LLM_API_KEY = "your_key_here"
   ```
4. Deploy. You'll get a public `*.streamlit.app` URL.

Free-tier Streamlit apps sleep after inactivity — if you're demoing live, open the URL a
few minutes beforehand to wake it up.

## Project structure

```
tawasolpay-risk-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── app.py                          # Streamlit entrypoint
├── data/                           # the data pack (CSVs + threat report)
├── src/
│   ├── models.py                    # RiskFinding, ScoreBreakdown dataclasses
│   ├── data_loader.py               # CSV loading + the structured-lane joins
│   ├── scoring.py                    # the composite risk score - no LLM
│   ├── nist_ingest.py                # NIST catalog -> chunks -> embeddings -> ChromaDB
│   ├── retrieval.py                  # query-time NIST control search
│   └── synthesis.py                  # grounded LLM prompt + completion call + fallback
├── scripts/
│   └── build_vector_store.py         # pre-builds the NIST vector store
├── eval/
│   ├── nist_retrieval_eval.py         # hand-labeled retrieval benchmark (hit@1 / hit@3)
│   └── check_chunk_lengths.py         # flags chunks too long for the embedding model's token limit
├── tests/
│   ├── test_scoring.py               # scoring formula in isolation
│   ├── test_ranking_ground_truth.py  # full pipeline vs. the real data pack
│   └── test_synthesis_grounding.py   # LLM output stays anchored to given facts
└── chroma_db/                       # generated on first run, gitignored - not in this repo
```

## How the vector store works

Only one thing in this project gets embedded: the **NIST SP 800-53 control catalog**
(`data/NIST_SP-800-53_rev5_catalog_load.csv`). Everything else — the 5 data-pack CSVs and
the threat report — stays as plain pandas/text, for the reasons in Supporting Question 1
below.

| | |
|---|---|
| **Embedding model** | `all-MiniLM-L6-v2` (sentence-transformers), run locally — no external embedding API or key needed. This is the same model the project started with; a larger-context `bge-small-en-v1.5` was tried once the chunking fix below was in place and measurably underperformed MiniLM on this catalog (hit@3 40% vs. 60% in a controlled isolation test), so it was rejected rather than kept as a "bigger model, must be better" default — see Supporting Question 3 |
| **Vector size** | 384 dimensions |
| **What's embedded** | One *or more* chunks per control, not a flat one-per-row mapping anymore — **1,567 chunks total** (from 1,189 catalog rows). Base controls (`SI-2`) and their enhancements (`SI-2(1)`, `SI-2(2)`, ...) are still separate, not merged, for the same reason as before. A control short enough to fit the model's context stays one chunk; anything longer is packed into as many chunks as it takes to stay inside the limit, all tagged with the same `control_id` in their metadata. **Zero of the 1,567 chunks exceed the model's 256-token limit** — verified with the model's own tokenizer by `eval/check_chunk_lengths.py`, which reports that 153 of the 1,189 controls would be silently cut short without this chunking (`SA-12` alone runs to ~3,500 tokens) |
| **Chunk text** | Built in `nist_ingest.build_control_chunks()`. Chunk boundaries are decided by **real token counts from the model's own tokenizer**, never by a word-count proxy — word and token counts diverge on NIST's jargon-heavy prose, and a word threshold tuned against one model's tokenizer doesn't transfer to another's, which is how this pipeline acquired the same truncation bug twice. Splits prefer natural boundaries (paragraphs, then lettered sub-points, then sentences) and fall back to word boundaries only for a single sentence that busts the budget alone; the control id and name are repeated on every piece so a fragment stays findable |
| **Where it's saved** | ChromaDB `PersistentClient`, on disk at `./chroma_db/`, collection name `nist_800_53_controls`. Gitignored — every environment (yours, a teammate's, a deployed instance) builds its own copy once, rather than committing a binary vector store to the repo |
| **When it's built** | Lazily, the first time `get_or_build_collection()` runs — it checks `collection.count() > 0` first and skips re-embedding on every subsequent run. Or build it ahead of time with `python scripts/build_vector_store.py` so the app's first load isn't slow |
| **How it's queried** | `retrieval.build_query_text()` turns a scored finding's own facts (vulnerability name, CVE, exposure, missing controls) into a natural-language query. From there, `search_controls()` runs a five-step pipeline: (1) `collection.query()` does a nearest-neighbor search over chunks for a wide raw candidate pool; (2) a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores every `(query, candidate text)` pair directly and reorders the pool by that score, rather than by embedding distance alone; (3) chunks are aggregated to one best chunk per control_id, "best" meaning highest cross-encoder score; (4) **base-vs-enhancement promotion** runs on that re-ranked order — generalized across *every* control family present in the pool, not just whichever control happens to rank first, so a base control buried behind several of its own enhancements still gets promoted ahead of them; (5) if the result is still weak, it's backfilled from the top control's `related` list. Top 3 controls returned. The cross-encoder needs no new dependency — `sentence-transformers` (already in `requirements.txt`) provides `CrossEncoder` the same way it provides the embedding model |

---

## Supporting Question 1 — The data split

We embedded only the NIST SP 800-53 control catalog, because it contains ~1,000 controls
with long prose descriptions, and the right control for a given finding is a semantic
match rather than a lookup key we already possess. Everything else — the five CSVs and the
one-page threat report — was queried as structured records: it's small, exactly joinable
on keys we control (`asset_id`, `cve`, `business_service`), and the ranking needs to be
deterministic and reproducible, which embeddings would work against.

## Supporting Question 2 — Where it goes wrong

1. **Synthetic identifiers never match the real CISA KEV catalog.** Every `CVE-SYN-*`,
   `CTRL-SYN-*`, `CICD-SYN-*`, etc. ID in this dataset is fictional and will never appear
   in the real KEV feed, since it's a genuine government catalog of real-world CVEs. A
   system that only trusted KEV lookups for "actively exploited" would silently under-rank
   several of the highest-impact findings here (e.g. the Kong Gateway admin API exposure).
   *What we did:* `scoring.py` treats the (synthetic, but scenario-internal)
   `threat_intelligence.csv` match as the primary exploitation signal, and uses the real
   KEV match only as an additive bonus on top — never a gate.
2. **The same underlying issue duplicated across redundant assets can inflate the
   ranking.** The Fortinet VPN CVE pair appears on two VPN nodes; CitrixBleed appears on
   two load balancers. A naive top-5 could end up with two near-identical entries instead
   of five distinct risks. *What we did:* `rank_top_findings()` deduplicates by
   (business service, vulnerability name), keeping only the highest-scoring instance —
   see `test_rank_top_findings_deduplicates_same_issue_on_redundant_assets`.
3. **A single, small local embedding model can confuse closely related NIST control
   families** — e.g. an "excessive permissions" finding could plausibly retrieve AC-2
   (Account Management), AC-6 (Least Privilege), or CM-7 (Least Functionality), and a
   single embedding pass isn't guaranteed to pick the best one. *What we did:*
   `retrieve_controls()` returns the top 3 matches rather than trusting a single best
   guess, and the raw retrieved text is shown in the UI (in an expander under each risk
   card) so a human can verify the match rather than trusting the LLM's paraphrase blindly.
4. **Every deployed risk card was rendering a blank explanation, and the test written to
   catch exactly that had never once executed.** `openai/gpt-oss-120b` is a reasoning
   model, and Groq bills its internal reasoning against `max_tokens`; at the old value of
   350, `eval/trace_pipeline.py` (see below) showed traced calls spending up to 348 of
   those 350 tokens on invisible reasoning, leaving 1-2 tokens for the actual answer -
   silently empty output, `finish_reason: "length"` on all 15 traced calls, on 4 of the
   5 real risk cards the app renders. This is a worse failure than a wrong answer: it's
   indistinguishable in the UI from the app simply having nothing to say.
   `tests/test_synthesis_grounding.py` asserts things like `"SI-2" in explanation`, which
   would have failed loudly on an empty string the first time it ran against a real key -
   except it never had. The test reads `LLM_API_KEY` from `os.environ` directly, while
   the key lives in `.env`, which only `app.py` loaded - so the test always skipped,
   silently, on every run, including whichever ones had a key configured. The test that
   would have caught this bug existed the whole time; it just never ran. *What we did:*
   raised `max_tokens` and added `reasoning_effort="low"` in `synthesis.py` (confirmed by
   re-tracing: reasoning spend dropped to 40-139 tokens, `finish_reason: "stop"` on all 15
   calls); made the grounding tests call `load_dotenv()` themselves instead of assuming
   the environment was already populated; and separately found the tests' own substring
   checks would have failed on *correct* output too, since the model writes typographic
   punctuation (`SA‑22` with U+2011, not `SA-22`) that a raw `"SI-2" in explanation` check
   doesn't recognize - fixed with a normalization helper, so the tests now run for real
   and check what they were meant to check.
5. **A cross-encoder and an embedding model can disagree sharply, and the cross-encoder
   isn't automatically the one that's right.** `ransomware-incident-response` (expects
   `IR-4`) is the concrete case: `IR-4`'s own control text ("...detection and analysis,
   containment, eradication, and recovery") is a near word-for-word match for the query's
   wording, yet `IR-4` itself never appears in the candidate pool at all — the base control
   is invisible to embedding-based retrieval for this query, even though it's arguably the
   best-worded match in the entire catalog. Between the two enhancements that *were*
   retrieved, `IR-4(12)` and `SI-7(7)`, the cross-encoder preferred `SI-7(7)` (by a 4.17-point
   margin) because its wording overlaps the query more directly on the surface
   ("...detection... into the... incident response capability" vs. `IR-4(12)`'s framing
   around post-incident forensic analysis of malicious code), even though `IR-4(12)` is the
   more correct control. Truncation was checked and ruled out first, not assumed away: both
   candidates were tokenized against the cross-encoder's real 512-token limit (130 and 151
   tokens respectively, with matching truncated/untruncated counts confirming neither was
   cut). *What we did:* nothing — this is left as a documented limitation rather than
   force-fixed. It's a specific, verified, human-checkable mechanism, not a guess that "the
   model might get it wrong."

## Supporting Question 3 — One thing to improve with another day

This section originally proposed adding a cross-encoder re-ranking step, framed as
something not yet done. It's been rewritten several times since, as systematic benchmarking
kept finding something more basic than ranking quality. Four findings, in the order they
were established:

**1. Root cause: truncation, not ranking.** Running `eval/nist_retrieval_eval.py` across
its full 10 hand-labeled cases originally scored **hit@1: 1/10, hit@3: 4/10** — far worse
than a ranking-quality problem would produce. Reading the actual retrieved text (not just
the control IDs) showed why: `all-MiniLM-L6-v2` silently truncates any input past 256
word-piece tokens, and several controls' combined control_text + discussion ran well past
that — `RA-5` alone is 600+ words. Those controls were being embedded from only their first
fraction of text, so the model never saw the part that actually matched the query. This was
an embedding-truncation bug, not a subtle ranking algorithm problem, and no amount of
re-ranking on top of it would have fixed it.

**2. The fix, proven model-agnostic.** The fix was: split a control's discussion into
multiple ~200-word chunks instead of force-concatenating it into one oversized chunk;
aggregate multi-chunk results back to one best chunk per control at query time; promote a
base control over its own enhancement when both are retrieved (e.g. preferring `SI-2` over
`SI-2(1)`); and backfill a weak result set from a control's own `related` list. This was
first tested paired with a model swap to `bge-small-en-v1.5` (512-token limit), scoring
**hit@1: 3/10, hit@3: 4/10** — an improvement, but a smaller one than expected, and one
previously-passing case (`SA-22`) regressed outright. To find out whether the model swap or
the chunking fix deserved the credit, the same chunking/promotion/backfill code was then
re-run on the *original* `all-MiniLM-L6-v2` model, isolating the one variable:

  | | hit@1 | hit@3 |
  |---|---|---|
  | Original baseline (MiniLM, no chunking fix) | 1/10 | 4/10 |
  | Word-based chunking fix + `bge-small-en-v1.5` | 3/10 | 4/10 |
  | Word-based chunking fix + `all-MiniLM-L6-v2` | 6/10 | 6/10 |
  | Token-based chunking fix + `all-MiniLM-L6-v2` | **5/10** | **5/10** |

  The chunking/promotion/backfill fix on its own — same model as the original baseline —
  outperformed the model swap on every axis. `SA-22`, the case that regressed under
  `bge-small-en-v1.5`, is a clean hit again under MiniLM. (The fourth row is finding 4
  below.)

**3. The model swap, tested and rejected.** `bge-small-en-v1.5` was not just unnecessary
here, it measurably underperformed plain MiniLM on this catalog once truncation was off the
table (row 3 vs. row 2 above). Rather than keep a bigger, slower model on the "bigger is
probably better" assumption, it was rejected and `all-MiniLM-L6-v2` was kept as a permanent
choice — see the comment on `EMBEDDING_MODEL_NAME` in `nist_ingest.py` for the same
conclusion in code.

**4. The same bug, a second time — and the mechanism fix.** The swap-and-revert left
`CHUNK_WORD_LIMIT = 200` behind: a word count, tuned by eye against `bge-small-en-v1.5`'s
512-token budget, still in place after reverting to a 256-token model. A direct tokenizer
check found **48 of 1,462 chunks still silently truncated** — the same bug as finding 1, at
smaller scale, caused the same way. Rather than re-tune the number, the mechanism was
replaced: chunk boundaries are now decided by counting real tokens with the model's own
tokenizer, against a budget read from the model itself (`max_seq_length`, less a small
safety margin) so a future model swap re-derives it instead of inheriting a stale constant.
Measuring rather than assuming also overturned a design assumption — control statements had
been exempt from splitting on the belief that they're always short, but 29 of them exceeded
the limit alone (`SA-12` at ~3,500 tokens was embedding about a fifteenth of itself), so
they're now packed the same way. Result: **0 of 1,567 chunks exceed the limit**, verified
by `eval/check_chunk_lengths.py`, which was itself converted from word counts to real
tokens — a word-count guard is what let this recur.

  The benchmark went **6/10 → 5/10** on that change, and the cause is worth stating plainly
  rather than hiding: `flat-network-lateral-movement` flipped from a hit to a miss because
  `AC-17::discussion::0` was *itself* one of the 48 truncated chunks. Embedding it in full
  made `AC-17` ("Remote Access") a genuinely better match for a query whose dominant term is
  "VPN" — it now beats `SC-7` ("Boundary Protection") at 0.428 vs. 0.580. The fix worked
  correctly and improved the losing control. That's a real trade-off, not a defect: one
  benchmark case was being carried by an accident of truncation.

**5. The remaining gap was narrower than it looked — two specific mechanisms, not "retrieval
is bad."** With chunking/truncation solved, the promotion rule turned out to be too narrow
(it only ever checked whether the single top-ranked result was an enhancement, so a family
that dominated the pool but sat buried behind its own siblings — as in
`flat-network-lateral-movement`, where `SC-7`'s base ranked behind 9 of its own
enhancements — was invisible to it), and some close-but-mis-ordered cases (`IA-5` sitting
just 0.021 behind rank 1 for `secrets-in-build-logs`) needed re-ranking precision a bi-encoder
distance alone can't provide. *What we did:* generalized promotion to check every control
family present in the pool, not just rank 0; and added a cross-encoder re-rank
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) between the dense retrieval step and aggregation
(see the vector-store table above for the exact pipeline order, which matters — promotion
has to run *after* re-ranking, or re-ranking would just undo it). Both fixes together moved
**hit@1 from 5/10 to 7/10 and hit@3 from 5/10 to 8/10, with zero regressions** on the 5 cases
that were already passing. One case, `ransomware-incident-response`, was investigated in
depth rather than force-fixed and left as a documented limitation — see Supporting Question 2,
finding 5.

**What's still open, honestly.** The current configuration scores **hit@1: 7/10, hit@3:
8/10** — 3 misses at hit@1, 2 at hit@3:

- `patch-available-not-applied` (SI-2) — still a miss on both. `build_query_text()` was
  fixed in an earlier round to state the patch situation in both directions (it used to say
  nothing at all when a patch exists but hasn't been applied — the single most common
  remediation state in this dataset). Tested directly against the real Fortinet query: `SI-2`
  moved from rank 98 to rank 65 of 1,567 — the right direction, nowhere near the top 3. The
  eval case's own hand-written description already said "vendor patch is available... not
  yet applied" before that fix, and cross-encoder re-ranking (this round) only reorders
  whatever the dense retrieval step already put in the pool — it can't rescue a control that
  never made the pool in the first place, and `SI-2` doesn't. That rules out "the query
  doesn't mention patching" as the explanation and points at `SI-2`'s own chunk text not
  being a strong semantic match for how these findings are phrased, unrelated to wording.
- `ransomware-incident-response` (IR-4) — still a miss on both, and investigated in full;
  see Supporting Question 2, finding 5, for the verified mechanism (not a guess).
- `flat-network-lateral-movement` (SC-7) — now a hit@3, still a miss at hit@1. Post-re-ranking,
  `SC-7`'s base is already the best-ranked member of its own family (rank 2 of the pool), so
  generalized promotion has nothing left to do here — it's two *unrelated* controls (`SI-20`,
  `AC-17`) that still out-score it, both plausibly favored by the query's dominant "VPN"
  framing over `SC-7`'s "boundary/segmentation" wording.

`shared-accounts` (IA-2) and `secrets-in-build-logs` (IA-5) — both misses as of the previous
round — are clean hits now; see finding 5 above and the `secrets-in-build-logs` example in
"Evaluating the retrieval lane."
=======
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> cee2a77ed0e01cf44f649f048d323a4e5280f71c
