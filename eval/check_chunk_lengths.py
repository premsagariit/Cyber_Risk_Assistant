"""Diagnostic: are any NIST control chunks long enough to get silently
truncated by the embedding model, the way they used to be?

This is a standalone script, same reasoning as eval/nist_retrieval_eval.py -
it needs to import the real ingestion code and walk the real catalog, which
is more setup than belongs in a fast pytest run, and it's a one-off check you
run after touching chunking or the embedding model, not on every save.

Why this exists: the retrieval benchmark once scored hit@1: 1/10, hit@3: 4/10.
Root cause, confirmed by reading the actual data - `all-MiniLM-L6-v2` silently
truncates any input past 256 word-piece tokens, and every control was being
concatenated into a single chunk regardless of length, so most of a long
control's text never got embedded at all. The fix (in src/nist_ingest.py) was
to pack a control's text into as many chunks as it takes to stay inside the
model's real token limit. This script re-checks that the fix holds: it imports
and calls the real `build_control_chunks()` rather than reimplementing chunking
logic, so it reports on the chunks that will actually get embedded.

This script counts the same way the chunker does - with the model's own
tokenizer, via `nist_ingest.count_tokens()`. An earlier version of it counted
words as a proxy for tokens, which is precisely the mistake it was written to
catch: word count and token count diverge (NIST's jargon-heavy prose splits
into far more word-pieces than it has words), so a word-count check reports
"all clear" on chunks the model is quietly cutting in half.

    python eval/check_chunk_lengths.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nist_ingest import (
    build_control_chunks,
    chunk_token_budget,
    count_tokens,
    load_nist_catalog,
    _load_embedding_model,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NIST_CSV_PATH = DATA_DIR / "NIST_SP-800-53_rev5_catalog_load.csv"

# How many of the longest chunks/controls to print, so the output stays
# short enough to paste directly into a README or PR description.
TOP_N_TO_SHOW = 8


def raw_control_token_counts(catalog) -> list[dict]:
    """Token count per control's full control_text + discussion, BEFORE any
    chunking. This is the number that made the original bug obvious - it's how
    much text a one-chunk-per-control scheme was asking the model to swallow -
    and it's kept here purely as before/after context.
    """
    rows = []
    for _, row in catalog.iterrows():
        control_id = str(row["identifier"]).strip()
        combined = f"{row['control_text']}\n\n{row['discussion']}"
        rows.append({"control_id": control_id, "tokens": count_tokens(combined)})
    return rows


def chunk_token_counts(chunks: list[dict]) -> list[dict]:
    """Token count per REAL chunk, i.e. the exact strings build_control_chunks()
    hands to the embedding function - what the model actually truncates on."""
    rows = []
    for chunk in chunks:
        rows.append(
            {
                "chunk_id": chunk["id"],
                "control_id": chunk["metadata"]["control_id"],
                "tokens": count_tokens(chunk["text"]),
            }
        )
    return rows


def print_top(rows: list[dict], label: str, n: int = TOP_N_TO_SHOW) -> None:
    longest = sorted(rows, key=lambda r: r["tokens"], reverse=True)[:n]
    print(f"\nLongest {n} {label} (by token count):")
    for row in longest:
        print(f"  {row.get('chunk_id', row['control_id']):<26} {row['tokens']:>5} tokens")


def main() -> None:
    model = _load_embedding_model()
    hard_limit = model.max_seq_length
    budget = chunk_token_budget()
    print(f"Embedding model: {model.__class__.__name__} - hard limit {hard_limit} tokens")
    print(f"Chunker's budget: {budget} tokens (hard limit less a safety margin)\n")

    print("Loading NIST catalog...")
    catalog = load_nist_catalog(str(NIST_CSV_PATH))
    print(f"{len(catalog)} controls loaded.\n")

    print("=" * 70)
    print("BEFORE CHUNKING: raw per-control text length (context, not the fix)")
    print("=" * 70)
    raw_rows = raw_control_token_counts(catalog)
    would_truncate = [r for r in raw_rows if r["tokens"] > hard_limit]
    print(
        f"\nControls whose full text alone exceeds {hard_limit} tokens: "
        f"{len(would_truncate)} of {len(raw_rows)}"
    )
    print("(i.e. how many would be silently cut short without chunking at all)")
    print_top(raw_rows, "controls by raw control_text + discussion")

    print()
    print("=" * 70)
    print("AFTER CHUNKING: real chunks from build_control_chunks()")
    print("=" * 70)
    chunks = build_control_chunks(catalog)
    chunk_rows = chunk_token_counts(chunks)

    truncated = [r for r in chunk_rows if r["tokens"] > hard_limit]

    print(f"\nTotal chunks: {len(chunk_rows)}")
    print(f"Chunks over the model's real {hard_limit}-token limit: {len(truncated)}")
    if truncated:
        print("TRUNCATED - these lose text at embedding time:")
        for row in sorted(truncated, key=lambda r: r["tokens"], reverse=True):
            print(f"  {row['chunk_id']:<26} {row['tokens']:>5} tokens")
    else:
        print("None - every chunk fits inside the model's limit in full.")

    print_top(chunk_rows, "chunks")


if __name__ == "__main__":
    main()
