"""One-time ingestion of the NIST SP 800-53 control catalog into ChromaDB.

This is the RAG lane. The control catalog has 1,000+ rows with long prose
descriptions - too much to hand an LLM directly, and "which control fits
this finding" is a semantic question rather than an exact lookup. So instead
of hardcoding a CVE-to-control mapping, we embed every control's real text
and search it at query time (see retrieval.py).

Embeddings run locally via sentence-transformers - no external API, no
rate limit, no extra account to sign up for. Retrieval quality here was
originally broken by silent truncation (see EMBEDDING_MODEL_NAME below and
the README's Supporting Question 3), not by the choice of model - splitting
long discussions into their own chunks instead of truncating them fixed
nearly all of it, on the same all-MiniLM-L6-v2 model this project started
with. A larger-context model (bge-small-en-v1.5) was tried once chunking was
fixed and measurably underperformed MiniLM on this catalog, so it was
rejected rather than kept as a "bigger model" default.
"""

import re
from functools import lru_cache
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "nist_800_53_controls"
# bge-small-en-v1.5 was tested and measurably underperformed MiniLM on this
# catalog once chunking/truncation was fixed (hit@3 40% vs 60% in a
# controlled isolation test - same chunking, promotion, and backfill code,
# only the model differed). Kept as a permanent decision, not a fallback:
# the original truncation bug was caused by chunking, not by MiniLM's
# smaller 256-token limit, so there was no real problem for a bigger model
# to solve here.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunk size is measured in real tokens from the model's own tokenizer rather
# than in words, because the two diverge and the gap isn't constant: NIST's
# jargon-heavy compliance prose splits into far more word-pieces than it has
# words, and a word threshold tuned against one model's tokenizer doesn't
# transfer to another's. That mismatch has now caused this pipeline's chunk
# truncation bug twice - once at 1 chunk per control, then again when a
# 200-word limit tuned for a 512-token model stayed in place after reverting
# to a 256-token one. Counting what the model actually counts removes the
# class of bug rather than re-tuning the number.
#
# Held back from the model's real limit so one longer-than-average sentence
# can't tip an otherwise-fitting chunk over it.
TOKEN_SAFETY_MARGIN = 16

# A control enhancement identifier appends a parenthesized number to its
# base control id (e.g. "AC-2(1)"); a base control (e.g. "AC-2") never has
# a trailing "(n)". Anchored to the end of the string so it doesn't
# misfire on ids that simply contain digits elsewhere.
_ENHANCEMENT_RE = re.compile(r"\(\d+\)$")

# Splits a discussion into sentences. Used only as a last-resort fallback
# when a discussion has no paragraph or lettered sub-point structure to
# split on, so at least a chunk boundary never lands mid-sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Matches a lettered NIST sub-point marker like "(a)" or "(b)" only when it
# stands alone as a list marker (whitespace or string-start before it,
# whitespace or string-end after). This deliberately excludes things like
# "AC-2(1)", where the parenthesized number is glued directly onto a
# control id with no surrounding space - that's a cross-reference, not a
# sub-point boundary, and splitting on it would fragment the text wrong.
_SUBPOINT_SPLIT_RE = re.compile(r"(?=(?:^|\s)\([a-z]\)(?:\s|$))")


def load_nist_catalog(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["identifier"])
    return df.fillna("")


def _parse_related(related_raw: str) -> str:
    """Cleans the CSV's `related` column into a comma-joined string of
    control ids.

    The catalog uses the literal string "[None]" as a placeholder for "this
    control has no related controls" rather than leaving the cell blank,
    and populated cells end with a trailing period (e.g. "AU-2, AU-6.").
    Both are data-format quirks of the source CSV, not real control ids, so
    they're stripped before we treat the cell as a plain list.
    """
    related_raw = related_raw.strip()
    if not related_raw or related_raw == "[None]":
        return ""
    related_raw = related_raw.rstrip(".")
    related_ids = [r.strip() for r in related_raw.split(",")]
    related_ids = [r for r in related_ids if r and r != "[None]"]
    return ", ".join(related_ids)


@lru_cache(maxsize=1)
def _load_embedding_model() -> SentenceTransformer:
    """Loads the embedding model once, for its tokenizer and its real maximum
    sequence length.

    get_or_build_collection() embeds through Chroma's SentenceTransformer
    wrapper, but that wrapper exposes neither the tokenizer nor the sequence
    limit - so chunking loads the same model directly instead of guessing at
    what it will accept.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def count_tokens(text: str) -> int:
    """The real word-piece token count the model will see, special tokens
    included - i.e. the exact number it truncates on."""
    return len(_load_embedding_model().tokenizer.encode(text))


def chunk_token_budget() -> int:
    """The model's own maximum sequence length, less a safety margin.

    Read from the loaded model rather than hardcoded, so changing
    EMBEDDING_MODEL_NAME re-derives the budget instead of silently inheriting
    the previous model's - which is how the last truncation bug survived a
    model swap.
    """
    return _load_embedding_model().max_seq_length - TOKEN_SAFETY_MARGIN


def _split_into_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _split_oversized_sentence(sentence: str, max_tokens: int) -> list[str]:
    """Last-resort split for a single sentence that busts the budget on its own
    - NIST occasionally runs a whole semicolon-separated list into one, and
    there's no smaller natural boundary left to respect.

    Cuts between words because that's the smallest boundary that doesn't leave
    mangled sub-word fragments; the word boundary is only *where* a cut may
    fall, never an estimate of how much fits. Every candidate is still measured
    with the real tokenizer.
    """
    pieces = []
    current: list[str] = []
    for word in sentence.split():
        candidate = " ".join(current + [word])
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_into_segments(text: str, max_tokens: int) -> list[str]:
    """Breaks text into its natural structural units - paragraphs, then
    lettered sub-points within a paragraph - before ever falling back to
    blind sentence splitting. Respecting these boundaries first means a
    chunk practically never cuts a control's sub-point in half, which
    plain fixed-length slicing would do all the time.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    segments = []
    for paragraph in paragraphs:
        sub_points = _SUBPOINT_SPLIT_RE.split(paragraph)
        sub_points = [s.strip() for s in sub_points if s.strip()]
        segments.extend(sub_points if len(sub_points) > 1 else [paragraph])

    # A segment with no internal structure (one big paragraph, no lettered
    # sub-points) can still be too long on its own - sentence-split those, then
    # word-split any single sentence that's still oversized, so the grouping
    # step below always has small enough pieces to pack.
    fine_grained = []
    for segment in segments:
        if count_tokens(segment) <= max_tokens:
            fine_grained.append(segment)
            continue
        for sentence in _split_into_sentences(segment):
            if count_tokens(sentence) <= max_tokens:
                fine_grained.append(sentence)
            else:
                fine_grained.extend(_split_oversized_sentence(sentence, max_tokens))
    return fine_grained


def _group_segments(segments: list[str], prefix: str, max_tokens: int) -> list[str]:
    """Greedily packs consecutive segments into the largest groups that still
    fit max_tokens once prefix is attached, so a control with many short
    sub-points doesn't end up as a dozen tiny, barely-distinct chunks instead
    of a handful of substantial ones.

    Measures the assembled candidate string on every step rather than summing
    per-segment counts: tokenization isn't additive (each encode carries the
    model's special tokens, and word-pieces can merge across a join), so a
    running total would drift away from what the model actually sees.
    """
    groups = []
    current: list[str] = []
    for segment in segments:
        candidate = " ".join(current + [segment])
        if current and count_tokens(prefix + candidate) > max_tokens:
            groups.append(" ".join(current))
            current = [segment]
        else:
            current.append(segment)
    if current:
        groups.append(" ".join(current))
    return groups


def _build_chunk_texts(prefix: str, body: str, max_tokens: int) -> list[str]:
    """Packs body into as few `prefix + body` strings as will fit max_tokens.

    The prefix (control id, name, and the Control:/Discussion: label) repeats
    on every piece so a fragment still reads as being about the right control
    on its own - a bare "(b) reviews organization-defined frequency" means
    very little once separated from the control it belongs to.
    """
    segments = _split_into_segments(body, max_tokens - count_tokens(prefix))
    return [prefix + group for group in _group_segments(segments, prefix, max_tokens)]


def build_control_chunks(catalog: pd.DataFrame) -> list[dict]:
    """Builds the embedded chunks for one control (or control enhancement,
    e.g. AC-2(1)).

    A control short enough to fit the model's context in one piece stays one
    chunk - splitting a three-sentence discussion off on its own would just
    add retrieval noise for no benefit. Past that, the control statement and
    the discussion are chunked separately, each packed to fill the token
    budget without crossing it.

    Control statements were originally never split, on the assumption that
    they're always short. Measuring rather than assuming showed 29 of them
    exceed the model's limit on their own - SA-12's runs to ~3,500 tokens,
    which meant embedding roughly a fifteenth of it - so they're packed the
    same way discussions are, with the id and name repeated on every piece to
    keep each one findable.
    """
    budget = chunk_token_budget()
    chunks = []
    for _, row in catalog.iterrows():
        control_id = str(row["identifier"]).strip()
        name = str(row["name"]).strip()
        control_text = str(row["control_text"]).strip()
        discussion = str(row["discussion"]).strip()
        related = _parse_related(str(row["related"]))
        is_enhancement = bool(_ENHANCEMENT_RE.search(control_id))

        base_metadata = {
            "control_id": control_id,
            "name": name,
            "is_enhancement": is_enhancement,
            "related": related,
        }

        control_prefix = f"{control_id} - {name}\n\nControl: "
        combined_text = control_prefix + control_text
        if discussion:
            combined_text += f"\n\nDiscussion: {discussion}"

        if count_tokens(combined_text) <= budget:
            chunks.append(
                {
                    "id": control_id,
                    "text": combined_text,
                    "metadata": {**base_metadata, "chunk_type": "control"},
                }
            )
            continue

        control_texts = _build_chunk_texts(control_prefix, control_text, budget)
        for i, text in enumerate(control_texts):
            # The first piece keeps the bare control id so the common case (a
            # control that fits in one chunk) has a stable, predictable id.
            chunks.append(
                {
                    "id": control_id if i == 0 else f"{control_id}::control::{i}",
                    "text": text,
                    "metadata": {**base_metadata, "chunk_type": "control"},
                }
            )

        discussion_prefix = f"{control_id} - {name}\n\nDiscussion: "
        discussion_texts = (
            _build_chunk_texts(discussion_prefix, discussion, budget) if discussion else []
        )
        for i, text in enumerate(discussion_texts):
            chunks.append(
                {
                    "id": f"{control_id}::discussion::{i}",
                    "text": text,
                    "metadata": {**base_metadata, "chunk_type": "discussion"},
                }
            )
    return chunks


def get_or_build_collection(persist_directory: str, nist_csv_path: str):
    """Returns the Chroma collection of embedded NIST controls, building it
    on first run and reusing the persisted version on every run after that.

    Re-embedding ~1,000 controls takes a minute or two - fine once, annoying
    on every page reload - so we check whether the collection is already
    populated before doing any of that work.
    """
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_directory)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    if collection.count() > 0:
        return collection

    catalog = load_nist_catalog(nist_csv_path)
    chunks = build_control_chunks(catalog)

    # Chroma wants ids/documents/metadatas as separate parallel lists rather
    # than a list of dicts, and it's happier loading in batches than one
    # giant call for ~1,000 rows.
    batch_size = 200
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    return collection
