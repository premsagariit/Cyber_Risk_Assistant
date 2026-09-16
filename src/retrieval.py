"""Turns a scored finding into a search query and retrieves matching NIST
controls from the already-built vector store.

Deliberately not hardcoding "this CVE type -> that control ID" anywhere in
here, even though the assignment brief helpfully lists likely candidates
(SI-2, RA-5, IR-4, AC-2, SA-22). Those are useful as a sanity check that
retrieval is working, not as a shortcut around actually retrieving.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from .models import RiskFinding

# A control can now be represented by more than one chunk in Chroma (a
# "control" chunk plus one or more "discussion" chunks, since long
# discussions are split rather than truncated). A narrow n_results could
# come back as several chunks that are all the same control - undercounting
# how many distinct controls we actually found - or clip off a control's
# best chunk entirely. Querying wide and aggregating down to distinct
# controls afterwards avoids both failure modes.
CANDIDATE_POOL_SIZE = 20

# Chroma distance is "lower is better" (see app.py: "Relevance distance ...
# lower is more relevant", which the old single-chunk-per-control code also
# passed through unchanged). Above this value a top-ranked result is
# considered a poor semantic match and becomes eligible for related-control
# backfill. This is a starting point, not a measured constant - bge-small-
# en-v1.5's cosine-distance scale differs from the old MiniLM model's, so
# re-check this against real distances once the store is rebuilt and tune
# it here if it's off.
POOR_MATCH_DISTANCE_THRESHOLD = 0.8

# Traced cases (eval/trace_pipeline.py) showed embedding distance alone
# getting the right *family* into the pool but not ranking it well:
# secrets-in-build-logs had IA-5 sitting 0.021 behind rank 1, just outside
# top_k. A cross-encoder scores the query against each candidate's actual
# text directly, rather than comparing two independently-embedded vectors,
# which is a strictly more informed (if slower) judgment of relevance - see
# search_controls() for exactly where this fits in the pipeline relative to
# aggregation and promotion.
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_cross_encoder() -> CrossEncoder:
    """Loads the re-ranking model once. Same caching shape as
    nist_ingest._load_embedding_model() - an lru_cache'd zero-arg loader -
    since this module has the same problem: don't reload a model on every
    query."""
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


def build_query_text(finding: RiskFinding) -> str:
    """Builds a search query from the finding's own facts, so retrieval is
    grounded in what's actually wrong rather than a generic template."""
    missing = []
    if finding.edr_installed == "No":
        missing.append("no endpoint detection and response installed")
    if finding.patch_available == "No":
        missing.append("no vendor patch currently available")
    elif finding.patch_available == "Yes":
        # patch_available == "Yes" is the single most common situation in this
        # dataset (e.g. both real Fortinet findings, open 27 and 14 days) and
        # it used to produce no patch-related words in the query at all - the
        # old check only fired on "No". A query with nothing describing the
        # actual remediation state can't match a control about that state; see
        # README Supporting Question 2 for the retrieval miss this caused.
        # This states a fact already in the data, in both directions, the
        # same way the "No" branch above always has - it does not name or
        # imply any control ID.
        missing.append(f"a vendor patch is available but has not been applied, open {finding.days_open} days")
    if finding.days_open > 90:
        missing.append(f"unresolved for {finding.days_open} days")

    missing_text = ", ".join(missing) if missing else "no notable compensating control gaps"

    return (
        f"{finding.vulnerability_name} ({finding.cve}) affecting a {finding.asset_type}, "
        f"{finding.asset_exposure.lower()}-facing, severity {finding.severity}. "
        f"Missing controls: {missing_text}."
    )


def _base_control_id(control_id: str) -> str:
    """Strips a trailing "(n)" enhancement suffix, e.g. "AC-7(4)" -> "AC-7"."""
    paren = control_id.find("(")
    return control_id[:paren] if paren != -1 else control_id


def search_controls(collection, query_text: str, top_k: int) -> list[dict]:
    """Runs the vector search and turns raw chunk-level hits into top_k
    distinct control recommendations, best match first.

    Pipeline, in this order - the order matters, see the comments at each
    step for why:
      1. Dense embedding search retrieves a wide raw candidate pool (chunk
         level, since a control's text can span multiple chunks).
      2. A cross-encoder re-scores every (query, chunk text) pair directly
         and reorders the pool by that score - a more informed judgment than
         comparing two independently-embedded vectors, at the cost of being
         slower, which is why it runs on the ~20-chunk pool, not the whole
         catalog.
      3. Chunks are aggregated down to one best chunk per control_id, "best"
         now meaning highest cross-encoder score rather than lowest raw
         embedding distance.
      4. Base-vs-enhancement promotion runs on that re-ranked, aggregated
         order. This must come after re-ranking, not before: promoting
         first and re-ranking second would just let the re-rank silently
         undo the promotion by reordering everything again.
      5. Related-control backfill (unchanged) fills a weak result set.
    """
    results = collection.query(query_texts=[query_text], n_results=CANDIDATE_POOL_SIZE)

    # Step 1 output: one dict per raw chunk hit, chunk-level (not yet
    # aggregated to controls).
    ids = results["ids"][0]
    raw_chunks = [
        {
            "control_id": results["metadatas"][0][i]["control_id"],
            "name": results["metadatas"][0][i]["name"],
            "text": results["documents"][0][i],
            # The original embedding distance is kept, not discarded - it's
            # still what the poor-match backfill check (step 5) and the
            # UI's "relevance distance" caption (app.py) read. Only the
            # aggregation/ranking order below switches to the cross-encoder
            # score; a raw cross-encoder logit lives on a completely
            # different scale than POOR_MATCH_DISTANCE_THRESHOLD was tuned
            # against, so reusing that threshold against it would silently
            # miscalibrate the check rather than genuinely upgrade it.
            "distance": results["distances"][0][i],
            "is_enhancement": results["metadatas"][0][i]["is_enhancement"],
            "related": results["metadatas"][0][i]["related"],
        }
        for i in range(len(ids))
    ]

    # Step 2: cross-encoder re-ranking. Scores the query against each
    # candidate's actual text, rather than the distance between two vectors
    # embedded independently of each other - catches cases like
    # secrets-in-build-logs, where the correct control (IA-5) sat in the
    # pool only 0.021 behind rank 1 by embedding distance: too close for any
    # structural rule to fix, exactly what re-ranking is for.
    cross_encoder = _load_cross_encoder()
    pairs = [(query_text, chunk["text"]) for chunk in raw_chunks]
    ce_scores = cross_encoder.predict(pairs) if pairs else []
    for chunk, score in zip(raw_chunks, ce_scores):
        chunk["ce_score"] = float(score)  # higher = more relevant

    # Step 3: aggregate chunk-level hits into one best hit per control_id,
    # "best" meaning highest cross-encoder score now, not lowest distance.
    best_by_control: dict[str, dict] = {}
    for chunk in raw_chunks:
        control_id = chunk["control_id"]
        if control_id not in best_by_control or chunk["ce_score"] > best_by_control[control_id]["ce_score"]:
            best_by_control[control_id] = chunk

    candidates = sorted(best_by_control.values(), key=lambda c: c["ce_score"], reverse=True)

    # Step 4: base-vs-enhancement promotion, generalized across every family
    # in the pool, not just whichever control happens to rank first.
    #
    # The original version of this rule only checked rank 0: if the top
    # result was an enhancement, promote its base. That missed cases where a
    # family dominates the pool but an *unrelated* control still edges out
    # all of them for rank 0 - flat-network-lateral-movement is the traced
    # example: AC-17 wins rank 1, then SC-7's own enhancements occupy 5 of
    # the next 6 slots (SC-7(7), SC-7(13), SC-7(25), SC-7(21), SC-7(27)),
    # with the SC-7 base itself buried at rank 10, behind 9 of its own
    # enhancements. Rank 0 being AC-17 (not an SC-7 enhancement) meant the
    # old check never even looked at the SC-7 family.
    #
    # This version checks every distinct base control_id present in the
    # pool: if that base is itself a candidate, and any of its own
    # enhancements ranks better than the base does, the base is moved up to
    # that best rank - not just to rank 0. Done as a fixed-point loop
    # (recomputing positions and restarting after each move) because
    # promoting one family can shift another family's positions relative to
    # each other; with at most CANDIDATE_POOL_SIZE candidates this converges
    # in a handful of passes.
    base_ids_present = sorted({_base_control_id(c["control_id"]) for c in candidates})
    moved = True
    while moved:
        moved = False
        for base_id in base_ids_present:
            base_index = next((i for i, c in enumerate(candidates) if c["control_id"] == base_id), None)
            if base_index is None:
                continue  # this base control was never itself retrieved - nothing to promote to
            family_indices = [i for i, c in enumerate(candidates) if _base_control_id(c["control_id"]) == base_id]
            best_family_index = min(family_indices)
            if best_family_index < base_index:
                candidates.insert(best_family_index, candidates.pop(base_index))
                moved = True
                break  # positions shifted - restart the scan rather than continue on stale indices

    selected = candidates[:top_k]

    if selected:
        # Related-backfill is a precision-preserving fallback, not a replacement for genuinely
        # strong matches: it only ever widens a weak result set (missing slots, or a low-confidence
        # trailing match), and only pulls in controls the top hit itself lists as related that are
        # already present in the candidate pool - never an arbitrary unrelated control just to pad
        # the list out to top_k.
        top_related_ids = [rid.strip() for rid in selected[0]["related"].split(",") if rid.strip()]
        selected_ids = {c["control_id"] for c in selected}
        related_pool = sorted(
            (
                best_by_control[rid]
                for rid in top_related_ids
                if rid in best_by_control and rid not in selected_ids
            ),
            key=lambda c: c["distance"],
        )

        if len(selected) < top_k:
            # Trigger (a): fewer than top_k distinct controls were found in the aggregated
            # candidates at all - fill the empty slot(s) with related controls, best first.
            while len(selected) < top_k and related_pool:
                selected.append(related_pool.pop(0))
        elif selected[-1]["distance"] > POOR_MATCH_DISTANCE_THRESHOLD and related_pool:
            # Trigger (b): slots are full but the last one is a poor match - swap it for the best
            # available related candidate, since a control related to the top hit is a more
            # informed fallback than an unrelated, low-confidence chunk match.
            selected[-1] = related_pool[0]

    return [
        {
            "control_id": c["control_id"],
            "name": c["name"],
            "text": c["text"],
            "distance": c["distance"],
        }
        for c in selected
    ]


def retrieve_controls(collection, finding: RiskFinding, top_k: int = 3) -> list[dict]:
    """Returns the top_k most relevant NIST controls for this finding, best
    match first. Each result includes the real control text - nothing here
    is generated, only retrieved."""
    query_text = build_query_text(finding)
    return search_controls(collection, query_text, top_k)
