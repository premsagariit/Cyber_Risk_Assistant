"""Regression test: does the retrieval lane's top pick for each of the 5
REAL findings avoid a confirmed wrong-domain match?

Confirmed regression this test guards against: after cross-encoder
re-ranking was added, the CitrixBleed session-token-leak findings (both
load balancers) started retrieving NIST control PE-19 (Information
Leakage) as their top match. PE-19's actual scope is electromagnetic
signal emanation (TEMPEST) - a physical-security control, unrelated to a
web session vulnerability. Diagnosed cause: the vulnerability's own name
("Citrix ADC Session Token Leak") shares the word "leak"/"leakage" with
PE-19's title, and the cross-encoder over-weighted that lexical overlap
against SC-23 (Session Authenticity) - the domain-correct family, already
ranked #1/#2 by raw embedding distance before re-ranking pushed it down.
Confirmed narrow, not a general cross-encoder problem: across the other 4
real findings and all 10 eval/nist_retrieval_eval.py cases, PE-19 never
even enters the top-30 raw candidate pool - only a query whose finding
name literally contains "Leak" triggers it. See README Supporting
Question 2 for the full diagnosis.

Why this test exists at all: these 5 real findings are not covered by
eval/nist_retrieval_eval.py's 10 hand-labeled cases - that benchmark
tracks retrieval quality on synthetic probes, and was never wired up to
check the app's own actual output. That's how this regression shipped
silently. This test runs the exact pipeline app.py renders in production
and checks it directly, instead of only a proxy benchmark.

Needs the real vector store (same one app.py builds/uses) and both local
models (embedding + cross-encoder) loaded - no network once they're
cached locally, but a genuinely first run needs it, the same tradeoff
eval/nist_retrieval_eval.py already makes. No API key needed - this only
exercises retrieval, never the LLM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import build_findings, load_all_data
from src.nist_ingest import get_or_build_collection
from src.retrieval import build_query_text, search_controls
from src.scoring import rank_top_findings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NIST_CSV_PATH = DATA_DIR / "NIST_SP-800-53_rev5_catalog_load.csv"
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

# Confirmed wrong-domain top matches, keyed by (asset_id, cve) - the same
# key shape as test_ranking_ground_truth.EXPECTED_TOP_5. Deliberately a
# small, explicit blocklist rather than a general "domain mismatch"
# classifier: a real, broader pattern would earn a real, general check, but
# writing one against a single evidenced case would be exactly the kind of
# speculative complexity this project has avoided elsewhere (see
# retrieval.py's own comments on not hardcoding query-to-control mappings -
# this is the same principle applied to what NOT to match, not just what
# to match). Add an entry here only once a specific wrong match has
# actually been confirmed the way these two were - not on a hunch.
KNOWN_BAD_TOP_MATCH = {
    ("A-1021", "CVE-2023-4966"): {"PE-19", "PE-19(1)"},
    ("A-1022", "CVE-2023-4966"): {"PE-19", "PE-19(1)"},
}


def _real_findings_and_top_controls():
    """Runs the exact pipeline app.py runs: real data -> ranked top 5 ->
    real retrieval against the real vector store. Returns a list of
    (finding, top_control) pairs."""
    data = load_all_data(str(DATA_DIR))
    collection = get_or_build_collection(str(CHROMA_PERSIST_DIR), str(NIST_CSV_PATH))
    top_findings = rank_top_findings(build_findings(data), n=5)

    results = []
    for finding in top_findings:
        query = build_query_text(finding)
        top_control = search_controls(collection, query, top_k=3)[0]
        results.append((finding, top_control))
    return results


def test_top_findings_avoid_confirmed_wrong_domain_matches():
    """Regression guard for the PE-19/CitrixBleed confusion - see module
    docstring for the full diagnosis. If this fails, don't just widen
    KNOWN_BAD_TOP_MATCH or patch around it here; re-diagnose the same way
    this one was (README Supporting Question 2) - a real fix belongs in
    retrieval.py, not in this test."""
    for finding, top_control in _real_findings_and_top_controls():
        known_bad = KNOWN_BAD_TOP_MATCH.get((finding.asset_id, finding.cve))
        if not known_bad:
            continue
        assert top_control["control_id"] not in known_bad, (
            f"{finding.vulnerability_name} on {finding.asset_name} retrieved "
            f"{top_control['control_id']} ({top_control['name']}) as its top match - "
            f"this is the confirmed wrong-domain regression this test guards against."
        )


def test_top_findings_return_a_real_control():
    """Cheaper sanity check across all 5, not just the two with a
    known-bad entry: retrieval should always return something with a real
    control_id and name, never an empty or malformed result. This doesn't
    assert correctness for the other 3 findings (no hand-verified ground
    truth exists for them, unlike the two above) - only that the pipeline
    didn't silently break."""
    for finding, top_control in _real_findings_and_top_controls():
        assert top_control.get("control_id"), f"No control_id returned for {finding.vulnerability_name}"
        assert top_control.get("name"), f"No control name returned for {finding.vulnerability_name}"


if __name__ == "__main__":
    # Quick way to see what the real pipeline currently retrieves for all 5,
    # e.g. after a retrieval.py or nist_ingest.py change:
    #   python tests/test_retrieval_real_findings.py
    for finding, top_control in _real_findings_and_top_controls():
        print(
            f"{finding.asset_id} {finding.cve:<20} {finding.vulnerability_name:<45} -> "
            f"{top_control['control_id']:<10} {top_control['name']}"
        )
