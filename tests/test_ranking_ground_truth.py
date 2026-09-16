"""Ground truth test: does the full pipeline (load CSVs -> join -> score ->
rank) reproduce a known-correct top 5 against the REAL data pack?

This is different from test_scoring.py, which tests the scoring formula in
isolation on made-up finding objects. This test runs everything - data
loading, all four joins, deduplication, scoring, ranking - against the
actual data/*.csv files, end to end. It's the closest thing this project
has to "does the whole system work," and it's the first thing to run after
any change to data_loader.py or scoring.py.

EXPECTED_TOP_5 below was generated once by running the real pipeline, then
verified by hand against the raw CSVs (asset exposure, threat intel match,
business service criticality) before being locked in here as ground truth.

If this test starts failing after a change:
  (a) it's probably a bug you just introduced, or
  (b) it's a deliberate, verified improvement to the scoring formula - in
      which case re-derive EXPECTED_TOP_5 (see the __main__ block below for
      a quick way to print the current output) and update it, with a note
      in your commit message about what changed and why.
Either way, don't just delete the assertion because it's inconvenient.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import build_findings, load_all_data
from src.scoring import rank_top_findings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# (asset_id, cve, expected_score) - each row hand-verified against the raw
# CSVs before being locked in:
#   1. CitrixBleed on both load balancers (A-1021 / A-1022): internet-facing,
#      Active Exploitation + ransomware per threat intel, confirmed in the
#      real CISA KEV catalog with known ransomware use, feeds two Critical/
#      Very-Low-risk-appetite business services.
#   2. Fortinet VPN chain (A-1005), both CVEs: internet-facing, Weaponized +
#      ransomware, also KEV-confirmed, feeds Remote Access (High impact).
#   3. Payment API IDOR (A-1003): internet-facing, Active Exploitation +
#      ransomware per threat intel, feeds Payment Processing (Critical,
#      Very Low risk appetite, PCI DSS) - not in the real KEV catalog since
#      its CVE ID is one of this exercise's synthetic identifiers.
EXPECTED_TOP_5 = [
    ("A-1021", "CVE-2023-4966", 140),
    ("A-1022", "CVE-2023-4966", 140),
    ("A-1005", "CVE-2024-21762", 125),
    ("A-1005", "CVE-2024-55591", 125),
    ("A-1003", "CVE-SYN-2026-0010", 115),
]


def _run_pipeline():
    data = load_all_data(str(DATA_DIR))
    findings = build_findings(data)
    return rank_top_findings(findings, n=5)


def test_top_5_matches_expected_findings_and_order():
    top5 = _run_pipeline()
    actual = [(f.asset_id, f.cve, f.score.total) for f in top5]
    assert actual == EXPECTED_TOP_5, (
        "Top 5 no longer matches ground truth. Run this file directly "
        "(`python tests/test_ranking_ground_truth.py`) to see the current "
        "output and compare against EXPECTED_TOP_5 above."
    )


def test_top_5_is_internet_exposed_or_business_critical():
    """A cheaper sanity check than exact-match: whatever the top 5 are,
    every single one should be internet-facing OR tied to a Critical/Very
    Low risk appetite business service. If a purely internal, low-stakes
    finding ever makes the top 5, something in the formula has gone wrong -
    this is the assignment's core requirement, checked directly."""
    top5 = _run_pipeline()
    for f in top5:
        is_exposed = f.asset_exposure == "Internet"
        is_business_critical = f.revenue_impact == "Critical" or f.risk_appetite == "Very Low"
        assert is_exposed or is_business_critical, (
            f"{f.asset_name} / {f.vulnerability_name} made the top 5 despite "
            f"being internal AND business-low-stakes - that shouldn't happen."
        )


def test_top_5_are_distinct_business_risks():
    """Confirms dedup worked: no two entries in the top 5 should be the
    identical (business_service, vulnerability_name) pair."""
    top5 = _run_pipeline()
    keys = [(f.business_service, f.vulnerability_name) for f in top5]
    assert len(keys) == len(set(keys)), "Top 5 contains a duplicate (business service, vulnerability) pair"


if __name__ == "__main__":
    # Quick way to see the current pipeline output without pytest, e.g.
    # after intentionally changing the scoring formula:
    #   python tests/test_ranking_ground_truth.py
    for f in _run_pipeline():
        print(f'("{f.asset_id}", "{f.cve}", {f.score.total}),  # {f.asset_name}, {f.vulnerability_name}')