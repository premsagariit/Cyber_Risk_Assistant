"""Grounding checks for the LLM synthesis step.

Ranking and retrieval have objective right answers, so they get real
ground-truth tests (see test_ranking_ground_truth.py and
eval/nist_retrieval_eval.py). The LLM's prose doesn't have one "correct"
wording, so instead of checking exact output, these tests check that the
output stays anchored to the facts it was given - the actual failure mode
worth catching here is the model drifting into invented specifics rather
than paraphrasing what it was handed.

Requires a real LLM_API_KEY - these are skipped automatically without one,
since they make real API calls and aren't meant to run in an offline CI
environment. The key normally lives in .env, which only app.py loaded - these
tests read os.environ directly, so they skipped on every run regardless of
whether a key was configured. load_dotenv() below is what makes that key
visible here too.
"""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from src.models import RiskFinding
from src.scoring import score_finding
from src.synthesis import generate_explanation

requires_api_key = pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"),
    reason="LLM_API_KEY not set - grounding tests need a live LLM call",
)

# The model writes with typographic punctuation - U+2011 non-breaking hyphen
# in "SA-22", U+2013/2014 dashes, U+202F narrow no-break space - not the plain
# ASCII this file's literals use. A raw `"SI-2" in explanation` check fails on
# correct output that merely spells the hyphen differently (confirmed via
# eval/trace_pipeline.py: "SA‑22" and "RA‑5" in real traced
# responses). Normalizing both sides to ASCII before comparing tests what the
# checks actually care about - which characters were mentioned - without
# caring how the model typeset them.
_DASH_CHARS = "‐‑–—"
_SPACE_CHARS = "  "


def _normalize(text: str) -> str:
    for dash in _DASH_CHARS:
        text = text.replace(dash, "-")
    for space in _SPACE_CHARS:
        text = text.replace(space, " ")
    return text


def _sample_finding() -> RiskFinding:
    finding = RiskFinding(
        vuln_id="V-TEST",
        vulnerability_name="Fortinet SSL-VPN Heap Buffer Overflow RCE",
        cve="CVE-2024-21762",
        cvss=9.8,
        severity="Critical",
        exploit_available="Yes",
        patch_available="Yes",
        days_open=27,
        asset_exposure="Internet",
        asset_id="A-1005",
        asset_name="vpn-edge-01",
        asset_type="VPN Gateway",
        environment="Production",
        location="UAE",
        edr_installed="No",
        owner_team="Network Team",
        business_service="Remote Access",
        business_impact="Remote employees and administrators lose secure network access",
        revenue_impact="High",
        compliance_scope="ISO 27001",
        risk_appetite="Low",
        rto_hours=2,
        threat_actor="CrimsonJackal",
        campaign_name="Gateway Breaker",
        exploit_maturity="Weaponized",
        ransomware_association="Yes",
        ti_confidence="High",
    )
    score_finding(finding)
    return finding


@requires_api_key
def test_explanation_mentions_the_actual_cve():
    finding = _sample_finding()
    control = {"control_id": "SI-2", "name": "Flaw Remediation", "text": "sample control text", "distance": 0.1}

    explanation = _normalize(generate_explanation(finding, control))

    assert _normalize(finding.cve) in explanation, "Explanation should reference the actual CVE, not a generic description"


@requires_api_key
def test_explanation_cites_the_retrieved_control_not_a_different_one():
    finding = _sample_finding()
    control = {"control_id": "SI-2", "name": "Flaw Remediation", "text": "sample control text", "distance": 0.1}

    explanation = _normalize(generate_explanation(finding, control))

    assert "SI-2" in explanation, "Explanation should cite the control it was actually given"
    # A cheap check against the model inventing a plausible-sounding but
    # different control ID instead of the one it was handed.
    other_common_controls = ["AC-2", "IR-4", "RA-5", "SA-22", "IA-2"]
    invented = [c for c in other_common_controls if _normalize(c) in explanation and c != "SI-2"]
    assert not invented, f"Explanation mentions control(s) it was never given: {invented}"


@requires_api_key
def test_explanation_does_not_invent_a_ransomware_group_that_was_not_matched():
    """Uses a finding with NO threat intel match, to check the model doesn't
    fill that gap with a plausible-sounding but fictional campaign name."""
    finding = _sample_finding()
    finding.threat_actor = None
    finding.campaign_name = None
    finding.exploit_maturity = None
    finding.ransomware_association = None
    score_finding(finding)

    control = {"control_id": "SI-2", "name": "Flaw Remediation", "text": "sample control text", "distance": 0.1}
    explanation = _normalize(generate_explanation(finding, control))

    # None of the real campaign names in this data pack should appear if we
    # told the model there was no match.
    known_campaign_names = [
        "Gateway Breaker", "Collaboration Breach", "Build Chain Theft",
        "CitrixBleed Exploitation", "API Gateway Takeover",
    ]
    invented = [name for name in known_campaign_names if _normalize(name) in explanation]
    assert not invented, f"Explanation invented a campaign match that wasn't given: {invented}"