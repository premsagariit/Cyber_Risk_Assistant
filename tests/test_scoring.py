"""Unit tests for the scoring engine.

The first test encodes the exact scenario the project brief warns about
directly: "a CVSS 10 on an internal dev server should rank lower than a
CVSS 8 on an internet-exposed payment gateway." These numbers are drawn
from two real rows in the data pack (the payment API IDOR and the
end-of-life Windows Server finding), traced by hand before being written
back into code here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import RiskFinding
from src.scoring import rank_top_findings, score_finding


def make_finding(**overrides) -> RiskFinding:
    defaults = dict(
        vuln_id="V-TEST",
        vulnerability_name="Test Vulnerability",
        cve="CVE-2024-00000",
        cvss=9.0,
        severity="Critical",
        exploit_available="No",
        patch_available="Yes",
        days_open=10,
        asset_exposure="Internal",
        asset_id="A-TEST",
        asset_name="test-asset",
        asset_type="Server",
        environment="Production",
        location="UAE",
        edr_installed="Yes",
        owner_team="Some Team",
        business_service="Test Service",
        business_impact="Test impact",
        revenue_impact="Medium",
        compliance_scope="",
        risk_appetite="Medium",
        rto_hours=24,
        threat_actor=None,
        campaign_name=None,
        exploit_maturity=None,
        ransomware_association=None,
        ti_confidence=None,
        ti_summary=None,
        in_kev=False,
        kev_ransomware_use=None,
    )
    defaults.update(overrides)
    return RiskFinding(**defaults)


def test_exposed_actively_exploited_finding_outranks_higher_cvss_internal_finding():
    payment_api_idor = make_finding(
        cve="CVE-SYN-2026-0010",
        cvss=9.1,
        asset_exposure="Internet",
        exploit_available="Yes",
        edr_installed="Yes",
        days_open=11,
        threat_actor="IronVeil",
        exploit_maturity="Active Exploitation",
        ransomware_association="Yes",
        business_service="Payment Processing",
        revenue_impact="Critical",
        compliance_scope="PCI DSS",
        risk_appetite="Very Low",
    )

    eol_windows_server = make_finding(
        cve="CVE-SYN-2024-0001",
        cvss=10.0,
        asset_exposure="Internal",
        exploit_available="No",
        patch_available="No",
        edr_installed="No",
        days_open=365,
        threat_actor="GhostByte",
        exploit_maturity="Commodity Exploit",
        ransomware_association="No",
        business_service="Financial Reporting",
        revenue_impact="High",
        compliance_scope="SOC 2, IFRS",
        risk_appetite="Very Low",
    )

    payment_score = score_finding(payment_api_idor)
    eol_score = score_finding(eol_windows_server)

    assert eol_windows_server.cvss > payment_api_idor.cvss, "sanity check on the test data itself"
    assert payment_score.total > eol_score.total, (
        "a lower-CVSS, internet-exposed, actively-exploited finding should still "
        "outrank a higher-CVSS internal one with no working exploit - that's the "
        "entire point of not ranking on CVSS alone"
    )


def test_missing_edr_increases_score():
    with_edr = score_finding(make_finding(edr_installed="Yes"))
    without_edr = score_finding(make_finding(edr_installed="No"))
    assert without_edr.total > with_edr.total


def test_ransomware_association_increases_score():
    without_ransomware = score_finding(make_finding(ransomware_association="No"))
    with_ransomware = score_finding(make_finding(ransomware_association="Yes"))
    assert with_ransomware.total > without_ransomware.total


def test_real_kev_match_adds_a_bonus_over_no_match():
    not_in_kev = score_finding(make_finding(in_kev=False))
    in_kev_no_ransomware = score_finding(make_finding(in_kev=True, kev_ransomware_use="Unknown"))
    in_kev_with_ransomware = score_finding(make_finding(in_kev=True, kev_ransomware_use="Known"))

    assert in_kev_no_ransomware.total > not_in_kev.total
    assert in_kev_with_ransomware.total > in_kev_no_ransomware.total


def test_rank_top_findings_deduplicates_same_issue_on_redundant_assets():
    # Same vulnerability, same business service, two different assets -
    # this mirrors the Fortinet VPN pair (A-1005 / A-1006) in the real data.
    node_one = make_finding(
        asset_id="A-1005",
        asset_name="vpn-edge-01",
        vulnerability_name="Fortinet SSL-VPN Heap Overflow",
        business_service="Remote Access",
        asset_exposure="Internet",
    )
    node_two = make_finding(
        asset_id="A-1006",
        asset_name="vpn-edge-02",
        vulnerability_name="Fortinet SSL-VPN Heap Overflow",
        business_service="Remote Access",
        asset_exposure="Internet",
        edr_installed="No",  # slightly different -> should score higher -> should be the one kept
    )
    unrelated_finding = make_finding(
        asset_id="A-2000",
        asset_name="unrelated-server",
        vulnerability_name="Some Other Bug",
        business_service="Some Other Service",
    )

    ranked = rank_top_findings([node_one, node_two, unrelated_finding], n=5)

    vpn_entries = [f for f in ranked if f.vulnerability_name == "Fortinet SSL-VPN Heap Overflow"]
    assert len(vpn_entries) == 1, "duplicate finding across redundant assets should be deduplicated"
    assert vpn_entries[0].asset_id == "A-1006", "the higher-scoring instance should be the one kept"
