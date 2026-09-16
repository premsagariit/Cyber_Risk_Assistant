"""Turns the analyst notes' stated priority order into an actual number.

The MDR advisory says to weight, in order: internet exposure, active
exploitation, ransomware association, business criticality, then missing
compensating controls. This module encodes exactly that ordering as
descending point values, so the formula is a direct, checkable translation
of the stated methodology rather than something invented from scratch.

Deliberately no LLM anywhere in this file. The ranking needs to be
deterministic and reproducible - the same input data should always produce
the same top 5, and that's much easier to defend in a README than "the
model decided."
"""

from .models import RiskFinding, ScoreBreakdown

REVENUE_IMPACT_POINTS = {"Critical": 15, "High": 10, "Medium": 5, "Low": 0}


def score_finding(finding: RiskFinding) -> ScoreBreakdown:
    """Computes and attaches a ScoreBreakdown to the given finding, and
    returns that same breakdown for convenience."""

    # 1. Internet exposure - heaviest weight. An internet-facing system can
    # be reached by literally anyone; an internal one requires the attacker
    # to already have a foothold inside the network first.
    exposure_points = 40 if finding.asset_exposure == "Internet" else 10

    # 2. Active exploitation. A campaign actively using this flaw right now
    # (or a ready-to-use "weaponized" exploit) matters far more than a
    # theoretical exploit that merely exists.
    if finding.exploit_maturity in ("Weaponized", "Active Exploitation"):
        exploitation_points = 30
    elif finding.exploit_available == "Yes":
        exploitation_points = 15
    else:
        exploitation_points = 0

    # 3. Ransomware association - the assignment calls this out specifically
    # as causing the most organizational disruption of anything on the list.
    ransomware_points = 20 if finding.ransomware_association == "Yes" else 0

    # 4. Business criticality & compliance scope.
    business_points = REVENUE_IMPACT_POINTS.get(finding.revenue_impact, 0)
    if finding.compliance_scope.strip():
        business_points += 5
    if finding.risk_appetite == "Very Low":
        business_points += 5

    # 5. Missing compensating controls.
    missing_controls_points = 0
    if finding.edr_installed == "No":
        missing_controls_points += 10
    if finding.days_open > 90:
        missing_controls_points += 5
    if not finding.owner_team.strip():
        missing_controls_points += 5

    # Bonus: independent, real-world confirmation via the actual CISA KEV
    # catalog (not the synthetic threat intel file). This only fires for
    # real CVE IDs, since the synthetic ones were invented for this exercise
    # and were never going to be in a real government catalog.
    if finding.in_kev and finding.kev_ransomware_use == "Known":
        kev_bonus_points = 10
    elif finding.in_kev:
        kev_bonus_points = 5
    else:
        kev_bonus_points = 0

    breakdown = ScoreBreakdown(
        exposure_points=exposure_points,
        exploitation_points=exploitation_points,
        ransomware_points=ransomware_points,
        business_points=business_points,
        missing_controls_points=missing_controls_points,
        kev_bonus_points=kev_bonus_points,
    )
    finding.score = breakdown
    return breakdown


def rank_top_findings(findings: list[RiskFinding], n: int = 5) -> list[RiskFinding]:
    """Scores every finding and returns the top n, one per distinct
    (business service, vulnerability) pair.

    Several findings in this data pack are duplicated across redundant
    assets - the same VPN CVE pair shows up on two VPN nodes, the same
    CitrixBleed CVE shows up on two load balancers, and so on. Without
    deduping, the top 5 could end up being two near-identical entries for
    what a human would call "one risk." We keep whichever instance scored
    highest and drop the rest.
    """
    for finding in findings:
        score_finding(finding)

    best_per_key: dict[tuple[str, str], RiskFinding] = {}
    for finding in findings:
        key = (finding.business_service, finding.vulnerability_name)
        current_best = best_per_key.get(key)
        if current_best is None or finding.score.total > current_best.score.total:
            best_per_key[key] = finding

    ranked = sorted(best_per_key.values(), key=lambda f: f.score.total, reverse=True)
    return ranked[:n]
