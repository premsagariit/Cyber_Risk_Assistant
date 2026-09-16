"""Plain data classes used to pass a fully-joined risk finding around the app.

We use dataclasses instead of raw pandas rows or dicts once the data leaves
the loading step, because from here on we want autocomplete, type checking,
and a single obvious place (this file) that defines what a "finding" even is.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreBreakdown:
    """The five weighted factors behind a finding's composite score, plus a
    small bonus for real-world confirmation via the CISA KEV catalog.

    The factor order and relative weights follow the priority order given in
    the MDR advisory's analyst notes: exposure > active exploitation >
    ransomware association > business criticality > missing controls.
    """

    exposure_points: int
    exploitation_points: int
    ransomware_points: int
    business_points: int
    missing_controls_points: int
    kev_bonus_points: int

    @property
    def total(self) -> int:
        return (
            self.exposure_points
            + self.exploitation_points
            + self.ransomware_points
            + self.business_points
            + self.missing_controls_points
            + self.kev_bonus_points
        )


@dataclass
class RiskFinding:
    """One vulnerability, joined with the asset it sits on, the business
    service it threatens, and (if one exists) a matching threat intel record.

    This is the single unit everything else in the app operates on: it's
    what gets scored, what gets turned into a NIST search query, and what
    gets handed to the LLM as grounding facts.
    """

    # From vulnerabilities.csv
    vuln_id: str
    vulnerability_name: str
    cve: str
    cvss: float
    severity: str
    exploit_available: str
    patch_available: str
    days_open: int
    asset_exposure: str

    # From assets.csv
    asset_id: str
    asset_name: str
    asset_type: str
    environment: str
    location: str
    edr_installed: str
    owner_team: str

    # From business_services.csv
    business_service: str
    business_impact: str
    revenue_impact: str
    compliance_scope: str
    risk_appetite: str
    rto_hours: int

    # From threat_intelligence.csv - all optional, since not every
    # vulnerability has a matching campaign
    threat_actor: Optional[str] = None
    campaign_name: Optional[str] = None
    exploit_maturity: Optional[str] = None
    ransomware_association: Optional[str] = None
    ti_confidence: Optional[str] = None
    ti_summary: Optional[str] = None

    # From the real CISA KEV catalog - an independent confirmation signal,
    # separate from (and more authoritative than) the synthetic threat intel
    in_kev: bool = False
    kev_ransomware_use: Optional[str] = None

    # Filled in later by scoring.score_finding()
    score: Optional[ScoreBreakdown] = None

    @property
    def has_threat_intel_match(self) -> bool:
        return self.threat_actor is not None
