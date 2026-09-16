"""Loads the raw data pack and joins it into a flat list of RiskFinding objects.

This is deliberately plain pandas - no ORM, no query builder, no database.
Five small CSVs joined on a handful of keys don't need more than that, and
keeping this step boring and readable makes the ranking easy to trust.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .models import RiskFinding


def load_all_data(data_dir: str) -> dict:
    """Reads every file in the data pack into memory. Called once per app run
    (and cached by Streamlit) since none of these files are large."""
    data_dir = Path(data_dir)

    # encoding="utf-8" explicit on every read below rather than left to the
    # platform default: pandas' C parser happens to already resolve encoding=
    # None to utf-8 regardless of OS (verified against this data pack), but
    # threat_report_path.read_text() below does not - it uses
    # locale.getpreferredencoding(), cp1252 on Windows - and that mismatch
    # silently corrupted the threat report's em-dashes and smart quotes on
    # every prompt sent to the LLM. Being explicit here means both readers
    # agree, rather than one of them being an accident of a C library default.
    assets = pd.read_csv(data_dir / "assets.csv", encoding="utf-8").fillna("")
    vulnerabilities = pd.read_csv(data_dir / "vulnerabilities.csv", encoding="utf-8").fillna("")
    business_services = pd.read_csv(data_dir / "business_services.csv", encoding="utf-8").fillna("")
    threat_intel = pd.read_csv(data_dir / "threat_intelligence.csv", encoding="utf-8").fillna("")
    kev = pd.read_csv(data_dir / "known_exploited_vulnerabilities.csv", encoding="utf-8").fillna("")

    threat_report_path = data_dir / "synthetic_threat_report.md"
    threat_report_text = threat_report_path.read_text(encoding="utf-8") if threat_report_path.exists() else ""

    # Only look at vulnerabilities that are still open - if this data pack
    # ever includes remediated findings, we don't want them cluttering a
    # "what should we fix" ranking.
    vulnerabilities = vulnerabilities[vulnerabilities["status"] == "Open"]

    return {
        "assets": assets,
        "vulnerabilities": vulnerabilities,
        "business_services": business_services,
        "threat_intel": threat_intel,
        "kev": kev,
        "threat_report_text": threat_report_text,
    }


def _best_threat_intel_match(cve: str, threat_intel: pd.DataFrame) -> Optional[pd.Series]:
    """A single CVE sometimes appears in more than one threat intel row (a
    campaign chaining several CVEs gets one row per CVE, but a CVE can also
    be referenced by more than one campaign). When there's more than one
    match, we keep the single strongest signal rather than picking arbitrarily:
    ransomware-associated and actively-exploited campaigns outrank everything
    else, since those are the ones the scoring formula cares about most.
    """
    matches = threat_intel[threat_intel["matched_cve_or_control"] == cve]
    if matches.empty:
        return None

    def strength(row) -> int:
        points = 0
        if row["ransomware_association"] == "Yes":
            points += 2
        if row["exploit_maturity"] in ("Weaponized", "Active Exploitation"):
            points += 1
        return points

    matches = matches.copy()
    matches["_strength"] = matches.apply(strength, axis=1)
    return matches.sort_values("_strength", ascending=False).iloc[0]


def build_findings(data: dict) -> list[RiskFinding]:
    """Joins vulnerabilities -> assets -> business services -> threat intel
    -> real CISA KEV data into one RiskFinding per open vulnerability.

    Join keys used:
      vulnerabilities.asset_id      -> assets.asset_id
      assets.business_service       -> business_services.business_service
      vulnerabilities.cve           -> threat_intelligence.matched_cve_or_control
      vulnerabilities.cve           -> known_exploited_vulnerabilities.cveID

    Note the threat intel join key works for both real CVE IDs and the
    synthetic non-CVE identifiers used in this exercise (e.g. CICD-SYN-001) -
    both files use the same identifier in the same column, so a plain
    equality join covers both cases without special handling.
    """
    vulnerabilities = data["vulnerabilities"]
    assets_by_id = data["assets"].set_index("asset_id")
    services_by_name = data["business_services"].set_index("business_service")
    threat_intel = data["threat_intel"]
    kev_by_cve = data["kev"].set_index("cveID")

    findings: list[RiskFinding] = []

    for _, vuln in vulnerabilities.iterrows():
        asset_id = vuln["asset_id"]
        if asset_id not in assets_by_id.index:
            # A vulnerability pointing at an asset we have no record for.
            # Shouldn't happen with clean data, but one bad row shouldn't be
            # able to crash the whole pipeline - skip it and keep going.
            continue
        asset = assets_by_id.loc[asset_id]

        service_name = asset["business_service"]
        service = services_by_name.loc[service_name] if service_name in services_by_name.index else None

        intel_match = _best_threat_intel_match(vuln["cve"], threat_intel)

        in_kev = vuln["cve"] in kev_by_cve.index
        kev_row = kev_by_cve.loc[vuln["cve"]] if in_kev else None

        findings.append(
            RiskFinding(
                vuln_id=vuln["vuln_id"],
                vulnerability_name=vuln["vulnerability_name"],
                cve=vuln["cve"],
                cvss=float(vuln["cvss"]) if vuln["cvss"] != "" else 0.0,
                severity=vuln["severity"],
                exploit_available=vuln["exploit_available"],
                patch_available=vuln["patch_available"],
                days_open=int(vuln["days_open"]) if vuln["days_open"] != "" else 0,
                asset_exposure=vuln["asset_exposure"],
                asset_id=asset_id,
                asset_name=asset["asset_name"],
                asset_type=asset["asset_type"],
                environment=asset["environment"],
                location=asset["location"],
                edr_installed=asset["edr_installed"],
                owner_team=asset["owner_team"],
                business_service=service_name,
                business_impact=service["business_impact"] if service is not None else "",
                revenue_impact=service["revenue_impact"] if service is not None else "",
                compliance_scope=service["compliance_scope"] if service is not None else "",
                risk_appetite=service["risk_appetite"] if service is not None else "",
                rto_hours=int(service["rto_hours"]) if service is not None and service["rto_hours"] != "" else 0,
                threat_actor=intel_match["threat_actor"] if intel_match is not None else None,
                campaign_name=intel_match["campaign_name"] if intel_match is not None else None,
                exploit_maturity=intel_match["exploit_maturity"] if intel_match is not None else None,
                ransomware_association=intel_match["ransomware_association"] if intel_match is not None else None,
                ti_confidence=intel_match["confidence"] if intel_match is not None else None,
                ti_summary=intel_match["summary"] if intel_match is not None else None,
                in_kev=in_kev,
                kev_ransomware_use=kev_row["knownRansomwareCampaignUse"] if kev_row is not None else None,
            )
        )

    return findings
