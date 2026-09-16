"""TawasolPay AI Cyber Risk Assistant - Streamlit entrypoint.

Run locally with:  streamlit run app.py
See README.md for setup, including the one-time NIST vector store build.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from src.data_loader import build_findings, load_all_data
from src.nist_ingest import get_or_build_collection
from src.retrieval import retrieve_controls
from src.scoring import rank_top_findings
from src.synthesis import generate_explanation

load_dotenv()

# Streamlit Cloud stores secrets in st.secrets rather than a .env file.
# Bridge them into environment variables here so the rest of the codebase
# can just read os.environ and not care which environment it's running in.
try:
    for key in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        if key in st.secrets:
            os.environ.setdefault(key, st.secrets[key])
except FileNotFoundError:
    pass  # no secrets.toml locally - .env handles it instead

DATA_DIR = "data"
NIST_CSV_PATH = os.path.join(DATA_DIR, "NIST_SP-800-53_rev5_catalog_load.csv")
CHROMA_PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="TawasolPay Cyber Risk Assistant", layout="wide")


@st.cache_resource(show_spinner="Loading NIST 800-53 control catalog into the vector store...")
def load_vector_store():
    return get_or_build_collection(CHROMA_PERSIST_DIR, NIST_CSV_PATH)


@st.cache_data(show_spinner="Joining assets, vulnerabilities, threat intel, and business context...")
def load_top_findings():
    data = load_all_data(DATA_DIR)
    findings = build_findings(data)
    top_findings = rank_top_findings(findings, n=5)
    # Returned alongside the findings (not embedded anywhere) since it's fed
    # to the LLM directly as short-form context - see src/synthesis.py.
    return top_findings, data["threat_report_text"]


def main():
    st.title("TawasolPay AI Cyber Risk Assistant")
    st.caption(
        "Top 5 risks, ranked by exposure, active exploitation, and business impact - "
        "not CVSS alone. Remediation guidance is retrieved live from the real "
        "NIST SP 800-53 control catalog, not hardcoded and not from the LLM's training data."
    )

    if not os.environ.get("LLM_API_KEY"):
        st.warning(
            "LLM_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set it in Streamlit secrets. Until then, explanations below will "
            "fall back to a plain template instead of an LLM-generated one."
        )

    collection = load_vector_store()
    top_findings, threat_report_text = load_top_findings()

    st.divider()

    for rank, finding in enumerate(top_findings, start=1):
        render_risk_card(rank, finding, collection, threat_report_text)


def render_risk_card(rank: int, finding, collection, threat_report_text: str) -> None:
    header = f"#{rank} - {finding.vulnerability_name} on {finding.asset_name}"

    with st.expander(header, expanded=(rank <= 2)):
        left, right = st.columns([2, 1])

        with left:
            st.markdown(
                f"**Asset:** {finding.asset_name} "
                f"({finding.asset_type}, {finding.environment}, {finding.location})"
            )
            st.markdown(
                f"**CVE:** {finding.cve}  |  **CVSS:** {finding.cvss}  |  "
                f"**Exposure:** {finding.asset_exposure}"
            )
            st.markdown(f"**Business service at risk:** {finding.business_service}")
            st.caption(finding.business_impact)

            if finding.has_threat_intel_match:
                ransomware_note = (
                    " — confirmed ransomware association"
                    if finding.ransomware_association == "Yes"
                    else ""
                )
                st.markdown(
                    f"**Threat intel match:** {finding.threat_actor} / "
                    f"{finding.campaign_name}{ransomware_note}"
                )
            else:
                st.markdown("**Threat intel match:** No confirmed campaign match")

            if finding.in_kev:
                st.markdown(
                    f"🛡️ **Confirmed in the real CISA KEV catalog** "
                    f"(known ransomware use: {finding.kev_ransomware_use})"
                )

        with right:
            st.metric("Composite risk score", finding.score.total)
            st.caption(
                f"Exposure +{finding.score.exposure_points} · "
                f"Exploitation +{finding.score.exploitation_points} · "
                f"Ransomware +{finding.score.ransomware_points}  \n"
                f"Business +{finding.score.business_points} · "
                f"Missing controls +{finding.score.missing_controls_points} · "
                f"KEV +{finding.score.kev_bonus_points}"
            )

        st.divider()

        with st.spinner("Retrieving NIST 800-53 guidance and writing the explanation..."):
            controls = retrieve_controls(collection, finding, top_k=3)
            explanation = generate_explanation(finding, controls[0], threat_report_text)

        st.markdown("#### Why this ranks here, and what to do about it")
        st.write(explanation)

        with st.expander("Retrieved NIST controls (raw text, for verification)"):
            for control in controls:
                st.markdown(f"**{control['control_id']} - {control['name']}**")
                st.text(control["text"][:800])
                st.caption(f"Relevance distance: {control['distance']:.3f} (lower is more relevant)")


if __name__ == "__main__":
    main()