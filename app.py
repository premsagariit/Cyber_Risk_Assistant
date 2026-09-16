"""TawasolPay AI Cyber Risk Assistant - Streamlit entrypoint.

Run locally with:  streamlit run app.py
See README.md for setup, including the one-time NIST vector store build.
"""

import os

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import build_findings, load_all_data
from src.nist_ingest import get_or_build_collection
from src.retrieval import retrieve_controls
from src.scoring import rank_top_findings
from src.synthesis import stream_explanation

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

# The data pack has 104 distinct (business service, vulnerability) risks
# after dedup - this caps how many the "how many to show" slider can reach.
# Uncapped would turn a top-risk briefing tool into a full vulnerability
# management dashboard, which isn't what this app is for.
MAX_RISKS_SHOWN = 20

# Fixed color per scoring factor, in a fixed order, so the same factor
# always reads as the same color across every chart on the page - color
# identifies the factor, never its rank within a given finding. Values are
# the dataviz skill's validated categorical palette (light-mode steps),
# slots 1-6 in their validated order.
FACTOR_COLORS = {
    "Exposure": "#2a78d6",
    "Exploitation": "#eb6834",
    "Ransomware": "#1baf7a",
    "Business": "#eda100",
    "Missing controls": "#e87ba4",
    "KEV bonus": "#008300",
}

# Fixed status palette (good/warning/serious/critical) for the severity
# badge - reserved colors, never reused for a scoring factor, always paired
# with an icon + label rather than color alone.
SEVERITY_BADGE = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "🟢",
}

# Sequential single hue (blue, mid step) for the overview chart - magnitude
# is encoded by bar length, not by a per-bar color gradient, so one flat
# hue is correct here rather than a heatmap-style ramp.
SCORE_BAR_COLOR = "#2a78d6"

st.set_page_config(page_title="TawasolPay Cyber Risk Assistant", layout="wide")


@st.cache_resource(show_spinner="Loading NIST 800-53 control catalog into the vector store...")
def load_vector_store():
    return get_or_build_collection(CHROMA_PERSIST_DIR, NIST_CSV_PATH)


@st.cache_data(show_spinner="Joining assets, vulnerabilities, threat intel, and business context...")
def load_all_ranked_findings():
    """Every distinct (business service, vulnerability) risk, scored and
    ranked, not just a fixed top 5 - the UI slices/filters/sorts this list
    interactively instead of re-running the join+scoring pipeline on every
    widget interaction, which is what actually makes the sidebar controls
    below feel instant rather than triggering a multi-second reload each
    time someone moves a slider.
    """
    data = load_all_data(DATA_DIR)
    findings = build_findings(data)
    # n=len(findings) is a safe upper bound: rank_top_findings dedupes down
    # to a strictly smaller list internally, so this just means "don't cap."
    all_ranked = rank_top_findings(findings, n=len(findings))
    return all_ranked, data["threat_report_text"]


@st.cache_data(show_spinner=False)
def cached_retrieve_controls(cache_key: str, _finding, top_k: int = 3):
    """Caches retrieval by the finding's own stable identity (vuln_id +
    asset_id), not by hashing the finding object itself. Streamlit reruns
    this whole script on every widget interaction - without this cache,
    dragging the "how many to show" slider would re-run the embedding +
    cross-encoder search for every already-computed card on every rerun.
    The leading underscore on _finding tells Streamlit not to hash it (and
    RiskFinding isn't cleanly hashable anyway); cache_key is what actually
    identifies the entry.
    """
    collection = load_vector_store()
    return retrieve_controls(collection, _finding, top_k=top_k)


def main():
    st.title("TawasolPay AI Cyber Risk Assistant")
    st.caption(
        "Explore the ranked risk list - adjust how many to show, filter to KEV-confirmed "
        "only, or change the sort order - not just a fixed top 5. Ranking is by exposure, "
        "active exploitation, and business impact, not CVSS alone. Remediation guidance is "
        "retrieved live from the real NIST SP 800-53 control catalog, not hardcoded and not "
        "from the LLM's training data."
    )

    if not os.environ.get("LLM_API_KEY"):
        st.warning(
            "LLM_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set it in Streamlit secrets. Until then, explanations below will "
            "fall back to a plain template instead of an LLM-generated one."
        )

    collection = load_vector_store()
    all_ranked, threat_report_text = load_all_ranked_findings()

    displayed, num_matching = _render_sidebar_and_filter(all_ranked)

    st.divider()

    if not displayed:
        st.info("No risks match the current filters - try widening them in the sidebar.")
        return

    _render_summary_metrics(displayed, num_matching)
    _render_overview_chart(displayed)

    st.divider()

    for rank, finding in enumerate(displayed, start=1):
        render_risk_card(rank, finding, collection, threat_report_text)


def _render_sidebar_and_filter(all_ranked):
    """Sidebar controls, applied to the full ranked list. Returns
    (displayed_findings, count_matching_before_the_show-N_slice) so the
    summary metrics below can report how many risks match the filters,
    not just how many are currently visible."""
    with st.sidebar:
        st.header("Explore the risk list")

        num_to_show = st.slider(
            "How many risks to show",
            min_value=3,
            max_value=min(MAX_RISKS_SHOWN, len(all_ranked)),
            value=min(5, len(all_ranked)),
        )

        kev_only = st.checkbox("Confirmed in the real CISA KEV catalog only")

        sort_by = st.selectbox(
            "Sort by",
            ["Composite risk score", "CVSS", "Days open"],
        )

        st.session_state.setdefault("expand_all", False)
        st.toggle("Expand all cards", key="expand_all")

    filtered = [f for f in all_ranked if f.in_kev] if kev_only else list(all_ranked)

    if sort_by == "CVSS":
        filtered = sorted(filtered, key=lambda f: f.cvss, reverse=True)
    elif sort_by == "Days open":
        filtered = sorted(filtered, key=lambda f: f.days_open, reverse=True)
    # "Composite risk score" needs no re-sort - all_ranked (and therefore
    # filtered, since filtering preserves order) is already score-descending.

    return filtered[:num_to_show], len(filtered)


def _render_summary_metrics(displayed, num_matching):
    avg_score = sum(f.score.total for f in displayed) / len(displayed)
    internet_count = sum(1 for f in displayed if f.asset_exposure == "Internet")
    kev_count = sum(1 for f in displayed if f.in_kev)

    cols = st.columns(4)
    cols[0].metric("Matching your filters", num_matching)
    cols[1].metric("Showing", len(displayed))
    cols[2].metric("Avg. composite score", f"{avg_score:.0f}")
    cols[3].metric("Internet-exposed / KEV-confirmed", f"{internet_count} / {kev_count}")


def _render_overview_chart(displayed):
    """Horizontal bar of composite scores for exactly what's displayed
    below - updates instantly as the sidebar filters/slider change, so the
    chart and the cards never show two different views of the data."""
    # Reversed so the highest-scoring finding renders at the TOP of the
    # chart - Plotly's default category order for a horizontal bar draws
    # bottom-to-top, which would otherwise put #1 at the bottom.
    ordered = list(reversed(displayed))
    labels = [f"{f.vulnerability_name} · {f.asset_name}" for f in ordered]
    scores = [f.score.total for f in ordered]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=SCORE_BAR_COLOR,
            text=scores,
            textposition="outside",
            hovertemplate="%{y}<br>Composite score: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Composite risk score - displayed risks",
        xaxis_title="Composite score",
        yaxis_title=None,
        margin=dict(l=0, r=40, t=40, b=0),
        height=max(160, 36 * len(displayed)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e1e0d9", zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _score_breakdown_chart(finding) -> go.Figure:
    """One thin horizontal stacked bar per card, segments in the fixed
    FACTOR_COLORS order - same color always means the same factor, whether
    it's this card or the next one. No legend per card (shown once, in the
    caption above the first card) since repeating a 6-item legend on every
    card would be pure repetition, not new information."""
    score = finding.score
    segments = [
        ("Exposure", score.exposure_points),
        ("Exploitation", score.exploitation_points),
        ("Ransomware", score.ransomware_points),
        ("Business", score.business_points),
        ("Missing controls", score.missing_controls_points),
        ("KEV bonus", score.kev_bonus_points),
    ]

    fig = go.Figure()
    for name, points in segments:
        if points == 0:
            continue  # a zero-width segment still shows a hover target - skip it
        fig.add_trace(
            go.Bar(
                x=[points],
                y=[""],
                orientation="h",
                name=name,
                marker_color=FACTOR_COLORS[name],
                hovertemplate=f"{name}: +{points}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=50,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def render_risk_card(rank: int, finding, collection, threat_report_text: str) -> None:
    badge = SEVERITY_BADGE.get(finding.severity, "")
    header = f"#{rank} · {badge} {finding.severity} · {finding.vulnerability_name} on {finding.asset_name}"

    with st.expander(header, expanded=st.session_state.get("expand_all", False) or rank <= 2):
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
            st.plotly_chart(
                _score_breakdown_chart(finding),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"breakdown-{finding.asset_id}-{finding.vuln_id}",
            )
            st.caption(" · ".join(f"{name} +{pts}" for name, pts in _nonzero_factors(finding)))

        st.divider()

        cache_key = f"{finding.asset_id}:{finding.vuln_id}"
        with st.spinner("Retrieving NIST 800-53 guidance..."):
            controls = cached_retrieve_controls(cache_key, finding, top_k=3)

        st.markdown("#### Why this ranks here, and what to do about it")

        # The explanation itself isn't behind st.cache_data like the retrieval
        # call above: st.write_stream() needs a live generator to display
        # progressively, and a cache decorator would have to fully consume
        # that generator before it could return/cache anything - defeating
        # the point. A plain session_state dict caches the *finished* text
        # instead: the first time a card is shown this session it streams
        # (visible, token-by-token progress instead of a silent wait behind a
        # spinner); every rerun after that - a sidebar tweak, expanding a
        # different card - reads the cached text straight away, so the
        # real API call still only happens once per card per session.
        if "explanation_cache" not in st.session_state:
            st.session_state.explanation_cache = {}

        if cache_key in st.session_state.explanation_cache:
            st.write(st.session_state.explanation_cache[cache_key])
        else:
            full_text = st.write_stream(stream_explanation(finding, controls[0], threat_report_text))
            st.session_state.explanation_cache[cache_key] = full_text

        with st.expander("Retrieved NIST controls (raw text, for verification)"):
            for control in controls:
                st.markdown(f"**{control['control_id']} - {control['name']}**")
                st.text(control["text"][:800])
                st.caption(f"Relevance distance: {control['distance']:.3f} (lower is more relevant)")


def _nonzero_factors(finding):
    score = finding.score
    factors = [
        ("Exposure", score.exposure_points),
        ("Exploitation", score.exploitation_points),
        ("Ransomware", score.ransomware_points),
        ("Business", score.business_points),
        ("Missing controls", score.missing_controls_points),
        ("KEV bonus", score.kev_bonus_points),
    ]
    return [(name, pts) for name, pts in factors if pts > 0]


if __name__ == "__main__":
    main()
