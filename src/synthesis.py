"""The only place in the app that calls an LLM.

By design, the model never decides the ranking (that's scoring.py, plain
Python) and never invents a control (that's retrieval.py, real NIST text).
All it does is turn facts we've already computed and text we've already
retrieved into two readable paragraphs. That's what keeps hallucination risk
low enough to trust for something a CISO would actually read.

Uses the OpenAI-compatible chat completions API - provider-agnostic by
design, currently pointed at Gemini 3.1 Flash-Lite. Any OpenAI-compatible
endpoint works unmodified: just change LLM_BASE_URL and LLM_MODEL (and
LLM_API_KEY) in your .env file. These are named generically rather than
after whichever provider happens to be configured today - see README's
"A note on the LLM provider" for why.
"""

import os

from openai import OpenAI

from .models import RiskFinding

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Previous default, kept as a record of a real past decision, not the
# current one: Groq deprecated llama-3.3-70b-versatile in mid-2026, and
# openai/gpt-oss-120b (also served via Groq) was their recommended
# replacement at the time - see https://console.groq.com/docs/models for
# Groq's live list.
#
# Current default: gemini-3.1-flash-lite. Its free tier allows 15 RPM
# (vs. 5 RPM on non-Lite Flash), and it defaults to MINIMAL thinking, which
# avoids the exact failure mode openai/gpt-oss-120b hit here - reasoning
# tokens consuming the entire max_tokens budget and leaving empty visible
# output (see MAX_COMPLETION_TOKENS below and README Supporting Question 2,
# finding 4, for the full story of diagnosing and fixing that). Override
# with LLM_MODEL if this changes again.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Historical origin of this budget: openai/gpt-oss-120b (the previous
# provider) bills its internal reasoning tokens against max_tokens along with
# the visible answer. At 350 this was silently empty in production: traced
# calls (eval/trace_pipeline.py) showed the model spending up to 348 of 350
# tokens reasoning and returning 2 tokens of visible output. 1200 was
# calibrated against that provider's observed reasoning spend and fixed it
# there.
#
# Recalibrated for the current provider: gemini-3.1-flash-lite hides its
# internal reasoning spend even more thoroughly than gpt-oss did - it never
# reports it via completion_tokens_details (always null on this provider;
# confirmed via the raw response's extra_content.google.thought_signature
# field, present on every call). A repeated trial (5 reps x the 5 real
# findings, 25 calls) with reasoning_effort="low" - the setting inherited
# from the previous provider - found that hidden spend was usually a steady
# ~125-token baseline but spiked to 1077-1231 tokens on 5 of the 25 calls,
# occasionally enough to truncate the visible answer mid-sentence at the old
# 1200 limit. 2500 was calibrated against that largest observed spike, not
# just the smallest margin that happened to work once.
#
# That same trial is also why _create_completion() below no longer passes
# reasoning_effort at all: dropping it entirely (letting the provider use its
# own MINIMAL-thinking default) produced a zero-token hidden-reasoning gap on
# all 25 calls, not just a smaller one. So under the config actually shipped,
# 2500 is a generous safety margin rather than a tightly load-bearing number
# - kept this large anyway as headroom against a future model or provider
# change, not because the current setup needs it. Don't lower this back down,
# and don't reintroduce reasoning_effort, without re-running the repeated
# trial - a single clean-looking run is not evidence here, since the failure
# is non-deterministic per call.
MAX_COMPLETION_TOKENS = 2500

PROMPT_TEMPLATE = """You are writing one entry in a board-level cyber risk briefing for TawasolPay, a fintech company.

Use ONLY the facts listed below. Do not invent details, statistics, or controls that are not present here.

BACKGROUND (this week's MDR threat advisory, for context only - do not restate it verbatim):
{threat_report_excerpt}

ASSET: {asset_name} ({asset_type}, {environment} environment, {location})
FINDING: {vulnerability_name} ({cve}), CVSS {cvss}, exposure: {asset_exposure}
BUSINESS SERVICE AT RISK: {business_service} - {business_impact}
  Revenue impact: {revenue_impact} | Compliance scope: {compliance_scope} | RTO: {rto_hours}h
THREAT INTEL MATCH: {threat_summary}
MISSING CONTROLS: {missing_controls}
RETRIEVED NIST CONTROL: {control_id} - {control_name}
  Control excerpt: {control_excerpt}

Write exactly two short paragraphs, plain text, no markdown headers:
1. One or two sentences on why this finding belongs in the top 5 risks, referencing the specific
   factors above (exposure, threat intel, business impact - whichever actually apply here).
   Cite the CVE ID ({cve}) verbatim somewhere in this paragraph.
2. Two to three sentences of remediation guidance grounded in the retrieved NIST control above.
   Paraphrase the control in your own words and mention the control ID by name. Do not quote the
   control text verbatim.
"""


def _get_client() -> OpenAI:
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set - add it to .env or Streamlit secrets.")

    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def _describe_threat_intel(finding: RiskFinding) -> str:
    if not finding.has_threat_intel_match:
        return "No confirmed campaign match in current threat intelligence."
    return (
        f"{finding.threat_actor} ({finding.campaign_name}), exploit maturity "
        f"{finding.exploit_maturity}, ransomware association: {finding.ransomware_association}, "
        f"confidence: {finding.ti_confidence}."
    )


def _describe_missing_controls(finding: RiskFinding) -> str:
    missing = []
    if finding.edr_installed == "No":
        missing.append("no EDR installed")
    if finding.patch_available == "No":
        missing.append("no vendor patch available")
    if finding.days_open > 90:
        missing.append(f"open for {finding.days_open} days")
    return ", ".join(missing) if missing else "none identified"


def _build_prompt(finding: RiskFinding, top_control: dict, threat_report_text: str) -> str:
    """Fills PROMPT_TEMPLATE from a finding + retrieved control. Shared by
    generate_explanation() and stream_explanation() so the two request paths
    can never drift into sending two different prompts for the same inputs.

    threat_report_text is the raw text of synthetic_threat_report.md, passed
    straight through as context - it's short enough (about a page) that it
    doesn't need chunking or retrieval, so it goes in directly rather than
    through the vector store. Truncated defensively in case a future report
    is much longer than this one.
    """
    return PROMPT_TEMPLATE.format(
        threat_report_excerpt=(threat_report_text[:3000] or "No advisory text available."),
        asset_name=finding.asset_name,
        asset_type=finding.asset_type,
        environment=finding.environment,
        location=finding.location,
        vulnerability_name=finding.vulnerability_name,
        cve=finding.cve,
        cvss=finding.cvss,
        asset_exposure=finding.asset_exposure,
        business_service=finding.business_service,
        business_impact=finding.business_impact,
        revenue_impact=finding.revenue_impact,
        compliance_scope=finding.compliance_scope or "None",
        rto_hours=finding.rto_hours,
        threat_summary=_describe_threat_intel(finding),
        missing_controls=_describe_missing_controls(finding),
        control_id=top_control["control_id"],
        control_name=top_control["name"],
        control_excerpt=top_control["text"][:600],
    )


def generate_explanation(finding: RiskFinding, top_control: dict, threat_report_text: str = "") -> str:
    """Calls the LLM to turn pre-computed facts into two readable paragraphs,
    as a single non-streaming string.

    If the call fails for any reason - missing key, rate limit, network
    blip - falls back to a plain template built from the same facts, so the
    app still produces a usable (if less polished) result instead of
    crashing the whole page.
    """
    prompt = _build_prompt(finding, top_control, threat_report_text)

    try:
        client = _get_client()
        model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        response = _create_completion(client, model, prompt)
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001 - a demo app shouldn't crash on a flaky API call
        return _fallback_explanation(finding, top_control, exc)


def stream_explanation(finding: RiskFinding, top_control: dict, threat_report_text: str = ""):
    """Same call as generate_explanation() - same prompt, model, and
    max_tokens - but yields text as it arrives instead of waiting for the
    whole response. Meant for a UI that shows visible progress (e.g.
    Streamlit's st.write_stream) rather than a blank wait behind a spinner.

    Falls back the same way generate_explanation() does: if the call fails
    before any content arrives, or the stream ends having produced nothing
    (an empty completion is indistinguishable from a hung request to
    someone watching the UI), yields the plain-template fallback as its
    only chunk - so a caller that does nothing but concatenate whatever
    this yields still gets the same guarantee generate_explanation() gives.
    """
    prompt = _build_prompt(finding, top_control, threat_report_text)

    try:
        client = _get_client()
        model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=MAX_COMPLETION_TOKENS,
            stream=True,
        )
        received_any = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                received_any = True
                yield delta
        if not received_any:
            yield _fallback_explanation(finding, top_control, RuntimeError("stream produced no content"))
    except Exception as exc:  # noqa: BLE001 - same reasoning as generate_explanation()
        yield _fallback_explanation(finding, top_control, exc)


def _create_completion(client: OpenAI, model: str, prompt: str):
    """Requests the completion. Deliberately does NOT pass reasoning_effort.

    An earlier version of this function passed reasoning_effort="low",
    inherited unmodified from the previous provider (Groq's gpt-oss models).
    A repeated trial (5 reps x the 5 real findings, 25 calls per
    configuration - see the comment on MAX_COMPLETION_TOKENS above) found
    that setting caused gemini-3.1-flash-lite to spend a large, unpredictable
    amount of hidden reasoning on 5 of 25 calls (1077-1231 tokens, vs. a
    ~120-150 token baseline on the other 20) - a bimodal spike, not gradual
    noise. Omitting the parameter entirely and trusting the provider's own
    documented default (MINIMAL thinking for Flash-Lite) produced an exact
    zero-token hidden-reasoning gap on all 25 calls in the same trial. The
    inherited setting from a different provider was not actually the most
    conservative option available for this one - the provider's own default
    was. Don't re-add reasoning_effort here without re-running that trial.
    """
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=MAX_COMPLETION_TOKENS,
    )


def _fallback_explanation(finding: RiskFinding, top_control: dict, error: Exception) -> str:
    return (
        f"[LLM call failed ({error}) - showing a plain summary instead of a generated one.]\n\n"
        f"This finding scored {finding.score.total} points, driven by "
        f"{finding.asset_exposure.lower()} exposure, "
        f"{'a matched active campaign' if finding.has_threat_intel_match else 'its severity and business context'}, "
        f"and its impact on {finding.business_service} ({finding.business_impact}).\n\n"
        f"Recommended control: {top_control['control_id']} - {top_control['name']}. "
        f"See the retrieved control text below for the full guidance."
    )