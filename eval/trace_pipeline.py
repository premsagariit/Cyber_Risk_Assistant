"""Diagnostic: trace one query all the way through the pipeline and write
down what every stage actually produced.

`eval/nist_retrieval_eval.py` answers "did retrieval get the right control?"
with a number. This answers "what did the machine actually see and say?" -
the raw candidate pool with distances, what survived aggregation/promotion/
backfill into the final top 3, the exact prompt that went to the LLM, and
what came back. It exists for the times a hit@1 number is technically fine
but the reason behind it is wrong, which is how every real bug in this
pipeline has been found so far.

This is a diagnostic, not a test: no assertions, nothing passes or fails, and
it is deliberately not named `test_*` so pytest never collects it. It writes
only to eval/traces/ and reads everything else.

Two sources are traced, because they answer different questions:
  A. The 10 hand-labeled cases from eval/nist_retrieval_eval.py, which have
     clean, deliberately-worded queries but no real finding behind them.
  B. The 5 findings the deployed app actually ranks top, from the real data
     pack - the only way to see production synthesis on this project's own
     data, since the retrieval benchmark never calls the LLM at all.

Everything is imported from the modules that run in production rather than
reimplemented here. A trace that built its own prompt would drift from the
real one silently, and a drifting trace is worse than no trace.

Runs 15 LLM calls (one per traced item) if LLM_API_KEY is set; without one,
generate_explanation() falls back to a plain template on its own and each
item is labelled accordingly. Either way it completes.

Three files are written to eval/traces/, all overwritten each run:
  trace_report.md   curated and readable - trimmed tables, for a human
  trace_full.json   every step's complete input and output, nothing trimmed:
                    full finding records, all 20 candidate chunks with their
                    document text, the full prompt, and the raw LLM request
                    and response (finish_reason and token usage included)
  console.log       the run's own stdout, verbatim

The raw LLM request/response capture works by wrapping synthesis._get_client
at runtime, inside this process only - src/synthesis.py is not modified. That
wrapper is the one way to see finish_reason and reasoning-token usage, which
is where an output that "succeeded" but came back empty explains itself.

    python eval/trace_pipeline.py
"""

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import src.synthesis as synthesis
from eval.nist_retrieval_eval import EVAL_CASES
from src.data_loader import build_findings, load_all_data
from src.models import RiskFinding
from src.nist_ingest import EMBEDDING_MODEL_NAME, chunk_token_budget, get_or_build_collection
from src.retrieval import (
    CANDIDATE_POOL_SIZE,
    POOR_MATCH_DISTANCE_THRESHOLD,
    _base_control_id,
    build_query_text,
    search_controls,
)
from src.scoring import rank_top_findings, score_finding
from src.synthesis import (
    PROMPT_TEMPLATE,
    _describe_missing_controls,
    _describe_threat_intel,
    generate_explanation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
NIST_CSV_PATH = DATA_DIR / "NIST_SP-800-53_rev5_catalog_load.csv"
CHROMA_PERSIST_DIR = REPO_ROOT / "chroma_db"

# The script's own output directory - the only thing it writes to. Overwritten
# on every run rather than timestamped: this is a working diagnostic you read
# once and regenerate, not an audit trail worth accumulating.
TRACES_DIR = Path(__file__).resolve().parent / "traces"
REPORT_PATH = TRACES_DIR / "trace_report.md"
FULL_LOG_PATH = TRACES_DIR / "trace_full.json"
CONSOLE_LOG_PATH = TRACES_DIR / "console.log"

TOP_K = 3

# How much of the raw pool to print. The pool itself is queried at production's
# own CANDIDATE_POOL_SIZE so the ordering comparison below is faithful; only
# the printed slice is trimmed, to keep the report readable.
RAW_POOL_DISPLAY = 10

# Fenced with four backticks because control text and LLM output can contain
# triple-backtick sequences of their own.
FENCE = "````"


def _cell(value: str) -> str:
    """Escapes a value for a markdown table cell. NIST enhancement names carry
    their parent's name and their own separated by a pipe ("Least Privilege |
    Privileged Accounts"), which silently adds columns to the rendered row."""
    return str(value).replace("|", "\\|")


class _Tee:
    """Mirrors stdout into a buffer so the run's console output can be saved
    verbatim alongside the structured log, while still appearing live in the
    terminal - a progress line you only see once isn't much of a log."""

    def __init__(self, stream):
        self._stream = stream
        self.lines = []

    def write(self, text):
        self.lines.append(text)
        return self._stream.write(text)

    def flush(self):
        self._stream.flush()


def _install_llm_recorder(records: dict) -> None:
    """Wraps synthesis's OpenAI client so every chat completion request and
    its raw response land in `records`, keyed by call order.

    Done by patching synthesis._get_client in this process rather than by
    editing src/synthesis.py: the trace is a read-only observer of production
    code, and production shouldn't carry instrumentation it doesn't need. The
    raw response is what makes an empty-but-successful answer diagnosable -
    finish_reason and usage.completion_tokens_details live there and nowhere
    else.
    """
    real_get_client = synthesis._get_client

    def recording_get_client():
        client = real_get_client()
        real_create = client.chat.completions.create

        def create(**kwargs):
            started = time.monotonic()
            call = {"request": kwargs, "elapsed_s": None, "response": None, "error": None}
            try:
                response = real_create(**kwargs)
            except Exception as exc:
                call["elapsed_s"] = round(time.monotonic() - started, 3)
                call["error"] = repr(exc)
                records[len(records)] = call
                raise
            call["elapsed_s"] = round(time.monotonic() - started, 3)
            call["response"] = response.model_dump()
            records[len(records)] = call
            return response

        client.chat.completions.create = create
        return client

    synthesis._get_client = recording_get_client


def _raw_candidate_pool(collection, query_text: str) -> list[dict]:
    """The chunk-level hits exactly as Chroma returns them, before
    search_controls() aggregates them down to one row per control.

    search_controls() deliberately doesn't expose this intermediate state -
    it's the thing being diagnosed, so the trace queries for it directly
    rather than inferring it from the final result.
    """
    results = collection.query(query_texts=[query_text], n_results=CANDIDATE_POOL_SIZE)
    pool = []
    for i, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i]
        pool.append(
            {
                "chunk_id": chunk_id,
                "control_id": metadata["control_id"],
                "name": metadata["name"],
                "chunk_type": metadata["chunk_type"],
                "is_enhancement": metadata["is_enhancement"],
                "related": metadata["related"],
                "distance": results["distances"][0][i],
                # The embedded text itself. Trimmed out of the markdown report
                # for readability, kept in full in the JSON log - "why did this
                # chunk match?" is unanswerable without it.
                "document": results["documents"][0][i],
            }
        )
    return pool


def _aggregate_by_control(pool: list[dict]) -> list[dict]:
    """One row per control, keeping its closest chunk, ordered by distance.

    This is the ordering the final top 3 *would* have if promotion and
    backfill did nothing - the reference the trace diffs against to show what
    those two rules actually changed.
    """
    best: dict[str, dict] = {}
    for hit in pool:
        control_id = hit["control_id"]
        if control_id not in best or hit["distance"] < best[control_id]["distance"]:
            best[control_id] = hit
    return sorted(best.values(), key=lambda hit: hit["distance"])


def _describe_reordering(aggregated: list[dict], final_ids: list[str]) -> str:
    """Explains how the final top 3 differs from plain distance order, naming
    the rule whose signature matches rather than asserting one ran."""
    distance_order = [hit["control_id"] for hit in aggregated]
    if final_ids == distance_order[: len(final_ids)]:
        return "No change - the final top 3 is plain distance order; neither promotion nor backfill altered it."

    notes = []
    if final_ids and distance_order and final_ids[0] != distance_order[0]:
        displaced = distance_order[0]
        if _base_control_id(displaced) == final_ids[0]:
            notes.append(
                f"**Promotion fired**: base control `{final_ids[0]}` moved above its own "
                f"enhancement `{displaced}`, which was closest by raw distance."
            )
        else:
            notes.append(f"Top result changed: `{displaced}` (closest by distance) -> `{final_ids[0]}`.")

    added = [cid for cid in final_ids if cid not in distance_order[:TOP_K]]
    if added:
        top_related = [r.strip() for r in aggregated[0]["related"].split(",") if r.strip()] if aggregated else []
        from_related = [cid for cid in added if cid in top_related]
        if from_related:
            notes.append(
                f"**Backfill fired**: {', '.join(f'`{c}`' for c in from_related)} pulled in from the "
                f"top result's `related` list, replacing a weaker match."
            )
        else:
            notes.append(f"Entered the top 3 from deeper in the pool: {', '.join(f'`{c}`' for c in added)}.")

    return " ".join(notes) if notes else f"Order changed: {distance_order[:TOP_K]} -> {final_ids}."


def _synthetic_finding_for_case(case: dict) -> RiskFinding:
    """A stand-in RiskFinding for an eval case, built only to drive
    generate_explanation().

    The eval cases are hand-written control-retrieval probes - they have a
    description and an expected control, but no asset, business service, or
    threat intel behind them. Synthesis needs all of that, so every field the
    case can't supply gets a deliberately conspicuous placeholder: a trace
    read six months from now should be impossible to mistake for real data.
    Follows the same shape as tests/test_synthesis_grounding._sample_finding().
    """
    finding = RiskFinding(
        vuln_id="V-SYNTHETIC",
        vulnerability_name=f"[synthetic eval case: {case['id']}]",
        cve="CVE-SYNTHETIC-0000",
        cvss=9.8,
        severity="Critical",
        exploit_available="Yes",
        patch_available="No",
        days_open=99,
        asset_exposure="Internet",
        asset_id="A-SYNTHETIC",
        asset_name="SYNTHETIC-ASSET-NOT-REAL",
        asset_type="SYNTHETIC-ASSET-TYPE",
        environment="SYNTHETIC-ENV",
        location="SYNTHETIC-LOCATION",
        edr_installed="No",
        owner_team="SYNTHETIC-TEAM",
        business_service="SYNTHETIC-BUSINESS-SERVICE",
        business_impact=f"Placeholder impact for the eval case: {case['description']}",
        revenue_impact="High",
        compliance_scope="SYNTHETIC-COMPLIANCE-SCOPE",
        risk_appetite="Low",
        rto_hours=1,
        threat_actor=None,
        campaign_name=None,
        exploit_maturity=None,
        ransomware_association=None,
        ti_confidence=None,
    )
    score_finding(finding)
    return finding


def _build_prompt(finding: RiskFinding, top_control: dict, threat_report_text: str) -> str:
    """Rebuilds the prompt generate_explanation() sends, for display.

    Uses synthesis.py's own PROMPT_TEMPLATE and its two description helpers
    rather than restating any of that text here. This mapping of finding
    fields to template placeholders is the one piece the trace has to mirror -
    if generate_explanation()'s .format() call ever gains or drops a field,
    this call needs the same edit, or the trace starts quietly showing a
    prompt the app doesn't send.
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


def _label_output(explanation: str) -> str:
    """Whether synthesis returned a real LLM answer or its own fallback.

    _fallback_explanation() prefixes its output with "[LLM call failed", so
    that prefix is the signal - the trace never builds a fallback of its own,
    it only reports which one synthesis chose.
    """
    if not explanation.startswith("[LLM call failed"):
        return "[LLM]"
    if not os.environ.get("LLM_API_KEY"):
        return "[FALLBACK - no API key]"
    return "[FALLBACK - LLM call failed]"


def trace_item(collection, item_id: str, heading: str, provenance: str, query_text: str,
               finding: RiskFinding, threat_report_text: str, llm_calls: dict,
               case: dict = None) -> dict:
    """Runs one item through every stage, capturing each stage's full input and
    output, and returns the markdown section plus the complete structured
    record that goes into trace_full.json."""
    pool = _raw_candidate_pool(collection, query_text)
    aggregated = _aggregate_by_control(pool)
    final = search_controls(collection, query_text, TOP_K)
    final_ids = [control["control_id"] for control in final]
    reordering = _describe_reordering(aggregated, final_ids)

    prompt = _build_prompt(finding, final[0], threat_report_text)

    # Anything the recorder picks up from here belongs to this item's call.
    calls_before = len(llm_calls)
    explanation = generate_explanation(finding, final[0], threat_report_text)
    label = _label_output(explanation)
    this_call = llm_calls.get(calls_before)

    lines = [
        f"### {heading}",
        "",
        f"`{provenance}`",
        "",
        "**1. Query text**",
        "",
        FENCE,
        query_text,
        FENCE,
        "",
        f"**2. Raw candidate pool** (top {RAW_POOL_DISPLAY} of {len(pool)} chunks, before aggregation)",
        "",
        "| # | chunk | control | type | dist |",
        "|---|---|---|---|---|",
    ]
    for i, hit in enumerate(pool[:RAW_POOL_DISPLAY]):
        lines.append(
            f"| {i} | `{_cell(hit['chunk_id'])}` | {_cell(hit['control_id'])} | "
            f"{hit['chunk_type']} | {hit['distance']:.4f} |"
        )

    lines += [
        "",
        f"**3. Final top {TOP_K}** (after aggregation, promotion, backfill)",
        "",
        "| # | control | name | dist |",
        "|---|---|---|---|",
    ]
    for i, control in enumerate(final):
        lines.append(
            f"| {i} | **{_cell(control['control_id'])}** | {_cell(control['name'])} | "
            f"{control['distance']:.4f} |"
        )
    lines += [
        "",
        f"{reordering}",
        "",
        "**4. Prompt sent to the LLM**",
        "",
        FENCE,
        prompt,
        FENCE,
        "",
        f"**5. Output** {label}",
        "",
        FENCE,
        explanation,
        FENCE,
        "",
    ]

    # The full record: every step, both sides of it, nothing trimmed. The
    # markdown above is the readable summary of exactly this.
    record = {
        "id": item_id,
        "heading": heading,
        "provenance": provenance,
        "expected_primary": case["expected_primary"] if case else None,
        "acceptable": case["acceptable"] if case else None,
        "finding": asdict(finding),
        "steps": {
            "1_query": {
                "input": {
                    "source": "eval case description (verbatim)" if case else "build_query_text(finding)",
                    "finding_id": finding.vuln_id,
                },
                "output": query_text,
            },
            "2_retrieval_raw": {
                "input": {"query_text": query_text, "n_results": CANDIDATE_POOL_SIZE},
                "output": pool,
            },
            "3_ranking": {
                "input": {
                    "distinct_controls_in_pool": len(aggregated),
                    "order_by_distance": [hit["control_id"] for hit in aggregated],
                    "top_k": TOP_K,
                    "poor_match_distance_threshold": POOR_MATCH_DISTANCE_THRESHOLD,
                },
                "output": {"final": final, "diagnosis": reordering},
            },
            "4_prompt": {
                "input": {
                    "top_control": final[0],
                    "threat_summary": _describe_threat_intel(finding),
                    "missing_controls": _describe_missing_controls(finding),
                    "threat_report_excerpt_chars": len(threat_report_text[:3000]),
                },
                "output": prompt,
            },
            "5_llm_call": this_call or {"note": "no API call recorded - synthesis used its fallback"},
            "6_explanation": {
                "output": explanation,
                "label": label,
                "is_empty": not explanation.strip(),
                "cites_top_control": final[0]["control_id"] in explanation,
                "paragraphs": len([p for p in explanation.strip().split("\n\n") if p.strip()]),
            },
        },
    }

    return {
        "markdown": "\n".join(lines),
        "record": record,
        "heading": heading,
        "label": label,
        "reordered": not reordering.startswith("No change"),
        "final_ids": final_ids,
        "distance_order": [hit["control_id"] for hit in aggregated][:TOP_K],
    }


def main() -> None:
    # Same as app.py: pick up LLM_API_KEY from .env so the trace reflects
    # whatever the app itself would do when run locally.
    load_dotenv()

    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    llm_calls: dict = {}
    _install_llm_recorder(llm_calls)

    print("Loading the vector store and the data pack...")
    collection = get_or_build_collection(str(CHROMA_PERSIST_DIR), str(NIST_CSV_PATH))
    data = load_all_data(str(DATA_DIR))
    threat_report_text = data["threat_report_text"]
    print(f"Vector store ready - {collection.count()} chunks.")

    if os.environ.get("LLM_API_KEY"):
        print("LLM_API_KEY found - explanations will be real LLM calls.\n")
    else:
        print("No LLM_API_KEY - synthesis will fall back to its plain template.\n")

    traces = []

    print(f"Source A: tracing {len(EVAL_CASES)} hand-labeled eval cases...")
    for i, case in enumerate(EVAL_CASES, start=1):
        print(f"  A{i} {case['id']}")
        traces.append(
            trace_item(
                collection,
                item_id=f"A{i}",
                heading=f"A{i}. {case['id']} (expects {case['expected_primary']})",
                provenance="[SYNTHETIC - eval case, not real data]",
                # Source A traces the benchmark's own hand-written probe, not a
                # query derived from the placeholder finding below - that finding
                # exists only to give synthesis something to render.
                query_text=case["description"],
                finding=_synthetic_finding_for_case(case),
                threat_report_text=threat_report_text,
                llm_calls=llm_calls,
                case=case,
            )
        )

    print("\nSource B: tracing the real top 5 from the data pack...")
    top_findings = rank_top_findings(build_findings(data), n=5)
    for i, finding in enumerate(top_findings, start=1):
        print(f"  B{i} {finding.vulnerability_name} on {finding.asset_name}")
        traces.append(
            trace_item(
                collection,
                item_id=f"B{i}",
                heading=f"B{i}. {finding.vulnerability_name} on {finding.asset_name}",
                provenance="[REAL - from data pack]",
                query_text=build_query_text(finding),
                finding=finding,
                threat_report_text=threat_report_text,
                llm_calls=llm_calls,
            )
        )

    source_a = traces[: len(EVAL_CASES)]
    source_b = traces[len(EVAL_CASES) :]

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        "# Pipeline trace report",
        "",
        "Generated by `python eval/trace_pipeline.py` - a read-only diagnostic with no ",
        "assertions. Regenerated in place on every run.",
        "",
        f"- Vector store: {collection.count()} chunks",
        f"- Traced: {len(source_a)} synthetic eval cases (source A), {len(source_b)} real findings (source B)",
        f"- LLM outputs: {sum(1 for t in traces if t['label'] == '[LLM]')} / {len(traces)}",
        "",
        "## Source A - hand-labeled eval cases",
        "",
        "Queries are the benchmark's own descriptions. The findings behind them are ",
        "placeholders built only to drive synthesis - every field not implied by the case ",
        "is an obvious fake. Nothing here is real data.",
        "",
    ]
    report += [trace["markdown"] for trace in source_a]
    report += [
        "## Source B - real findings from the data pack",
        "",
        "The same five findings the deployed app ranks top, with queries built by ",
        "`retrieval.build_query_text()` exactly as the app builds them.",
        "",
    ]
    report += [trace["markdown"] for trace in source_b]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    llm_count = sum(1 for trace in traces if trace["label"] == "[LLM]")
    fallback_count = len(traces) - llm_count
    reordered = [trace for trace in source_a if trace["reordered"]]
    records = [trace["record"] for trace in traces]
    empty = [r["id"] for r in records if r["steps"]["6_explanation"]["is_empty"]]
    uncited = [r["id"] for r in records if not r["steps"]["6_explanation"]["cites_top_control"]]

    full_log = {
        "run": {
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.monotonic() - started, 1),
            "python": sys.version.split()[0],
            "config": {
                "embedding_model": EMBEDDING_MODEL_NAME,
                "chunk_token_budget": chunk_token_budget(),
                "collection_chunks": collection.count(),
                "candidate_pool_size": CANDIDATE_POOL_SIZE,
                "top_k": TOP_K,
                "poor_match_distance_threshold": POOR_MATCH_DISTANCE_THRESHOLD,
                "llm_model": os.environ.get("LLM_MODEL", synthesis.DEFAULT_MODEL),
                "llm_base_url": os.environ.get("LLM_BASE_URL", synthesis.DEFAULT_BASE_URL),
                # Presence only, never the key itself - this file is a log.
                "llm_api_key_present": bool(os.environ.get("LLM_API_KEY")),
            },
            "data_pack": {
                "findings_built": len(build_findings(data)),
                "threat_report_chars": len(threat_report_text),
                "threat_report_non_ascii_codepoints": sorted(
                    {hex(ord(c)) for c in threat_report_text if ord(c) > 127}
                ),
            },
            "summary": {
                "items_traced": len(records),
                "llm_outputs": llm_count,
                "fallback_outputs": fallback_count,
                "empty_outputs": empty,
                "outputs_not_citing_their_control": uncited,
                "reordered_by_promotion_or_backfill": [t["heading"] for t in traces if t["reordered"]],
            },
        },
        "items": records,
    }
    FULL_LOG_PATH.write_text(json.dumps(full_log, indent=2, default=str), encoding="utf-8")

    print(f"\nWrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {FULL_LOG_PATH.relative_to(REPO_ROOT)} ({FULL_LOG_PATH.stat().st_size:,} bytes)")
    print(f"  LLM outputs:      {llm_count}")
    print(f"  Fallback outputs: {fallback_count}")
    print(f"  Empty outputs:    {len(empty)} {empty}")
    print(f"  Not citing their retrieved control: {len(uncited)} of {len(records)}")
    print(f"  Source A cases where promotion/backfill changed the top 3: {len(reordered)}")
    for trace in reordered:
        print(f"    {trace['heading']}")
        print(f"      by distance: {trace['distance_order']}")
        print(f"      final:       {trace['final_ids']}")


if __name__ == "__main__":
    # Tee stdout so the console transcript is saved too. Wrapped in try/finally
    # so a crash mid-run still leaves the log of how far it got - a run that
    # died is exactly when you want the console output.
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    _tee = _Tee(sys.stdout)
    sys.stdout = _tee
    try:
        main()
    finally:
        sys.stdout = _tee._stream
        CONSOLE_LOG_PATH.write_text("".join(_tee.lines), encoding="utf-8")
        print(f"Wrote {CONSOLE_LOG_PATH.relative_to(REPO_ROOT)}")
