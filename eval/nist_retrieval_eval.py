"""Ground truth benchmark for the retrieval lane: given a finding
description, does semantic search over the NIST 800-53 catalog surface the
control a security analyst would actually pick?

This is a benchmark script, not a pytest unit test, on purpose - it needs
network access on first run (to download the embedding model) and takes a
minute or two to build the vector store the first time, so it doesn't belong
in a fast test suite that runs on every save. Run it manually:

    python eval/nist_retrieval_eval.py

Each case below is grounded in a real finding pattern from this project's
data pack, with an expected control ID chosen by reading the ACTUAL NIST
control text (not guessed from the control's name alone) - see the comment
on each case for the reasoning. `acceptable` lists controls that would also
be a defensible answer, since more than one control is often relevant to a
real finding - a strict single-answer benchmark would punish the system for
being reasonable.

Four metrics are reported, as two clearly separate pairs:
  hit@1 / hit@3         - strict: was the retrieved control_id an EXACT
                           match for expected_primary/acceptable?
  family hit@1 / hit@3  - forgiving: does the retrieved control's BASE id
                           (its own "(n)" enhancement suffix stripped, if
                           any) match the base id of expected_primary or
                           an acceptable control?

The family metric exists because retrieving IR-4(12) when IR-4 was expected
is a human-reviewable correct answer, not a wrong one - IR-4(12) is a
specific enhancement of exactly the right control family, findable by a
security analyst reading the control ID alone, which is a fundamentally
different kind of "close" than retrieving an unrelated control. But it is
still a looser match than the strict metric asks for, so the two are kept
separate and both printed, rather than quietly folding family credit into
the headline number - that would change what hit@1/hit@3 mean without
saying so.

hit@3 is the more forgiving of the two *width* settings (top-1 vs. top-3),
not to be confused with the *strictness* axis above - the app shows all 3
retrieved controls in an expander for exactly this reason (see app.py),
so a control landing at #2 or #3 is still a usable result, not a miss.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nist_ingest import get_or_build_collection
from src.retrieval import _base_control_id, search_controls

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NIST_CSV_PATH = DATA_DIR / "NIST_SP-800-53_rev5_catalog_load.csv"
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


EVAL_CASES = [
    {
        "id": "eol-no-patch",
        # Mirrors A-1034 / V-2089 in the real data: Windows Server 2012 R2,
        # end of support, no patch exists because the OS itself is EOL.
        "description": (
            "Windows Server 2012 R2 end of support, no vendor security patches "
            "available, internal database server, exploit maturity commodity."
        ),
        "expected_primary": "SA-22",  # "Replace system components when support ... no longer available"
        "acceptable": ["SI-2"],
    },
    {
        "id": "patch-available-not-applied",
        # Mirrors A-1005 / V-2015: a patch exists, it just hasn't been installed.
        "description": (
            "Fortinet SSL-VPN heap buffer overflow RCE, CVSS 9.8, vendor patch "
            "is available, internet-facing, not yet applied, open 27 days."
        ),
        "expected_primary": "SI-2",  # "Install security-relevant software ... updates within [period]"
        "acceptable": [],
    },
    {
        "id": "no-account-lockout",
        "description": (
            "Login endpoint has no limit on failed password attempts, "
            "allowing unlimited brute-force login attempts against user accounts."
        ),
        "expected_primary": "AC-7",  # "Enforce a limit of ... consecutive invalid logon attempts"
        "acceptable": [],
    },
    {
        "id": "excessive-privileges",
        "description": (
            "Service account for the CI/CD pipeline has full administrator "
            "rights across the environment, far more access than its automated "
            "build tasks require."
        ),
        "expected_primary": "AC-6",  # "Employ the principle of least privilege"
        "acceptable": ["AC-2"],
    },
    {
        "id": "no-vuln-scanning-cadence",
        "description": (
            "No regular vulnerability scanning process in place; critical "
            "flaws go undetected for months before anyone notices them."
        ),
        "expected_primary": "RA-5",  # "Monitor and scan for vulnerabilities ... [defined frequency]"
        "acceptable": [],
    },
    {
        "id": "ransomware-incident-response",
        "description": (
            "Confirmed ransomware deployment following VPN compromise; "
            "need a formal capability to detect, contain, and recover from "
            "the active incident."
        ),
        "expected_primary": "IR-4",  # "Implement an incident handling capability ... containment, eradication, and recovery"
        "acceptable": [],
    },
    {
        "id": "backup-targeting",
        # Mirrors the report's own "Intelligence gaps" note about NightHarbor
        # profiling backup and storage infrastructure.
        "description": (
            "Threat actor observed profiling backup and storage infrastructure; "
            "backups are not currently immutable and could be altered or "
            "deleted by a compromised admin account."
        ),
        "expected_primary": "CP-9",  # "Conduct backups of user-level information ... consistent with recovery objectives"
        "acceptable": [],
    },
    {
        "id": "shared-accounts",
        "description": (
            "Multiple engineers share a single generic login for the internal "
            "admin tool; individual user actions cannot be distinguished in logs."
        ),
        "expected_primary": "IA-2",  # "Uniquely identify and authenticate organizational users"
        "acceptable": [],
    },
    {
        "id": "secrets-in-build-logs",
        # Mirrors the SilentForge campaign pattern in the threat report.
        "description": (
            "API credentials and private keys found stored in plaintext "
            "environment variables and CI build logs, accessible to anyone "
            "with build system access."
        ),
        "expected_primary": "IA-5",  # "Manage system authenticators" - covers credential lifecycle/protection
        "acceptable": [],
    },
    {
        "id": "flat-network-lateral-movement",
        "description": (
            "Once inside the network via the compromised VPN, an attacker can "
            "reach internal databases and admin tools directly - there is no "
            "segmentation between the perimeter and internal systems."
        ),
        "expected_primary": "SC-7",  # "Monitor and control communications at ... managed interfaces"
        "acceptable": [],
    },
]


def run_eval():
    print("Building/loading the NIST vector store (first run takes a minute or two)...")
    collection = get_or_build_collection(str(CHROMA_PERSIST_DIR), str(NIST_CSV_PATH))
    print(f"Vector store ready - {collection.count()} controls embedded.\n")

    hit_at_1 = 0
    hit_at_3 = 0
    family_hit_at_1 = 0
    family_hit_at_3 = 0
    rows = []

    for case in EVAL_CASES:
        acceptable_ids = {case["expected_primary"], *case["acceptable"]}
        acceptable_families = {_base_control_id(cid) for cid in acceptable_ids}
        results = search_controls(collection, case["description"], top_k=3)
        retrieved_ids = [m["control_id"] for m in results]

        top1_hit = retrieved_ids[0] in acceptable_ids
        top3_hit = any(rid in acceptable_ids for rid in retrieved_ids)
        # Family credit: same base control id, ignoring any "(n)" enhancement
        # suffix - see the module docstring for why this is a distinct,
        # deliberately looser metric rather than folded into the strict one.
        family_top1_hit = _base_control_id(retrieved_ids[0]) in acceptable_families
        family_top3_hit = any(_base_control_id(rid) in acceptable_families for rid in retrieved_ids)

        hit_at_1 += top1_hit
        hit_at_3 += top3_hit
        family_hit_at_1 += family_top1_hit
        family_hit_at_3 += family_top3_hit

        rows.append(
            {
                "id": case["id"],
                "expected": case["expected_primary"],
                "retrieved": retrieved_ids,
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "family_top1_hit": family_top1_hit,
                "family_top3_hit": family_top3_hit,
            }
        )

    print(f"{'case':<32} {'expected':<10} {'retrieved (top 3)':<28} {'@1':<4} {'@3':<4} {'fam@1':<6} {'fam@3'}")
    print("-" * 100)
    for row in rows:
        mark1 = "✅" if row["top1_hit"] else "❌"
        mark3 = "✅" if row["top3_hit"] else "❌"
        fam_mark1 = "✅" if row["family_top1_hit"] else "❌"
        fam_mark3 = "✅" if row["family_top3_hit"] else "❌"
        print(
            f"{row['id']:<32} {row['expected']:<10} {', '.join(row['retrieved']):<28} "
            f"{mark1:<4} {mark3:<4} {fam_mark1:<6} {fam_mark3}"
        )

    n = len(EVAL_CASES)
    print("-" * 100)
    print("Strict (exact control_id match):")
    print(f"  hit@1: {hit_at_1}/{n} ({100 * hit_at_1 / n:.0f}%)")
    print(f"  hit@3: {hit_at_3}/{n} ({100 * hit_at_3 / n:.0f}%)")
    print("Family (base control_id match, enhancements credited to their base - see module docstring):")
    print(f"  family hit@1: {family_hit_at_1}/{n} ({100 * family_hit_at_1 / n:.0f}%)")
    print(f"  family hit@3: {family_hit_at_3}/{n} ({100 * family_hit_at_3 / n:.0f}%)")
    print(
        "\nCross-encoder re-ranking and generalized base-vs-enhancement promotion are already "
        "applied inside search_controls() - see README Supporting Question 3 for what's still "
        "open after those."
    )


if __name__ == "__main__":
    run_eval()