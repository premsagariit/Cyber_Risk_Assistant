"""Pre-builds the NIST 800-53 vector store, so the first `streamlit run`
doesn't have to spend a minute or two embedding ~1,000 controls while
someone's watching a loading spinner.

Usage:
    python scripts/build_vector_store.py

Safe to run more than once - it's a no-op if the store already exists.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nist_ingest import get_or_build_collection

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NIST_CSV_PATH = DATA_DIR / "NIST_SP-800-53_rev5_catalog_load.csv"
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


def main():
    print(f"Reading NIST catalog from {NIST_CSV_PATH}")
    collection = get_or_build_collection(str(CHROMA_PERSIST_DIR), str(NIST_CSV_PATH))
    print(f"Vector store ready at {CHROMA_PERSIST_DIR} - {collection.count()} controls embedded.")


if __name__ == "__main__":
    main()
