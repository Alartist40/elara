#!/usr/bin/env python3
"""Build FAISS index from documents."""

import argparse
from pathlib import Path
from elara_core.tiers.tier2 import Tier2Engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("documents", help="Directory of .txt files")
    parser.add_argument("--output_index", default="data/faiss.index")
    parser.add_argument("--output_docs", default="data/documents.json")
    args = parser.parse_args()

    # Load documents
    docs = []
    doc_paths = list(Path(args.documents).glob("*.txt"))
    for path in doc_paths:
        try:
            docs.append(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not docs:
        print(f"No .txt files found in {args.documents}")
        return

    print(f"Loading {len(docs)} documents...")

    # Build index
    # We pass None as the generator because we only need the encoder for indexing
    tier2 = Tier2Engine(
        tier1_engine=None,
        index_path=args.output_index,
        docs_path=args.output_docs
    )
    tier2.add_documents(docs)

    print(f"Index saved to {Path(args.output_index).parent}/")

if __name__ == "__main__":
    main()
