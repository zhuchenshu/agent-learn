#!/usr/bin/env python3
"""检索测试：query → embedding → Chroma top-k。

用法：
    uv run python scripts/search.py 黄果树瀑布怎么安排
    uv run python scripts/search.py 咖啡推荐 --top-k 3
"""

import argparse

import chromadb

from ingest import ROOT, embed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="检索问题")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    collection = chromadb.PersistentClient(path=str(ROOT / "data/chroma")).get_collection(
        "knowledge"
    )
    result = collection.query(
        query_embeddings=[embed([args.query])[0]],
        n_results=args.top_k,
        include=["documents", "metadatas", "distances"],
    )

    print(f"query: {args.query}\n")
    for i, (doc, meta, distance) in enumerate(
        zip(result["documents"][0], result["metadatas"][0], result["distances"][0]), 1
    ):
        print(f"[{i}] score={1 - distance:.3f}  {meta['source']}")
        print(doc)
        print("-" * 60)


if __name__ == "__main__":
    main()
