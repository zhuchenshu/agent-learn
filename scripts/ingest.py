#!/usr/bin/env python3
"""读 data/doc → 切片 → qwen3-vl-embedding → 写入 Chroma。

用法：
    uv run python scripts/ingest.py
    uv run python scripts/ingest.py --reset
"""

import argparse
import json
import os
from pathlib import Path

import chromadb
import httpx
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("api_key", "")
URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)
MODEL = "qwen3-vl-embedding"
DIM = 1024
CHUNK_SIZE = 250
OVERLAP = 50
BATCH_SIZE = 10
OUTPUT_DIR = ROOT / "data/intermediate"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    keep_separator=True,
)


def split_text(text: str) -> list[str]:
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        response = httpx.post(
            URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "input": {"contents": [{"text": t} for t in texts[i : i + BATCH_SIZE]]},
                "parameters": {"dimension": DIM},
            },
            timeout=60,
        )
        response.raise_for_status()
        vectors += [item["embedding"] for item in response.json()["output"]["embeddings"]]
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="先清空 collection")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("缺少 api_key，请先写入 .env")

    files = [
        path
        for path in sorted((ROOT / "data/doc").rglob("*"))
        if path.suffix.lower() in {".txt", ".md"}
    ]

    ids, chunks, sources = [], [], []
    for path in files:
        pieces = split_text(path.read_text(encoding="utf-8", errors="ignore"))
        ids += [f"{path.name}-{i}" for i in range(len(pieces))]
        chunks += pieces
        sources += [path.name] * len(pieces)
        print(f"{path.name}: {len(pieces)} 个切片")

    if not chunks:
        raise SystemExit("data/doc 下没有 txt/md 文件")

    chunks_file = OUTPUT_DIR / "chunks.jsonl"
    save_jsonl(
        chunks_file,
        [
            {"id": chunk_id, "source": source, "text": text}
            for chunk_id, source, text in zip(ids, sources, chunks)
        ],
    )
    print(f"切片已保存: {chunks_file.relative_to(ROOT)}")

    print(f"共 {len(chunks)} 个切片，开始 embedding")
    vectors = embed(chunks)

    embeddings_file = OUTPUT_DIR / "embeddings.jsonl"
    save_jsonl(
        embeddings_file,
        [{"id": chunk_id, "embedding": vector} for chunk_id, vector in zip(ids, vectors)],
    )
    print(f"embedding 已保存: {embeddings_file.relative_to(ROOT)}")

    client = chromadb.PersistentClient(path=str(ROOT / "data/chroma"))
    if args.reset:
        try:
            client.delete_collection("knowledge")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        "knowledge", metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=[{"source": source} for source in sources],
        embeddings=vectors,
    )
    print(f"完成，collection 共 {collection.count()} 条")


if __name__ == "__main__":
    main()
