#!/usr/bin/env python3

import os
import pickle
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, util
from packages.sgrep_visitor import sgrepVisitor

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CACHE_DIR = os.path.expanduser("~/.cache/sgrep")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")

model = SentenceTransformer(MODEL_NAME)


def search_codebase(dir: str, query: str, chunks: list, top_k: int = 3):
    print(f"\nSearching for: '{query}' in '{dir}'")

    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_embeddings = np.stack([c['embedding'] for c in chunks])
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k)
    results = hits[0]

    print(f"Top {top_k} Matches:")
    print("-" * 40)
    for hit in results:
        idx = hit['corpus_id']
        score = hit['score']
        chunk = chunks[idx]

        print(f"Score: {score:.4f} | Line {chunk['line_start']}")
        print(f"Context: {chunk['context']}")
        first_line = chunk['code'].split('\n')[0].strip()
        print(f"Code: {first_line}")
        print("-" * 40)


def init_path(path: str, reindex: bool = False):
    if not reindex and os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "rb") as f:
                print("Loading index from cache")
                return pickle.load(f)
        except Exception:
            print("Cache corrupted, rebuilding")

    parsed_chunks = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(".py"): continue

            print(f"chunking: {file}")
            file_path = os.path.join(root, file)

            with open(file_path, "r", encoding='utf-8') as f:
                sample_code = f.read()

            visitor = sgrepVisitor()
            file_chunks = visitor.parse(sample_code)

            for chunk in file_chunks:
                chunk['file_path'] = file_path
                parsed_chunks.append(chunk)

    if not parsed_chunks:
        print("No chunks found!")
        return []

    print(f"Generating embeddings for {len(parsed_chunks)} code chunks")

    texts_to_embed = [
        f"{chunk['context']}\n{chunk['code']}"
        for chunk in parsed_chunks
    ]

    embeddings = model.encode(texts_to_embed, batch_size=64, show_progress_bar=True)

    for i, chunk in enumerate(parsed_chunks):
        chunk['embedding'] = embeddings[i]

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(parsed_chunks, f)
        print(f"Index saved to {INDEX_PATH}")

    return parsed_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Query to search for")
    parser.add_argument("--dir", default="./", help="Directory context to search")
    parser.add_argument("--reindex", action="store_true", help="Flag to tell sgrep to rebuild the index (must be set following any code change)")
    parser.add_argument("--top_K", action="store", type=int, default=3, help="Top K matches to return")
    args = parser.parse_args()

    chunks = init_path(args.dir, reindex=args.reindex)
    search_codebase(args.dir, args.query, chunks, top_k=args.top_K)