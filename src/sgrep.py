#!/usr/bin/env python3

import os
import pickle
import argparse
import hashlib
import sys


CACHE_DIR = os.path.expanduser("~/.cache/sgrep")
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def get_model():
    """lazy loading FTW"""
    print("Loading AI model...")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def get_index_path(target_dir: str) -> str:
    abs_path = os.path.abspath(target_dir)
    path_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"index_{path_hash}.pkl")


def search_codebase(dir: str, query: str, chunks: list, top_k: int = 3):
    from sentence_transformers import util
    import numpy as np

    # Load model just for the query encoding
    model = get_model()

    print(f"\nSearching for: '{query}' in '{os.path.abspath(dir)}'")

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

        display_path = os.path.relpath(chunk['file_path'], os.getcwd())

        print(f"Score: {score:.4f} | {display_path}:{chunk['line_start']}")
        print(f"Context: {chunk['context']}")
        first_line = chunk['code'].split('\n')[0].strip()
        print(f"Code: {first_line}")
        print("-" * 40)


def init_path(path: str, reindex: bool = False):
    from packages.sgrep_visitor import sgrepVisitor

    index_path = get_index_path(path)

    if not reindex and os.path.exists(index_path):
        try:
            with open(index_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            print("Cache corrupted, rebuilding")

    parsed_chunks = []

    print("Parsing codebase...")
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(".py"): continue

            file_path = os.path.abspath(os.path.join(root, file))

            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    sample_code = f.read()

                visitor = sgrepVisitor()
                file_chunks = visitor.parse(sample_code)

                for chunk in file_chunks:
                    chunk['file_path'] = file_path
                    parsed_chunks.append(chunk)
            except Exception as e:
                #there was a syntax error probably since AST failed
                pass

    if not parsed_chunks:
        print("No chunks found!")
        return []

    print(f"Generating embeddings for {len(parsed_chunks)} code chunks...")
    model = get_model()

    texts_to_embed = [
        f"{chunk['context']}\n{chunk['code']}"
        for chunk in parsed_chunks
    ]
    # BATCH SIZE SHOULD BE CONFIGURABLE SINCE NOT ALL HARDWARE IS CREATED EQUAL
    embeddings = model.encode(texts_to_embed, batch_size=64, show_progress_bar=True)

    for i, chunk in enumerate(parsed_chunks):
        chunk['embedding'] = embeddings[i]

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    with open(index_path, "wb") as f:
        pickle.dump(parsed_chunks, f)
        print(f"Index saved to {index_path}")

    return parsed_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Query to search for")
    parser.add_argument("--dir", default="./", help="Directory context to search")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild index")
    parser.add_argument("--top_K", action="store", type=int, default=3, help="Top K matches")
    args = parser.parse_args()

    abs_dir = os.path.abspath(args.dir)

    chunks = init_path(abs_dir, reindex=args.reindex)
    search_codebase(abs_dir, args.query, chunks, top_k=args.top_K)