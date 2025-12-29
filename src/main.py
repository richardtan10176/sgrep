from sentence_transformers import SentenceTransformer, util
from packages.sgrep_visitor import sgrepVisitor
import numpy as np
import os

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# automatically uses best API
model = SentenceTransformer(MODEL_NAME)


def search_codebase(query: str, chunks: list, top_k: int = 3):
    """
    Return top k results for given query
    """
    print(f"\nSearching for: '{query}'")

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
        print(f"Code: {first_line}...")
        print("-" * 40)


def init_path(path: str):
    parsed_chunks = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(".py"): continue

            print(f"Parsing file: {file}")
            file_path = os.path.join(root, file)

            with open(file_path, "r", encoding='utf-8') as f:
                sample_code = f.read()

            visitor = sgrepVisitor()
            file_chunks = visitor.parse(sample_code)

            for chunk in file_chunks:
                chunk['file_path'] = file_path
                parsed_chunks.append(chunk)

    if not parsed_chunks:
        print("No code chunks found.")
        return []
    print(f"Generating embeddings for {len(parsed_chunks)} code chunks...")

    texts_to_embed = [
        f"{chunk['context']}\n{chunk['code']}"
        for chunk in parsed_chunks
    ]

    embeddings = model.encode(texts_to_embed, batch_size=64, show_progress_bar=True)

    for i, chunk in enumerate(parsed_chunks):
        chunk['embedding'] = embeddings[i]

    return parsed_chunks


if __name__ == "__main__":
    user_query = "How do I connect to the database"

    target_dir = "sample_dir"
    chunks = init_path(target_dir)
    search_codebase(user_query, chunks, top_k=3)