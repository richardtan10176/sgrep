from llama_cpp import Llama
from packages.sgrep_visitor import sgrepVisitor
import numpy as np
import os



MODEL_PATH = "/Users/richardtan/models/Qwen3-Embedding-0.6B-Q8_0.gguf"
MODEL_INSTRUCT = "Encode this code chunk for retrieval"
QUERY_INSTRUCT = "Retrieve the Python code that matches this query"

llm = Llama(
    model_path = MODEL_PATH,
    embedding  = True,
    n_threads  = 8,
    verbose    = False,
    n_ctx      = 32768,
    n_batch    = 512,
)
def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def search_codebase(query: str, chunks: list, top_k: int = 3):
    """
    Return top k results for given query
    """
    print(f"\nSearching for: '{query}'...")
    query_embedding = get_embedding([query], instruction=QUERY_INSTRUCT)[0]
    results = []
    for chunk in chunks:
        score = compute_cosine_similarity(query_embedding, chunk['embedding'])
        results.append((score, chunk))
    results.sort(key=lambda x: x[0], reverse=True)
    print(f"Top {top_k} Matches:")
    print("-" * 40)
    for i, (score, chunk) in enumerate(results[:top_k]):
        print(f"{i+1}. Score: {score:.4f} | {chunk['context']} (Line {chunk['line_start']})")
        first_line = chunk['code'].split('\n')[0].strip()
        print(f"   Code Preview: {first_line}...")
        print("-" * 40)


def get_embedding(texts: list[str], instruction: str) -> list[np.ndarray]:
    """
    Create embeds from a list of code chunks
    """
    processed_texts = [instruction + text for text in texts]
    embeddings = []
    for text in processed_texts:
        response = llm.create_embedding(text)
        embedding_vector = response['data'][0]['embedding']
        embeddings.append(np.array(embedding_vector))

    return embeddings

def init_path(path: str):
    for root, dirs, files in os.walk(path):
        for file in files:
            print("im in file", file)
            sample_code = ""
            with open(os.path.join(root, file), "r") as f:
                sample_code = f.read()

            visitor = sgrepVisitor()
            parsed_chunks = visitor.parse(sample_code)

            code_texts = [
                f"# Context: {chunk['context']}\n{chunk['code']}"
                for chunk in parsed_chunks
            ]

            print(f"Generating embeddings for {len(code_texts)} code chunks...")
            embeddings = get_embedding(code_texts, instruction=MODEL_INSTRUCT)

            for i, chunk in enumerate(parsed_chunks):
                chunk['embedding'] = embeddings[i]

            return parsed_chunks

user_query = "How do I connect to the database?"
parsed_chunks = init_path("sample_dir")
search_codebase(user_query, parsed_chunks, top_k=5)

