import faiss
import pickle
from sentence_transformers import SentenceTransformer

INDEX_PATH = "rag/index.faiss"
DOCS_PATH = "rag/docs.pkl"

def retrieve(query: str, k: int = 3):
    # Load index + docs
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    # Embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])

    # Search
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(docs[idx])

    return results

if __name__ == "__main__":
    query = "What programming languages does Selim know?"
    results = retrieve(query)

    print("Query:", query)
    print("\nTop retrieved chunks:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}\n")