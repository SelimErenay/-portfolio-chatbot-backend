import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

RAW_DIR = "data/raw"
INDEX_PATH = "rag/index.faiss"
DOCS_PATH = "rag/docs.pkl"

def load_documents():
    docs = []
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(RAW_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
            print(f"Loaded {filename}")
    return docs

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def main():
    print("Starting ingestion...")

    docs = load_documents()
    if not docs:
        print("No documents found.")
        return

    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    print(f"Created {len(chunks)} chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("Ingestion complete")
    print(f"Saved FAISS index → {INDEX_PATH}")
    print(f"Saved docs → {DOCS_PATH}")

if __name__ == "__main__":
    main()