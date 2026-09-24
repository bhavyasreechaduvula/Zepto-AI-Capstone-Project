from pathlib import Path


from sentence_transformers import SentenceTransformer
import chromadb


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"


def load_documents():
    documents = []
    ids = []

    for file_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append(text)
            ids.append(file_path.stem)

    return ids, documents


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def main():
    ids, documents = load_documents()

    all_chunks = []
    all_ids = []

    for doc_id, document in zip(ids, documents):
        chunks = chunk_text(document)

        for number, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_id}_chunk_{number}")

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(all_chunks)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        all_chunks,
        normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name="zepto_policies"
    )

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
    )

    print("Embeddings created successfully.")
    print("ChromaDB collection created successfully.")
    print(f"Stored chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()