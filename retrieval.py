import faiss
import numpy as np
from embedding import embed_documents, embed_query


class Retriever:

    def __init__(self, chunks):

        self.chunks = chunks
        self.texts = [c["content"] for c in chunks]

        print("Creating embeddings...")

        self.embeddings = embed_documents(self.texts).astype("float32")

        # normalize vector (rất quan trọng khi dùng cosine similarity)
        faiss.normalize_L2(self.embeddings)

        self.dimension = self.embeddings.shape[1]
        print("Vector dimension:", self.dimension)

        # cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(self.embeddings)

        print("Index vector count:", self.index.ntotal)

    def retrieve(self, query, top_k=3):

        # embed query
        query_vector = embed_query(query).astype("float32")

        # reshape (FAISS cần 2D vector)
        query_vector = query_vector.reshape(1, -1)

        # normalize
        faiss.normalize_L2(query_vector)

        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, k)

        results = []

        for i, idx in enumerate(indices[0]):

            results.append({
                "score": float(scores[0][i]),
                "id": int(idx),
                "content": self.chunks[idx]["content"],
                "metadata": self.chunks[idx].get("metadata", {})
            })

        return results