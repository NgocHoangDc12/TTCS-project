import streamlit as st
from sentence_transformers import CrossEncoder

import torch

@st.cache_resource(show_spinner=False)
def load_reranker_model():
    print("Loading reranker...")
    # Kiểm tra trạng thái GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Trạng thái GPU cho Reranker: Đang dùng {device.upper()}")
    
    reranker = CrossEncoder(
        "BAAI/bge-reranker-base",
        local_files_only=True,
        device=device
    )
    print("Reranker loaded!")
    return reranker

reranker = load_reranker_model()

def rerank(query, results, top_k=3):

    pairs = []

    for r in results:
        pairs.append((query, r["content"]))

    scores = reranker.predict(pairs)

    for i in range(len(results)):
        results[i]["rerank_score"] = float(scores[i])

    results.sort(key=lambda x: x["rerank_score"], reverse=True)

    return results[:top_k]