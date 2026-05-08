import streamlit as st
from sentence_transformers import SentenceTransformer

import torch

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    print("Loading BGE-M3...")
    # Kiểm tra xem PyTorch có nhận diện được Card màn hình không
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Trạng thái GPU cho Embedding: Đang dùng {device.upper()}")
    
    model = SentenceTransformer(
        "BAAI/bge-m3",
        trust_remote_code=True,
        local_files_only=True,
        device=device
    )
    print("Model loaded!")
    return model

model = load_embedding_model()


def embed_documents(texts):
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )


def embed_query(query):
    return model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )