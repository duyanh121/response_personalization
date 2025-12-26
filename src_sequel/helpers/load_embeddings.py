import torch
import os
import pickle
import faiss

def load_embeddings(embeddings_dir="../embeddings"):
    """
    Loads:
    - FAISS index
    - metadata records (author + content)
    - embeddings tensor (optional but useful)

    Returns:
    - index: FAISS index
    - records: list[dict]
    - embeddings: torch.Tensor
    """

    index_path = os.path.join(embeddings_dir, "style_index.faiss")
    meta_path = os.path.join(embeddings_dir, "style_metadata.pkl")
    emb_path = os.path.join(embeddings_dir, "style_embeddings.pt")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load metadata
    with open(meta_path, "rb") as f:
        records = pickle.load(f)

    # Load embeddings
    embeddings = torch.load(emb_path, map_location="cpu")

    assert index.ntotal == len(records) == embeddings.shape[0], \
        "Mismatch between index, metadata, and embeddings"

    return index, records, embeddings
