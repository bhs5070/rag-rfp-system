import json
import numpy as np
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ==============================================
# CONFIG
# ==============================================
CHUNK_FILE = "/home/bhs1581/rag-rfp-system/chunking/chunks/chunks_multi_aspect (1).jsonl"
EMBED_PATH = "/home/bhs1581/rag-rfp-system/scenario_a/embeddings.npy"
META_PATH = "/home/bhs1581/rag-rfp-system/scenario_a/metadata.pkl"

print("Loading bge-m3 model...")
model = SentenceTransformer(
    "BAAI/bge-m3",
    trust_remote_code=True
)

# ==============================================
# 1) Load chunks
# ==============================================
chunks = []
with open(CHUNK_FILE, "r") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"Loaded chunks: {len(chunks)}")

# ==============================================
# 2) Extract text
# ==============================================
texts = []
metadata = []

for i, c in enumerate(chunks):
    txt = c.get("text", "")
    texts.append(txt)

    metadata.append({
        "id": i,
        "chunk_id": c.get("chunk_id", i),
        "filename": c.get("filename", ""),
        "doc_id": c.get("doc_id", "")
    })

print("Prepared text + metadata.")

# ==============================================
# 3) Embedding
# ==============================================
print("Embedding...")
embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding completed:", embeddings.shape)

# ==============================================
# 4) Save
# ==============================================
np.save(EMBED_PATH, embeddings)
with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("Saved embeddings to:", EMBED_PATH)
print("Saved metadata to:", META_PATH)
