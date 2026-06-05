# scripts/02_embed.py
import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the dataset
print(f"Loading dataset from {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)

# Prepare texts for encoding: title + " [SEP] " + abstract
print("Preparing texts for encoding...")
texts = (df['title'] + " [SEP] " + df['abstract']).tolist()

# Load the pre-trained model
print(f"Loading model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

# Generate embeddings with specific requirements
print("Generating embeddings...")
start_time = time.time()

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

# Output statistics to console
num_processed = len(embeddings)
embedding_dim = embeddings.shape[1]
first_embedding_norm = np.linalg.norm(embeddings[0])

print("\n--- Encoding Statistics ---")
print(f"Total processed texts: {num_processed}")
print(f"Embedding dimensionality: {embedding_dim} (expected 768)")
print(f"Norm of the first embedding: {first_embedding_norm:.4f} (expected ~1.0)")
print(f"Time taken for encoding: {minutes} minutes and {seconds} seconds")

# Save embeddings to numpy format
print(f"\nSaving embeddings to {OUTPUT_FILE}...")
np.save(OUTPUT_FILE, embeddings)
print("Done!")