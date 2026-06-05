# scripts/03_load_to_pinecone.py
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone recommends batches up to 200 vectors

# Initialize client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create index (if it does not exist)
print(f"Checking if index '{INDEX_NAME}' exists...")
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating index '{INDEX_NAME}'... (this might take a minute)")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    # Wait until the index is fully ready
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)
    print("Index created successfully.")
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load data
print(f"Loading data from {INPUT_PARQUET}...")
df = pd.read_parquet(INPUT_PARQUET)

print(f"Loading embeddings from {INPUT_EMBEDDINGS}...")
embeddings = np.load(INPUT_EMBEDDINGS)

# Prepare and upload data in batches
print("Preparing and uploading vectors to Pinecone (this may take a few minutes)...")
start_time = time.time()

total_records = len(df)

for i in tqdm(range(0, total_records, BATCH_SIZE), desc="Uploading batches"):
    batch_df = df.iloc[i:i+BATCH_SIZE]
    batch_embeddings = embeddings[i:i+BATCH_SIZE]
    
    vectors_to_upsert = []
    
    for j, (row_idx, row) in enumerate(batch_df.iterrows()):
        # Calculate absolute index for the unique ID
        abs_idx = i + j
        vector_id = f"paper_{abs_idx}"
        
        # Truncate metadata to avoid exceeding Pinecone size limits
        abstract_trunc = str(row["abstract"])[:500]
        authors_trunc = str(row["authors"])[:200]
        
        metadata = {
            "arxiv_id": str(row["id"]),
            "title": str(row["title"]),
            "abstract": abstract_trunc,
            "authors": authors_trunc,
            "year": int(row["year"]),
            "category": str(row["category"])
        }
        
        vectors_to_upsert.append({
            "id": vector_id,
            "values": batch_embeddings[j].tolist(),
            "metadata": metadata
        })
        
    index.upsert(vectors=vectors_to_upsert)

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

# Display final vector count
time.sleep(3) # Give Pinecone some time to update its stats
stats = index.describe_index_stats()
print("\n--- Upload Statistics ---")
print(f"Total vectors in index '{INDEX_NAME}': {stats.total_vector_count}")
print(f"Time taken for upload: {minutes} minutes and {seconds} seconds")