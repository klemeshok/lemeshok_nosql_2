# scripts/04_search.py
import os
import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

print(f"Loading model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

print("Loading local dataset and embeddings...")
df = pd.read_parquet("data/arxiv_subset.parquet")
local_embeddings = np.load("embeddings/embeddings.npy")

# Function to encode query
def encode_query(query_text):
    return model.encode(query_text, normalize_embeddings=True)

def print_pinecone_results(response, title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")
    if not response['matches']:
        print("No results found matching the criteria.")
        return
        
    for i, match in enumerate(response['matches']):
        score = match['score']
        meta = match['metadata']
        arxiv_id = meta['arxiv_id']
        
        full_abstract = df[df['id'] == arxiv_id]['abstract'].values[0]
        snippet = full_abstract[:150] + "..."
        
        print(f"{i+1}. [Score: {score:.4f}] {meta['title']}")
        print(f"   Year: {meta['year']} | Category: {meta['category']} | ID: {arxiv_id}")
        print(f"   Abstract: {snippet}\n")

# Pure Semantic Search
query_1 = "teaching machines to recognize objects in pictures"
print(f"\nExecuting pure semantic search for: '{query_1}'")
q_emb_1 = encode_query(query_1).tolist()
res_1 = index.query(vector=q_emb_1, top_k=TOP_K, include_metadata=True)
print_pinecone_results(res_1, "Pure Semantic Search Results")

# Search with Filtering
query_2 = "reinforcement learning"
print(f"\nExecuting filtered search for: '{query_2}'")
q_emb_2 = encode_query(query_2).tolist()

# RL, last 5 years, category cs.LG
current_year = datetime.datetime.now().year
target_year = current_year - 4

filter_a = {
    "year": {"$gte": target_year},
    "category": {"$eq": "cs.LG"}
}
res_2a = index.query(vector=q_emb_2, top_k=TOP_K, include_metadata=True, filter=filter_a)
print_pinecone_results(res_2a, f"Filtered Search A (>={target_year}, cs.LG)")

# RL, older articles (< 2015), any category
filter_b = {
    "year": {"$lt": 2015}
}
res_2b = index.query(vector=q_emb_2, top_k=TOP_K, include_metadata=True, filter=filter_b)
print_pinecone_results(res_2b, "Filtered Search B (< 2015, any category)")

# Local Metrics Comparison
print(f"\n{'='*50}\nLocal Metrics Comparison\n{'='*50}")
query_3 = "quantum computing in cryptography"
print(f"Query: '{query_3}'")
q_emb_3 = encode_query(query_3)

# Calculate Dot Product
dot_scores = np.dot(local_embeddings, q_emb_3)
dot_top_k_idx = np.argsort(dot_scores)[::-1][:TOP_K]

# Calculate Cosine Similarity
norms = np.linalg.norm(local_embeddings, axis=1) * np.linalg.norm(q_emb_3)
cos_scores = dot_scores / norms
cos_top_k_idx = np.argsort(cos_scores)[::-1][:TOP_K]

# Calculate L2 Distance
l2_dists = np.linalg.norm(local_embeddings - q_emb_3, axis=1)
l2_top_k_idx = np.argsort(l2_dists)[:TOP_K]

def print_local_results(indices, values, metric_name, is_distance=False):
    print(f"\n--- Top 5 by {metric_name} ---")
    label = "Distance" if is_distance else "Score"
    for i, idx in enumerate(indices):
        val = values[idx]
        title = df.iloc[idx]['title']
        print(f"{i+1}. [{label}: {val:.4f}] {title[:80]}...")

print_local_results(dot_top_k_idx, dot_scores, "Dot Product", is_distance=False)
print_local_results(cos_top_k_idx, cos_scores, "Cosine Similarity", is_distance=False)
print_local_results(l2_top_k_idx, l2_dists, "L2 Distance", is_distance=True)

print("\nSearch tests completed successfully!")