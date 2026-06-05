# scripts/06_hybrid_search.py
import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K_RETRIEVE = 10
TOP_K_DISPLAY = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

print(f"Loading model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

print("Loading local dataset...")
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# Build local BM25 index
print("Building local BM25 index (this may take a moment)...")

def tokenize(text):
    if pd.isna(text):
        return []
    # Lowercase and extract alphanumeric words
    return [word.lower() for word in re.findall(r'\w+', str(text))]

corpus = (df['title'] + " " + df['abstract']).tolist()
tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

id_to_meta = df.set_index('id')[['title', 'abstract', 'year', 'category']].to_dict('index')

# Implement Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    rrf_scores = {}
    
    # Process BM25 ranks
    for item in bm25_results:
        doc_id = item['arxiv_id']
        rank = item['rank']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    # Process Vector ranks
    for item in vector_results:
        doc_id = item['arxiv_id']
        rank = item['rank']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_rrf

# Search Functions

def search_bm25(query_text, top_k=TOP_K_RETRIEVE):
    tokenized_query = tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)
    # Get indices of top scores
    top_n_idx = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_n_idx):
        arxiv_id = str(df.iloc[idx]['id'])
        score = scores[idx]
        results.append({"arxiv_id": arxiv_id, "score": score, "rank": rank + 1})
    return results

def search_vector(query_text, top_k=TOP_K_RETRIEVE):
    query_emb = model.encode(query_text, normalize_embeddings=True).tolist()
    res = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    
    results = []
    for rank, match in enumerate(res['matches']):
        arxiv_id = match['metadata']['arxiv_id']
        score = match['score']
        results.append({"arxiv_id": arxiv_id, "score": score, "rank": rank + 1})
    return results

def print_results(results_list, title, is_rrf=False):
    print(f"\n--- {title} ---")
    for i, item in enumerate(results_list[:TOP_K_DISPLAY]):
        if is_rrf:
            arxiv_id = item[0]
            score_str = f"RRF Score: {item[1]:.4f}"
        else:
            arxiv_id = item['arxiv_id']
            score_str = f"Score: {item['score']:>7.4f}"
            
        meta = id_to_meta.get(arxiv_id, {"title": "Unknown Title"})
        # Truncate title if it's too long for clean output
        title_text = meta['title'][:90] + "..." if len(meta['title']) > 90 else meta['title']
        
        print(f"{i+1}. [{score_str}] {title_text}")

# Run Queries
queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]

for q in queries:
    print(f"\n{'='*80}\nQUERY: '{q}'\n{'='*80}")
    
    bm25_res = search_bm25(q, top_k=TOP_K_RETRIEVE)
    vec_res = search_vector(q, top_k=TOP_K_RETRIEVE)
    hybrid_res = reciprocal_rank_fusion(bm25_res, vec_res, k=60)
    
    # Display Comparisons
    print_results(bm25_res, "Top 5: BM25 (Lexical)")
    print_results(vec_res, "Top 5: Vector Search (Semantic)")
    print_results(hybrid_res, "Top 5: Hybrid Search (RRF)", is_rrf=True)

print("\nHybrid search tests completed successfully!")