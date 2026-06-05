# scripts/05_chunking.py
import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

print(f"Loading model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

print("Loading local dataset...")
df = pd.read_parquet("data/arxiv_subset.parquet")

# Select the 30 articles with the longest abstracts
print("Selecting top 30 longest abstracts...")
df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))
top_30_df = df.nlargest(30, 'abstract_word_count').copy()

# Define chunking strategies
def fixed_size_chunking(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    if not words: return chunks
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def semantic_chunking(text, max_words=50):
    sentences = re.split(r'(?<=[.?!])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)
        
        if current_word_count + sentence_len > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_len
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# Create Pinecone Indexes
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

def ensure_index_exists(index_name):
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index '{index_name}' is ready.")
    else:
        print(f"Index '{index_name}' already exists.")

ensure_index_exists(INDEX_FIXED)
ensure_index_exists(INDEX_SEMANTIC)

index_fixed = pc.Index(INDEX_FIXED)
index_semantic = pc.Index(INDEX_SEMANTIC)

# Prepare, Encode, and Upsert Chunks
def process_and_upload(chunking_method, index_obj, method_name):
    print(f"\nProcessing {method_name} chunks...")
    vectors_to_upsert = []
    
    for _, row in top_30_df.iterrows():
        arxiv_id = str(row['id'])
        abstract = str(row['abstract'])
        
        chunks = chunking_method(abstract)
        embeddings = model.encode(chunks, normalize_embeddings=True)
        
        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{arxiv_id}_{method_name}_{i}"
            
            metadata = {
                "arxiv_id": arxiv_id,
                "title": str(row['title']),
                "chunk_text": chunk_text,
                "chunk_index": i,
                "year": int(row['year']),
                "category": str(row['category'])
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": emb.tolist(),
                "metadata": metadata
            })

    # Batch upload to Pinecone
    BATCH_SIZE = 200
    for i in tqdm(range(0, len(vectors_to_upsert), BATCH_SIZE), desc=f"Uploading {method_name}"):
        batch = vectors_to_upsert[i:i + BATCH_SIZE]
        index_obj.upsert(vectors=batch)
        
    print(f"Total {method_name} chunks uploaded: {len(vectors_to_upsert)}")

process_and_upload(fixed_size_chunking, index_fixed, "fixed")
process_and_upload(semantic_chunking, index_semantic, "semantic")

# Search Functionality
def test_search(query_text):
    print(f"\n{'='*60}\nSEARCH QUERY: '{query_text}'\n{'='*60}")
    
    query_emb = model.encode(query_text, normalize_embeddings=True).tolist()
    
    # Query Fixed Index
    res_fixed = index_fixed.query(vector=query_emb, top_k=5, include_metadata=True)
    print("\n--- Top 5 Results: FIXED CHUNKING ---")
    for i, match in enumerate(res_fixed['matches']):
        meta = match['metadata']
        print(f"{i+1}. [Score: {match['score']:.4f}] {meta['title']}")
        print(f"   Chunk {meta['chunk_index']}: {meta['chunk_text'][:100]}...\n")

    # Query Semantic Index
    res_sem = index_semantic.query(vector=query_emb, top_k=5, include_metadata=True)
    print("--- Top 5 Results: SEMANTIC CHUNKING ---")
    for i, match in enumerate(res_sem['matches']):
        meta = match['metadata']
        print(f"{i+1}. [Score: {match['score']:.4f}] {meta['title']}")
        print(f"   Chunk {meta['chunk_index']}: {meta['chunk_text'][:100]}...\n")

# Run search tests
time.sleep(3)
test_search("challenges in quantum physics algorithms")
test_search("natural language processing and understanding")

print("\nChunking and search operations complete!")