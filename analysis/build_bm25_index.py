"""Build BM25 index from our mini corpus"""
import json
import os
import bm25s
import Stemmer

corpus_path = os.path.expanduser("~/ReARTeR/analysis/corpus/mini_corpus.jsonl")
index_path = os.path.expanduser("~/ReARTeR/analysis/corpus/bm25_index")

# Load corpus
corpus_data = []
with open(corpus_path) as f:
    for line in f:
        doc = json.loads(line.strip())
        corpus_data.append(doc)

print(f"Loaded {len(corpus_data)} documents")

# Tokenize
stemmer = Stemmer.Stemmer('english')
corpus_texts = [doc['contents'] for doc in corpus_data]
corpus_tokens = bm25s.tokenize(corpus_texts, stemmer=stemmer, show_progress=True)

# Build index
retriever = bm25s.BM25(corpus=corpus_data)
retriever.index(corpus_tokens, show_progress=True)

# Save
os.makedirs(index_path, exist_ok=True)
retriever.save(index_path, corpus=corpus_data)
print(f"BM25 index saved to {index_path}")

# Test retrieval
query = "Who was president when Berlin Wall fell"
query_tokens = bm25s.tokenize([query], stemmer=stemmer)
results, scores = retriever.retrieve(query_tokens, k=3)
print(f"\nTest retrieval for: '{query}'")
for r, s in zip(results[0], scores[0]):
    print(f"  Score {s:.3f}: {r['contents'][:100]}")
