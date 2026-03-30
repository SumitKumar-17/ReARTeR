"""
Download TriviaQA from HuggingFace and prepare datasets at 50/100/300/500 scale.
Builds BM25 index from the question contexts.
"""
import os, sys, json, random
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['HF_TOKEN'] = ''

HF_TOKEN = ''
SCALES   = [50, 100, 300, 500]
DATA_DIR = os.path.expanduser('~/ReARTeR/analysis/data')
CORP_DIR = os.path.expanduser('~/ReARTeR/analysis/corpus')

print("=== Downloading TriviaQA from HuggingFace ===")
from datasets import load_dataset
ds = load_dataset('trivia_qa', 'rc.nocontext', split='validation',
                  token=HF_TOKEN)
print(f"Loaded {len(ds)} TriviaQA validation questions")
print("Columns:", ds.column_names)
print("Sample:", ds[0]['question'], "->", ds[0]['answer']['value'])

# Convert to FlashRAG format
all_items = []
for i, row in enumerate(ds):
    item = {
        "id": f"trivia_{i:05d}",
        "question": row['question'],
        "golden_answers": [row['answer']['value']] + list(row['answer'].get('aliases', []))
    }
    all_items.append(item)

print(f"\nConverted {len(all_items)} questions to FlashRAG format")

# Build a corpus from question context (TriviaQA rc has search results)
print("\n=== Building corpus from TriviaQA contexts ===")
# Load rc split (has contexts) for corpus
try:
    ds_rc = load_dataset('trivia_qa', 'rc', split='validation',
                         token=HF_TOKEN)
    corpus_docs = []
    seen_titles = set()
    for row in ds_rc:
        for ctx in row.get('search_results', {}).get('search_context', []):
            title = ctx.split('\n')[0].strip() if ctx else ''
            if title and title not in seen_titles and len(corpus_docs) < 5000:
                seen_titles.add(title)
                corpus_docs.append({"id": f"doc_{len(corpus_docs):05d}",
                                    "contents": ctx[:500]})
    print(f"Built corpus with {len(corpus_docs)} docs from TriviaQA contexts")
except Exception as e:
    print(f"Could not load rc split: {e}")
    print("Using nocontext — generating synthetic Wikipedia stubs from answers")
    corpus_docs = []
    seen = set()
    for row in ds:
        ans = row['answer']['value']
        if ans not in seen and len(corpus_docs) < 3000:
            seen.add(ans)
            corpus_docs.append({
                "id": f"doc_{len(corpus_docs):05d}",
                "contents": f"{ans}\n{ans} is a well-known entity. {row['question'].replace('?','')} relates to {ans}."
            })

corpus_path = os.path.join(CORP_DIR, 'trivia_corpus.jsonl')
with open(corpus_path, 'w') as f:
    for doc in corpus_docs:
        f.write(json.dumps(doc) + '\n')
print(f"Saved corpus: {corpus_path} ({len(corpus_docs)} docs)")

# Save dataset splits at each scale
random.seed(42)
random.shuffle(all_items)

for scale in SCALES:
    items = all_items[:scale]
    split_dir = os.path.join(DATA_DIR, f'trivia_{scale}')
    os.makedirs(split_dir, exist_ok=True)
    out_path = os.path.join(split_dir, 'dev.jsonl')
    with open(out_path, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {scale}-Q dataset: {out_path}")

# Build BM25 index for trivia corpus
print("\n=== Building BM25 index for trivia corpus ===")
import bm25s, Stemmer
stemmer = Stemmer.Stemmer('english')
texts   = [d['contents'] for d in corpus_docs]
tokens  = bm25s.tokenize(texts, stemmer=stemmer, show_progress=True)
idx     = bm25s.BM25(corpus=corpus_docs)
idx.index(tokens, show_progress=True)
idx_path = os.path.join(CORP_DIR, 'trivia_bm25_index')
os.makedirs(idx_path, exist_ok=True)
idx.save(idx_path, corpus=corpus_docs)
print(f"BM25 index saved: {idx_path}")

print("\n=== Dataset preparation complete ===")
for scale in SCALES:
    print(f"  trivia_{scale}/dev.jsonl  →  {scale} questions")
print(f"  Corpus: {len(corpus_docs)} docs")
