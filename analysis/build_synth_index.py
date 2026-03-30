import os, json
os.environ['JAX_PLATFORMS'] = 'cpu'
import bm25s, Stemmer

CORP_DIR = os.path.expanduser('~/ReARTeR/analysis/corpus')

corpus_data = []
with open(os.path.join(CORP_DIR, 'synth_corpus.jsonl')) as f:
    for line in f:
        corpus_data.append(json.loads(line.strip()))
print(f'Corpus: {len(corpus_data)} docs')

stemmer = Stemmer.Stemmer('english')
corpus_texts = [d['contents'] for d in corpus_data]

print('Tokenizing...')
corpus_tokens = bm25s.tokenize(corpus_texts, stemmer=stemmer, show_progress=False)
print('Tokenized OK')

print('Indexing...')
retriever = bm25s.BM25(corpus=corpus_data)
retriever.index(corpus_tokens, show_progress=False)
print('Indexed OK')

idx_path = os.path.join(CORP_DIR, 'synth_bm25_index')
os.makedirs(idx_path, exist_ok=True)
retriever.save(idx_path, corpus=corpus_data)
print(f'Saved: {idx_path}')

query_tokens = bm25s.tokenize(['What country won 2018 FIFA World Cup'], stemmer=stemmer)
results, scores = retriever.retrieve(query_tokens, k=2)
for r in results[0]:
    print(' ->', r['contents'][:80])
