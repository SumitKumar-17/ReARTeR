"""
Scale Analysis Runner for ReARTeR
Runs Zero-Shot / Naive RAG / ReARTeR Reasoning at a given scale and
appends results to ~/ReARTeR/analysis/README.md
"""
import os, sys, json, time, datetime, argparse, re
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['HF_TOKEN'] = ''

sys.path.insert(0, os.path.expanduser('~/ReARTeR/FlashRAG'))

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.evaluator import Evaluator

ANALYSIS_DIR = os.path.expanduser('~/ReARTeR/analysis')
RESULTS_DIR  = os.path.join(ANALYSIS_DIR, 'results')
README_PATH  = os.path.join(ANALYSIS_DIR, 'README.md')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
def make_config(dataset_name, corpus_name='trivia'):
    """Build config dict for given dataset."""
    corpus_dir = os.path.join(ANALYSIS_DIR, 'corpus')
    if corpus_name == 'trivia' or corpus_name == 'synth':
        index_path  = os.path.join(corpus_dir, 'synth_bm25_index')
        corpus_path = os.path.join(corpus_dir, 'synth_corpus.jsonl')
    else:
        index_path  = os.path.join(corpus_dir, 'bm25_index')
        corpus_path = os.path.join(corpus_dir, 'mini_corpus.jsonl')
    return {
        'model2path': {'e5': ''},
        'model2pooling': {'e5': 'mean'},
        'method2index': {'bm25': index_path},
        'data_dir': os.path.join(ANALYSIS_DIR, 'data') + '/',
        'save_dir': RESULTS_DIR + '/',
        'gpu_id': '0',
        'dataset_name': dataset_name,
        'split': ['dev'],
        'test_sample_num': None,
        'random_sample': False,
        'seed': 2024,
        'save_intermediate_data': True,
        'save_note': f'{dataset_name}_analysis',
        'retrieval_method': 'bm25',
        'retrieval_model_path': '',
        'index_path': index_path,
        'faiss_gpu': False,
        'corpus_path': corpus_path,
        'retrieval_topk': 3,
        'retrieval_batch_size': 32,
        'retrieval_use_fp16': False,
        'retrieval_query_max_length': 128,
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'retrieval_cache_path': None,
        'retrieval_pooling_method': None,
        'bm25_backend': 'bm25s',
        'use_sentence_transformer': False,
        'use_reranker': False,
        'rerank_model_name': None,
        'rerank_model_path': None,
        'rerank_topk': 3,
        'framework': 'openai',
        'generator_model': 'qwen2.5:3b',
        'openai_setting': {
            'api_key': 'ollama',
            'base_url': 'http://localhost:11434/v1',
            'timeout': 120,
        },
        'generator_model_path': None,
        'generator_max_input_len': 4096,
        'generator_batch_size': 1,
        'generation_params': {'max_tokens': 512, 'temperature': 0.7},
        'use_fid': False,
        'gpu_memory_utilization': 0.85,
        'refiner_name': None,
        'refiner_model_path': None,
        'refiner_topk': 5,
        'metrics': ['em', 'f1', 'acc', 'precision', 'recall'],
        'metric_setting': {'retrieval_recall_topk': 3},
        'save_metric_score': True,
    }

# ──────────────────────────────────────────────
def load_data(dataset_name, corpus_name='trivia'):
    cfg = Config(config_dict=make_config(dataset_name, corpus_name))
    splits = get_dataset(cfg)
    return cfg, splits['dev']

def run_zero_shot(dataset_name, corpus_name='trivia'):
    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate
    cfg, data = load_data(dataset_name, corpus_name)
    tmpl = PromptTemplate(cfg,
        system_prompt='Answer the question based on your own knowledge. Give a short, direct answer.',
        user_prompt='Question: {question}\nAnswer:')
    t0 = time.time()
    result = SequentialPipeline(cfg, tmpl).naive_run(data)
    elapsed = time.time() - t0
    metrics = Evaluator(cfg).evaluate(result)
    items = [{'id': it.id, 'question': it.question, 'golden': it.golden_answers, 'pred': it.pred}
             for it in result]
    return {'method': 'zero_shot', 'elapsed': round(elapsed, 2), 'metrics': metrics, 'items': items}

def run_naive_rag(dataset_name, corpus_name='trivia'):
    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate
    cfg, data = load_data(dataset_name, corpus_name)
    tmpl = PromptTemplate(cfg,
        system_prompt='Answer the question based on the provided documents. Give a short, direct answer.',
        user_prompt='Documents:\n{reference}\n\nQuestion: {question}\nAnswer:')
    t0 = time.time()
    result = SequentialPipeline(cfg, tmpl).run(data)
    elapsed = time.time() - t0
    metrics = Evaluator(cfg).evaluate(result)
    items = [{'id': it.id, 'question': it.question, 'golden': it.golden_answers, 'pred': it.pred}
             for it in result]
    return {'method': 'naive_rag', 'elapsed': round(elapsed, 2), 'metrics': metrics, 'items': items}

def run_rearter(dataset_name, corpus_name='trivia'):
    from flashrag.pipeline import ReasoningPipeline
    cfg, data = load_data(dataset_name, corpus_name)
    pipeline = ReasoningPipeline(cfg)
    t0 = time.time()
    items = []
    for i, item in enumerate(data):
        print(f'  [{i+1}/{len(data)}] {item.question[:65]}...', flush=True)
        try:
            steps, bad = pipeline.run_item_mcts(item.question, '')
            pred = steps[-1] if steps else ''
            item.update_output('pred', pred)
            items.append({'id': item.id, 'question': item.question,
                          'golden': item.golden_answers, 'pred': pred,
                          'steps': len(steps), 'bad_gen': bad})
        except Exception as e:
            item.update_output('pred', '')
            items.append({'id': item.id, 'question': item.question,
                          'golden': item.golden_answers, 'pred': '', 'error': str(e)})
    elapsed = time.time() - t0
    metrics = Evaluator(cfg).evaluate(data)
    return {'method': 'rearter', 'elapsed': round(elapsed, 2), 'metrics': metrics, 'items': items}

# ──────────────────────────────────────────────
def save_run(scale, results):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'scale_{scale}_{ts}.json')
    with open(path, 'w') as f:
        json.dump({'scale': scale, 'timestamp': ts, 'results': results}, f, indent=2)
    print(f'Saved: {path}')
    return path

def append_to_readme(scale, results, dataset_name):
    """Append a results section to the README."""
    zs = next(r for r in results if r['method'] == 'zero_shot')
    nr = next(r for r in results if r['method'] == 'naive_rag')
    ra = next(r for r in results if r['method'] == 'rearter')

    def m(r, k): return round(r['metrics'].get(k, 0), 4)

    section = f"""
---

## Scale Experiment: {scale} Questions ({dataset_name})

**Run date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Dataset:** TriviaQA validation (shuffled, seed=42)  
**Corpus:** TriviaQA search contexts (BM25)

| Method | EM ↑ | F1 ↑ | Accuracy ↑ | Precision ↑ | Recall ↑ | Time |
|--------|------|------|------------|-------------|---------|------|
| Zero-Shot | {m(zs,'em')} | {m(zs,'f1')} | {m(zs,'acc')} | {m(zs,'precision')} | {m(zs,'recall')} | {zs['elapsed']}s |
| Naive RAG | {m(nr,'em')} | {m(nr,'f1')} | {m(nr,'acc')} | {m(nr,'precision')} | {m(nr,'recall')} | {nr['elapsed']}s |
| ReARTeR Reasoning | {m(ra,'em')} | {m(ra,'f1')} | {m(ra,'acc')} | {m(ra,'precision')} | {m(ra,'recall')} | {ra['elapsed']}s |

**Best EM:** {'Naive RAG' if m(nr,'em') >= m(ra,'em') else 'ReARTeR'} · 
**Best F1:** {'Naive RAG' if m(nr,'f1') >= m(ra,'f1') else 'ReARTeR'} · 
**Best Recall:** {'Naive RAG' if m(nr,'recall') >= m(ra,'recall') else 'ReARTeR'}

### Sample Predictions ({scale}Q)

| Question (first 60 chars) | Gold | Zero-Shot | Naive RAG | ReARTeR |
|---------------------------|------|-----------|-----------|---------|
"""
    for i in range(min(5, len(zs['items']))):
        q  = zs['items'][i]['question'][:55] + '...'
        g  = str(zs['items'][i]['golden'][0])[:20]
        p0 = str(zs['items'][i]['pred'])[:25].replace('\n',' ')
        p1 = str(nr['items'][i]['pred'])[:25].replace('\n',' ')
        p2 = str(ra['items'][i]['pred'])[:25].replace('\n',' ')
        section += f"| {q} | {g} | {p0} | {p1} | {p2} |\n"

    section += '\n'
    with open(README_PATH, 'a') as f:
        f.write(section)
    print(f'README updated with {scale}Q results')

# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, required=True, help='Number of questions: 50/100/300/500')
    parser.add_argument('--corpus', default='trivia', help='mini or trivia')
    args = parser.parse_args()

    scale = args.scale
    dataset_name = f'synth_{scale}' if args.corpus == 'synth' else f'trivia_{scale}'
    corpus_name  = args.corpus

    print(f'\n{"="*60}')
    print(f'ReARTeR Scale Analysis — {scale} Questions')
    print(f'Dataset: {dataset_name}  |  Corpus: {corpus_name}')
    print(f'Started: {datetime.datetime.now()}')
    print('='*60)

    results = []

    print(f'\n[1/3] Zero-Shot...')
    r1 = run_zero_shot(dataset_name, corpus_name)
    results.append(r1)
    print(f"  EM={r1['metrics'].get('em',0):.3f}  F1={r1['metrics'].get('f1',0):.3f}  time={r1['elapsed']}s")

    print(f'\n[2/3] Naive RAG...')
    r2 = run_naive_rag(dataset_name, corpus_name)
    results.append(r2)
    print(f"  EM={r2['metrics'].get('em',0):.3f}  F1={r2['metrics'].get('f1',0):.3f}  time={r2['elapsed']}s")

    print(f'\n[3/3] ReARTeR Multi-Step Reasoning...')
    r3 = run_rearter(dataset_name, corpus_name)
    results.append(r3)
    print(f"  EM={r3['metrics'].get('em',0):.3f}  F1={r3['metrics'].get('f1',0):.3f}  time={r3['elapsed']}s")

    save_run(scale, results)
    append_to_readme(scale, results, dataset_name)

    print(f'\n{"="*60}')
    print(f'SCALE {scale}Q DONE — {datetime.datetime.now()}')
    print('='*60)
    print(f"{'Method':<25} {'EM':>7} {'F1':>7} {'Acc':>7} {'Time':>8}")
    print('-'*60)
    for r in results:
        m = r['metrics']
        print(f"{r['method']:<25} {m.get('em',0):>7.3f} {m.get('f1',0):>7.3f} {m.get('acc',0):>7.3f} {r['elapsed']:>7.1f}s")

if __name__ == '__main__':
    main()
