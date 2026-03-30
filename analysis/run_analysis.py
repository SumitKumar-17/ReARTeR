"""
ReARTeR Small-Scale Analysis Script
Runs the ReasoningPipeline on 15 multi-hop questions using:
- BM25 retrieval (mini corpus, 35 docs)
- Ollama (qwen2.5:3b) as generator via OpenAI-compatible API

Compares 3 methods:
1. Zero-shot (no retrieval)
2. Naive RAG (single retrieval step)
3. ReARTeR Reasoning (multi-step reasoning with retrieval)
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Use CPU for JAX/BM25; GPU is handled by Ollama

import sys
import os
import json
import time
import datetime

# Add FlashRAG to path
sys.path.insert(0, os.path.expanduser("~/ReARTeR/FlashRAG"))

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.evaluator import Evaluator

CONFIG_PATH = os.path.expanduser("~/ReARTeR/analysis/analysis_config.yaml")
RESULTS_DIR = os.path.expanduser("~/ReARTeR/analysis/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_results(results_dict, method_name):
    """Save results to JSON file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{method_name}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Results saved to: {path}")
    return path

def run_zero_shot(config, test_data):
    """Baseline: Answer questions without any retrieval"""
    print("\n" + "="*60)
    print("METHOD 1: Zero-Shot (no retrieval)")
    print("="*60)
    
    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate

    template = PromptTemplate(
        config=config,
        system_prompt="Answer the question based on your own knowledge. Give a short, direct answer.",
        user_prompt="Question: {question}\nAnswer:",
    )
    pipeline = SequentialPipeline(config, template)
    
    start = time.time()
    result = pipeline.naive_run(test_data)
    elapsed = time.time() - start
    
    # Collect per-item results
    items = []
    for item in result:
        items.append({
            "id": item.id,
            "question": item.question,
            "golden_answers": item.golden_answers,
            "pred": item.pred,
        })
    
    evaluator = Evaluator(config)
    eval_result = evaluator.evaluate(result)
    
    summary = {
        "method": "zero_shot",
        "elapsed_seconds": round(elapsed, 2),
        "metrics": eval_result,
        "items": items
    }
    print(f"  Time: {elapsed:.1f}s | Metrics: {eval_result}")
    save_results(summary, "zero_shot")
    return summary

def run_naive_rag(config, test_data):
    """Naive RAG: Single retrieval step, then answer"""
    print("\n" + "="*60)
    print("METHOD 2: Naive RAG (single retrieval + answer)")
    print("="*60)
    
    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate
    
    template = PromptTemplate(
        config=config,
        system_prompt="Answer the question based on the provided documents. Give a short, direct answer.",
        user_prompt="Documents:\n{reference}\n\nQuestion: {question}\nAnswer:",
    )
    pipeline = SequentialPipeline(config, template)
    
    start = time.time()
    result = pipeline.run(test_data)
    elapsed = time.time() - start
    
    items = []
    for item in result:
        retrieval = item.retrieval_result if hasattr(item, 'retrieval_result') else []
        items.append({
            "id": item.id,
            "question": item.question,
            "golden_answers": item.golden_answers,
            "pred": item.pred,
            "retrieved_docs": [r.get("contents","")[:150] for r in retrieval] if retrieval is not None and len(retrieval) > 0 else [],
        })
    
    evaluator = Evaluator(config)
    eval_result = evaluator.evaluate(result)
    
    summary = {
        "method": "naive_rag",
        "elapsed_seconds": round(elapsed, 2),
        "metrics": eval_result,
        "items": items
    }
    print(f"  Time: {elapsed:.1f}s | Metrics: {eval_result}")
    save_results(summary, "naive_rag")
    return summary

def run_rearter_reasoning(config, test_data):
    """ReARTeR: Multi-step reasoning with iterative retrieval"""
    print("\n" + "="*60)
    print("METHOD 3: ReARTeR Multi-Step Reasoning")
    print("="*60)
    
    from flashrag.pipeline import ReasoningPipeline
    
    pipeline = ReasoningPipeline(config)
    
    items = []
    start = time.time()
    
    for i, item in enumerate(test_data):
        print(f"\n  [{i+1}/{len(test_data)}] Q: {item.question[:70]}...")
        try:
            item_start = time.time()
            reasoning_steps, bad_gen = pipeline.run_item_mcts(item.question, "")
            item_elapsed = time.time() - item_start
            
            # Final prediction is the last step
            pred = reasoning_steps[-1] if reasoning_steps else ""
            item.update_output("pred", pred)
            item.update_output("reasoning_steps", reasoning_steps)
            
            print(f"    Steps: {len(reasoning_steps)} | Bad gen: {bad_gen} | Time: {item_elapsed:.1f}s")
            print(f"    Prediction: {str(pred)[:100]}")
            
            items.append({
                "id": item.id,
                "question": item.question,
                "golden_answers": item.golden_answers,
                "pred": pred,
                "reasoning_steps": reasoning_steps,
                "bad_gen": bad_gen,
                "time_seconds": round(item_elapsed, 2)
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            item.update_output("pred", "")
            items.append({
                "id": item.id,
                "question": item.question,
                "golden_answers": item.golden_answers,
                "pred": "",
                "error": str(e)
            })
    
    elapsed = time.time() - start
    
    evaluator = Evaluator(config)
    eval_result = evaluator.evaluate(test_data)
    
    summary = {
        "method": "rearter_reasoning",
        "elapsed_seconds": round(elapsed, 2),
        "metrics": eval_result,
        "items": items
    }
    print(f"\n  Total Time: {elapsed:.1f}s | Metrics: {eval_result}")
    save_results(summary, "rearter_reasoning")
    return summary

def print_comparison(results):
    """Print a comparison table of all methods"""
    print("\n" + "="*70)
    print("RESULTS COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Method':<30} {'EM':>6} {'F1':>6} {'Acc':>6} {'Time(s)':>8}")
    print("-"*70)
    for r in results:
        m = r["metrics"]
        em = m.get("em", m.get("EM", 0))
        f1 = m.get("f1", m.get("F1", 0))
        acc = m.get("acc", m.get("ACC", 0))
        t = r["elapsed_seconds"]
        print(f"{r['method']:<30} {em:>6.3f} {f1:>6.3f} {acc:>6.3f} {t:>8.1f}")
    print("="*70)
    
    # Save comparison
    comp_path = os.path.join(RESULTS_DIR, "comparison_summary.json")
    with open(comp_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison saved to: {comp_path}")

def main():
    print("ReARTeR Small-Scale Analysis")
    print(f"Started: {datetime.datetime.now()}")
    
    # Load config and dataset
    config = Config(CONFIG_PATH, {})
    all_split = get_dataset(config)
    test_data = all_split["dev"]
    
    print(f"\nDataset: {len(test_data)} questions loaded")
    print(f"Retrieval: BM25 (bm25s) with mini corpus")
    print(f"Generator: {config['generator_model']} via Ollama")
    
    results = []
    
    # Run each method (reload fresh dataset for each)
    config1 = Config(CONFIG_PATH, {})
    all_split1 = get_dataset(config1)
    r1 = run_zero_shot(config1, all_split1["dev"])
    results.append(r1)
    
    config2 = Config(CONFIG_PATH, {})
    all_split2 = get_dataset(config2)
    r2 = run_naive_rag(config2, all_split2["dev"])
    results.append(r2)
    
    config3 = Config(CONFIG_PATH, {})
    all_split3 = get_dataset(config3)
    r3 = run_rearter_reasoning(config3, all_split3["dev"])
    results.append(r3)
    
    print_comparison(results)
    print(f"\nFinished: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()
