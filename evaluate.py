import math
import time
import tracemalloc
import torch

def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ", "") for _ in predictions]

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results

def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res

def ndcg_k(topk_results, k):
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg

def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit

def evaluate_model(predictions, scores, targets, k, metrics, all_items=None):
    # Start tracking time and memory usage
    start_time = time.time()
    tracemalloc.start()

    # Run evaluation functions
    topk_results = get_topk_results(predictions, scores, targets, k, all_items)
    metrics_results = get_metrics_results(topk_results, metrics)

    # Stop tracking time and memory usage
    inference_duration = time.time() - start_time
    memory_used = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # Memory in MB
    tracemalloc.stop()
    gpu_memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Print and log metrics
    print(f"Inference Time: {inference_duration:.2f} seconds")
    print(f"Memory Used (CPU): {memory_used:.2f} MB")
    print(f"GPU Memory Used: {gpu_memory_used:.2f} MB")
    print(f"Evaluation Metrics: {metrics_results}")

    # Save results to a file
    with open("inference_baseline_metrics.txt", "w") as f:
        f.write(f"Inference Time: {inference_duration:.2f} seconds\n")
        f.write(f"Memory Used (CPU): {memory_used:.2f} MB\n")
        f.write(f"GPU Memory Used: {gpu_memory_used:.2f} MB\n")
        f.write(f"Evaluation Metrics: {metrics_results}\n")

    return metrics_results
