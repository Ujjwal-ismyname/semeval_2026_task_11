import json
import random
import os
import argparse
from collections import Counter

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def run_baselines(data_path, output_dir):
    data = load_data(data_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analyze Distribution
    labels = [item['validity'] for item in data]
    counts = Counter(labels)
    majority_label = counts.most_common(1)[0][0]
    print(f"Dataset Size: {len(data)}")
    print(f"Class Distribution: {counts}")
    print(f"Majority Label: {majority_label} ({counts[majority_label]/len(data):.2%})")
    
    # 2. Majority Baseline
    majority_preds = []
    for item in data:
        majority_preds.append({
            "id": item['id'],
            "validity": majority_label,
            # For Subtask 2, majority baseline for retrieval is tricky. 
            # We'll just return empty list as "safe" baseline or maybe first sentence?
            # Let's return empty list for now.
            "relevant_premises": []
        })
        
    maj_path = os.path.join(output_dir, "baseline_majority.json")
    with open(maj_path, 'w') as f:
        json.dump(majority_preds, f, indent=4)
    print(f"Saved Majority Baseline to {maj_path}")

    # 3. Random Baseline
    random_preds = []
    for item in data:
        random_preds.append({
            "id": item['id'],
            "validity": random.choice([True, False]),
            "relevant_premises": []
        })
        
    rand_path = os.path.join(output_dir, "baseline_random.json")
    with open(rand_path, 'w') as f:
        json.dump(random_preds, f, indent=4)
    print(f"Saved Random Baseline to {rand_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="semeval_2026_task_11/data/raw/train_data.json")
    parser.add_argument("--output_dir", default="semeval_2026_task_11/output")
    args = parser.parse_args()
    
    run_baselines(args.data_path, args.output_dir)
