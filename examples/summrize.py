import json
import math
import numpy as np
from itertools import chain
import os
from typing import Literal
import argparse
from scipy import stats

from scipy.stats import norm
def mean_and_conf(data):
    mean = np.mean(data)
    n = len(data)
    se = stats.sem(data)
    confidence_interval = stats.norm.interval(0.95,loc=mean, scale=se)
    return np.mean(data), confidence_interval

def get_score(input_path: str, 
              data_type: Literal["post", "pre"], 
              metric: Literal["accuracy", "delta_p", "p"],
              num: int):
    scores = {
        "rewrite": [[], 0],

        "p-reasoning": [[], 0],
        "p-subject_alias": [[], 0],
        "p-paraphrase": [[], 0],

        "l-relation": [[], 0],
        "l-relation_aft_edit_text": [[], 0],
        "l-relation-wrong": [[], 0],
        "l-relation_aft_edit_text-wrong": [[], 0],
        "l-neighbor": [[],0],
        "l-neighbor_aft_edit_text": [[], 0],
        "l-neighbor-wrong": [[],0],
        "l-neighbor_aft_edit_text-wrong": [[], 0],

        "ngram_entropy": [[], 0],
        "reference_score": [[], 0],
        "o-commonsense": [[],0],
        "o-commonsense_aft_edit_text": [[], 0]
    }

    res_lst = json.load(open(input_path))[:num]
    print(len(res_lst))
    for res in res_lst:
        data = res[data_type]
        data_keys = [i.replace('-wrong', '') for i in data.keys()]
        for key in scores:
            if key not in data_keys: continue
            if key == "rewrite" or key.split('-')[0] in ["p", "l", "o"]:
                right_scores, wrong_scores = data[f"{key}-right"], data[f"{key}-wrong"]
                scores[key][1] += 1
                r_p, w_p = np.exp(-np.array(right_scores)), np.exp(-np.array(wrong_scores))
                if metric == "accuracy":
                    scores[key][0].extend([i for i in (w_p < r_p).tolist()])
                elif metric == "p":
                    scores[key][0].extend([i for i in (r_p).tolist()])
                    if "commonsense" in key:
                        scores[key][0].append((r_p - w_p).mean())
                    if key.split('-')[0] == 'l':
                        scores[f"{key}-wrong"][0].append((w_p).mean())
                        scores[f"{key}-wrong"][1] += 1
                else:
                    raise NotImplementedError(f"Unknown {metric}")
            elif key == "ppl_score":
                scores[key][1] += 1
                scores[key][2].append(data[key])
                scores[key][0].appedn(0)
            else:
                scores[key][1] += 1
                scores[key][0].append(data[key])
    for key, value in scores.items():
        score = mean_and_conf(np.array(value[0])) if value[1] != 0 else (0, (0,0))
        scores[key] = score
    return scores


def process_directory(base_path, metric, num):
    results = {}
    pre_flag = False
    for folder in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, folder)):
            json_path = os.path.join(base_path, folder, 'results.json')
            if not os.path.exists(json_path): continue
            if not pre_flag:
                results['pre'] = get_score(json_path, 'pre', metric, num)
                pre_flag = True
            if os.path.exists(json_path):
                folder_name_parts = folder.split('-')
                label = '-'.join(folder_name_parts[-3:])
            else:
                label = "Unknown"
            results[label] = get_score(json_path, 'post', metric, num)
    
    return results
    
def generate_markdown_table(results, mode):
    keys = list(next(iter(results.values())).keys())
    labels = sorted(results.keys())
    md_table = "| Metric | " + " | ".join(labels) + " |\n"
    md_table += "|---|" + "|".join(["---"] * len(labels)) + "|\n"
    
    for key in keys:
        row = f"| {key} | " + " | ".join([f"{results[label].get(key, 0)[0] * 100:.2f}({(results[label].get(key, 0)[1][1] - results[label].get(key, 0)[0])*100:.1f})" for label in labels]) + " |"
        md_table += row + "\n"
    
    for label in labels:
        if mode == "score":
            rkeys = ['rewrite', 'p-paraphrase', 'l-neighbor', 'l-relation', 'l-neighbor_aft_edit_text', 'ngram_entropy']
            metrics = [f"{results[label].get(key, 0)[0] * 100:.2f}({(results[label].get(key, 0)[1][1] - results[label].get(key, 0)[0])*100:.1f})" for key in rkeys]
            s = '& ' + '& '.join(metrics)
            print(label + "  :" + s)
        else:
            rkeys = ['rewrite', 'p-paraphrase', 'l-neighbor', 'l-relation', 'l-neighbor_aft_edit_text', 'ngram_entropy']
            metrics = [f"{results[label].get(key, 0)[0] * 100:.2f}" for key in rkeys]
            s = '& ' + '& '.join(metrics)
            print(label + "  :" + s)
    
    return md_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', required=True, type=str, choices=['accuracy', 'delta_p', 'p'])
    parser.add_argument('--base_path', required=True, type=str)
    parser.add_argument('--tojson', action="store_true")
    parser.add_argument('--num', default=100000, type=int)
    parser.add_argument('--mode', default='score', type=str)
    args = parser.parse_args()

    results = process_directory(args.base_path, args.metric, args.num)
    if not args.tojson:
        markdown_table = generate_markdown_table(results, mode=args.mode)
        with open('./results.md', 'w') as f:
            f.write(str(args) + '\n\n')
            f.write(markdown_table)
    else:
        with open(os.path.join(args.base_path, f'{args.metric}_result.json'), 'w') as f:
            f.write(json.dumps(results, indent = 4))

"""
python summrize.py \
--metric accuracy \
--mode scores \
--base_path ./results/gpt-j-6B
"""
    
