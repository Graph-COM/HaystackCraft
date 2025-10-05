import collections
import json
import os
import re
import string

from tqdm import tqdm

# Code from https://github.com/StonyBrookNLP/musique/blob/main/metrics/answer.py
def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return s.split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def eval_f1(outputs):
    total_f1 = 0
    
    for item in tqdm(outputs):    
        if item['pred'] is None:
            continue
        
        pred_normalized = normalize_answer(item['pred'])
        ans_list_normalized = [normalize_answer(ans) for ans in [item['answer']] + item['answer_aliases']]
        f1_score = metric_max_over_ground_truths(compute_f1, pred_normalized, ans_list_normalized)
        
        total_f1 += f1_score
    
    overall_f1 = total_f1 / len(outputs)
    print("f1: ", overall_f1)

def main(args):        
    pred_file = os.path.join(args.result_dir, 'pred.jsonl')
    assert os.path.exists(pred_file), f"Pred file {pred_file} does not exist"
    with open(pred_file, 'r', encoding='utf-8') as f:
        outputs = [json.loads(line) for line in f]
    
    eval_f1(outputs)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    args = parser.parse_args()
    
    main(args)