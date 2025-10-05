import json
import os

from utils.data import load_data_100
from eval import eval_f1

def main(args):
    data_all = load_data_100()
    valid_ids = set()
    for item in data_all:
        valid_ids.add(item['id'])
    
    pred_file = os.path.join(args.result_dir, 'pred.jsonl')
    assert os.path.exists(pred_file), f"Pred file {pred_file} does not exist"
    with open(pred_file, 'r', encoding='utf-8') as f:
        all_outputs = [json.loads(line) for line in f]
    
    # Only keep entries whose IDs are in data_all
    outputs = [item for item in all_outputs if item['id'] in valid_ids]
    print(f"Filtered outputs: {len(outputs)} out of {len(all_outputs)} entries kept")
    
    eval_f1(outputs)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    args = parser.parse_args()
    
    main(args)
