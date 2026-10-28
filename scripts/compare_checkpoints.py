"""
比较各 checkpoint 在验证集上的表现，选出最佳 checkpoint（替代缺失的 test_all_checkpoints.bash）

输入: <pred_root>/checkpoint-*/test_predictions.json  （main_llm_cls.py --do_predict 在验证集上的输出）
输出: CSV 对比表；最后一行打印最佳 checkpoint 名称，供 shell 脚本读取

只用验证集选模型，避免在测试集上挑 checkpoint 导致结果偏高。
"""

import os
import re
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.data import EHRTokenizer
from evaluate import evaluate_jsonlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", required=True, help="包含 checkpoint-*/test_predictions.json 的目录")
    parser.add_argument("--train_file", required=True, help="训练集，用于构建与训练时一致的 L4 词表")
    parser.add_argument("--drug_embedding_file", default=None)
    parser.add_argument("--threshold", type=float, default=0.35, help="论文 4.5 节的推理阈值")
    parser.add_argument("--metric", default="jaccard", choices=["jaccard", "f1", "prauc"])
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    ehr_tokenizer = EHRTokenizer(args.train_file, drug_embedding_file=args.drug_embedding_file)

    ckpts = [d for d in os.listdir(args.pred_root)
             if re.fullmatch(r"checkpoint-\d+", d)
             and os.path.isfile(os.path.join(args.pred_root, d, "test_predictions.json"))]
    if not ckpts:
        sys.exit(f"❌ {args.pred_root} 下没有 checkpoint-*/test_predictions.json")
    ckpts.sort(key=lambda d: int(d.split("-")[1]))

    rows = []
    for ckpt in ckpts:
        pred_file = os.path.join(args.pred_root, ckpt, "test_predictions.json")
        ja, prauc, avg_p, avg_r, avg_f1 = evaluate_jsonlines(pred_file, ehr_tokenizer, threshold=args.threshold)
        rows.append({"checkpoint": ckpt, "step": int(ckpt.split("-")[1]), "jaccard": ja, "f1": avg_f1,
                     "prauc": prauc, "precision": avg_p, "recall": avg_r})

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 同分时取更早的 checkpoint
    best = max(rows, key=lambda r: (r[args.metric], -r["step"]))
    print(f"\n验证集对比（阈值 {args.threshold}，按 {args.metric} 选择）:")
    for r in rows:
        mark = "  ← best" if r is best else ""
        print(f"  {r['checkpoint']:>16}  Jaccard={r['jaccard']:.4f}  F1={r['f1']:.4f}  PRAUC={r['prauc']:.4f}{mark}")
    print(f"对比表已保存: {args.output_csv}")
    print(best["checkpoint"])


if __name__ == "__main__":
    main()
