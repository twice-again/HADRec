"""
冒烟测试素材生成（完全离线，不含任何 MIMIC 数据）

生成内容（默认输出到 smoke/）:
    l_data/{train,val,test}_atc_hierarchy2.json   与正式数据同格式的合成样本
    l_data/drug_embeddings.pt                      随机药物嵌入（384 维，与 ChemBERTa 一致）
    auxiliary/drug-atc.csv, auxiliary/drug-DDI.csv 合成的 DDI 评估文件
    tiny-llama/                                    随机初始化的小 LLaMA 与本地训练的分词器

用途只是验证 训练 → 推理 → 评估 全流程能跑通，指标没有意义。
"""

import os
import json
import random
import argparse

import torch

# 真实 ATC 分类中的第 3 级代码（公开分类标准，与病人无关）
ATC_CODES = [
    "A02B", "A06A", "A10A", "A12C", "B01A", "B03B", "B05C", "C01C", "C03C", "C07A",
    "C08C", "C09A", "C10A", "H02A", "J01C", "J01D", "J01M", "N01A", "N02A", "N02B",
    "N03A", "N05B", "N05C", "N06A", "R03A", "R06A", "S01A", "V03A",
]

# 虚构的诊断及其常用药物类别
CONDITIONS = {
    "Essential hypertension": ["C07A", "C09A", "C08C"],
    "Type 2 diabetes": ["A10A", "C10A"],
    "Atrial fibrillation": ["B01A", "C07A", "C01C"],
    "Pneumonia": ["J01C", "J01D", "R03A"],
    "Sepsis": ["J01D", "J01M", "B05C", "C01C"],
    "Congestive heart failure": ["C03C", "C07A", "C09A"],
    "Acute kidney failure": ["B05C", "A12C", "V03A"],
    "Major depressive disorder": ["N06A", "N05B"],
    "Epilepsy": ["N03A", "N05B"],
    "Chronic pain": ["N02A", "N02B", "A06A"],
    "Gastric ulcer": ["A02B"],
    "Anemia": ["B03B"],
    "Asthma": ["R03A", "R06A", "H02A"],
    "Insomnia": ["N05C"],
    "Postoperative state": ["N01A", "N02A", "B01A", "A02B"],
    "Conjunctivitis": ["S01A"],
}
PROCEDURES = ["Venous catheterization", "Transfusion", "Hemodialysis", "Endoscopy", "Intubation", "None"]


def make_sample(rng, subject_id):
    def visit(n_cond):
        conds = rng.sample(list(CONDITIONS), n_cond)
        codes = {c for cond in conds for c in CONDITIONS[cond]}
        codes |= set(rng.sample(ATC_CODES, rng.randint(0, 2)))
        procs = rng.sample(PROCEDURES, rng.randint(1, 2))
        return conds, procs, sorted(codes)

    history = []
    for no in range(1, rng.randint(1, 3) + 1):
        conds, procs, codes = visit(rng.randint(1, 3))
        history.append(
            f"During the {no} visit, the patient was diagnosed with {', '.join(conds)} and underwent the "
            f"following procedures: {', '.join(procs)}. Medications administered included {', '.join(codes)}. "
        )
    conds, procs, codes = visit(rng.randint(2, 4))
    text = "".join(history) + (
        f"Currently, in the most recent visit, the patient presents with {', '.join(conds)} and receives the "
        f"following procedures: {', '.join(procs)}. Based on these findings, the recommended medications are: "
    )
    return {
        "input": text,
        "target": ", ".join(codes),
        "subject_id": subject_id,
        "atc_level_1": sorted({c[0] for c in codes}),
        "atc_level_2": sorted({c[:2] for c in codes}),
        "atc_level_3": sorted({c[:3] for c in codes}),
        "atc_level_4": codes,
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_data(out_dir, rng, n_train, n_eval):
    l_data = os.path.join(out_dir, "l_data")
    os.makedirs(l_data, exist_ok=True)
    sid = 0
    for split, n in [("train", n_train), ("val", n_eval), ("test", n_eval)]:
        rows = []
        for _ in range(n):
            rows.append(make_sample(rng, sid))
            sid += 1
        write_jsonl(os.path.join(l_data, f"{split}_atc_hierarchy2.json"), rows)

    g = torch.Generator().manual_seed(0)
    torch.save({
        "atc4_codes": ATC_CODES,
        "atc4_embeddings": torch.randn(len(ATC_CODES), 384, generator=g),
        "atc4_to_idx": {c: i for i, c in enumerate(ATC_CODES)},
    }, os.path.join(l_data, "drug_embeddings.pt"))


def build_ddi_files(out_dir, rng):
    aux = os.path.join(out_dir, "auxiliary")
    os.makedirs(aux, exist_ok=True)
    cids = [f"CID{i:09d}" for i in range(len(ATC_CODES))]
    with open(os.path.join(aux, "drug-atc.csv"), "w") as f:
        for cid, code in zip(cids, ATC_CODES):
            f.write(f"{cid},{code}AA01\n")
    side_effects = [f"side effect {i}" for i in range(30)]
    with open(os.path.join(aux, "drug-DDI.csv"), "w") as f:
        f.write("STITCH 1,STITCH 2,Polypharmacy Side Effect,Side Effect Name\n")
        for _ in range(200):
            a, b = rng.sample(cids, 2)
            se = rng.choice(side_effects)
            f.write(f"{a},{b},C{side_effects.index(se):07d},{se}\n")


def build_tiny_model(out_dir, train_file):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    model_dir = os.path.join(out_dir, "tiny-llama")
    os.makedirs(model_dir, exist_ok=True)

    texts = [json.loads(line)["input"] for line in open(train_file, encoding="utf-8")]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    tokenizer.save_pretrained(model_dir)

    # hidden_size 需能被 4（层级 cross-attention）和 8（药物知识模块）整除
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=1024, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(model_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="smoke")
    parser.add_argument("--n_train", type=int, default=64)
    parser.add_argument("--n_eval", type=int, default=16)
    parser.add_argument("--skip_model", action="store_true", help="只生成数据，不生成小模型（改用 MODEL_NAME_OR_PATH）")
    args = parser.parse_args()

    rng = random.Random(42)
    build_data(args.out_dir, rng, args.n_train, args.n_eval)
    build_ddi_files(args.out_dir, rng)
    if not args.skip_model:
        build_tiny_model(args.out_dir, os.path.join(args.out_dir, "l_data", "train_atc_hierarchy2.json"))
    print(f"✅ 冒烟测试素材已生成: {args.out_dir}/")


if __name__ == "__main__":
    main()
