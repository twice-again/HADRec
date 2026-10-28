"""
本地预处理完成后，把云端训练/评估所需的最小文件打成一个压缩包，再上传到云端

只打包处理后的结果，绝不包含 data/mimic3/raw/ 下的 MIMIC 原始表。
包内路径与仓库目录一致，在云端仓库根目录解压即可。

用法（在仓库根目录，Windows / Linux / macOS 均可）:
    python scripts/pack_for_cloud.py                  # 默认包含 DDI 评估文件
    python scripts/pack_for_cloud.py --skip_ddi       # 不含 drug-DDI.csv（体积大；可在本地做最终评估）

云端:
    python scripts/pack_for_cloud.py --verify hadrec_cloud_data.tar.gz   # 校验
    tar xzf hadrec_cloud_data.tar.gz
"""

import os
import sys
import json
import hashlib
import tarfile
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 训练与推理必需
REQUIRED = [
    "data/mimic3/l_data/train_atc_hierarchy2.json",
    "data/mimic3/l_data/val_atc_hierarchy2.json",
    "data/mimic3/l_data/test_atc_hierarchy2.json",
    "data/mimic3/l_data/drug_embeddings.pt",
]
# 评估用，存在则打包
OPTIONAL = [
    "data/mimic3/handled/ddi_A_final.pkl",
    "data/mimic3/handled/atc4_vocab.pkl",
    "data/mimic3/auxiliary/drug-atc.csv",
]
DDI_CSV = "data/mimic3/auxiliary/drug-DDI.csv"
MANIFEST = "hadrec_cloud_manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def check_embedding_coverage():
    """药物嵌入未覆盖的 L4 代码会被 EHRTokenizer 从标签中过滤掉，改变评估口径"""
    try:
        import torch
    except ImportError:
        print("⚠️  未安装 torch，跳过药物嵌入覆盖率检查")
        return
    emb = torch.load(os.path.join(ROOT, REQUIRED[3]), map_location="cpu")
    codes = set(emb.get("atc4_to_idx", {}))
    if float(emb["atc4_embeddings"].abs().sum()) == 0:
        print("❌ drug_embeddings.pt 全为零，请重新生成")
        sys.exit(1)
    train = {c for line in open(os.path.join(ROOT, REQUIRED[0]), encoding="utf-8")
             for c in json.loads(line)["atc_level_4"]}
    test_labels = [c for line in open(os.path.join(ROOT, REQUIRED[2]), encoding="utf-8")
                   for c in json.loads(line)["atc_level_4"]]
    missing = sorted(train - codes)
    dropped = sum(1 for c in test_labels if c not in codes)
    if missing:
        print(f"⚠️  药物嵌入未覆盖 {len(missing)}/{len(train)} 个 L4 代码: {missing}")
        print(f"   测试集 {dropped}/{len(test_labels)} 个标签将被过滤，评估口径会与全量标签不同")
    else:
        print(f"✅ 药物嵌入覆盖全部 {len(train)} 个 L4 代码")


def pack(output, skip_ddi):
    missing = [p for p in REQUIRED if not os.path.isfile(os.path.join(ROOT, p))]
    if missing:
        print("❌ 缺少必需文件:")
        for p in missing:
            print(f"   {p}")
        print("   drug_embeddings.pt 可由 python data_process.py 或 bash download.sh embeddings 生成")
        sys.exit(1)

    check_embedding_coverage()
    files = list(REQUIRED)
    optional = OPTIONAL + ([] if skip_ddi else [DDI_CSV])
    for p in optional:
        if os.path.isfile(os.path.join(ROOT, p)):
            files.append(p)
        else:
            print(f"⚠️  未找到（跳过）: {p}")

    manifest = {p: {"sha256": sha256(os.path.join(ROOT, p)), "size": os.path.getsize(os.path.join(ROOT, p))}
                for p in files}
    manifest_path = os.path.join(ROOT, MANIFEST)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with tarfile.open(output, "w:gz") as tar:
        for p in files:
            tar.add(os.path.join(ROOT, p), arcname=p)
        tar.add(manifest_path, arcname=MANIFEST)
    os.remove(manifest_path)

    total = sum(v["size"] for v in manifest.values())
    print(f"\n✅ 已打包 {len(files)} 个文件（原始 {human(total)}，压缩后 {human(os.path.getsize(output))}）: {output}")
    for p in files:
        print(f"   {human(manifest[p]['size']):>8}  {p}")
    print("\n上传与解压示例:")
    print(f"   scp {os.path.basename(output)} <user>@<host>:<云端仓库路径>/")
    print(f"   cd <云端仓库路径> && python scripts/pack_for_cloud.py --verify {os.path.basename(output)}"
          f" && tar xzf {os.path.basename(output)}")
    print("\n⚠️  包内是 MIMIC 衍生数据：只上传到自己控制的私有实例，勿放入公开仓库、镜像或共享盘。")


def verify(archive):
    with tarfile.open(archive, "r:gz") as tar:
        manifest = json.load(tar.extractfile(MANIFEST))
        bad = []
        for p, meta in manifest.items():
            h = hashlib.sha256()
            f = tar.extractfile(p)
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
            if h.hexdigest() != meta["sha256"]:
                bad.append(p)
    if bad:
        print("❌ 校验失败:", *bad, sep="\n   ")
        sys.exit(1)
    print(f"✅ 校验通过，共 {len(manifest)} 个文件")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "hadrec_cloud_data.tar.gz"))
    parser.add_argument("--skip_ddi", action="store_true", help="不打包 drug-DDI.csv")
    parser.add_argument("--verify", metavar="ARCHIVE", help="校验压缩包完整性")
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        pack(args.output, args.skip_ddi)


if __name__ == "__main__":
    main()
