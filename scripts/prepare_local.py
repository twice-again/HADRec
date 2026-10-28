"""
本地准备预处理所需文件（Windows / Linux / macOS 均可，纯 Python，不依赖 bash）

1. 从已有目录（如其他基线仓库的数据目录）复制辅助文件，自动识别常见别名并改成代码使用的文件名
2. 缺失的公开文件从固定版本的上游仓库下载（与 download.sh 相同来源）
3. 从 MIMIC-III 目录复制所需的 6 张表（支持 .csv 与 .csv.gz）
4. 可选 --run：依次执行 data_process.py 与 scripts/pack_for_cloud.py

用法（在仓库根目录）:
    python scripts/prepare_local.py ^
        --aux_src "E:\\A_ResearchRoutine\\MCR\\baselines\\DMRNet-main\\data\\mimic_iii" ^
        --mimic_dir "<MIMIC-III 1.4 所在目录>" ^
        --run
"""

import os
import sys
import gzip
import shutil
import argparse
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "mimic3", "raw")
AUX_DIR = os.path.join(ROOT, "data", "mimic3", "auxiliary")
UTILS_PY = os.path.join(ROOT, "data", "mimic3", "utils.py")

SAFEDRUG_RAW = "https://raw.githubusercontent.com/ycq091044/SafeDrug/88ce5c377dcdc2aa01aaa88f5478dfa4373ba49a"
LEADER_RAW = "https://raw.githubusercontent.com/liuqidong07/LEADER-pytorch/dbde80ece74c3d5bd72258005c2e91391f3672e7"
ATCD_RAW = "https://raw.githubusercontent.com/fabkury/atcd/c5c6833428619b58380fe8a06dcb00e6dc511087"

# 目标文件名: (可接受的源文件名, 下载地址或 Google Drive 说明)
AUX_FILES = {
    "drug-atc.csv": (["drug-atc.csv"], SAFEDRUG_RAW + "/data/input/drug-atc.csv"),
    "drug-DDI.csv": (["drug-DDI.csv", "drug-ddi.csv"],
                     "gdrive:1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz"),
    "ndc2RXCUI.txt": (["ndc2RXCUI.txt", "ndc2rxcui.txt", "ndc2rxnorm_mapping.txt", "ndc2rxnorm_mapping.csv"],
                      SAFEDRUG_RAW + "/data/input/ndc2RXCUI.txt"),
    "RXCUI2atc4.csv": (["RXCUI2atc4.csv", "rxcui2atc4.csv", "ndc2atc_level4.csv"],
                       SAFEDRUG_RAW + "/data/input/RXCUI2atc4.csv"),
    "drugbank_drugs_info.csv": (["drugbank_drugs_info.csv"], "gdrive:1EzIlVeiIR6LFtrBnhzAth4fJt6H_ljxk"),
    "WHO ATC-DDD 2021-12-03.csv": (["WHO ATC-DDD 2021-12-03.csv"],
                                   ATCD_RAW + "/WHO%20ATC-DDD%202021-12-03.csv"),
}
MIMIC_TABLES = ["PRESCRIPTIONS", "DIAGNOSES_ICD", "PROCEDURES_ICD", "ADMISSIONS", "D_ICD_DIAGNOSES", "D_ICD_PROCEDURES"]


def find_file(src_dirs, names):
    """在源目录（含子目录）中按文件名查找，不区分大小写"""
    wanted = {n.lower() for n in names}
    for src in src_dirs:
        for dirpath, _, files in os.walk(src):
            for f in files:
                if f.lower() in wanted:
                    return os.path.join(dirpath, f)
    return None


def download(url, dest):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)


def download_gdrive(file_id, dest):
    try:
        import gdown
    except ImportError:
        return False
    return gdown.download(id=file_id, output=dest, quiet=False) is not None


def prepare_aux(src_dirs):
    os.makedirs(AUX_DIR, exist_ok=True)
    status = {}
    for target, (aliases, source) in AUX_FILES.items():
        dest = os.path.join(AUX_DIR, target)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            status[target] = "已存在"
            continue
        found = find_file(src_dirs, aliases)
        if found:
            shutil.copyfile(found, dest)
            status[target] = f"复制自 {found}"
            continue
        try:
            if source.startswith("gdrive:"):
                ok = download_gdrive(source.split(":", 1)[1], dest)
                status[target] = "已下载 (Google Drive)" if ok else \
                    f"缺失：请手动下载 https://drive.google.com/file/d/{source.split(':', 1)[1]}/view 到 {dest}"
            else:
                download(source, dest)
                status[target] = "已下载"
        except Exception as e:  # 网络问题时给出明确提示，不中断其他文件
            status[target] = f"下载失败（{e}），请手动放到 {dest}"
    if not os.path.isfile(UTILS_PY):
        try:
            download(LEADER_RAW + "/data/mimic3/utils.py", UTILS_PY)
            status["data/mimic3/utils.py"] = "已下载"
        except Exception as e:
            status["data/mimic3/utils.py"] = f"下载失败（{e}）"
    else:
        status["data/mimic3/utils.py"] = "已存在"
    return status


def prepare_mimic(mimic_dir):
    os.makedirs(RAW_DIR, exist_ok=True)
    status = {}
    for t in MIMIC_TABLES:
        dest = os.path.join(RAW_DIR, f"{t}.csv")
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            status[t] = "已存在"
            continue
        if not mimic_dir:
            status[t] = "缺失（未指定 --mimic_dir）"
            continue
        src = find_file([mimic_dir], [f"{t}.csv"])
        src_gz = None if src else find_file([mimic_dir], [f"{t}.csv.gz"])
        if src:
            shutil.copyfile(src, dest)
            status[t] = f"复制自 {src}"
        elif src_gz:
            with gzip.open(src_gz, "rb") as fi, open(dest + ".part", "wb") as fo:
                shutil.copyfileobj(fi, fo)
            os.replace(dest + ".part", dest)
            status[t] = f"解压自 {src_gz}"
        else:
            status[t] = f"缺失：{mimic_dir} 中未找到 {t}.csv(.gz)"
    return status


def report(title, status):
    print(f"\n{title}")
    for k, v in status.items():
        mark = "❌" if v.startswith(("缺失", "下载失败")) else "✅"
        print(f"  {mark} {k}: {v}")
    return all(not v.startswith(("缺失", "下载失败")) for v in status.values())


def main():
    parser = argparse.ArgumentParser(description="准备本地预处理所需文件")
    parser.add_argument("--aux_src", action="append", default=[],
                        help="已有辅助文件所在目录，可重复指定（如 DMRNet-main/data/mimic_iii）")
    parser.add_argument("--mimic_dir", default=None, help="MIMIC-III 原始表所在目录")
    parser.add_argument("--run", action="store_true", help="文件齐全后执行 data_process.py 与打包")
    parser.add_argument("--skip_ddi", action="store_true", help="打包时不含 drug-DDI.csv")
    args = parser.parse_args()

    for d in args.aux_src:
        if not os.path.isdir(d):
            sys.exit(f"❌ 目录不存在: {d}")

    ok_aux = report("辅助文件 -> data/mimic3/auxiliary/", prepare_aux(args.aux_src))
    ok_raw = report("MIMIC-III 原始表 -> data/mimic3/raw/", prepare_mimic(args.mimic_dir))

    if not (ok_aux and ok_raw):
        sys.exit("\n⚠️  仍有文件缺失，补齐后重新运行本脚本")
    print("\n✅ 预处理所需文件已齐全")
    if not args.run:
        print("   下一步: python data_process.py && python scripts/pack_for_cloud.py")
        return

    subprocess.run([sys.executable, "data_process.py"], cwd=ROOT, check=True)
    pack = [sys.executable, os.path.join("scripts", "pack_for_cloud.py")] + (["--skip_ddi"] if args.skip_ddi else [])
    subprocess.run(pack, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
