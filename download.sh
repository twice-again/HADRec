#!/usr/bin/env bash
# 下载复现所需的外部资源，并放到代码约定的位置。
#
# 用法:
#   bash download.sh                 # 等同于 bash download.sh aux
#   bash download.sh aux             # 辅助文件 + 预处理工具 data/mimic3/utils.py
#   bash download.sh models          # LLaMA-7B 与 ChemBERTa 权重 -> resources/
#   bash download.sh mimic           # MIMIC-III 原始表（需 PhysioNet 账号: PHYSIONET_USER=xxx）
#   bash download.sh embeddings      # 由 SMILES 计算 data/mimic3/l_data/drug_embeddings.pt
#   bash download.sh all             # 以上全部
#
# 已存在的文件会跳过，可重复执行。

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RAW_DIR=data/mimic3/raw
AUX_DIR=data/mimic3/auxiliary
HANDLED_DIR=data/mimic3/handled
L_DATA_DIR=data/mimic3/l_data
RES_DIR=resources

# 固定上游提交，保证下载内容可复现
SAFEDRUG_RAW=https://raw.githubusercontent.com/ycq091044/SafeDrug/88ce5c377dcdc2aa01aaa88f5478dfa4373ba49a
LEADER_RAW=https://raw.githubusercontent.com/liuqidong07/LEADER-pytorch/dbde80ece74c3d5bd72258005c2e91391f3672e7
ATCD_RAW=https://raw.githubusercontent.com/fabkury/atcd/c5c6833428619b58380fe8a06dcb00e6dc511087

# SafeDrug README 中给出的 Google Drive 文件
DRUG_DDI_GDRIVE_ID=1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz
DRUGBANK_GDRIVE_ID=1EzIlVeiIR6LFtrBnhzAth4fJt6H_ljxk

LLAMA_REPO=${LLAMA_REPO:-huggyllama/llama-7b}
CHEMBERTA_REPO=${CHEMBERTA_REPO:-DeepChem/ChemBERTa-77M-MLM}

MIMIC_TABLES=(PRESCRIPTIONS DIAGNOSES_ICD PROCEDURES_ICD ADMISSIONS D_ICD_DIAGNOSES D_ICD_PROCEDURES)

info() { echo -e "\033[32m[download]\033[0m $*"; }
warn() { echo -e "\033[33m[warn]\033[0m $*" >&2; }

fetch() {  # fetch <url> <dest>
    local url=$1 dest=$2
    if [ -s "$dest" ]; then info "已存在，跳过: $dest"; return; fi
    mkdir -p "$(dirname "$dest")"
    info "下载 $dest"
    curl -fL --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

fetch_gdrive() {  # fetch_gdrive <file_id> <dest>
    local id=$1 dest=$2
    if [ -s "$dest" ]; then info "已存在，跳过: $dest"; return; fi
    if ! command -v gdown >/dev/null 2>&1; then
        warn "未安装 gdown（pip install gdown），请手动下载后放到 $dest:"
        warn "  https://drive.google.com/file/d/$id/view"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    info "下载 $dest (Google Drive)"
    gdown --fuzzy "https://drive.google.com/file/d/$id/view" -O "$dest" \
        || warn "Google Drive 下载失败，请手动下载 https://drive.google.com/file/d/$id/view 到 $dest"
}

hf_download() {  # hf_download <repo> <local_dir>
    local repo=$1 dest=$2
    if [ -f "$dest/config.json" ]; then info "已存在，跳过: $dest"; return; fi
    info "下载 $repo -> $dest"
    if command -v hf >/dev/null 2>&1; then
        hf download "$repo" --local-dir "$dest"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$repo" --local-dir "$dest"
    else
        warn "未找到 huggingface CLI（pip install -U huggingface_hub），跳过 $repo"
    fi
}

download_aux() {
    # GAMENet / SafeDrug 的映射文件（SafeDrug 中已改为代码使用的文件名）
    fetch "$SAFEDRUG_RAW/data/input/RXCUI2atc4.csv" "$AUX_DIR/RXCUI2atc4.csv"
    fetch "$SAFEDRUG_RAW/data/input/drug-atc.csv"   "$AUX_DIR/drug-atc.csv"
    fetch "$SAFEDRUG_RAW/data/input/ndc2RXCUI.txt"  "$AUX_DIR/ndc2RXCUI.txt"
    # ATC 代码 -> 药物名称
    fetch "$ATCD_RAW/WHO%20ATC-DDD%202021-12-03.csv" "$AUX_DIR/WHO ATC-DDD 2021-12-03.csv"
    # 体积较大的 DDI 表和 DrugBank 信息表
    fetch_gdrive "$DRUG_DDI_GDRIVE_ID" "$AUX_DIR/drug-DDI.csv"
    fetch_gdrive "$DRUGBANK_GDRIVE_ID" "$AUX_DIR/drugbank_drugs_info.csv"
    # data_process.py / construction.py 依赖的预处理函数（来自 LEADER 原仓库）
    fetch "$LEADER_RAW/data/mimic3/utils.py" "data/mimic3/utils.py"
}

download_models() {
    hf_download "$LLAMA_REPO" "$RES_DIR/llama-7b"
    hf_download "$CHEMBERTA_REPO" "$RES_DIR/ChemBERTa-77M-MLM"
}

download_mimic() {
    local missing=()
    for t in "${MIMIC_TABLES[@]}"; do
        [ -s "$RAW_DIR/$t.csv" ] || missing+=("$t")
    done
    if [ ${#missing[@]} -eq 0 ]; then info "MIMIC-III 原始表已齐全"; return; fi
    if [ -z "${PHYSIONET_USER:-}" ]; then
        warn "缺少 MIMIC-III 表: ${missing[*]}"
        warn "请在 https://physionet.org/content/mimiciii/1.4/ 获得授权后，"
        warn "设置 PHYSIONET_USER=<用户名> 重新执行本命令，或手动解压到 $RAW_DIR/"
        return
    fi
    mkdir -p "$RAW_DIR"
    for t in "${missing[@]}"; do
        info "下载 $t.csv.gz (PhysioNet)"
        wget -q --show-progress -c --user "$PHYSIONET_USER" --ask-password \
            -O "$RAW_DIR/$t.csv.gz" "https://physionet.org/files/mimiciii/1.4/$t.csv.gz"
        gunzip -f "$RAW_DIR/$t.csv.gz"
    done
}

build_embeddings() {
    local out="$L_DATA_DIR/drug_embeddings.pt"
    if [ -s "$out" ]; then info "已存在，跳过: $out"; return; fi
    local pkl="$HANDLED_DIR/atc3toSMILES.pkl"
    if [ ! -s "$pkl" ]; then
        # data_process.py 会基于 MIMIC 重新生成此文件；没有 MIMIC 时用 SafeDrug 发布的版本近似
        warn "$pkl 不存在，改用 SafeDrug 发布的 atc3toSMILES.pkl（可能与作者版本有差异）"
        fetch "$SAFEDRUG_RAW/data/output/atc3toSMILES.pkl" "$pkl"
    fi
    local model="$CHEMBERTA_REPO"
    [ -f "$RES_DIR/ChemBERTa-77M-MLM/config.json" ] && model="$RES_DIR/ChemBERTa-77M-MLM"
    python scripts/precompute_from_pkl.py --pkl_file "$pkl" --output_file "$out" --model_name "$model"
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(aux)
for t in "${targets[@]}"; do
    case "$t" in
        aux)        download_aux ;;
        models)     download_models ;;
        mimic)      download_mimic ;;
        embeddings) build_embeddings ;;
        all)        download_aux; download_models; download_mimic; build_embeddings ;;
        *) echo "未知目标: $t（可选: aux models mimic embeddings all）" >&2; exit 1 ;;
    esac
done
info "完成。"
