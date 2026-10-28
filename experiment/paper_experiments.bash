#!/bin/bash
# 论文实验总入口：训练 → 最后一个 checkpoint 预测 → 按验证集选 checkpoint → bootstrap 评估
#
# 用法: bash experiment/paper_experiments.bash <实验名> [<实验名> ...]
#       bash experiment/paper_experiments.bash list
#
# 每个实验是一次完整训练（5000 步），单卡 24GB 约数小时，建议先跑 asis_main 和 paper_main。
# 结果:
#   final_evaluation_results/<实验名>/final_ckpt/   最后一个 checkpoint（原代码默认评估 checkpoint-5000）
#   final_evaluation_results/<实验名>/val_best/     验证集上最佳的 checkpoint

set -e

DATA_DIR="${DATA_DIR:-data/mimic3/l_data}"
HIER="--hierarchical_prediction --use_cross_attention --cross_threshold 0.1"
MKR="--drug_embedding_file $DATA_DIR/drug_embeddings.pt"
# 论文与原代码不一致处按论文取值：融合层参与训练、Eq.5 的 λ 可学习、Table 3 的一致性权重、Eq.14 无对齐损失
PAPER="--train_fusion_modules --learnable_fusion_scale --consistency_weight 0.01 --consistency_level_weights 1.0,1.5,2.0 --drug_align_weight 0"

describe() {
    cat <<'EOF'
asis_main      代码原样（复现论文数字的首选对照）
paper_main     按论文修正的完整模型（Table 4 / Table 5 末行）
paper_base     Table 5: MKR× HP× CL×（LoRA LLaMA-7B 扁平预测）
paper_mkr      Table 5: MKR✓ HP× CL×
paper_mkr_hp   Table 5: MKR✓ HP✓ CL×
paper_hp_cl    Table 5: MKR× HP✓ CL✓
paper_l1_l4    Fig. 3: 只监督 L1 + L4
paper_l1_l3_l4 Fig. 3: 监督 L1 + L3 + L4
EOF
}

configure() {  # 设置 MODEL_ARGS（结构，训练与预测共用）和 EXTRA_ARGS（训练配置）
    case "$1" in
        asis_main)      MODEL_ARGS="$HIER $MKR"; EXTRA_ARGS="" ;;
        paper_main)     MODEL_ARGS="$HIER $MKR"; EXTRA_ARGS="$PAPER" ;;
        paper_base)     MODEL_ARGS="";           EXTRA_ARGS="" ;;
        paper_mkr)      MODEL_ARGS="$MKR";       EXTRA_ARGS="--train_fusion_modules --learnable_fusion_scale --drug_align_weight 0" ;;
        paper_mkr_hp)   MODEL_ARGS="$HIER $MKR"; EXTRA_ARGS="$PAPER --consistency_weight 0" ;;
        paper_hp_cl)    MODEL_ARGS="$HIER";      EXTRA_ARGS="$PAPER" ;;
        paper_l1_l4)    MODEL_ARGS="$HIER $MKR"; EXTRA_ARGS="$PAPER --bce_weights 0.4,0,0,1.5" ;;
        paper_l1_l3_l4) MODEL_ARGS="$HIER $MKR"; EXTRA_ARGS="$PAPER --bce_weights 0.4,0,1.0,1.5" ;;
        *) echo "❌ 未知实验: $1"; describe; exit 1 ;;
    esac
}

evaluate() {  # evaluate <实验名> <预测目录> <验证集预测目录> <输出子目录>
    local drug_file=""
    [[ "$MODEL_ARGS" == *--drug_embedding_file* ]] && drug_file="$DATA_DIR/drug_embeddings.pt"
    python final_evaluation_with_bootstrap.py \
        --test_pred_file "$2/test_predictions.json" \
        --val_pred_file "$3/test_predictions.json" \
        --train_file $DATA_DIR/train_atc_hierarchy2.json \
        --drug_embedding_file "$drug_file" \
        --output_dir "final_evaluation_results/$1/$4"
}

if [ "$1" = "list" ] || [ $# -eq 0 ]; then
    describe
    exit 0
fi

for name in "$@"; do
    configure "$name"
    echo ""
    echo "==================== $name ===================="
    echo "MODEL_ARGS: $MODEL_ARGS"
    echo "EXTRA_ARGS: $EXTRA_ARGS"

    RUN_NAME=$name MODEL_ARGS="$MODEL_ARGS" EXTRA_ARGS="$EXTRA_ARGS" DATA_DIR=$DATA_DIR \
        bash experiment/llm_cls.bash
    evaluate $name results/$name results/${name}_val final_ckpt

    DATE=$name MODEL_ARGS="$MODEL_ARGS" EXTRA_ARGS="$EXTRA_ARGS" DATA_DIR=$DATA_DIR RESULTS_ROOT=results/selected \
        bash experiment/select_checkpoint.bash
    evaluate $name results/selected/$name results/selected/${name}_val val_best
done
