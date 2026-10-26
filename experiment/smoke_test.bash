#!/bin/bash
# 冒烟测试：用合成数据 + 随机初始化的小 LLaMA 跑通 训练 → 选 checkpoint → 测试集预测 → bootstrap 评估
# 不需要 MIMIC、不需要下载模型、不需要 GPU（CPU 约 1–2 分钟）；指标没有意义，只验证代码完整可运行。
#
# 用法: bash experiment/smoke_test.bash
# 可选: MODEL_NAME_OR_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0 bash experiment/smoke_test.bash  # 换成真实小模型

set -e

SMOKE=${SMOKE_DIR:-smoke}
MAX_SOURCE_LENGTH=256
rm -rf $SMOKE

if [ -n "$MODEL_NAME_OR_PATH" ]; then
    python scripts/smoke_test/prepare.py --out_dir $SMOKE --skip_model
    model=$MODEL_NAME_OR_PATH
else
    python scripts/smoke_test/prepare.py --out_dir $SMOKE
    model=$SMOKE/tiny-llama
fi

echo ""
echo "========== 1/3 训练 =========="
python main_llm_cls.py \
    --do_train \
    --train_file $SMOKE/l_data/train_atc_hierarchy2.json \
    --test_file $SMOKE/l_data/test_atc_hierarchy2.json \
    --cache_dir $SMOKE/l_data \
    --prompt_column input \
    --response_column atc_level_4 \
    --overwrite_cache \
    --model_name_or_path $model \
    --output_dir $SMOKE/saved/lora-smoke \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_steps 8 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 4 \
    --logging_steps 2 \
    --learning_rate 2e-4 \
    --lora_rank 8 \
    --trainable q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --modules_to_save null \
    --lora_dropout 0.1 \
    --hierarchical_prediction \
    --balance_strategy combined \
    --use_cross_attention \
    --cross_threshold 0.1 \
    --drug_embedding_file $SMOKE/l_data/drug_embeddings.pt \
    --report_to none

echo ""
echo "========== 2/3 按验证集选 checkpoint 并预测测试集 =========="
MODEL_NAME_OR_PATH=$model DATA_DIR=$SMOKE/l_data CKPT_ROOT=$SMOKE/saved RESULTS_ROOT=$SMOKE/results \
DATE=smoke MAX_SOURCE_LENGTH=$MAX_SOURCE_LENGTH EXTRA_ARGS="--report_to none" \
    bash experiment/select_checkpoint.bash

echo ""
echo "========== 3/3 Bootstrap 评估 =========="
python final_evaluation_with_bootstrap.py \
    --test_pred_file $SMOKE/results/smoke/test_predictions.json \
    --val_pred_file $SMOKE/results/smoke_val/test_predictions.json \
    --train_file $SMOKE/l_data/train_atc_hierarchy2.json \
    --drug_embedding_file $SMOKE/l_data/drug_embeddings.pt \
    --vocab_pkl_file $SMOKE/atc4_vocab.pkl \
    --ddi_file $SMOKE/auxiliary/drug-DDI.csv \
    --drug_atc_file $SMOKE/auxiliary/drug-atc.csv \
    --output_dir $SMOKE/final_evaluation_results

echo ""
echo "✅ 冒烟测试通过：训练、checkpoint 选择、预测、评估全流程可运行"
