#!/bin/bash
# 完整流程：训练 → 测试 → 可视化
# 用法: bash experiment/full_pipeline.bash

set -e  # 遇到错误立即退出

echo ""
echo "========================================"
echo "   完整流程：训练 → 测试 → 可视化"
echo "========================================"
echo ""

# ============= 配置参数 =============
lora_rank=8
lora_trainable="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="${MODEL_NAME_OR_PATH:-resources/llama-7b}"
your_data_path="${DATA_DIR:-data/mimic3/l_data}"
your_checkpopint_path="saved"
MAX_STEPS=5000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
date="${RUN_NAME:-atc_hierarchy2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_SOURCE_LENGTH=2048

checkpoint_dir="${your_checkpopint_path}/lora-${date}"

# ============= Step 1: 训练（自动恢复）=============
echo "📝 Step 1/3: 训练模型"
echo "========================================"
echo ""

# 查找最新checkpoint
latest_checkpoint=$(ls -d ${checkpoint_dir}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$latest_checkpoint" ]; then
    echo "🆕 从头开始训练..."
    resume_arg=""
else
    step_num=$(basename $latest_checkpoint | sed 's/checkpoint-//')
    remaining=$((MAX_STEPS - step_num))
    
    if [ $remaining -le 0 ]; then
        echo "✅ 训练已完成 (${step_num}/${MAX_STEPS})，跳过训练"
        echo ""
    else
        echo "🔄 从 checkpoint-${step_num} 恢复训练 (剩余 ${remaining} 步)..."
        resume_arg="--resume_from_checkpoint ${latest_checkpoint} --ignore_data_skip"
    fi
fi

# 只有在训练未完成时才执行训练
final_checkpoint=$(ls -d ${checkpoint_dir}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
final_step=$(basename $final_checkpoint 2>/dev/null | sed 's/checkpoint-//')

if [ -z "$final_step" ] || [ $final_step -lt $MAX_STEPS ]; then
    echo "开始训练..."
    echo ""
    
    deepspeed --num_gpus=1 --master_port $MASTER_PORT main_llm_cls.py \
        --deepspeed llm/ds.config \
        --do_train \
        --train_file $your_data_path/train_atc_hierarchy2.json \
        --test_file $your_data_path/test_atc_hierarchy2.json \
        --cache_dir $your_data_path \
        --prompt_column input \
        --response_column atc_level_4 \
        --overwrite_cache \
        --model_name_or_path $model_name_or_path \
        --output_dir $your_checkpopint_path/lora-$date \
        --max_source_length $MAX_SOURCE_LENGTH \
        --max_target_length 196 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --max_steps ${MAX_STEPS} \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 30 \
        --logging_steps 100 \
        --learning_rate $LR \
        --lora_rank ${lora_rank} \
        --trainable ${lora_trainable} \
        --modules_to_save ${modules_to_save} \
        --lora_dropout ${lora_dropout} \
        --fp16 \
        --hierarchical_prediction \
        --balance_strategy combined \
        --use_cross_attention \
        --cross_threshold 0.1 \
        --drug_embedding_file $your_data_path/drug_embeddings.pt \
        $EXTRA_ARGS \
        $resume_arg
    
    echo ""
    echo "✅ 训练完成"
    echo ""
fi

# ============= Step 2: 按验证集选 checkpoint 并预测测试集 =============
echo ""
echo "📝 Step 2/3: 按验证集选 checkpoint 并预测测试集"
echo "========================================"
echo ""

MODEL_NAME_OR_PATH=$model_name_or_path DATA_DIR=$your_data_path CKPT_ROOT=$your_checkpopint_path \
DATE=$date MAX_SOURCE_LENGTH=$MAX_SOURCE_LENGTH EXTRA_ARGS="$EXTRA_ARGS" \
    bash experiment/select_checkpoint.bash
best_checkpoint=$(cat results/${date}_best_checkpoint.txt)

echo ""
echo "✅ 测试完成"
echo ""

# ============= Step 3: 生成可视化（可选，脚本未随仓库提供）=============
echo ""
echo "📝 Step 3/3: 生成可视化"
echo "========================================"
echo ""

if [ -f visualize_real_attention.py ] && [ -f advanced_visualization.py ]; then
    echo "生成真实注意力可视化..."
    python visualize_real_attention.py results/${date}/test_predictions.json

    echo ""
    echo "生成深度分析图表..."
    python advanced_visualization.py

    echo ""
    echo "✅ 可视化完成"
else
    echo "⚠️  未找到可视化脚本，跳过（不影响评估指标）"
fi

# ============= 最终总结 =============
echo ""
echo "========================================"
echo "   🎉 所有任务完成！"
echo "========================================"
echo ""

echo "📊 结果文件:"
echo "   - results/${date}_checkpoint_comparison.csv (各 checkpoint 验证集性能)"
echo "   - results/${date}/test_predictions.json (最佳 checkpoint 测试集预测)"
echo "   - results/${date}_val/test_predictions.json (最佳 checkpoint 验证集预测)"
echo ""

echo "💡 下一步:"
echo "   python final_evaluation_with_bootstrap.py"
echo ""

echo "🎯 验证集最佳checkpoint: ${best_checkpoint}"
echo ""
