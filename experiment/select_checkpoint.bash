#!/bin/bash
# 按验证集选最佳 checkpoint，再用它在测试集上预测（替代缺失的 test_all_checkpoints.bash）
#
#   1. 对 saved/lora-$date/ 下每个 checkpoint 在验证集上预测
#   2. scripts/compare_checkpoints.py 比较验证集指标，选出最佳 checkpoint
#   3. 用最佳 checkpoint 在测试集上预测 -> results/$date/
#      并把它的验证集预测复制到 results/${date}_val/，供 final_evaluation_with_bootstrap.py 调阈值
#
# 用法: bash experiment/select_checkpoint.bash
# 可用环境变量覆盖: MODEL_NAME_OR_PATH DATA_DIR CKPT_ROOT RESULTS_ROOT DATE MAX_SOURCE_LENGTH SELECT_METRIC EXTRA_ARGS

set -e

model_name_or_path="${MODEL_NAME_OR_PATH:-resources/llama-7b}"
your_data_path="${DATA_DIR:-data/mimic3/l_data}"
date="${DATE:-atc_hierarchy2}"
ckpt_dir="${CKPT_ROOT:-saved}/lora-$date"
results_root="${RESULTS_ROOT:-results}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-2048}"
SELECT_METRIC="${SELECT_METRIC:-jaccard}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

val_root="$results_root/${date}_val_checkpoints"

predict() {  # predict <peft_path> <test_file> <output_dir>
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main_llm_cls.py \
        --do_predict \
        --train_file $your_data_path/train_atc_hierarchy2.json \
        --test_file "$2" \
        --cache_dir $your_data_path \
        --overwrite_cache \
        --prompt_column input \
        --response_column atc_level_4 \
        --model_name_or_path $model_name_or_path \
        --peft_path "$1" \
        --output_dir "$3" \
        --overwrite_output_dir \
        --max_source_length $MAX_SOURCE_LENGTH \
        --max_target_length 196 \
        --per_device_eval_batch_size 4 \
        --hierarchical_prediction \
        --use_cross_attention \
        --cross_threshold 0.1 \
        --drug_embedding_file $your_data_path/drug_embeddings.pt \
        $EXTRA_ARGS
}

checkpoints=$(ls -d $ckpt_dir/checkpoint-* 2>/dev/null | sort -V)
if [ -z "$checkpoints" ]; then
    echo "❌ 未找到 checkpoint: $ckpt_dir/checkpoint-*"
    exit 1
fi

# Step 1: 每个 checkpoint 在验证集上预测（已有结果则跳过）
for ckpt in $checkpoints; do
    name=$(basename $ckpt)
    if [ -f "$val_root/$name/test_predictions.json" ]; then
        echo "⏭️  已有验证集预测，跳过: $name"
        continue
    fi
    echo "🔍 验证集预测: $name"
    predict "$ckpt" "$your_data_path/val_atc_hierarchy2.json" "$val_root/$name"
done

# Step 2: 比较并选出最佳 checkpoint
compare_log="$val_root/compare.log"
python scripts/compare_checkpoints.py \
    --pred_root "$val_root" \
    --train_file $your_data_path/train_atc_hierarchy2.json \
    --drug_embedding_file $your_data_path/drug_embeddings.pt \
    --metric $SELECT_METRIC \
    --output_csv "$results_root/${date}_checkpoint_comparison.csv" | tee "$compare_log"
best=$(tail -n 1 "$compare_log")
echo "$best" > "$results_root/${date}_best_checkpoint.txt"
echo "🏆 验证集最佳 checkpoint: $best"

# Step 3: 最佳 checkpoint 在测试集上预测
predict "$ckpt_dir/$best" "$your_data_path/test_atc_hierarchy2.json" "$results_root/$date"
mkdir -p "$results_root/${date}_val"
cp "$val_root/$best/test_predictions.json" "$results_root/${date}_val/test_predictions.json"

echo ""
echo "✅ 完成"
echo "   对比表:     $results_root/${date}_checkpoint_comparison.csv"
echo "   测试集预测: $results_root/$date/test_predictions.json"
echo "   验证集预测: $results_root/${date}_val/test_predictions.json"
