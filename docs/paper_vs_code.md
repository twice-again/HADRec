# HADRec 论文与代码对照

论文：*HADRec: A Hierarchy-Aware Drug Recommendation Framework by Fusing Molecular Knowledge and Electronic Health Record*（arXiv:2610.00984v1）。
论文第 30 页给出的官方代码地址为 `github.com/wjk78623-cell/HADRec`，本仓库的代码与其作者一致（脚本中的路径含 `wjk`），但缺少部分文件，见第 4 节。

复现分两套配置并行（`experiment/paper_experiments.bash`）：

- **代码原样**（`asis_main`）：不改任何行为，用于对照论文数字。
- **按论文修正**（`paper_*`）：只修改下表中论文写明、而代码不一致的部分。所有修正都是命令行参数，默认值等于原代码。

## 1. 复现目标（论文报告的数字，MIMIC-III 测试集）

**Table 4 / Table 6（阈值 0.35）**

| Jaccard | F1 | PR-AUC | DDI | Avg.Med |
|---|---|---|---|---|
| 0.5517±0.0059 | 0.7010±0.0054 | 0.7994±0.0057 | 0.0774±0.011 | 20.51±0.27 |

Table 6 中阈值 0.35 一行与 Table 4 完全相同，±std 应来自 `final_evaluation_with_bootstrap.py` 的"10 次随机抽取 30% 测试集"，而不是论文表注所说的多次训练。

**Table 5 消融（Jaccard / F1 / PR-AUC）**

| MKR | HP | CL | Jaccard | F1 | PR-AUC | 对应实验名 |
|---|---|---|---|---|---|---|
| × | × | × | 0.5397 | 0.6904 | 0.7946 | `paper_base` |
| ✓ | × | × | 0.5405 | 0.6915 | 0.7944 | `paper_mkr` |
| ✓ | ✓ | × | 0.5472 | 0.6967 | 0.7986 | `paper_mkr_hp` |
| × | ✓ | ✓ | 0.5450 | 0.6950 | 0.7971 | `paper_hp_cl` |
| ✓ | ✓ | ✓ | 0.5517 | 0.7010 | 0.7994 | `paper_main` |

**Fig. 3 各层级 Jaccard（L1 / L2 / L3 / L4）**

| 监督 | L1 | L2 | L3 | L4 | 对应实验名 |
|---|---|---|---|---|---|
| L1+L4 | 0.7752 | 0.1858 | 0.2398 | 0.5385 | `paper_l1_l4` |
| L1+L3+L4 | 0.7741 | 0.2130 | 0.6260 | 0.5389 | `paper_l1_l3_l4` |
| L1+L2+L3+L4 | 0.7854 | 0.7772 | 0.6348 | 0.5517 | `paper_main` |

**Table 9**：MIMIC-III 上 ECE = 0.09，Brier = 0.12（评估脚本已补充输出）。

## 2. 一致的部分

| 项目 | 论文 | 代码 / 数据 |
|---|---|---|
| 数据规模（Table 1） | 6350 人；5080/635/635；L1–L4 类别 14/17/59/110；平均每次就诊 19.57 种药 | 仓库中的 `*_atc_hierarchy2.json` 完全一致 |
| ATC 层级 | A → A0 → A01 → A01A（第 2 级取前 2 个字符） | 一致 |
| 输入模板（Fig. 2） | "During the … visit …" + "Currently, in the most recent visit …" | 一致 |
| 历史窗口 | 最近 3 次就诊 | 数据中每个样本最多 3 次历史就诊 |
| 骨干与训练 | LLaMA-7B + LoRA，AdamW，lr 2e-4，2048 tokens，5000 步，FP16，单卡 RTX 3090 | 一致（论文未写 LoRA rank，代码为 8） |
| BCE 权重 λ_i（Table 3） | 0.4 / 0.4 / 1.0 / 1.5 | 一致 |
| 层级 cross-attention 阈值 τ_cross | 0.1 | 一致 |
| 层级残差缩放（Table 3 的 λ = 0.7） | 0.7 | 一致 |
| 软掩码（Eq. 12） | m_i = 0.5·mean(σ(Z_{i-1})) + 0.5·σ(Z_i) | 一致 |
| 推理阈值 | 0.35 | bootstrap 评估脚本一致；训练脚本内置评估用 0.4 |

## 3. 不一致的部分及修正

| # | 项目 | 论文 | 代码原样 | 论文修正配置 |
|---|---|---|---|---|
| 1 | 融合层是否训练 | Eq. 3–4 的 W_Q、W_K、W_v 为可训练参数；层级注意力同理 | LoRA 只解冻 `lora_*` 和 `cls_head`：药物知识模块、层级 cross-attention、LayerNorm 一直是随机初始化，也不保存，预测时靠同一随机种子重建 | `--train_fusion_modules`（训练、保存并在预测时加载） |
| 2 | 融合残差 λ（Eq. 5） | 可学习 | 固定 0.5 | `--learnable_fusion_scale`（初值 0.5） |
| 3 | 一致性损失总权重 λ_cons | 0.01 | 0.1 | `--consistency_weight 0.01` |
| 4 | 各层一致性权重 L2/L3/L4 | 1.0 / 1.5 / 2.0 | 2.0 / 1.5 / 1.0（顺序相反） | `--consistency_level_weights 1.0,1.5,2.0` |
| 5 | 药物对齐 MSE 损失 | Eq. 14 的总损失中没有 | 额外加 0.1 × MSE | `--drug_align_weight 0` |
| 6 | DDI 指标定义 | 预测药对中存在已知相互作用的比例 | 存在任一相互作用的**样本**比例，且只看前 20 个药；DDI 对来自"最罕见的 40 条记录"；`atc4_vocab.pkl` 缺失时索引与预测错位 | 评估脚本额外输出论文口径的 DDI（`paper_metrics.json`），原口径保留 |
| 7 | 推理阈值 | 0.35 | 训练脚本内置评估用 0.4 | checkpoint 选择与最终评估统一用 0.35 |

修正配置汇总（与 `experiment/paper_experiments.bash` 中的 `PAPER` 相同）：

```
--train_fusion_modules --learnable_fusion_scale --consistency_weight 0.01
--consistency_level_weights 1.0,1.5,2.0 --drug_align_weight 0
```

## 4. 尚未确定或无法核实

| 项目 | 情况 |
|---|---|
| 分子嵌入维度 | 论文 Table 3 写 d_mol = 768，代码使用的 `DeepChem/ChemBERTa-77M-MLM` 输出 384 维。论文未写具体的 ChemBERTa 版本；若要 768 维，可用 `CHEMBERTA_MODEL` 指定其他版本后重新生成嵌入 |
| DDI 矩阵的副作用筛选 | `data_process.py` 取**最常见**的 40 类副作用，GAMENet / SafeDrug 取**最罕见**的 40 类，bootstrap 评估又取最罕见的 40 条记录；论文只说来自 TWOSIDES |
| 无历史就诊的样本 | 训练集 734/5080（14%）样本没有任何历史就诊，而论文的队列是"至少 2 次入院"的患者，按理每个样本都应有历史。生成 `hierarchy2` 数据的脚本缺失，无法核实 |
| checkpoint 选择 | 论文未说明。原脚本注释为"测试所有 checkpoint 找最佳"（`full_pipeline.bash` 按测试集选），评估脚本默认 checkpoint-5000。两种结果都会输出（`final_ckpt`、`val_best`） |
| ±std 的含义 | 论文写"多次运行"，代码为单次训练的 bootstrap 抽样 |
| 类别不平衡 | README 写的"四层防御"论文中没有提到。论文认为层级结构和阈值已能缓解；代码里的 `balance_strategy` 在本数据上实际不生效，频率权重也未使用。不影响复现 |
| 层级 cross-attention 的实现 | 论文 Eq. 7：用 Z_i 作为权重聚合 W_i。代码相同，但其后的多头注意力只有 1 个 key，实际等价于线性变换 |
| Fig. 4 注意力热图 | 图为 L1×L2 等二维矩阵，代码每个样本只保存一维权重；可视化脚本缺失 |
| MIMIC-IV 实验（Table 8）、反事实分析（Table 7）、效率（Table 10）、基线模型数字 | 仓库中没有对应代码或数据 |
| 缺失文件 | `llm/qwen.py`、`test_all_checkpoints.bash`、`visualize_real_attention.py`、`advanced_visualization.py`、`atc4_vocab.pkl`、生成 `hierarchy2` 数据的脚本；可能在官方仓库中 |

## 5. 运行顺序

```bash
bash experiment/paper_experiments.bash list                      # 查看全部实验
bash experiment/paper_experiments.bash asis_main paper_main      # 先跑主实验
bash experiment/paper_experiments.bash paper_base paper_mkr paper_mkr_hp paper_hp_cl   # Table 5
bash experiment/paper_experiments.bash paper_l1_l4 paper_l1_l3_l4                      # Fig. 3
```

每个实验是一次 5000 步的完整训练。结果在 `final_evaluation_results/<实验名>/{final_ckpt,val_best}/`，
其中 `bootstrap_results.*` 为原代码口径，`paper_metrics.json` 为论文口径的 DDI、ECE 和 Brier。
