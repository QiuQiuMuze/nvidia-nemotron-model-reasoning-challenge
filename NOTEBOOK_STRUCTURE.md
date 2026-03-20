# Advanced Notebook Cell Structure

这份说明配套 `train_advanced_notebook.py`，用于帮助你把代码直接整理成 Kaggle 公共 Notebook，并满足竞赛对“公开 notebook + 方法写作”的要求。

## Notebook 建议标题

**NVIDIA Nemotron Advanced Rank-32 LoRA: Group Split + Curriculum + Family Reweighting**

## Cell 分层建议

### Part A — Competition framing
1. **Cell 1: 依赖安装**
   - 安装 `transformers / peft / trl / bitsandbytes / datasets`。
2. **Cell 2: 全局配置**
   - 显式声明 `LoRA rank <= 32`、`max_model_len=8192`、`temperature=0.0` 等比赛约束。
3. **Cell 3: 环境检查**
   - 打印 GPU / Torch / CUDA / 显存环境。

### Part B — Data strategy
4. **Cell 4: 读取官方数据与外挂数据接口**
   - 主训练集必须来自比赛 `train.csv`。
   - 额外数据作为可选混合输入，而不是硬编码依赖。
5. **Cell 5: 数据体检**
   - 统计 `prompt_family`、`answer_type`、长度分桶。
6. **Cell 6: 去重 + Group split**
   - 强调为什么不用随机切分。
7. **Cell 7: 近似官方 metric**
   - `\boxed{}` 提取、数值容差、回退规则。

### Part C — Prompt and supervision design
8. **Cell 8: Prompt 模板工程**
   - compact 模板 vs reasoning 模板。
9. **Cell 9: answer-only loss 样本构造**
   - 监督目标只覆盖最终 boxed answer。
10. **Cell 10: 两阶段 curriculum 切分**
    - Stage 1 短样本，Stage 2 全量样本。

### Part D — Model and training
11. **Cell 11: tokenizer**
12. **Cell 12: tokenize**
13. **Cell 13: 4-bit 基座 + 自动发现 LoRA 模块**
14. **Cell 14: family reweight / length bonus / difficulty 权重**
15. **Cell 15: 自定义 WeightedTrainer**
16. **Cell 16: 本地生成评估 callback**
17. **Cell 17: 训练参数工厂**
18. **Cell 18: Stage 1 训练**
19. **Cell 19: Stage 2 训练**
20. **Cell 20: 训练后近似评估**

### Part E — Submission packaging
21. **Cell 21: 保存 adapter + 校验 rank**
22. **Cell 22: 打包 submission.zip**
23. **Cell 23: smoke test**

### Part F — Prize-oriented write-up
24. **Cell 24: 冲奖策略总结**
   - 数据、prompt、RL、bagging、CV、template ensemble。
25. **Cell 25: 导出配置 / 清理显存**

## 为什么这套结构比基础版更适合比赛

- **更贴合评分方式**：训练目标直接对齐 boxed final answer，而不是泛化不清的长文本输出。
- **更贴合真实泛化**：按 prompt family split，避免验证集泄漏造成“看起来很强”。
- **更利于冲榜迭代**：外挂数据、伪标签、自蒸馏、模板 A/B、stage curriculum 都有现成扩展位。
- **更利于公开复现**：每个 cell 都有清晰职责，方便你做 Kaggle write-up 和公开 Notebook 注释。

## 推荐的公开 Notebook 说明写法

建议在 Notebook 开头写清楚这 4 点：

1. **Competition constraints respected**
   - Base model: Nemotron-3-Nano-30B
   - LoRA rank <= 32
   - Submission artifact includes `adapter_config.json`
2. **Validation design**
   - Group split by prompt family
   - Approximate local metric aligned to boxed-answer extraction
3. **Training strategy**
   - QLoRA + answer-only loss + curriculum + reweighting
4. **Future improvement path**
   - synthetic data / self-distillation / RL / prompt ensemble
