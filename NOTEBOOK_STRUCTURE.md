# Advanced Notebook Cell Structure

这份说明配套 `train_advanced_notebook.py`，用于帮助你把代码直接整理成 Kaggle 公共 Notebook，并满足竞赛对“公开 notebook + 方法写作”的要求。

## Notebook 建议标题

**NVIDIA Nemotron Advanced Rank-32 LoRA: Fingerprint Split + A/B Supervision + Hard Mining**

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
   - 统计 `prompt_family`、`template_group`、`answer_type`、长度分桶。
6. **Cell 6: 去重 + Fingerprint Group split**
   - 先显式规则分组，再对剩余模板做规范化指纹分组。
7. **Cell 7: 近似官方 metric**
   - `\boxed{}` 提取、`1e-2` 数值容差、回退 heuristic 尽量贴近官方。
   - 保留 fast extractor，同时 serious eval 走更 strict 的 official-like extractor。

### Part C — Prompt and supervision design
8. **Cell 8: Prompt 模板池**
   - ultra compact / hidden reasoning / numeric specialized / text specialized。
9. **Cell 9: A/B supervision 构造**
   - `answer-only` vs `short-reasoning + final boxed`，并支持 `family_aware_mix`。
10. **Cell 10: 更激进的 curriculum**
   - Stage 1 先稳定 family + 短答案，Stage 2 再补长题和弱项。

### Part D — Model and training
11. **Cell 11: tokenizer**
12. **Cell 12: tokenize**
13. **Cell 13: 4-bit 基座 + 自动发现 LoRA 模块**
14. **Cell 14: family reweight / length bonus / difficulty 权重**
15. **Cell 15: 自定义 WeightedTrainer**
16. **Cell 16: 本地生成评估 callback**
   - 输出 `overall / family / template_group / answer_type / len_bucket / source / template` 面板。
   - 区分 fast eval 与 serious eval。
17. **Cell 17: 训练参数工厂**
18. **Cell 18: A/B supervision 预检 + 模板先验**
   - 训练前只展示 `answer_only / short_reasoning / family_aware_mix` 的监督差异。
   - 模板选择先使用 family-aware heuristic priors，不在未训练模型上做 ablation。
19. **Cell 19: Stage 1 训练**
20. **Cell 20: Stage 1 后 serious eval + conservative template refresh**
   - Stage 1 后再跑 family-balanced template ablation。
   - 模板切换需要同时满足 gain / min_rows / boxed metrics / secondary view 稳定性。
   - 一旦更新 `BEST_TEMPLATE_BY_FAMILY`，立即重建 `train_records / valid_records / stage2 datasets`。
   - 这里也会接入 sample-level replay bootstrap 与 consensus+verifier pseudolabel refresh。
21. **Cell 21: Stage 2 多轮重加权训练**
   - 每轮 eval 后，按 group hard profile + sample replay buffer 联合刷新采样权重。
   - 额外支持低频率轮间资产刷新（模板 map / pseudolabel / stage2 dataset），并可由弱 family 停滞或 replay 过度集中触发。
22. **Cell 22: 分组近似评估**
   - 增加 test-time template consensus 对照。
   - 把 consensus 更多当成伪标签/脆弱 family 发现工具，而不是最终提交流程本身。

### Part E — Submission packaging
23. **Cell 23: 保存 adapter + 校验 rank**
24. **Cell 24: 打包 submission.zip**
25. **Cell 25: smoke test**

### Part F — Prize-oriented write-up
26. **Cell 26: 冲奖策略总结**
   - 数据、prompt、RL、bagging、CV、template ensemble。
27. **Cell 27: 导出配置 / 清理显存**

## 为什么这套结构比基础版更适合比赛

- **更贴合评分方式**：训练目标直接对齐 boxed final answer，而不是泛化不清的长文本输出。
- **更贴合真实泛化**：显式规则 + 模板指纹分组，比纯手工 family split 更稳。
- **更利于冲榜迭代**：外挂数据、A/B supervision、template pool、动态错题驱动重加权、stage curriculum 都有现成扩展位。
- **更利于数据增益**：external/synthetic mixture 与带 family-specific verifier 的 consensus pseudolabel refresh 已经接入主训练路径。
- **更利于离线判断**：fast eval 看趋势，serious eval 做最终方案选择，并优先尝试挂载官方评测后端。
- **更利于路线判断**：可以先用小规模对照实验判断 supervision 该走 `answer_only` 还是 `short_reasoning`。
- **更利于公开复现**：每个 cell 都有清晰职责，方便你做 Kaggle write-up 和公开 Notebook 注释。

## 推荐的公开 Notebook 说明写法

建议在 Notebook 开头写清楚这 4 点：

1. **Competition constraints respected**
   - Base model: Nemotron-3-Nano-30B
   - LoRA rank <= 32
   - Submission artifact includes `adapter_config.json`
2. **Validation design**
   - Fingerprint-aware group split
   - Fixed sanity subset + grouped local metrics
   - Boxed-format stability panel (`has_boxed` / `boxed_parse_rate`)
   - Approximate local metric aligned to boxed-answer extraction
3. **Training strategy**
   - QLoRA + A/B supervision + curriculum + hard mining + reweighting
4. **Future improvement path**
   - synthetic data / self-distillation / RL / prompt ensemble


cfg.smoke_test_mode = False   # 正常模式
cfg.smoke_test_mode = True    # 进入 smoke 模式
cfg.smoke_profile是 smoke 模式里的子模式选择器，只有在：
cfg.smoke_test_mode = True
时它才生效。
可选值

"fast"
超轻量快速测试模式
会做的事：
测数据处理
测 tokenizer / model / LoRA 加载
测基础 forward / generate
测本地 eval 函数
测导出 zip
不会做的事：
不走 trainer.train()
不走 stage2 主循环
不测导出后 reload
适合：
只是想快速看 notebook 有没有明显语法/流程 bug
改了前处理、tokenize、加载部分后快速体检
写法：
cfg.smoke_test_mode = True
cfg.smoke_profile = "fast"

"pipeline"
轻量全链路测试模式
会做的事：
跑 stage1 的 trainer.train()
跑 stage1 evaluate
跑 stage2 主循环（轻量版）
导出 adapter
重新加载导出的 adapter 并测试推理
不会做的事：
不跑 heavy eval
不跑 rerank
不跑 template ablation
不跑 pseudolabel refresh
不跑多 seed 重评估
适合：
改了训练逻辑后检查会不会炸
提交前检查“主链路是否能跑通”
想抓代码级 bug，而不是追求分数
写法：
cfg.smoke_test_mode = True
cfg.smoke_profile = "pipeline"


cfg.run_supervision_ablation = True
run_supervision_ablation 不是主训练开关，
而是：
“要不要额外跑 supervision 消融对照实验”的开关。
开了它，通常会多跑训练；
关了它，主流程更干净、更适合 smoke test 和正式提交。