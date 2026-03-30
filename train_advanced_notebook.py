# %% [markdown]
# # NVIDIA Nemotron Model Reasoning Challenge — Advanced LoRA Notebook
# 
# 这份脚本按 Kaggle Notebook 的 cell 风格编排，目标不是做“最基础能跑”的 LoRA，
# 而是做一份**更贴近竞赛约束、可公开复现、适合冲榜迭代**的高级训练模板。
# 
# 设计重点：
# 1. **严格贴合比赛约束**：Nemotron-3-Nano-30B、LoRA rank <= 32、最终产出 adapter_config.json + submission.zip。
# 2. **本地评估尽量贴近官方 metric**：优先提取 `\\boxed{}`，否则回退字符串 / 数值容差匹配。
# 3. **避免过拟合式随机切分**：按 prompt family 做 group split，更接近真实泛化。
# 4. **高级训练策略**：QLoRA、answer-only loss、curriculum、长度分桶、family reweighting、两阶段训练。
# 5. **可扩展冲榜接口**：支持外挂额外数据、伪标签、自蒸馏、候选 prompt 模板 A/B test。
# 
# > 使用方式：在 Kaggle Notebook 中将本文件拆成 cell，或直接复制每个 `# %%` 代码块运行。

# %% [markdown]
# ## Cell 1 — 安装依赖
# Kaggle 环境通常已有大部分库；这里使用 Python 安装，避免 notebook magic 对脚本兼容性的影响。

# %%
import os
import sys
import subprocess

INSTALL_PACKAGES = False

if INSTALL_PACKAGES:
    raise RuntimeError("This notebook is configured for offline/preinstalled environments only; do not install extra packages at runtime.")

# %% [markdown]
# ## Cell 2 — 导入依赖与全局配置
# 这里把所有“竞赛硬约束”显式写进配置，避免训练流程偏离提交规则。

# %%
import ast
import gc
import hashlib
import importlib.util
import io
import json
import math
import operator
import random
import re
import shutil
import textwrap
import zipfile
from collections import Counter
from fractions import Fraction
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from IPython.display import display
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# %%
@dataclass
class CFG:
    # ===== competition =====
    base_model: str = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
    local_files_only: bool = True
    trust_remote_code: bool = True
    preferred_attn_implementation: str = "flash_attention_2"
    max_lora_rank: int = 32
    official_max_new_tokens: int = 7680
    official_temperature: float = 0.0
    official_top_p: float = 1.0
    official_max_model_len: int = 8192
    warmup_steps_override: Optional[int] = None

    # ===== paths =====
    competition_path: str = "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge"
    extra_data_dir: str = "/kaggle/input"
    work_dir: str = "/kaggle/working/nemotron_advanced"
    output_dir: str = "/kaggle/working/nemotron_advanced/checkpoints"
    adapter_dir: str = "/kaggle/working/nemotron_advanced/adapter"
    submission_zip: str = "/kaggle/working/submission.zip"

    # ===== candidate / final selection =====
    topk_candidate_keep: int = 12
    topk_candidate_dir: str = "/kaggle/working/nemotron_advanced/topk_candidates"
    topk_candidate_manifest_path: str = "/kaggle/working/nemotron_advanced/topk_candidates_manifest.csv"
    final_rerank_report_path: str = "/kaggle/working/nemotron_advanced/final_rerank_report.csv"
    final_export_info_path: str = "/kaggle/working/nemotron_advanced/final_export_info.json"
    snapshot_root_dir: str = "/kaggle/working/nemotron_advanced/snapshots"
    best_score_epsilon: float = 1e-12
    topk_min_global_step: int = 0
    final_rerank_run_submission_style: bool = True
    final_rerank_submission_rows: int = 384

    # ===== stage1 -> stage2 inheritance =====
    stage2_inherit_best_stage1_candidate: bool = True
    stage2_inherit_fallback_to_final_stage1_state: bool = True

    stage1_inherit_use_serious_rerank: bool = True
    stage1_inherit_serious_top_n: int = 5
    stage1_inherit_serious_eval_rows: int = 192

    # ===== reproducibility =====
    seed: int = 3407
    candidate_train_seeds: Tuple[int, ...] = (3407, 1337, 2029)

    # ===== training =====
    use_bf16: bool = True
    use_4bit: bool = True
    max_seq_len: int = 4096
    answer_target_max_tokens: int = 256
    micro_batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_steps: int = 100
    eval_steps: int = 100
    max_grad_norm: float = 0.3
    enable_gradient_checkpointing: bool = True
    force_nemotron_torch_fallback: bool = False
    cuda_load_headroom_gib: int = 6

    # ===== lora =====
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # ===== split / eval =====
    valid_size: float = 0.08
    fast_eval_examples: int = 96
    serious_eval_examples: int = 256
    serious_eval_seeds: Tuple[int, ...] = (11, 23, 47)
    numeric_rel_tol: float = 1e-2
    prefer_official_metric_backend: bool = True
    official_metric_backend_path: Optional[str] = None

    # ===== curriculum =====
    stage1_epochs: float = 0.8
    stage2_epochs: float = 3.0
    stage2_rounds: int = 6
    stage1_max_prompt_chars: int = 1800
    stage1_relaxed_max_prompt_chars: int = 2200
    stage1_lr: float = 1.2e-4
    stage2_lr: float = 4e-5

    # ===== optional leaderboard tricks =====
    enable_external_mixture: bool = True
    enable_prompt_template_ablation: bool = True
    enable_family_reweight: bool = True
    enable_length_bucket_bonus: bool = True
    run_supervision_ablation: bool = False

    # ===== stage-specific supervision =====
    stage1_supervision_variant: str = "answer_only"
    stage2_supervision_variant: str = "family_aware_mix"
    valid_supervision_variant: str = "family_aware_mix"

    # 保留这个字段仅用于兼容旧代码 / ablation 打印，不再作为主训练选择开关
    primary_supervision_variant: str = "family_aware_mix"

    supervision_ablation_variants: Tuple[str, ...] = ("answer_only", "short_reasoning", "family_aware_mix")
    supervision_ablation_stage1_epochs: float = 0.15
    supervision_ablation_stage2_epochs: float = 0.25
    supervision_ablation_stage2_rounds: int = 1
    reasoning_template_eval_rows: int = 48
    enable_consensus_pseudolabel_refresh: bool = False
    consensus_pseudolabel_max_rows: int = 320
    consensus_pseudolabel_allowed_families: Tuple[str, ...] = ("bit_transform", "sequence", "matrix_reasoning")
    consensus_template_ids: Tuple[str, ...] = (
        "T1_ultra_compact",
        "T3_hidden_reasoning",
        "T4_numeric_specialized",
        "T5_text_specialized",
    )
    template_router_top_k: int = 5
    test_time_router_top_k: int = 4
    train_time_use_template_augmentation: bool = True
    train_time_template_top_k: int = 3
    fallback_template_id: str = "T2_compact"
    pseudolabel_min_confidence: float = 0.90
    pseudolabel_source_loss_weight: float = 0.4
    hard_mining_bottom_family_k: int = 5
    hard_mining_bottom_answer_type_k: int = 3
    hard_mining_bottom_bucket_k: int = 3
    template_ablation_after_stage1_rows: int = 192
    template_ablation_stage2_min_gain: float = 0.02
    template_ablation_family_min_rows: int = 10
    template_ablation_require_non_worse_boxed_metrics: bool = True
    template_ablation_secondary_seed: int = 97
    template_ablation_secondary_max_drop: float = 0.01
    template_ablation_strong_family_accuracy: float = 0.85
    template_ablation_strong_family_min_gain: float = 0.04
    stage2_asset_refresh_interval_rounds: int = 2
    stage2_refresh_template_eval_rows: int = 128
    stage2_refresh_weak_family_top_k: int = 3
    stage2_refresh_min_weak_family_gain: float = 0.005
    stage2_refresh_replay_family_error_threshold: float = 0.33
    fixed_sanity_rows: int = 64
    stage1_family_frequency_quantile: float = 0.45
    stage1_open_template_extra_ratio: float = 0.08
    stage1_multi_token_extra_ratio: float = 0.04
    hard_mining_family_boost: float = 0.35
    hard_mining_template_group_boost: float = 0.28
    hard_mining_answer_type_boost: float = 0.25
    hard_mining_bucket_boost: float = 0.20
    hard_mining_sample_boost: float = 0.55
    replay_probe_rows: int = 128
    short_text_weight_boost: float = 1.45
    open_template_short_text_extra_boost: float = 1.30
    cipher_decrypt_weight_boost: float = 1.35
    multi_token_text_weight_boost: float = 1.15
    allow_target_auto_discovery: bool = False

    # ===== runtime control =====
    run_stage1_multi_seed_eval: bool = False
    run_stage1_submission_style_eval: bool = True
    run_post_train_heavy_eval: bool = False
    save_stage1_snapshot: bool = True
    save_final_snapshot_before_post_eval: bool = True
    post_eval_fail_open: bool = True

    # ===== smoke test mode =====
    smoke_test_mode: bool = False
    smoke_train_rows: int = 24
    smoke_valid_rows: int = 12
    smoke_fast_eval_rows: int = 8
    smoke_serious_eval_rows: int = 12
    smoke_run_forward_check: bool = True
    smoke_run_generate_check: bool = True
    smoke_disable_rerank: bool = True

    smoke_profile: str = "pipeline"   # "fast" | "pipeline"
    smoke_train_max_steps_stage1: int = 1
    smoke_train_max_steps_stage2: int = 1
    smoke_run_export_reload_check: bool = True
    smoke_export_reload_rows: int = 2

    target_modules_final: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    test_time_template_candidates: Tuple[str, ...] = (
        "T1_ultra_compact",
        "T3_hidden_reasoning",
        "T4_numeric_specialized",
    )

# ===== 手动切换当前训练 seed：每次只改这里，然后 Run All =====
TRAIN_SEED = 3407   # 想试别的 seed，就改这里：3407 / 1337 / 2029

cfg = CFG()
cfg.seed = int(TRAIN_SEED)
cfg.force_nemotron_torch_fallback = False

# ===== 最终输出仍然固定为官方要求的 submission.zip =====
cfg.work_dir = "/kaggle/working/nemotron_advanced"
cfg.output_dir = f"{cfg.work_dir}/checkpoints"
cfg.adapter_dir = f"{cfg.work_dir}/adapter"
cfg.topk_candidate_dir = f"{cfg.work_dir}/topk_candidates"
cfg.topk_candidate_manifest_path = f"{cfg.work_dir}/topk_candidates_manifest.csv"
cfg.final_rerank_report_path = f"{cfg.work_dir}/final_rerank_report.csv"
cfg.final_export_info_path = f"{cfg.work_dir}/final_export_info.json"
cfg.snapshot_root_dir = f"{cfg.work_dir}/snapshots"
cfg.submission_zip = "/kaggle/working/submission.zip"

print(f"TRAIN SEED = {cfg.seed}")
print(f"WORK DIR = {cfg.work_dir}")
print(f"SUBMISSION ZIP = {cfg.submission_zip}")

if cfg.smoke_test_mode:
    print(f"SMOKE TEST MODE ENABLED: profile={cfg.smoke_profile}")

    # 这些重功能在 smoke 中继续关闭
    cfg.run_stage1_multi_seed_eval = False
    cfg.run_stage1_submission_style_eval = False
    cfg.run_post_train_heavy_eval = False
    cfg.run_supervision_ablation = False
    cfg.enable_consensus_pseudolabel_refresh = False
    cfg.enable_prompt_template_ablation = False

    cfg.save_stage1_snapshot = False
    cfg.save_final_snapshot_before_post_eval = False

    cfg.fast_eval_examples = min(cfg.fast_eval_examples, cfg.smoke_fast_eval_rows)
    cfg.serious_eval_examples = min(cfg.serious_eval_examples, cfg.smoke_serious_eval_rows)

    if cfg.smoke_disable_rerank:
        cfg.topk_candidate_keep = 0
        cfg.final_rerank_run_submission_style = False

    if cfg.smoke_profile == "fast":
        # 旧的极快 smoke：不测 trainer 主链
        cfg.stage1_epochs = 0.0
        cfg.stage2_epochs = 0.0
        cfg.stage2_rounds = 1

    elif cfg.smoke_profile == "pipeline":
        # 新的轻量全链路 smoke：只跑极少步数，但把主链走通
        cfg.stage1_epochs = 1.0
        cfg.stage2_epochs = 1.0
        cfg.stage2_rounds = 1

        cfg.grad_accum = 1
        cfg.logging_steps = 1
        cfg.save_steps = 1
        cfg.eval_steps = 1
        cfg.warmup_steps_override = 0

        # 再缩小一点数据量，让 smoke 更轻
        cfg.smoke_train_rows = min(cfg.smoke_train_rows, 16)
        cfg.smoke_valid_rows = min(cfg.smoke_valid_rows, 8)
        cfg.smoke_fast_eval_rows = min(cfg.smoke_fast_eval_rows, 4)
        cfg.smoke_serious_eval_rows = min(cfg.smoke_serious_eval_rows, 6)

    else:
        raise ValueError(f"Unknown smoke_profile: {cfg.smoke_profile}")

Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.adapter_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.topk_candidate_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.snapshot_root_dir).mkdir(parents=True, exist_ok=True)

assert cfg.lora_r <= cfg.max_lora_rank, "LoRA rank 超过比赛上限 32"
assert cfg.max_seq_len <= cfg.official_max_model_len, "训练 max_seq_len 超过官方 max_model_len"

# %% [markdown]
# ## Cell 3 — 固定随机种子 / 环境检查

# %%
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

seed_everything(cfg.seed)

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")



# %% [markdown]
# ## Cell 4 — 读取比赛数据 + 可选外挂数据
# 
# 这一步是高级版本与基础版的重要区别：
# - 主数据一定来自比赛官方 train.csv。
# - 额外数据采用**可插拔**接口，不强依赖，但为冲榜留足接口。
# - 如果你后续做 synthetic / self-distill / external reasoning mix，只需要拼成相同列结构即可。

# %%
all_train = list(Path("/kaggle/input").rglob("train.csv"))
all_test = list(Path("/kaggle/input").rglob("test.csv"))

print("Found train files:")
for p in all_train:
    print("  ", p)

print("Found test files:")
for p in all_test:
    print("  ", p)

# 优先使用明确指定的比赛目录，避免误读到其他 Dataset（例如 demo / extra 数据）
preferred_root = Path(cfg.competition_path)
preferred_train = preferred_root / "train.csv"
preferred_test = preferred_root / "test.csv"

if preferred_train.exists() and preferred_test.exists():
    data_root = preferred_root
    train_path = preferred_train
    test_path = preferred_test
else:
    # 回退：只保留同时存在 train.csv 和 test.csv 的目录
    candidate_roots = []
    for p in all_train:
        root = p.parent
        if (root / "test.csv").exists():
            candidate_roots.append(root)

    if not candidate_roots:
        raise FileNotFoundError("在 /kaggle/input 下没找到同时包含 train.csv 和 test.csv 的目录")

    # 若未命中 cfg.competition_path，则显式按路径排序，避免 rglob 顺序不稳定
    candidate_roots = sorted(candidate_roots, key=lambda x: str(x))
    data_root = candidate_roots[0]
    train_path = data_root / "train.csv"
    test_path = data_root / "test.csv"

print("Using data root:", data_root)
print("Using train:", train_path)
print("Using test :", test_path)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

assert {"id", "prompt", "answer"}.issubset(train_df.columns)
assert {"id", "prompt"}.issubset(test_df.columns)

train_df["source"] = "official_train"

def load_optional_external_data(extra_root: str) -> pd.DataFrame:
    """可选外挂数据接口。

    约定：如果你在 Kaggle Dataset 中准备了额外数据，整理成至少包含
    prompt / answer 两列的 parquet/csv 后，可以在这里自动加载。
    """
    root = Path(extra_root)
    candidates = sorted(root.glob("**/nemotron_extra_train*.csv")) + sorted(root.glob("**/nemotron_extra_train*.parquet"))
    frames: List[pd.DataFrame] = []

    for path in candidates:
        if path.suffix == ".csv":
            tmp = pd.read_csv(path)
        else:
            tmp = pd.read_parquet(path)
        if {"prompt", "answer"}.issubset(tmp.columns):
            tmp = tmp.copy()
            tmp["id"] = tmp.get("id", [f"extra_{i}" for i in range(len(tmp))])
            tmp["source"] = path.stem
            frames.append(tmp[["id", "prompt", "answer", "source"]])

    if not frames:
        return pd.DataFrame(columns=["id", "prompt", "answer", "source"])
    return pd.concat(frames, ignore_index=True)


def filter_external_data(df: pd.DataFrame) -> pd.DataFrame:
    """外部数据质量门槛：宁缺毋滥。"""
    if df.empty:
        return df

    tmp = df.copy()
    tmp["prompt"] = tmp["prompt"].astype(str)
    tmp["answer"] = tmp["answer"].astype(str).str.strip()
    tmp = tmp[tmp["prompt"].str.len().between(40, 12000)]
    tmp = tmp[tmp["answer"].str.len().between(1, 160)]
    tmp = tmp[tmp["prompt"].str.contains("->|example|Examples|Now|Solve|Task|Question", regex=True, na=False)]
    tmp = tmp[~tmp["answer"].str.contains("todo|unknown|n/a", case=False, na=False)]
    return tmp.reset_index(drop=True)


def load_optional_unlabeled_pool(extra_root: str) -> pd.DataFrame:
    root = Path(extra_root)
    candidates = sorted(root.glob("**/nemotron_extra_unlabeled*.csv")) + sorted(root.glob("**/nemotron_extra_unlabeled*.parquet"))
    frames: List[pd.DataFrame] = []

    for path in candidates:
        if path.suffix == ".csv":
            tmp = pd.read_csv(path)
        else:
            tmp = pd.read_parquet(path)
        if {"prompt"}.issubset(tmp.columns):
            tmp = tmp.copy()
            tmp["id"] = tmp.get("id", [f"unlabeled_{path.stem}_{i}" for i in range(len(tmp))])
            tmp["source"] = path.stem
            frames.append(tmp[["id", "prompt", "source"]])

    if not frames:
        return pd.DataFrame(columns=["id", "prompt", "source"])
    return pd.concat(frames, ignore_index=True)


def filter_unlabeled_pool(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["prompt"] = tmp["prompt"].astype(str)
    tmp = tmp[tmp["prompt"].str.len().between(40, 12000)]
    tmp = tmp[tmp["prompt"].str.contains("->|example|Examples|Now|Solve|Task|Question", regex=True, na=False)]
    return tmp.reset_index(drop=True)

external_df = load_optional_external_data(cfg.extra_data_dir) if cfg.enable_external_mixture else pd.DataFrame(columns=["id", "prompt", "answer", "source"])
external_df = filter_external_data(external_df)
unlabeled_external_df = filter_unlabeled_pool(load_optional_unlabeled_pool(cfg.extra_data_dir))
full_train_df = pd.concat([train_df, external_df], ignore_index=True)

print("official train:", train_df.shape)
print("external train:", external_df.shape)
print("external unlabeled pool:", unlabeled_external_df.shape)
print("full train:", full_train_df.shape)
display(full_train_df.head(2))

# %% [markdown]
# ## Cell 5 — 数据体检：prompt family / 答案类型 / 长度分布
# 
# 基础方案最大的问题是“随机切分 + 全量硬喂给模型”，这会让本地分数虚高、泛化偏差大。
# 这里先做结构分析，再按 family 切分和采样。

# %%
ANSWER_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)$")
BINARY_RE = re.compile(r"^[01]{4,}$")


def normalize_whitespace(x: str) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()


def infer_prompt_family(prompt: str) -> str:
    prompt_low = prompt.lower()
    if "determine the output for" in prompt_low and "bit manipulation" in prompt_low:
        return "bit_transform"
    if "decrypt the following text" in prompt_low:
        return "cipher_decrypt"
    if "here are some examples" in prompt_low and "now" in prompt_low:
        return "fewshot_pattern"
    if "matrix" in prompt_low:
        return "matrix_reasoning"
    if "sequence" in prompt_low or "next number" in prompt_low:
        return "sequence"
    return "open_template"


def normalized_template_fingerprint(prompt: str, prefix_chars: int = 160) -> str:
    prompt_low = normalize_whitespace(prompt).lower()
    prompt_low = re.sub(r"[01]{4,}", "<bin>", prompt_low)
    prompt_low = re.sub(r"\b\d+(\.\d+)?\b", "<num>", prompt_low)
    prompt_low = re.sub(r"\b[a-z]\b", "<var>", prompt_low)
    prompt_low = re.sub(r"[^a-z0-9<> ]+", " ", prompt_low)
    prompt_low = normalize_whitespace(prompt_low)[:prefix_chars]
    digest = hashlib.md5(prompt_low.encode("utf-8")).hexdigest()[:12]
    return f"fp_{digest}"


def infer_template_group(prompt: str) -> str:
    family = infer_prompt_family(prompt)
    if family != "open_template":
        return family
    return normalized_template_fingerprint(prompt)


def infer_answer_shape_from_gold(answer: str) -> str:
    answer = str(answer).strip()
    if BINARY_RE.fullmatch(answer):
        return "binary"
    if ANSWER_NUMBER_RE.fullmatch(answer.replace(",", "")):
        return "numeric"
    if " " in answer:
        return "multi_token_text"
    return "short_text"


def infer_expected_answer_type_from_prompt(prompt: str) -> str:
    text = normalize_whitespace(prompt).lower()
    binary_keywords = ["bit", "binary", "xor", "0/1", "flip", "bits", "ones", "zeros"]
    numeric_keywords = ["number", "digit", "value", "compute", "sequence", "matrix", "sum", "count", "next number"]
    text_keywords = ["word", "text", "string", "cipher", "decode", "decrypt", "letters", "phrase"]

    if any(keyword in text for keyword in binary_keywords):
        return "binary"
    if any(keyword in text for keyword in numeric_keywords):
        return "numeric"
    if any(keyword in text for keyword in text_keywords):
        return "short_text"
    return "short_text"

full_train_df["prompt_norm"] = full_train_df["prompt"].map(normalize_whitespace)
full_train_df["answer_norm"] = full_train_df["answer"].astype(str).str.strip()
full_train_df["prompt_family"] = full_train_df["prompt"].map(infer_prompt_family)
full_train_df["template_group"] = full_train_df["prompt"].map(infer_template_group)
full_train_df["answer_shape"] = full_train_df["answer"].map(infer_answer_shape_from_gold)
full_train_df["expected_answer_type"] = full_train_df["prompt"].map(infer_expected_answer_type_from_prompt)
full_train_df["answer_type"] = full_train_df["answer_shape"]
full_train_df["prompt_chars"] = full_train_df["prompt"].str.len()
full_train_df["answer_chars"] = full_train_df["answer_norm"].str.len()
full_train_df["example_len_bucket"] = pd.cut(
    full_train_df["prompt_chars"],
    bins=[0, 400, 800, 1400, 4000, 20000],
    labels=["xs", "s", "m", "l", "xl"],
    include_lowest=True,
)

print("prompt families top 15")
display(full_train_df["prompt_family"].value_counts().head(15).to_frame("count"))
print("template groups top 15")
display(full_train_df["template_group"].value_counts().head(15).to_frame("count"))
print("answer types")
display(full_train_df["answer_type"].value_counts().to_frame("count"))
print("length bucket")
display(full_train_df["example_len_bucket"].value_counts(dropna=False).sort_index().to_frame("count"))


# %% [markdown]
# ## Cell 6 — 去重 + group split
# 
# 按 prompt 文本去重，再按 family 做 group split，可以显著降低“模板泄漏”带来的验证幻觉。

# %%
full_train_df = full_train_df.drop_duplicates(subset=["prompt_norm", "answer_norm"]).reset_index(drop=True)

def family_aware_group_split(
    df: pd.DataFrame,
    valid_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    valid_parts = []

    for family, block in df.groupby("prompt_family", sort=True):
        block = block.reset_index(drop=True)
        group_count = block["template_group"].nunique()

        # 至少给这个 family 留一个有效 holdout
        target_valid_rows = max(1, int(round(len(block) * valid_size)))

        if len(block) <= 2 or group_count < 2:
            # 极小 family 的兜底
            if len(block) == 1:
                train_part = block.copy()
                valid_part = block.iloc[:0].copy()
            else:
                valid_part = block.sample(n=1, random_state=seed)
                train_part = block.drop(valid_part.index)
        else:
            # 先按 template_group 做 group split，避免模板泄漏
            group_test_size = min(max(valid_size, 1.0 / group_count), 0.5)
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=group_test_size,
                random_state=seed,
            )
            tr_idx, va_idx = next(splitter.split(block, groups=block["template_group"]))
            train_part = block.iloc[tr_idx].copy()
            valid_part = block.iloc[va_idx].copy()

            # 如果这个 family 在 valid 里太少，再从 train 中补一点不同 group
            if len(valid_part) < target_valid_rows:
                remain = train_part.loc[
                    ~train_part["template_group"].isin(valid_part["template_group"])
                ].copy()

                if not remain.empty:
                    extra_need = min(target_valid_rows - len(valid_part), len(remain))
                    extra = remain.sample(n=extra_need, random_state=seed)
                    valid_part = pd.concat([valid_part, extra], ignore_index=True)
                    train_part = train_part.drop(extra.index)

        train_parts.append(train_part)
        valid_parts.append(valid_part)

    train_out = (
        pd.concat(train_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    valid_out = (
        pd.concat(valid_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return train_out, valid_out


train_fold, valid_fold = family_aware_group_split(
    full_train_df,
    valid_size=cfg.valid_size,
    seed=cfg.seed,
)


def shrink_df_for_smoke(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.reset_index(drop=True)

    sampled = []
    groups = max(df["prompt_family"].nunique(), 1)
    per_family = max(1, max_rows // groups)

    for _, block in df.groupby("prompt_family", sort=True):
        sampled.append(block.head(per_family))

    out = pd.concat(sampled, ignore_index=True)

    if len(out) < max_rows:
        if "id" in df.columns and "id" in out.columns:
            used_ids = set(out["id"].tolist())
            remain = df.loc[~df["id"].isin(used_ids)].head(max_rows - len(out))
        else:
            remain = df.head(max_rows - len(out))
        out = pd.concat([out, remain], ignore_index=True)

    return out.head(max_rows).reset_index(drop=True)


if cfg.smoke_test_mode:
    train_fold = shrink_df_for_smoke(train_fold, cfg.smoke_train_rows)
    valid_fold = shrink_df_for_smoke(valid_fold, cfg.smoke_valid_rows)
    print("[smoke] shrunk train_fold:", train_fold.shape)
    print("[smoke] shrunk valid_fold:", valid_fold.shape)

print("train fold:", train_fold.shape)
print("valid fold:", valid_fold.shape)
print("train families:", train_fold["prompt_family"].nunique())
print("valid families:", valid_fold["prompt_family"].nunique())
print("train template groups:", train_fold["template_group"].nunique())
print("valid template groups:", valid_fold["template_group"].nunique())

# %% [markdown]
# ## Cell 7 — 竞赛近似 metric
# 
# 官方说明里最关键的要点：
# - 生成文本会优先抽取 `\\boxed{}` 中内容；
# - 若没有 boxed，则回退到其他启发式；
# - 支持精确字符串匹配或数值相对容差匹配。
# 
# 所以训练模板必须从 prompt 构造、答案格式、评估逻辑三方面同时对齐这个规则。

# %%
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
LAST_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
FINAL_ANSWER_PATTERNS = [
    re.compile(r"final answer\s*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"answer\s*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"therefore[, ]+the answer is\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"the answer is\s+(.+)$", re.IGNORECASE | re.MULTILINE),
]
LATEX_WRAPPER_RE = re.compile(r"^(?:\\text\{(.+)\}|\\mathrm\{(.+)\}|\\left\((.+)\\right\))$")
SAFE_EVAL_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
SAFE_EVAL_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def strip_latex_wrappers(ans: str) -> str:
    prev = None
    cur = ans.strip()
    while prev != cur:
        prev = cur
        match = LATEX_WRAPPER_RE.match(cur)
        if not match:
            break
        cur = next(group for group in match.groups() if group is not None).strip()
    return cur


def safe_numeric_eval(expr: str) -> Optional[float]:
    expr = expr.strip()
    if not expr:
        return None
    expr = expr.replace('^', '**')
    expr = expr.replace('{', '(').replace('}', ')')
    expr = re.sub(r"\\frac\s*\(([^()]+)\)\s*\(([^()]+)\)", r"(\1)/(\2)", expr)
    expr = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", expr)
    expr = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"math.sqrt(\1)", expr)
    expr = re.sub(r"\s+", "", expr)

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError('bad constant')
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_EVAL_BIN_OPS:
            return SAFE_EVAL_BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_EVAL_UNARY_OPS:
            return SAFE_EVAL_UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'math' and node.func.attr == 'sqrt' and len(node.args) == 1:
                return math.sqrt(_eval(node.args[0]))
        raise ValueError('unsupported expression')

    try:
        parsed = ast.parse(expr, mode='eval')
        return float(_eval(parsed))
    except Exception:
        return None


def canonicalize_answer(ans: Any) -> str:
    ans = str(ans).strip()
    ans = ans.replace('，', ',')
    ans = ans.rstrip('.。;； ')
    ans = re.sub(r'^[$`]+|[$`]+$', '', ans).strip()
    ans = re.sub(r'^\s*(?:final answer|answer)\s*[:：-]\s*', '', ans, flags=re.IGNORECASE)
    ans = strip_latex_wrappers(ans)

    if ans.startswith('\\boxed{') and ans.endswith('}'):
        ans = ans[len('\\boxed{'):-1].strip()

    numeric_candidate = ans.replace(',', '')
    if ANSWER_NUMBER_RE.fullmatch(numeric_candidate):
        numeric = float(numeric_candidate)
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, '.12g')

    fraction_candidate = ans.replace(' ', '')
    if re.fullmatch(r'[-+]?\d+/\d+', fraction_candidate):
        try:
            frac = Fraction(fraction_candidate)
            return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
        except Exception:
            pass

    evaluated = safe_numeric_eval(ans)
    if evaluated is not None and math.isfinite(evaluated):
        if float(evaluated).is_integer():
            return str(int(evaluated))
        return format(float(evaluated), '.12g')

    return normalize_whitespace(ans)


def boxed(ans: Any) -> str:
    clean = canonicalize_answer(ans)
    return clean if clean.startswith('\\boxed{') else f"\\boxed{{{clean}}}"


def extract_boxed(text: str) -> Optional[str]:
    text = str(text).strip()
    boxed_hits = BOXED_RE.findall(text)
    if boxed_hits:
        return canonicalize_answer(boxed_hits[-1])
    return None


def extract_final_answer_pattern(text: str) -> Optional[str]:
    text = str(text).strip()

    for pattern in FINAL_ANSWER_PATTERNS:
        hits = pattern.findall(text)
        if hits:
            candidate = normalize_whitespace(hits[-1])
            candidate = candidate.rstrip('.。 ')
            candidate = re.sub(r'^[$`]+|[$`]+$', '', candidate)
            candidate = re.sub(r'^[=:：-]\s*', '', candidate)
            if candidate:
                return canonicalize_answer(candidate)
    return None


def extract_other_heuristics(text: str) -> Optional[str]:
    text = str(text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        short_lines = [line for line in lines[-3:] if len(line) <= 128]
        for line in reversed(short_lines):
            candidate = re.sub(r'^[$`]+|[$`]+$', '', line).strip()
            if candidate:
                return canonicalize_answer(candidate)
    return None


def extract_last_numeric(text: str) -> Optional[str]:
    text = str(text).strip()
    last_num_hits = LAST_NUMBER_RE.findall(text.replace(',', ''))
    if last_num_hits:
        return canonicalize_answer(last_num_hits[-1])
    return None


def fast_extract_prediction(text: str) -> str:
    for fn in (extract_boxed, extract_last_numeric, extract_other_heuristics):
        result = fn(text)
        if result is not None:
            return result
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return canonicalize_answer(lines[-1] if lines else text)


def official_like_extract_prediction(text: str) -> str:
    for fn in (extract_boxed, extract_final_answer_pattern, extract_other_heuristics, extract_last_numeric):
        result = fn(text)
        if result is not None:
            return result
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return canonicalize_answer(lines[-1] if lines else text)


def extract_prediction(text: str) -> str:
    return official_like_extract_prediction(text)


def load_official_metric_backend() -> Optional[Any]:
    if not cfg.prefer_official_metric_backend:
        return None

    candidate_paths: List[Path] = []
    if cfg.official_metric_backend_path:
        candidate_paths.append(Path(cfg.official_metric_backend_path))
    candidate_paths.extend([
        Path(cfg.competition_path) / "evaluation.py",
        Path(cfg.competition_path) / "metric.py",
        Path("evaluation.py"),
        Path("metric.py"),
    ])

    for path in candidate_paths:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("nemotron_official_metric", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        extract_fn = getattr(module, "extract_prediction", None) or getattr(module, "extract_answer", None)
        score_fn = getattr(module, "score_prediction", None) or getattr(module, "score", None)
        if extract_fn is not None:
            print(f"loaded official metric backend from: {path}")
            return {"path": str(path), "extract_fn": extract_fn, "score_fn": score_fn}
    print("official metric backend not found; fallback to local official_like extractor")
    return None


OFFICIAL_METRIC_BACKEND = load_official_metric_backend()


def metric_extract_prediction(text: str) -> str:
    if OFFICIAL_METRIC_BACKEND is not None:
        return canonicalize_answer(OFFICIAL_METRIC_BACKEND["extract_fn"](text))
    return official_like_extract_prediction(text)


def approx_equal(pred: str, gold: str, rel_tol: float = cfg.numeric_rel_tol) -> bool:
    pred_c = canonicalize_answer(pred)
    gold_c = canonicalize_answer(gold)
    if OFFICIAL_METRIC_BACKEND is not None and OFFICIAL_METRIC_BACKEND.get("score_fn") is not None:
        return bool(OFFICIAL_METRIC_BACKEND["score_fn"](pred_c, gold_c))
    if pred_c == gold_c:
        return True

    pred_num = pred_c.replace(",", "")
    gold_num = gold_c.replace(",", "")
    if ANSWER_NUMBER_RE.fullmatch(pred_num) and ANSWER_NUMBER_RE.fullmatch(gold_num):
        p = float(pred_num)
        g = float(gold_num)
        if g == 0:
            return abs(p - g) <= rel_tol
        return abs(p - g) / max(abs(g), 1e-12) <= rel_tol
    return False



# %% [markdown]
# ## Cell 8 — Prompt 模板工程
# 
# 为什么这一步对冲榜重要：
# - 竞赛只评分最终答案，不奖励冗长解释；
# - 但模型仍然需要“推理倾向”；
# - 因此我们训练时使用**简洁、可控、强约束**模板，避免输出漫游。
# 
# 这里提供两个模板：
# - `template_compact`：更偏 leaderboard，尽量短输出。
# - `template_reasoning`：更偏鲁棒性，允许内部分析，但只暴露最终 boxed 答案。

# %%
SYSTEM_PROMPT = (
    "You are a careful reasoning model. Solve the task accurately. "
    "Return the final answer only, enclosed in \\boxed{} with no extra commentary."
)


def template_compact(problem: str) -> str:
    return textwrap.dedent(
        f"""\
        <system>
        {SYSTEM_PROMPT}
        </system>
        <user>
        {problem}
        </user>
        <assistant>
        """
    )


def template_reasoning(problem: str) -> str:
    return textwrap.dedent(
        f"""\
        <system>
        You are an expert competition reasoning assistant.
        Think carefully, keep hidden scratch work internal, and output only the final boxed answer.
        </system>
        <user>
        {problem}
        </user>
        <assistant>
        """
    )


def template_ultra_compact(problem: str) -> str:
    return textwrap.dedent(
        f"""\
        <system>
        Solve accurately. Output only the final answer in \\boxed{{}}.
        </system>
        <user>
        {problem}
        </user>
        <assistant>
        """
    )


def template_numeric(problem: str) -> str:
    return textwrap.dedent(
        f"""\
        <system>
        Infer the pattern precisely. Compute the final numeric or symbolic result and return only \\boxed{{answer}}.
        </system>
        <user>
        {problem}
        </user>
        <assistant>
        """
    )


def template_text(problem: str) -> str:
    return textwrap.dedent(
        f"""\
        <system>
        Infer the hidden rule from the examples.
        Return the shortest exact final text answer only in \\boxed{{}}.
        No explanation, no paraphrase, no extra words, no full sentence.
        </system>
        <user>
        {problem}
        </user>
        <assistant>
        """
    )


TEMPLATE_POOL = {
    "T1_ultra_compact": template_ultra_compact,
    "T2_compact": template_compact,
    "T3_hidden_reasoning": template_reasoning,
    "T4_numeric_specialized": template_numeric,
    "T5_text_specialized": template_text,
}


FAMILY_TEMPLATE_PRIORS: Dict[str, str] = {
    "bit_transform": "T4_numeric_specialized",
    "cipher_decrypt": "T3_hidden_reasoning",
    "fewshot_pattern": "T3_hidden_reasoning",
    "matrix_reasoning": "T3_hidden_reasoning",
    "sequence": "T4_numeric_specialized",
    "open_template": "T5_text_specialized",
}
BEST_TEMPLATE_BY_FAMILY: Dict[str, str] = dict(FAMILY_TEMPLATE_PRIORS)
FAMILY_TEMPLATE_ROUTER: Dict[str, List[str]] = {
    "bit_transform": ["T4_numeric_specialized", "T1_ultra_compact", "T3_hidden_reasoning"],
    "cipher_decrypt": ["T3_hidden_reasoning", "T5_text_specialized", "T1_ultra_compact"],
    "fewshot_pattern": ["T3_hidden_reasoning", "T1_ultra_compact", "T4_numeric_specialized"],
    "matrix_reasoning": ["T3_hidden_reasoning", "T4_numeric_specialized", "T1_ultra_compact"],
    "sequence": ["T3_hidden_reasoning", "T1_ultra_compact", "T4_numeric_specialized"],
    "open_template": ["T5_text_specialized", "T1_ultra_compact", "T3_hidden_reasoning", "T4_numeric_specialized"],
    "default": ["T1_ultra_compact", "T5_text_specialized", "T3_hidden_reasoning", "T4_numeric_specialized"],
}
BEST_TEMPLATE_ROUTER_BY_FAMILY: Dict[str, List[str]] = {
    family: list(router) for family, router in FAMILY_TEMPLATE_ROUTER.items()
}
CURRENT_WEAK_FAMILIES: List[str] = []


def choose_template_id(answer_type: str, prompt_family: str) -> str:
    # 先按“题型 + 答案类型”走强规则，不要先被 family prior 锁死
    if prompt_family == "cipher_decrypt":
        if answer_type in {"short_text", "multi_token_text"}:
            return "T3_hidden_reasoning"
        return "T5_text_specialized"

    if prompt_family == "open_template":
        if answer_type in {"short_text", "multi_token_text"}:
            return "T5_text_specialized"
        if answer_type in {"numeric", "binary"}:
            return "T4_numeric_specialized"
        return "T1_ultra_compact"

    if answer_type in {"numeric", "binary"}:
        return "T4_numeric_specialized"

    if answer_type in {"short_text", "multi_token_text"}:
        return "T5_text_specialized"

    if prompt_family in BEST_TEMPLATE_BY_FAMILY:
        return BEST_TEMPLATE_BY_FAMILY[prompt_family]

    if not cfg.enable_prompt_template_ablation:
        return "T2_compact"

    return "T3_hidden_reasoning"


def choose_template(problem: str, answer_type: str, prompt_family: str) -> Tuple[str, str]:
    template_id = choose_template_id(answer_type, prompt_family)
    return template_id, TEMPLATE_POOL[template_id](problem)


def router_get_templates(prompt_family: str, answer_type: Optional[str] = None, top_k: Optional[int] = None) -> List[str]:
    base_router = BEST_TEMPLATE_ROUTER_BY_FAMILY.get(
        prompt_family,
        BEST_TEMPLATE_ROUTER_BY_FAMILY.get("default", list(cfg.test_time_template_candidates)),
    )
    candidates = list(base_router)

    if answer_type in {"numeric", "binary"}:
        preferred = ["T4_numeric_specialized", "T1_ultra_compact", "T3_hidden_reasoning"]
        candidates = preferred + candidates

    elif prompt_family == "open_template" and answer_type in {"short_text", "multi_token_text"}:
        preferred = ["T5_text_specialized", "T3_hidden_reasoning", "T1_ultra_compact"]
        candidates = preferred + candidates

    elif prompt_family == "cipher_decrypt":
        preferred = ["T3_hidden_reasoning", "T5_text_specialized", "T1_ultra_compact"]
        candidates = preferred + candidates

    elif answer_type in {"short_text", "multi_token_text"}:
        preferred = ["T5_text_specialized", "T3_hidden_reasoning", "T1_ultra_compact"]
        candidates = preferred + candidates

    else:
        candidates = ["T1_ultra_compact", "T3_hidden_reasoning"] + candidates

    deduped: List[str] = []
    for template_id in candidates:
        if template_id in TEMPLATE_POOL and template_id not in deduped:
            deduped.append(template_id)

    keep = top_k if top_k is not None else cfg.test_time_router_top_k
    return deduped[: max(1, keep)]


def choose_train_template_ids(answer_type: str, prompt_family: str) -> List[str]:
    if not cfg.train_time_use_template_augmentation:
        return [choose_template_id(answer_type, prompt_family)]

    return router_get_templates(
        prompt_family,
        answer_type=answer_type,
        top_k=cfg.train_time_template_top_k,
    )

def get_weak_families_from_pred_df(pred_df: pd.DataFrame, top_k: Optional[int] = None) -> List[str]:
    if pred_df is None or pred_df.empty:
        return []
    family_acc = pred_df.groupby("prompt_family")["hit"].mean().sort_values()
    keep = min(max(top_k or cfg.hard_mining_bottom_family_k, 1), len(family_acc))
    return family_acc.head(keep).index.tolist()


def build_short_reasoning_scaffold(answer: Any, answer_type: str, prompt_family: str) -> str:
    final_boxed = boxed(answer)
    family_hint = {
        "bit_transform": "Pattern: identify the bit-level transformation shown in the examples.",
        "cipher_decrypt": "Pattern: infer the substitution rule from the examples.",
        "matrix_reasoning": "Pattern: detect the row/column relationship before computing the target entry.",
        "sequence": "Pattern: infer the sequence rule before applying it once.",
        "open_template": "Pattern: infer the latent rule from the demonstrations.",
    }.get(prompt_family, "Pattern: infer the latent rule from the demonstrations.")

    step_hint = {
        "binary": "Apply the rule once to the target and keep the output format exact.",
        "numeric": "Compute only the decisive intermediate step, then finalize.",
        "multi_token_text": "Apply the inferred mapping to the target phrase directly.",
        "short_text": "Apply the inferred mapping and keep the answer concise.",
    }.get(answer_type, "Apply the inferred rule directly.")

    return f"{family_hint}\n{step_hint}\nFinal answer: {final_boxed}"


# %% [markdown]
# ## Cell 9 — 构造训练样本（answer-only loss）
# 
# 竞赛的核心不是把整个 prompt 背下来，而是：
# - 在 prompt 条件下输出**正确 final answer**；
# - 并且答案格式与 metric 完全对齐。
# 
# 所以这里统一把监督目标收敛到：`prompt + boxed(answer)`，
# 且 label 仅覆盖答案 token，本质上是“条件答案建模”。

# %%

def build_training_record(
    row: pd.Series,
    template_id_override: Optional[str] = None,
) -> Dict[str, Any]:
    if template_id_override is None:
        template_id = choose_template_id(row.answer_type, row.prompt_family)
        record_id = row.id
    else:
        template_id = template_id_override
        record_id = f"{row.id}__{template_id}"

    prompt_text = TEMPLATE_POOL[template_id](row.prompt)
    answer_text = boxed(row.answer)
    short_reasoning_text = build_short_reasoning_scaffold(
        row.answer,
        row.answer_type,
        row.prompt_family,
    )

    full_text = prompt_text + answer_text
    reasoning_full_text = prompt_text + short_reasoning_text

    difficulty = 1.0
    difficulty += min(row.prompt_chars / 1800.0, 1.5)
    difficulty += 0.2 if row.answer_type == "multi_token_text" else 0.0
    difficulty += 0.15 if row.example_len_bucket in {"l", "xl"} else 0.0
    difficulty += 0.2 if row.prompt_family == "open_template" else 0.0

    return {
        "id": record_id,
        "source_example_id": row.id,
        "prompt_family": row.prompt_family,
        "template_group": row.template_group,
        "answer_type": row.answer_type,
        "template_id": template_id,
        "prompt_text": prompt_text,
        "answer_text": answer_text,
        "full_text": full_text,
        "reasoning_full_text": reasoning_full_text,
        "gold_answer": canonicalize_answer(row.answer),
        "source": row.source,
        "source_loss_weight": float(getattr(row, "source_loss_weight", 1.0)),
        "pseudo_confidence": float(getattr(row, "confidence", 1.0)),
        "difficulty": difficulty,
        "prompt_chars": row.prompt_chars,
        "len_bucket": str(row.example_len_bucket),
    }


def build_records_frame(
    df: pd.DataFrame,
    augment_templates: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "source_example_id",
                "prompt_family",
                "template_group",
                "answer_type",
                "template_id",
                "prompt_text",
                "answer_text",
                "full_text",
                "reasoning_full_text",
                "gold_answer",
                "source",
                "source_loss_weight",
                "pseudo_confidence",
                "difficulty",
                "prompt_chars",
                "len_bucket",
            ]
        )

    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        if augment_templates:
            template_ids = choose_train_template_ids(
                row.answer_type,
                row.prompt_family,
            )
            for template_id in template_ids:
                records.append(
                    build_training_record(
                        row,
                        template_id_override=template_id,
                    )
                )
        else:
            records.append(build_training_record(row))

    return pd.DataFrame(records)


def split_stage_records(records_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    family_freq = records_df["prompt_family"].value_counts(normalize=True)
    stable_families = set(
        family_freq[
            family_freq >= family_freq.quantile(cfg.stage1_family_frequency_quantile)
        ].index.tolist()
    )

    base_stage1_mask = (
        records_df["prompt_chars"].le(cfg.stage1_max_prompt_chars)
        & records_df["prompt_family"].isin(stable_families)
        & records_df["answer_type"].isin(["numeric", "binary", "short_text"])
    )

    stage1_base = records_df.loc[base_stage1_mask].copy()
    residual = records_df.loc[~base_stage1_mask].copy()

    extra_open_pool = residual.loc[
        residual["prompt_chars"].le(cfg.stage1_relaxed_max_prompt_chars)
        & residual["prompt_family"].isin(stable_families)
        & residual["prompt_family"].eq("open_template")
        & residual["answer_type"].isin(["numeric", "short_text"])
    ].copy()

    extra_multi_pool = residual.loc[
        residual["prompt_chars"].le(cfg.stage1_relaxed_max_prompt_chars)
        & residual["prompt_family"].isin(stable_families)
        & residual["answer_type"].eq("multi_token_text")
        & residual["prompt_family"].ne("cipher_decrypt")
    ].copy()

    extra_open_n = min(
        len(extra_open_pool),
        int(round(len(stage1_base) * cfg.stage1_open_template_extra_ratio)),
    )
    extra_multi_n = min(
        len(extra_multi_pool),
        int(round(len(stage1_base) * cfg.stage1_multi_token_extra_ratio)),
    )

    extra_frames = []
    if extra_open_n > 0:
        extra_frames.append(
            extra_open_pool.sample(n=extra_open_n, random_state=cfg.seed)
        )
    if extra_multi_n > 0:
        extra_frames.append(
            extra_multi_pool.sample(n=extra_multi_n, random_state=cfg.seed + 1)
        )

    if extra_frames:
        stage1_records_local = pd.concat(
            [stage1_base] + extra_frames,
            ignore_index=True,
        )
        stage1_records_local = (
            stage1_records_local
            .drop_duplicates(subset=["id"])
            .reset_index(drop=True)
        )
    else:
        stage1_records_local = stage1_base.reset_index(drop=True)

    stage2_records_local = records_df.reset_index(drop=True)
    return stage1_records_local, stage2_records_local, stable_families


train_records = build_records_frame(
    train_fold,
    augment_templates=cfg.train_time_use_template_augmentation,
)
valid_records = build_records_frame(
    valid_fold,
    augment_templates=False,
)

stage1_records, stage2_records, stable_families = split_stage_records(train_records)

print(train_records.shape, valid_records.shape)
display(train_records.head(2))

# %% [markdown]
# ## Cell 10 — Stage 1 / Stage 2 curriculum 切分
# 
# 两阶段训练比“一把梭全量训练”更稳定：
# - Stage 1 先学短 prompt、规则明显的题；
# - Stage 2 再纳入长文本和复杂 family。

# %%
print("stage1 records:", stage1_records.shape)
print(
    "stage1 unique source examples:",
    stage1_records["source_example_id"].nunique()
    if "source_example_id" in stage1_records.columns
    else stage1_records["id"].nunique(),
)

print("stage2 records:", stage2_records.shape)
print(
    "stage2 unique source examples:",
    stage2_records["source_example_id"].nunique()
    if "source_example_id" in stage2_records.columns
    else stage2_records["id"].nunique(),
)

print("stable families used in stage1:", sorted(stable_families)[:20])


# %% [markdown]
# ## Cell 11 — 加载 tokenizer

# %%
def load_tokenizer_local() -> AutoTokenizer:
    model_path = Path(cfg.base_model)

    if not model_path.exists():
        raise RuntimeError(
            f"Local model path does not exist: {cfg.base_model}"
        )

    print("Loading tokenizer from:", model_path)

    tokenizer_local = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    return tokenizer_local


tokenizer = load_tokenizer_local()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"
print("pad token:", tokenizer.pad_token)
print("eos token:", tokenizer.eos_token)


# %% [markdown]
# ## Cell 12 — Tokenize（只对答案部分算 loss）

# %%
def tokenize_answer_only(example: Dict[str, Any]) -> Dict[str, Any]:
    target_key = "reasoning_full_text" if example.get("supervision_variant") == "short_reasoning" else "full_text"

    prompt_text = example["prompt_text"]
    full_text = example[target_key]

    # full_text = prompt_text + target_suffix
    target_suffix = full_text[len(prompt_text):] + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    target_ids = tokenizer(
        target_suffix,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # 先保答案，再裁 prompt；避免长 prompt 把答案全部挤没
    target_ids = target_ids[: cfg.answer_target_max_tokens]
    prompt_budget = max(cfg.max_seq_len - len(target_ids), 0)

    # 保留 prompt 末尾
    prompt_ids = prompt_ids[-prompt_budget:]

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids.copy()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "gold_answer": example["gold_answer"],
        "prompt_text": example["prompt_text"],
        "difficulty": float(example["difficulty"]),
        "prompt_family": example["prompt_family"],
        "template_group": example["template_group"],
        "answer_type": example["answer_type"],
        "len_bucket": example["len_bucket"],
        "source": example["source"],
        "template_id": example["template_id"],
        "supervision_variant": example["supervision_variant"],
    }

def make_supervision_frame(df: pd.DataFrame, supervision_variant: str) -> pd.DataFrame:
    tmp = df.copy()

    if supervision_variant == "family_aware_mix":
        use_short_reasoning = (
            tmp["prompt_family"].eq("matrix_reasoning")
            & tmp["answer_type"].isin({"numeric", "binary"})
        )

        use_short_reasoning |= (
            tmp["prompt_family"].eq("cipher_decrypt")
            & tmp["answer_type"].isin({"short_text", "multi_token_text"})
        )

        use_short_reasoning |= (
            tmp["prompt_family"].eq("open_template")
            & tmp["answer_type"].isin({"short_text", "multi_token_text"})
            & tmp["prompt_chars"].ge(220)
        )

        tmp["supervision_variant"] = np.where(
            use_short_reasoning,
            "short_reasoning",
            "answer_only",
        )
    else:
        tmp["supervision_variant"] = supervision_variant

    return tmp

def build_variant_datasets(records_df: pd.DataFrame, variant: str) -> Dataset:
    ds = Dataset.from_pandas(make_supervision_frame(records_df, variant))
    ds = ds.map(tokenize_answer_only)
    removable = [
        c
        for c in ds.column_names
        if c
        not in {
            "input_ids",
            "attention_mask",
            "labels",
            "gold_answer",
            "prompt_text",
            "difficulty",
            "prompt_family",
            "template_group",
            "answer_type",
            "len_bucket",
            "source",
            "template_id",
            "supervision_variant",
        }
    ]
    return ds.remove_columns(removable)


def build_supervision_variant_panel(records_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in ["answer_only", "short_reasoning", "family_aware_mix"]:
        tmp = make_supervision_frame(records_df.copy(), variant)
        target_text = np.where(tmp["supervision_variant"].eq("answer_only"), tmp["full_text"], tmp["reasoning_full_text"])
        label_tokens = [
            max(
                len(tokenizer(text, add_special_tokens=False)["input_ids"]) - len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                0,
            )
            for text, prompt in zip(target_text, tmp["prompt_text"])
        ]
        tmp = tmp.assign(label_tokens=label_tokens)
        rows.append(
            {
                "variant": variant,
                "rows": len(tmp),
                "avg_label_tokens": float(np.mean(label_tokens)),
                "p90_label_tokens": float(np.quantile(label_tokens, 0.9)),
                "long_prompt_avg_label_tokens": float(tmp.loc[tmp["prompt_chars"] >= 1400, "label_tokens"].mean() if (tmp["prompt_chars"] >= 1400).any() else 0.0),
                "numeric_avg_label_tokens": float(tmp.loc[tmp["answer_type"] == "numeric", "label_tokens"].mean() if (tmp["answer_type"] == "numeric").any() else 0.0),
                "short_reasoning_ratio": float(tmp["supervision_variant"].eq("short_reasoning").mean()),
            }
        )
    return pd.DataFrame(rows)


def refresh_variant_datasets(
    stage1_records_df: pd.DataFrame,
    stage2_records_df: pd.DataFrame,
    valid_records_df: pd.DataFrame,
) -> Tuple[Dict[str, Dataset], Dict[str, Dataset], Dict[str, Dataset]]:
    variants = ["answer_only", "short_reasoning", "family_aware_mix"]
    stage1_variant_local = {
        variant: build_variant_datasets(stage1_records_df, variant)
        for variant in variants
    }
    stage2_variant_local = {
        variant: build_variant_datasets(stage2_records_df, variant)
        for variant in variants
    }
    valid_variant_local = {
        variant: build_variant_datasets(valid_records_df, variant)
        for variant in variants
    }
    return stage1_variant_local, stage2_variant_local, valid_variant_local


stage1_variant_ds, stage2_variant_ds, valid_variant_ds = refresh_variant_datasets(
    stage1_records,
    stage2_records,
    valid_records,
)

stage1_ds = stage1_variant_ds[cfg.stage1_supervision_variant]
stage2_ds = stage2_variant_ds[cfg.stage2_supervision_variant]
valid_ds = valid_variant_ds[cfg.valid_supervision_variant]

print("stage1 supervision variant:", cfg.stage1_supervision_variant)
print("stage2 supervision variant:", cfg.stage2_supervision_variant)
print("valid supervision variant :", cfg.valid_supervision_variant)

print(stage2_ds[0].keys())
print("non-masked labels:", sum(x != -100 for x in stage2_ds[0]["labels"]))

sample_non_masked = [
    sum(x != -100 for x in stage2_ds[i]["labels"])
    for i in range(min(64, len(stage2_ds)))
]
print("stage2 non-masked label stats:")
print("min =", min(sample_non_masked))
print("mean =", float(np.mean(sample_non_masked)))
print("p10 =", float(np.quantile(sample_non_masked, 0.1)))




# %% [markdown]
# ## Cell 12.5 — 强制关闭 Nemotron-H fast CUDA path（更稳妥版）
#
# 目标：
# 1. 关闭模块级 fast path 标志
# 2. 关闭类级别 forward -> cuda_kernels_forward 分支
# 3. 对已经实例化的模块再次做实例级补丁
#
# 这样做不会改变训练目标、LoRA 结构和 loss，只是把不兼容的 fused CUDA kernel
# 改成 torch fallback，保证 Kaggle 当前 GPU / 预编译环境可运行。

# %%
import types
import transformers.models.nemotron_h.modeling_nemotron_h as nemotron_h_mod

import functools
import inspect
import transformers.models.nemotron_h.modeling_nemotron_h as nemotron_h_mod


def patch_nemotron_create_causal_mask() -> None:
    """
    兼容旧参数名 input_embeds -> 新参数名 inputs_embeds
    不改变实际功能，只是消除 FutureWarning。
    """
    if not hasattr(nemotron_h_mod, "create_causal_mask"):
        print("create_causal_mask not found in nemotron_h module; skip patch.")
        return

    original_fn = nemotron_h_mod.create_causal_mask

    # 防止重复 patch
    if getattr(original_fn, "_patched_input_embeds_compat", False):
        print("create_causal_mask already patched.")
        return

    @functools.wraps(original_fn)
    def wrapped_create_causal_mask(*args, **kwargs):
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        return original_fn(*args, **kwargs)

    wrapped_create_causal_mask._patched_input_embeds_compat = True
    nemotron_h_mod.create_causal_mask = wrapped_create_causal_mask
    print("Patched nemotron_h.create_causal_mask: input_embeds -> inputs_embeds")


patch_nemotron_create_causal_mask()


def disable_nemotron_fast_path_globally() -> None:
    # 1) 模块级开关
    if hasattr(nemotron_h_mod, "is_fast_path_available"):
        nemotron_h_mod.is_fast_path_available = False
        print("Disabled Nemotron-H module-level fast CUDA path.")
    else:
        print("Nemotron-H module-level fast path flag not found.")

    # 2) 类级 monkey patch：直接让相关 mixer 永远走 torch_forward
    patched_classes = 0
    for name in dir(nemotron_h_mod):
        obj = getattr(nemotron_h_mod, name)
        if not isinstance(obj, type):
            continue

        has_torch_forward = hasattr(obj, "torch_forward")
        has_cuda_forward = hasattr(obj, "cuda_kernels_forward")
        has_forward = hasattr(obj, "forward")

        if has_torch_forward and has_cuda_forward and has_forward:
            # 保存原始 forward 便于排查（可选）
            if not hasattr(obj, "_orig_forward_for_debug"):
                obj._orig_forward_for_debug = obj.forward

            def _forward_fallback(self, hidden_states, cache_params=None, attention_mask=None):
                return self.torch_forward(hidden_states, cache_params, attention_mask)

            def _cuda_fallback(self, hidden_states, cache_params=None, attention_mask=None):
                return self.torch_forward(hidden_states, cache_params, attention_mask)

            obj.forward = _forward_fallback
            obj.cuda_kernels_forward = _cuda_fallback
            patched_classes += 1

    print(f"Patched {patched_classes} Nemotron class(es) to torch fallback.")


def patch_nemotron_mamba_modules_to_torch_fallback(model: torch.nn.Module) -> int:
    """
    对已经实例化的 Nemotron mixer 再做实例级补丁。
    这样即使类级 patch 没完全覆盖，实例也会被强制走 torch_forward。
    """
    patched = 0

    for module in model.modules():
        if (
            hasattr(module, "torch_forward")
            and hasattr(module, "cuda_kernels_forward")
            and hasattr(module, "forward")
            and hasattr(module, "in_proj")
        ):
            def _forward_fallback(self, hidden_states, cache_params=None, attention_mask=None):
                return self.torch_forward(hidden_states, cache_params, attention_mask)

            def _cuda_fallback(self, hidden_states, cache_params=None, attention_mask=None):
                return self.torch_forward(hidden_states, cache_params, attention_mask)

            module.forward = types.MethodType(_forward_fallback, module)
            module.cuda_kernels_forward = types.MethodType(_cuda_fallback, module)

            if hasattr(module, "is_fast_path_available"):
                try:
                    module.is_fast_path_available = False
                except Exception:
                    pass

            patched += 1

    return patched


def force_nemotron_torch_fallback(model: torch.nn.Module, tag: str = "") -> int:
    if not cfg.force_nemotron_torch_fallback:
        prefix = f"[{tag}] " if tag else ""
        print(f"{prefix}Keeping Nemotron native CUDA fast path enabled.")
        return 0

    disable_nemotron_fast_path_globally()
    patched = patch_nemotron_mamba_modules_to_torch_fallback(model)
    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}Patched {patched} instantiated Nemotron mixer module(s) to torch fallback.")
    return patched


# 仅在显式要求时才执行全局 patch；默认保留原生 CUDA fast path，避免 torch fallback 的显存膨胀。
if cfg.force_nemotron_torch_fallback:
    disable_nemotron_fast_path_globally()
# %% [markdown]
# ## Cell 13 — 稳定版模型加载 + LoRA target 自动发现 + OOM 清理

# %%
import gc
import math
import os
import importlib
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


# 加载时保守一点，避免 max_memory 估计过激进
cfg.cuda_load_headroom_gib = max(12, int(getattr(cfg, "cuda_load_headroom_gib", 12)))


def discover_lora_targets(named_modules: Iterable[Tuple[str, torch.nn.Module]]) -> List[str]:
    candidates = []
    wanted_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    for name, _module in named_modules:
        suffix = name.split(".")[-1]
        if suffix in wanted_suffixes:
            candidates.append(suffix)

    ordered = []
    for suffix in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        if suffix in candidates:
            ordered.append(suffix)
    return ordered


def package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def cleanup_cuda_before_model_load(verbose: bool = True) -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if verbose:
            print(
                f"CUDA free before load: "
                f"{free_bytes / (1024 ** 3):.2f} / {total_bytes / (1024 ** 3):.2f} GiB"
            )


def build_quant_config() -> Optional[BitsAndBytesConfig]:
    if not cfg.use_4bit:
        return None

    if not package_available("bitsandbytes"):
        print("bitsandbytes is not available locally; disabling 4-bit quantization.")
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def resolve_attn_implementation() -> Optional[str]:
    # 保守稳定路径，避免额外 kernel 问题
    return "eager"


def build_model_max_memory() -> Optional[Dict[Union[int, str], str]]:
    if not torch.cuda.is_available():
        return None

    max_memory: Dict[Union[int, str], str] = {}
    reserve_gib = max(4, int(cfg.cuda_load_headroom_gib))

    for device_idx in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
        free_gib = free_bytes / (1024 ** 3)
        total_gib = total_bytes / (1024 ** 3)

        # 用“当前空闲显存”而不是“总显存”估计
        usable_gib = max(1, math.floor(min(free_gib, total_gib) - reserve_gib))
        max_memory[device_idx] = f"{usable_gib}GiB"

    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        total_cpu_gib = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
        cpu_usable = max(16, math.floor(total_cpu_gib - 8))
        max_memory["cpu"] = f"{cpu_usable}GiB"

    print("max_memory plan:", max_memory)
    return max_memory


def build_model_load_kwargs(
    quant_config: Optional[BitsAndBytesConfig],
    attn_implementation: Optional[str],
) -> Dict[str, Any]:
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if cfg.use_bf16 else torch.float16,
        "trust_remote_code": False,
        "local_files_only": True,
    }

    if torch.cuda.is_available():
        if quant_config is not None:
            # 训练主模型时，避免 auto device_map + accelerate/offload 把部分参数留在 meta
            # 单卡量化加载更稳，和后面 candidate reload 的思路保持一致
            load_kwargs["device_map"] = {"": 0}
            load_kwargs["low_cpu_mem_usage"] = False
        else:
            # 非量化 fallback 只作为兜底；保留 auto，尽量让它还能尝试加载
            load_kwargs["device_map"] = "auto"
            load_kwargs["low_cpu_mem_usage"] = True
            max_memory = build_model_max_memory()
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["low_cpu_mem_usage"] = True

    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    return load_kwargs

def find_meta_tensors(model: torch.nn.Module, max_show: int = 20) -> List[str]:
    bad = []

    for name, param in model.named_parameters():
        try:
            if param.device.type == "meta":
                bad.append(f"param::{name}")
        except Exception:
            pass

    for name, buf in model.named_buffers():
        try:
            if buf.device.type == "meta":
                bad.append(f"buffer::{name}")
        except Exception:
            pass

    if len(bad) > max_show:
        bad = bad[:max_show] + [f"... ({len(bad)} total meta tensors)"]

    return bad


def load_training_model() -> Tuple[torch.nn.Module, List[str]]:
    cleanup_cuda_before_model_load()

    quant_config = build_quant_config()
    attn_implementation = resolve_attn_implementation()

    attempt_specs: List[Dict[str, Any]] = []

    # 只有真的有 4-bit 配置时，才尝试量化加载
    if quant_config is not None:
        attempt_specs.append(
            {
                "label": "quantized_eager_builtin",
                "quant_config": quant_config,
                "attn_implementation": attn_implementation,
            }
        )

    # 永远保留一个非量化 fallback
    attempt_specs.append(
        {
            "label": "fp16_or_bf16_eager_builtin_fallback",
            "quant_config": None,
            "attn_implementation": attn_implementation,
        }
    )

    errors: List[str] = []
    model: Optional[torch.nn.Module] = None
    used_spec: Optional[Dict[str, Any]] = None

    for spec in attempt_specs:
        cleanup_cuda_before_model_load(verbose=True)

        try:
            print(f"Trying load mode: {spec['label']}")
            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model,
                **build_model_load_kwargs(
                    quant_config=spec["quant_config"],
                    attn_implementation=spec["attn_implementation"],
                ),
            )
            used_spec = spec
            break

        except Exception as exc:
            errors.append(f"{spec['label']}: {type(exc).__name__}: {exc}")

            if model is not None:
                try:
                    del model
                except Exception:
                    pass
                model = None

            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    if model is None:
        raise RuntimeError(
            "Unable to load the base model in the current offline environment.\n"
            "Attempts:\n" + "\n".join(errors)
        )

    print(
        "model load mode:",
        {
            "label": used_spec["label"],
            "quantized": used_spec["quant_config"] is not None,
            "trust_remote_code": False,
            "attn_implementation": used_spec["attn_implementation"],
            "local_files_only": True,
        },
    )

    # 1) 刚加载完先打一次 Nemotron fallback 补丁
    force_nemotron_torch_fallback(model, tag="after_load")

    model.config.use_cache = False

    # 2) 只有真的量化加载了，才做 k-bit prepare
    if used_spec["quant_config"] is not None:
        model = prepare_model_for_kbit_training(model)
        force_nemotron_torch_fallback(model, tag="after_kbit_prepare")

    # 某些场景下需要输入支持梯度
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    supports_gc = bool(getattr(model, "supports_gradient_checkpointing", False))
    if cfg.enable_gradient_checkpointing and supports_gc:
        model.gradient_checkpointing_enable()
        print("gradient checkpointing enabled.")
    else:
        cfg.enable_gradient_checkpointing = False
        print(f"{model.__class__.__name__} does not support gradient checkpointing; disabled.")

    discovered_targets = discover_lora_targets(model.named_modules())

    if cfg.allow_target_auto_discovery:
        lora_targets = discovered_targets
    else:
        lora_targets = [
            module_name
            for module_name in cfg.target_modules_final
            if module_name in discovered_targets
        ]

    assert lora_targets, "未找到可用的 LoRA target modules，请检查模型结构"

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )

    model = get_peft_model(model, lora_config)

    # 3) PEFT 包装后再补一次，防止真正训练对象回到 fast CUDA path
    force_nemotron_torch_fallback(model, tag="after_peft")

    meta_tensors = find_meta_tensors(model)
    if meta_tensors:
        raise RuntimeError(
            "Model still contains meta tensors after load/PEFT patch. "
            "This usually means auto device_map/offload left some parameters unmaterialized.\n"
            f"Examples:\n" + "\n".join(meta_tensors)
        )

    return model, lora_targets

def get_model_input_device(model: torch.nn.Module) -> torch.device:
    # 优先找第一个非 meta 参数的设备
    for p in model.parameters():
        try:
            if p.device.type != "meta":
                return p.device
        except Exception:
            pass

    # 再看 hf_device_map
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for v in hf_device_map.values():
            if isinstance(v, int):
                return torch.device(f"cuda:{v}")
            if isinstance(v, str) and v.startswith("cuda"):
                return torch.device(v)

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_smoke_model_sanity_checks(model, tokenizer, sanity_df: pd.DataFrame) -> None:
    if sanity_df is None or sanity_df.empty:
        print("[smoke] sanity_df is empty, skip.")
        return

    meta_tensors = find_meta_tensors(model)
    if meta_tensors:
        raise RuntimeError(
            "Smoke sanity aborted because model still has meta tensors.\n"
            + "\n".join(meta_tensors)
        )

    row = sanity_df.iloc[0]
    prompt_text = template_compact(row["prompt"])

    toks = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=min(cfg.max_seq_len, 512),
    )

    device = get_model_input_device(model)
    toks = {k: v.to(device) for k, v in toks.items()}

    if cfg.smoke_run_forward_check:
        with torch.no_grad():
            out = model(**toks)
        print("[smoke] forward ok; logits shape:", tuple(out.logits.shape))

    if cfg.smoke_run_generate_check:
        with torch.no_grad():
            gen = model.generate(
                **toks,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        preview = tokenizer.decode(gen[0], skip_special_tokens=True)
        print("[smoke] generate ok; preview:")
        print(preview[:400])

model, lora_targets = load_training_model()
print("LoRA targets:", lora_targets)
model.print_trainable_parameters()

if cfg.smoke_test_mode:
    print("[smoke] running model sanity checks...")
    run_smoke_model_sanity_checks(model, tokenizer, valid_fold.head(1))

# %% [markdown]
# ## Cell 14 — 自定义 sampler：family reweight + length bonus + difficulty curriculum
# 
# 这一步是真正的“高级训练味道”：
# - 稀有 family 给予更高采样权重，避免被头部模板淹没；
# - 中长样本给予适度 bonus，提高真实测试稳健性；
# - difficulty 进入 curriculum 权重，而不是只做硬过滤。

# %%

def compute_sample_weights(
    df_like: pd.DataFrame,
    hard_profile: Optional[Dict[str, Dict[str, float]]] = None,
    replay_buffer: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    family_counts = df_like["prompt_family"].value_counts().to_dict()
    bucket_counts = df_like["len_bucket"].value_counts().to_dict()
    weights = []

    for row in df_like.itertuples(index=False):
        weight = 1.0
        if cfg.enable_family_reweight:
            weight *= 1.0 / math.sqrt(family_counts[row.prompt_family])
        if cfg.enable_length_bucket_bonus:
            weight *= {"xs": 0.9, "s": 1.0, "m": 1.08, "l": 1.15, "xl": 1.2}.get(row.len_bucket, 1.0)
            weight *= 1.0 / math.sqrt(bucket_counts.get(row.len_bucket, 1))
        weight *= float(row.difficulty)
        weight *= float(getattr(row, "source_loss_weight", 1.0))
        weight *= 0.8 + 0.2 * float(getattr(row, "pseudo_confidence", 1.0))

        if row.answer_type == "short_text":
            weight *= cfg.short_text_weight_boost

        if row.answer_type == "multi_token_text":
            weight *= cfg.multi_token_text_weight_boost

        if row.prompt_family == "open_template" and row.answer_type in {"short_text", "multi_token_text"}:
            weight *= cfg.open_template_short_text_extra_boost

        if row.prompt_family == "cipher_decrypt":
            weight *= cfg.cipher_decrypt_weight_boost

        if hard_profile is not None:
            weight *= 1.0 + cfg.hard_mining_family_boost * hard_profile["family"].get(row.prompt_family, 0.0)
            weight *= 1.0 + cfg.hard_mining_template_group_boost * hard_profile["template_group"].get(row.template_group, 0.0)
            weight *= 1.0 + cfg.hard_mining_answer_type_boost * hard_profile["answer_type"].get(row.answer_type, 0.0)
            weight *= 1.0 + cfg.hard_mining_bucket_boost * hard_profile["len_bucket"].get(row.len_bucket, 0.0)
        if replay_buffer is not None:
            weight *= 1.0 + cfg.hard_mining_sample_boost * replay_buffer.get(row.id, 0)
        weights.append(weight)

    arr = np.asarray(weights, dtype=np.float64)
    arr = arr / arr.mean()
    return arr

stage1_weights = compute_sample_weights(stage1_records) if len(stage1_records) else np.array([])
stage2_weights = compute_sample_weights(stage2_records)

print("stage2 weight summary:", np.quantile(stage2_weights, [0, 0.25, 0.5, 0.75, 1]))

# %% [markdown]
# ## Cell 15 — 自定义 Trainer：WeightedRandomSampler

# %%
class WeightedTrainer(Trainer):
    def __init__(self, *args, sample_weights: Optional[Sequence[float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if self.sample_weights is None:
            return super().get_train_dataloader()

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(self.sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            pin_memory=torch.cuda.is_available(),
        )


# %% [markdown]
# ## Cell 16 — 本地生成评估 callback
# 
# 训练 loss 并不等价于 leaderboard accuracy，所以需要在验证集上做小样本生成评估。
# 这里按 family 均衡抽样，而不是只抽前 N 条。

# %%

def stratified_valid_subset(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    per_family = max(1, max_rows // max(df["prompt_family"].nunique(), 1))
    parts = []
    for _family, block in df.groupby("prompt_family"):
        parts.append(block.head(per_family))
    out = pd.concat(parts, ignore_index=True)
    return out.head(max_rows)


def build_fixed_sanity_subset(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    sampled = []
    rng = np.random.default_rng(cfg.seed)
    for _family, block in df.groupby("prompt_family", sort=True):
        answer_order = {"numeric": 0, "binary": 1, "short_text": 2, "multi_token_text": 3}
        block = block.assign(_answer_rank=block["answer_type"].map(answer_order).fillna(99))
        block = block.sort_values(["_answer_rank", "prompt_chars", "id"]).drop(columns="_answer_rank")
        take = min(max(1, max_rows // max(df["prompt_family"].nunique(), 1)), len(block))
        sampled.append(block.head(take))
    out = pd.concat(sampled, ignore_index=True)
    if len(out) > max_rows:
        keep_idx = rng.choice(len(out), size=max_rows, replace=False)
        out = out.iloc[np.sort(keep_idx)].reset_index(drop=True)
    return out


def build_seeded_family_balanced_subset(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = []
    groups = max(df["prompt_family"].nunique(), 1)
    per_family = max(1, max_rows // groups)
    for _family, block in df.groupby("prompt_family", sort=True):
        if len(block) <= per_family:
            sampled.append(block)
        else:
            sampled.append(block.sample(n=per_family, random_state=seed))
    out = pd.concat(sampled, ignore_index=True)
    if len(out) > max_rows:
        keep_idx = rng.choice(len(out), size=max_rows, replace=False)
        out = out.iloc[np.sort(keep_idx)].reset_index(drop=True)
    return out


fast_eval_rows = cfg.smoke_fast_eval_rows if cfg.smoke_test_mode else cfg.fast_eval_examples
serious_eval_rows = cfg.smoke_serious_eval_rows if cfg.smoke_test_mode else cfg.serious_eval_examples
fixed_sanity_rows = min(cfg.fixed_sanity_rows, 8) if cfg.smoke_test_mode else cfg.fixed_sanity_rows
replay_probe_rows = min(cfg.replay_probe_rows, 8) if cfg.smoke_test_mode else cfg.replay_probe_rows

fast_eval_df = stratified_valid_subset(valid_fold, fast_eval_rows)
serious_eval_df = build_seeded_family_balanced_subset(valid_fold, serious_eval_rows, seed=cfg.seed)
serious_eval_views = {
    seed: build_seeded_family_balanced_subset(valid_fold, serious_eval_rows, seed=seed)
    for seed in cfg.serious_eval_seeds
}
fixed_sanity_df = build_fixed_sanity_subset(valid_fold, fixed_sanity_rows)
train_replay_probe_df = build_fixed_sanity_subset(train_fold, replay_probe_rows)

print("fast eval subset:", fast_eval_df.shape)
print("serious eval subset:", serious_eval_df.shape)
print("serious eval view seeds:", {seed: df.shape for seed, df in serious_eval_views.items()})
print("fixed sanity subset:", fixed_sanity_df.shape)
print("train replay probe subset:", train_replay_probe_df.shape)


@torch.no_grad()
def generate_answer_text(model: torch.nn.Module, prompt: str, max_new_tokens: int = 96) -> str:
    prev_training = model.training
    prev_use_cache = getattr(model.config, "use_cache", None)

    try:
        model.eval()
        if prev_use_cache is not None:
            model.config.use_cache = True

        # 推理时也和训练一样：优先保留 prompt 末尾
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        prompt_budget = max(min(cfg.max_seq_len, cfg.official_max_model_len - max_new_tokens), 1)
        prompt_ids = prompt_ids[-prompt_budget:]

        inputs = {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=model.device),
            "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=model.device),
        }
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        gen_ids = outputs[0][input_len:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    finally:
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache
        if prev_training:
            model.train()

@torch.no_grad()
def inspect_generation_completion_sanity(
    model: torch.nn.Module,
    df: pd.DataFrame,
    rows: int = 3,
    max_new_tokens: int = 96,
) -> pd.DataFrame:
    samples = []
    for row in df.head(rows).itertuples(index=False):
        prompt_family = infer_prompt_family(row.prompt)
        answer_type = infer_expected_answer_type_from_prompt(row.prompt)
        template_id, prompt = choose_template(row.prompt, answer_type, prompt_family)
        raw = generate_answer_text(model, prompt, max_new_tokens=max_new_tokens).strip()
        samples.append(
            {
                "id": row.id,
                "template_id": template_id,
                "prompt_prefix": prompt[:160],
                "raw_completion": raw[:400],
                "contains_prompt_echo": normalize_whitespace(prompt[:120]) in normalize_whitespace(raw),
                "pred": metric_extract_prediction(raw),
            }
        )
    return pd.DataFrame(samples)


@torch.no_grad()
def predict_one_row(
    model: torch.nn.Module,
    row: Any,
    template_ids: Optional[Sequence[str]] = None,
    fallback_template_id: Optional[str] = None,
    max_new_tokens_grid: Sequence[int] = (64, 96),
    extractor_fn=metric_extract_prediction,
) -> Dict[str, Any]:
    prompt_text = row.prompt if hasattr(row, "prompt") else row["prompt"]
    if hasattr(row, "answer"):
        gold = canonicalize_answer(row.answer)
    elif isinstance(row, dict) and "answer" in row:
        gold = canonicalize_answer(row["answer"])
    else:
        gold = ""
    prompt_family = infer_prompt_family(prompt_text)
    answer_type = infer_expected_answer_type_from_prompt(prompt_text)
    routed_templates = list(template_ids or router_get_templates(prompt_family, answer_type=answer_type))
    final_candidates: List[Dict[str, Any]] = []
    for max_new_tokens in max_new_tokens_grid:
        final_candidates = infer_with_multi_templates(
            model,
            {"prompt": prompt_text},
            template_ids=routed_templates,
            max_new_tokens=max_new_tokens,
            extractor_fn=extractor_fn,
        )
        aggregate = aggregate_candidates(final_candidates, answer_type=answer_type, preferred_template_ids=routed_templates)
        if aggregate["decision_type"] == "majority":
            break

    aggregate = aggregate_candidates(final_candidates, answer_type=answer_type, preferred_template_ids=routed_templates)
    fallback_used = False
    fallback_id = fallback_template_id or cfg.fallback_template_id
    if aggregate["needs_fallback"] and fallback_id not in routed_templates and fallback_id in TEMPLATE_POOL:
        fallback_used = True
        fallback_candidates = infer_with_multi_templates(
            model,
            {"prompt": prompt_text},
            template_ids=[fallback_id],
            max_new_tokens=max_new_tokens_grid[-1],
            extractor_fn=extractor_fn,
        )
        final_candidates = list(final_candidates) + list(fallback_candidates)
        aggregate = aggregate_candidates(
            final_candidates,
            answer_type=answer_type,
            preferred_template_ids=[fallback_id] + routed_templates,
        )

    return {
        "pred": aggregate["pred"],
        "consensus_votes": aggregate["vote_count"],
        "num_templates": len(final_candidates),
        "decision_type": aggregate["decision_type"],
        "fallback_used": fallback_used,
        "template_ids": [c["template_id"] for c in final_candidates],
        "candidates": final_candidates,
        "prompt_family": prompt_family,
        "answer_type": answer_type,
        "gold": gold,
    }


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    df: pd.DataFrame,
    template_override: Optional[str] = None,
    max_new_tokens_grid: Sequence[int] = (64, 96),
    extractor_fn=metric_extract_prediction,
    router_top_k: Optional[int] = None,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for row in df.itertuples(index=False):
        prompt_family = infer_prompt_family(row.prompt)
        answer_type = infer_expected_answer_type_from_prompt(row.prompt)
        routed_templates = [template_override] if template_override is not None else router_get_templates(
            prompt_family,
            answer_type=answer_type,
            top_k=router_top_k,
        )
        result = predict_one_row(
            model,
            row,
            template_ids=routed_templates,
            fallback_template_id=None if template_override is not None else cfg.fallback_template_id,
            max_new_tokens_grid=max_new_tokens_grid,
            extractor_fn=extractor_fn,
        )
        matched_candidate = next((c for c in result["candidates"] if c.get("answer") == result["pred"]), None)
        candidate_raw = matched_candidate["raw"] if matched_candidate is not None else (result["candidates"][0]["raw"] if result["candidates"] else "")
        boxed_hit = any(candidate.get("has_boxed", False) for candidate in result["candidates"])
        boxed_parsed_success = any(bool(candidate.get("boxed_answer")) for candidate in result["candidates"])
        gold = canonicalize_answer(row.answer)
        candidate_pred = result["pred"]
        if approx_equal(candidate_pred, gold):
            failure_type = "correct"
        elif not boxed_hit:
            failure_type = "missing_boxed"
        elif answer_type in {"numeric", "binary"}:
            pred_num = candidate_pred.replace(",", "")
            gold_num = gold.replace(",", "")
            if ANSWER_NUMBER_RE.fullmatch(pred_num) and ANSWER_NUMBER_RE.fullmatch(gold_num):
                rel_err = abs(float(pred_num) - float(gold_num)) / max(abs(float(gold_num)), 1e-12)
                failure_type = "wrong_numeric_close" if rel_err <= 0.05 else "wrong_numeric_far"
            else:
                failure_type = "extraction_mismatch"
        else:
            failure_type = "wrong_text"
        rows.append(
            {
                "id": row.id,
                "prompt_family": prompt_family,
                "template_group": infer_template_group(row.prompt),
                "answer_type": answer_type,
                "len_bucket": pd.cut(
                    [len(str(row.prompt))],
                    bins=[0, 400, 800, 1400, 4000, 20000],
                    labels=["xs", "s", "m", "l", "xl"],
                    include_lowest=True,
                )[0],
                "source": getattr(row, "source", "official_train"),
                "template_id": result["template_ids"][0] if result["template_ids"] else "",
                "gold": gold,
                "pred": candidate_pred,
                "hit": approx_equal(candidate_pred, gold),
                "has_boxed": boxed_hit,
                "boxed_parsed_success": boxed_parsed_success,
                "failure_type": failure_type,
                "raw": candidate_raw[:1200],
                "consensus_votes": result["consensus_votes"],
                "num_templates": result["num_templates"],
                "decision_type": result["decision_type"],
                "fallback_used": result["fallback_used"],
            }
        )
    out = pd.DataFrame(rows)
    acc = float(out["hit"].mean()) if len(out) else 0.0
    return acc, out


def summarize_eval_metrics(pred_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    summaries = {
        "overall": pd.DataFrame(
            [
                {
                    "metric": "accuracy",
                    "value": pred_df["hit"].mean(),
                    "has_boxed_rate": pred_df["has_boxed"].mean(),
                    "boxed_parse_rate": pred_df["boxed_parsed_success"].mean(),
                }
            ]
        )
    }
    for column in ["prompt_family", "template_group", "answer_type", "len_bucket", "source", "template_id", "failure_type"]:
        group_df = (
            pred_df.groupby(column, dropna=False)
            .agg(
                samples=("hit", "size"),
                accuracy=("hit", "mean"),
                has_boxed_rate=("has_boxed", "mean"),
                boxed_parse_rate=("boxed_parsed_success", "mean"),
            )
            .sort_values(["accuracy", "samples"], ascending=[False, False])
            .reset_index()
        )
        summaries[column] = group_df
    return summaries


def print_eval_summaries(pred_df: pd.DataFrame, prefix: str) -> None:
    summaries = summarize_eval_metrics(pred_df)
    print(f"\n[{prefix}] overall")
    display(summaries["overall"])
    for column in ["prompt_family", "template_group", "answer_type", "len_bucket", "source", "template_id", "failure_type"]:
        print(f"\n[{prefix}] grouped by {column}")
        display(summaries[column].head(12))


@torch.no_grad()
def evaluate_multi_seed_views(
    model: torch.nn.Module,
    eval_views: Dict[int, pd.DataFrame],
    extractor_fn=metric_extract_prediction,
) -> pd.DataFrame:
    rows = []
    for seed, view_df in eval_views.items():
        acc, pred_df = evaluate_accuracy(model, view_df, extractor_fn=extractor_fn)
        rows.append(
            {
                "seed": seed,
                "rows": len(view_df),
                "accuracy": acc,
                "has_boxed_rate": pred_df["has_boxed"].mean(),
                "boxed_parse_rate": pred_df["boxed_parsed_success"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "seed": "mean±std",
        "rows": out["rows"].mean(),
        "accuracy": out["accuracy"].mean(),
        "has_boxed_rate": out["has_boxed_rate"].mean(),
        "boxed_parse_rate": out["boxed_parse_rate"].mean(),
    }
    summary["accuracy_std"] = out["accuracy"].std(ddof=0)
    out = pd.concat([out, pd.DataFrame([summary])], ignore_index=True)
    return out


def build_hard_profile(pred_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    def _bottom_k_profile(series: pd.Series, bottom_k: int) -> Dict[str, float]:
        if series.empty:
            return {}
        scores = (1.0 - series).sort_values(ascending=False)
        keep = scores.head(min(max(bottom_k, 1), len(scores)))
        return keep.to_dict()

    family_error = pred_df.groupby("prompt_family")["hit"].mean()
    template_group_error = pred_df.groupby("template_group")["hit"].mean()
    answer_type_error = pred_df.groupby("answer_type")["hit"].mean()
    len_bucket_error = pred_df.groupby("len_bucket")["hit"].mean()
    return {
        "family": _bottom_k_profile(family_error, cfg.hard_mining_bottom_family_k),
        "template_group": (1.0 - template_group_error).to_dict(),
        "answer_type": _bottom_k_profile(answer_type_error, cfg.hard_mining_bottom_answer_type_k),
        "len_bucket": _bottom_k_profile(len_bucket_error, cfg.hard_mining_bottom_bucket_k),
    }


def run_serious_eval_suite(
    model: torch.nn.Module,
    primary_df: pd.DataFrame,
    eval_views: Dict[int, pd.DataFrame],
    prefix: str,
    extractor_fn=metric_extract_prediction,
    run_multi_seed: bool = True,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    primary_acc, primary_pred_df = evaluate_accuracy(model, primary_df, extractor_fn=extractor_fn)
    print(f"{prefix} serious accuracy: {primary_acc:.4f}")
    print_eval_summaries(primary_pred_df, prefix=prefix)
    primary_pred_df.to_csv(Path(cfg.work_dir) / f"{prefix}_predictions.csv", index=False)

    if run_multi_seed and eval_views:
        multi_seed_df = evaluate_multi_seed_views(model, eval_views, extractor_fn=extractor_fn)
        display(multi_seed_df)
        multi_seed_df.to_csv(Path(cfg.work_dir) / f"{prefix}_multi_seed_eval.csv", index=False)
    else:
        multi_seed_df = pd.DataFrame()

    return primary_acc, primary_pred_df, multi_seed_df


def update_replay_buffer(
    replay_buffer: Dict[str, int],
    pred_df: pd.DataFrame,
    max_size: int = 512,
) -> Dict[str, int]:
    failure_rows = pred_df.loc[~pred_df["hit"], ["id", "failure_type"]].copy()
    for row in failure_rows.itertuples(index=False):
        replay_buffer[row.id] = replay_buffer.get(row.id, 0) + 1
    if len(replay_buffer) > max_size:
        replay_buffer = dict(sorted(replay_buffer.items(), key=lambda x: x[1], reverse=True)[:max_size])
    return replay_buffer


def run_template_ablation(model: torch.nn.Module, df: pd.DataFrame, max_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_df = df.head(max_rows).copy()
    rows = []
    mapping_rows = []
    for template_id in TEMPLATE_POOL:
        acc, pred_df = evaluate_accuracy(model, eval_df, template_override=template_id, max_new_tokens_grid=(64, 96))
        rows.append(
            {
                "template_id": template_id,
                "accuracy": acc,
                "has_boxed_rate": pred_df["has_boxed"].mean(),
                "boxed_parse_rate": pred_df["boxed_parsed_success"].mean(),
                "rows": len(pred_df),
            }
        )
        family_scores = (
            pred_df.groupby("prompt_family")
            .agg(
                accuracy=("hit", "mean"),
                family_rows=("hit", "size"),
                has_boxed_rate=("has_boxed", "mean"),
                boxed_parse_rate=("boxed_parsed_success", "mean"),
            )
            .reset_index()
        )
        family_scores["template_id"] = template_id
        mapping_rows.append(family_scores)
    score_df = pd.DataFrame(rows).sort_values(["accuracy", "boxed_parse_rate", "has_boxed_rate"], ascending=False).reset_index(drop=True)
    mapping_df = pd.concat(mapping_rows, ignore_index=True)
    best_mapping = (
        mapping_df.sort_values(["prompt_family", "accuracy", "family_rows"], ascending=[True, False, False])
        .groupby("prompt_family", as_index=False)
        .first()
        .rename(columns={"template_id": "best_template_id", "accuracy": "best_accuracy"})
    )
    return score_df, best_mapping, mapping_df


def lookup_template_family_metrics(mapping_df: Optional[pd.DataFrame], family: str, template_id: str) -> Dict[str, float]:
    if mapping_df is None or mapping_df.empty:
        return {
            "accuracy": 0.0,
            "family_rows": 0.0,
            "has_boxed_rate": 0.0,
            "boxed_parse_rate": 0.0,
        }
    match = mapping_df.loc[
        (mapping_df["prompt_family"] == family) & (mapping_df["template_id"] == template_id)
    ]
    if match.empty:
        return {
            "accuracy": 0.0,
            "family_rows": 0.0,
            "has_boxed_rate": 0.0,
            "boxed_parse_rate": 0.0,
        }
    row = match.iloc[0]
    return {
        "accuracy": float(row.get("accuracy", 0.0)),
        "family_rows": float(row.get("family_rows", 0.0)),
        "has_boxed_rate": float(row.get("has_boxed_rate", 0.0)),
        "boxed_parse_rate": float(row.get("boxed_parse_rate", 0.0)),
    }


def apply_template_ablation_updates(
    current_map: Dict[str, str],
    best_mapping_df: pd.DataFrame,
    full_mapping_df: pd.DataFrame,
    secondary_mapping_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    decisions = []
    updated_map = dict(current_map)
    for row in best_mapping_df.itertuples(index=False):
        family = row.prompt_family
        new_template = row.best_template_id
        family_rows = int(row.family_rows)
        old_template = current_map.get(family, FAMILY_TEMPLATE_PRIORS.get(family, "T2_compact"))

        old_metrics = lookup_template_family_metrics(full_mapping_df, family, old_template)
        new_metrics = lookup_template_family_metrics(full_mapping_df, family, new_template)
        gain = new_metrics["accuracy"] - old_metrics["accuracy"]
        required_gain = cfg.template_ablation_stage2_min_gain
        if old_metrics["accuracy"] >= cfg.template_ablation_strong_family_accuracy:
            required_gain = max(required_gain, cfg.template_ablation_strong_family_min_gain)
        boxed_ok = (
            not cfg.template_ablation_require_non_worse_boxed_metrics
            or (
                new_metrics["has_boxed_rate"] >= old_metrics["has_boxed_rate"]
                and new_metrics["boxed_parse_rate"] >= old_metrics["boxed_parse_rate"]
            )
        )

        secondary_old_metrics = lookup_template_family_metrics(secondary_mapping_df, family, old_template) if secondary_mapping_df is not None else old_metrics
        secondary_new_metrics = lookup_template_family_metrics(secondary_mapping_df, family, new_template) if secondary_mapping_df is not None else new_metrics
        secondary_gain = secondary_new_metrics["accuracy"] - secondary_old_metrics["accuracy"]
        passed_multi_view_check = secondary_gain >= -cfg.template_ablation_secondary_max_drop
        should_update = (
            family_rows >= cfg.template_ablation_family_min_rows
            and gain >= required_gain
            and boxed_ok
            and passed_multi_view_check
        )
        if should_update:
            updated_map[family] = new_template
        decisions.append(
            {
                "prompt_family": family,
                "family_rows": family_rows,
                "old_template_id": old_template,
                "old_accuracy": old_metrics["accuracy"],
                "old_has_boxed_rate": old_metrics["has_boxed_rate"],
                "old_boxed_parse_rate": old_metrics["boxed_parse_rate"],
                "candidate_template_id": new_template,
                "candidate_accuracy": new_metrics["accuracy"],
                "candidate_has_boxed_rate": new_metrics["has_boxed_rate"],
                "candidate_boxed_parse_rate": new_metrics["boxed_parse_rate"],
                "gain": gain,
                "required_gain": required_gain,
                "secondary_old_accuracy": secondary_old_metrics["accuracy"],
                "secondary_candidate_accuracy": secondary_new_metrics["accuracy"],
                "secondary_gain": secondary_gain,
                "boxed_ok": boxed_ok,
                "passed_multi_view_check": passed_multi_view_check,
                "min_gain": cfg.template_ablation_stage2_min_gain,
                "min_rows": cfg.template_ablation_family_min_rows,
                "updated": should_update,
                "applied_template_id": updated_map.get(family, old_template),
            }
        )
    BEST_TEMPLATE_BY_FAMILY.clear()
    BEST_TEMPLATE_BY_FAMILY.update(updated_map)
    return pd.DataFrame(decisions).sort_values(["updated", "gain", "family_rows"], ascending=[False, False, False]).reset_index(drop=True)


def build_family_template_router_from_mapping(
    mapping_df: pd.DataFrame,
    top_k: Optional[int] = None,
) -> Dict[str, List[str]]:
    if mapping_df is None or mapping_df.empty:
        return {family: list(router) for family, router in FAMILY_TEMPLATE_ROUTER.items()}

    keep = top_k if top_k is not None else cfg.template_router_top_k
    router_map: Dict[str, List[str]] = {}
    for family, block in mapping_df.groupby("prompt_family", sort=True):
        ordered = block.sort_values(
            ["accuracy", "boxed_parse_rate", "has_boxed_rate", "family_rows"],
            ascending=[False, False, False, False],
        )["template_id"].tolist()
        router_map[family] = []
        for template_id in ordered + FAMILY_TEMPLATE_ROUTER.get(family, []) + FAMILY_TEMPLATE_ROUTER["default"]:
            if template_id in TEMPLATE_POOL and template_id not in router_map[family]:
                router_map[family].append(template_id)
        router_map[family] = router_map[family][: max(1, keep)]

    router_map["default"] = list(FAMILY_TEMPLATE_ROUTER["default"])[: max(1, keep)]
    return router_map


def is_numeric_candidate(answer: str) -> bool:
    answer = canonicalize_answer(answer)
    return bool(ANSWER_NUMBER_RE.fullmatch(answer.replace(",", "")) or re.fullmatch(r"[-+]?\d+/\d+", answer.replace(" ", "")))


def infer_with_multi_templates(
    model: torch.nn.Module,
    example: Any,
    template_ids: Sequence[str],
    max_new_tokens: int = 96,
    extractor_fn=metric_extract_prediction,
) -> List[Dict[str, Any]]:
    prompt_text = example.prompt if hasattr(example, "prompt") else example["prompt"]
    candidates: List[Dict[str, Any]] = []
    for template_id in template_ids:
        raw = generate_answer_text(model, TEMPLATE_POOL[template_id](prompt_text), max_new_tokens=max_new_tokens).strip()
        answer = extractor_fn(raw)
        candidates.append(
            {
                "template_id": template_id,
                "raw": raw[:1200],
                "answer": canonicalize_answer(answer),
                "has_boxed": "\\boxed{" in raw,
                "boxed_answer": extract_boxed(raw),
            }
        )
    return candidates


def aggregate_candidates(
    candidates: Sequence[Dict[str, Any]],
    answer_type: Optional[str] = None,
    preferred_template_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    valid_answers = [c["answer"] for c in candidates if c.get("answer") not in [None, ""]]
    if not valid_answers:
        return {"pred": "", "vote_count": 0, "needs_fallback": True, "decision_type": "empty"}

    answer_counts = Counter(valid_answers)
    top_answer, top_votes = answer_counts.most_common(1)[0]
    if top_votes >= 2:
        return {"pred": top_answer, "vote_count": top_votes, "needs_fallback": False, "decision_type": "majority"}

    if answer_type in {"numeric", "binary"}:
        numeric_answers = [c["answer"] for c in candidates if c.get("answer") and is_numeric_candidate(c["answer"])]
        if numeric_answers:
            numeric_counts = Counter(numeric_answers)
            numeric_answer, numeric_votes = numeric_counts.most_common(1)[0]
            if numeric_votes >= 1:
                return {
                    "pred": numeric_answer,
                    "vote_count": numeric_votes,
                    "needs_fallback": numeric_votes == 1,
                    "decision_type": "numeric_priority",
                }

    for candidate in candidates:
        if candidate.get("boxed_answer"):
            return {
                "pred": canonicalize_answer(candidate["boxed_answer"]),
                "vote_count": answer_counts.get(candidate["answer"], 1),
                "needs_fallback": True,
                "decision_type": "boxed_priority",
            }

    preferred = list(preferred_template_ids or ["T4_numeric_specialized", "T1_ultra_compact", "T3_hidden_reasoning", "T5_text_specialized"])
    for template_id in preferred:
        for candidate in candidates:
            if candidate["template_id"] == template_id and candidate.get("answer"):
                return {
                    "pred": candidate["answer"],
                    "vote_count": answer_counts.get(candidate["answer"], 1),
                    "needs_fallback": True,
                    "decision_type": f"preferred::{template_id}",
                }

    return {"pred": top_answer, "vote_count": top_votes, "needs_fallback": True, "decision_type": "fallback_first"}


@torch.no_grad()
def submission_style_predict_row(
    model: torch.nn.Module,
    row: Any,
    template_ids: Optional[Sequence[str]] = None,
    fallback_template_id: Optional[str] = None,
    max_new_tokens_grid: Sequence[int] = (64, 96),
    extractor_fn=metric_extract_prediction,
) -> Dict[str, Any]:
    return predict_one_row(
        model,
        row,
        template_ids=template_ids,
        fallback_template_id=fallback_template_id,
        max_new_tokens_grid=max_new_tokens_grid,
        extractor_fn=extractor_fn,
    )


@torch.no_grad()
def official_single_pass_predict_row(
    model: torch.nn.Module,
    row: Any,
    max_new_tokens: int = 768,
    extractor_fn=metric_extract_prediction,
) -> Dict[str, Any]:
    """更贴近官方评测：单提示词、单次生成、单答案提取（无模板路由/投票）。"""
    prompt_text = row.prompt if hasattr(row, "prompt") else row["prompt"]
    answer_type = infer_expected_answer_type_from_prompt(prompt_text)
    prompt = (
        f"{prompt_text}\n\n"
        "Please reason internally and output only the final answer in \\boxed{}."
    )
    raw = generate_answer_text(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
    )
    pred = extractor_fn(raw)
    return {
        "pred": pred,
        "raw": raw,
        "answer_type": answer_type,
        "prompt_family": infer_prompt_family(prompt_text),
        "template_ids": ["official_single_pass"],
        "consensus_votes": 1,
        "num_templates": 1,
        "decision_type": "single_pass",
        "fallback_used": False,
    }


@torch.no_grad()
def evaluate_with_consensus(
    model: torch.nn.Module,
    df: pd.DataFrame,
    template_ids: Optional[Sequence[str]] = None,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for row in df.itertuples(index=False):
        result = predict_one_row(model, row, template_ids=template_ids)
        gold = canonicalize_answer(row.answer)
        rows.append(
            {
                "id": row.id,
                "prompt_family": result["prompt_family"],
                "answer_type": result["answer_type"],
                "gold": gold,
                "pred": result["pred"],
                "hit": approx_equal(result["pred"], gold),
                "consensus_votes": result["consensus_votes"],
                "num_templates": result["num_templates"],
                "decision_type": result["decision_type"],
                "fallback_used": result["fallback_used"],
            }
        )
    out = pd.DataFrame(rows)
    return float(out["hit"].mean()) if len(out) else 0.0, out


@torch.no_grad()
def offline_submission_style_eval(
    model: torch.nn.Module,
    valid_df: pd.DataFrame,
    top_k: Optional[int] = None,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for row in valid_df.itertuples(index=False):
        # submission-style 评估改为官方近似单路评测，避免本地分数被“多模板投票”抬高
        # 保留 top_k 参数仅为了兼容旧调用签名
        _ = top_k
        result = official_single_pass_predict_row(model, row)
        gold = canonicalize_answer(row.answer)
        rows.append(
            {
                "id": row.id,
                "prompt_family": result["prompt_family"],
                "answer_type": result["answer_type"],
                "gold": gold,
                "pred": result["pred"],
                "hit": approx_equal(result["pred"], gold),
                "template_ids": ",".join(result["template_ids"]),
                "consensus_votes": result["consensus_votes"],
                "num_templates": result["num_templates"],
                "decision_type": result["decision_type"],
                "fallback_used": result["fallback_used"],
            }
        )
    out = pd.DataFrame(rows)
    return float(out["hit"].mean()) if len(out) else 0.0, out


def pseudolabel_passes_family_filter(prompt_family: str, answer_type: str, pred: str) -> bool:
    pred = canonicalize_answer(pred)
    if prompt_family not in set(cfg.consensus_pseudolabel_allowed_families):
        return False
    if answer_type == "multi_token_text":
        return False
    if answer_type == "binary":
        return bool(BINARY_RE.fullmatch(pred))
    if answer_type == "numeric":
        return bool(ANSWER_NUMBER_RE.fullmatch(pred.replace(",", "")))
    if prompt_family == "bit_transform":
        return bool(BINARY_RE.fullmatch(pred))
    if prompt_family in {"sequence", "matrix_reasoning"}:
        return bool(ANSWER_NUMBER_RE.fullmatch(pred.replace(",", "")) or re.fullmatch(r"[A-Za-z0-9]{1,16}", pred))
    return False


def family_specific_verifier(prompt_family: str, prompt: str, answer: str) -> bool:
    prompt = str(prompt)
    answer = canonicalize_answer(answer)
    if prompt_family == "bit_transform":
        if not BINARY_RE.fullmatch(answer):
            return False
        binary_spans = re.findall(r"\b[01]{3,}\b", prompt)
        if not binary_spans:
            return len(answer) <= 64
        allowed_lengths = {len(span) for span in binary_spans}
        return len(answer) in allowed_lengths
    if prompt_family == "sequence":
        if not ANSWER_NUMBER_RE.fullmatch(answer.replace(",", "")):
            return False
        return len(answer.replace(",", "")) <= 24 and not bool(re.search(r"[^0-9+\-.]", answer))
    if prompt_family == "matrix_reasoning":
        if BINARY_RE.fullmatch(answer):
            return len(answer) <= 32
        if ANSWER_NUMBER_RE.fullmatch(answer.replace(",", "")):
            return len(answer.replace(",", "")) <= 24
        return False
    return False


@torch.no_grad()
def build_consensus_pseudolabels(
    model: torch.nn.Module,
    unlabeled_df: pd.DataFrame,
    template_ids: Sequence[str],
    max_rows: int = 128,
    weak_families: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    rows = []
    weak_family_set = set(weak_families or [])
    for row in unlabeled_df.head(max_rows).itertuples(index=False):
        prompt_family = infer_prompt_family(row.prompt)
        if weak_family_set and prompt_family not in weak_family_set:
            continue
        result = predict_one_row(model, row, template_ids=template_ids)
        answer_type = infer_answer_shape_from_gold(result["pred"])
        template_vote_ratio = result["consensus_votes"] / max(result["num_templates"], 1)
        self_consistency_ratio = 1.0 if result["decision_type"] == "majority" else template_vote_ratio
        format_valid_score = 1.0 if any(c.get("boxed_answer") for c in result["candidates"]) else 0.5
        confidence = template_vote_ratio * 0.5 + self_consistency_ratio * 0.3 + format_valid_score * 0.2
        if confidence < cfg.pseudolabel_min_confidence:
            continue
        if not pseudolabel_passes_family_filter(prompt_family, answer_type, result["pred"]):
            continue
        if not family_specific_verifier(prompt_family, row.prompt, result["pred"]):
            continue
        rows.append(
            {
                "id": getattr(row, "id", f"pseudo_{len(rows)}"),
                "prompt": row.prompt,
                "answer": result["pred"],
                "source": f"consensus_pseudolabel::{prompt_family}",
                "confidence": confidence,
                "template_vote_ratio": template_vote_ratio,
                "self_consistency_ratio": self_consistency_ratio,
                "format_valid_score": format_valid_score,
                "prompt_family": prompt_family,
                "answer_type": answer_type,
                "source_loss_weight": cfg.pseudolabel_source_loss_weight,
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def summarize_template_disagreement(model: torch.nn.Module, df: pd.DataFrame, template_ids: Sequence[str], max_rows: int = 128) -> pd.DataFrame:
    rows = []
    for row in df.head(max_rows).itertuples(index=False):
        result = predict_one_row(model, row, template_ids=template_ids)
        disagreement = 1.0 - (result["consensus_votes"] / max(result["num_templates"], 1))
        rows.append(
            {
                "id": row.id,
                "prompt_family": result["prompt_family"],
                "answer_type": result["answer_type"],
                "disagreement": disagreement,
                "decision_type": result["decision_type"],
                "fallback_used": result["fallback_used"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.groupby("prompt_family", as_index=False)
        .agg(samples=("id", "size"), mean_disagreement=("disagreement", "mean"))
        .sort_values(["mean_disagreement", "samples"], ascending=[False, False])
    )


class LocalAccuracyCallback(TrainerCallback):
    def __init__(
        self,
        eval_df: pd.DataFrame,
        save_path: str,
        extractor_fn=metric_extract_prediction,
        save_top_k: int = 1,
        candidates_dir: Optional[str] = None,
        tokenizer=None,
        metric_name: str = "local_accuracy",
        min_global_step: int = 0,
        router_top_k: Optional[int] = None,
        max_new_tokens_grid: Sequence[int] = (64, 96),
    ):
        self.router_top_k = router_top_k
        self.max_new_tokens_grid = tuple(max_new_tokens_grid)
        self.eval_df = eval_df
        self.save_path = save_path
        self.extractor_fn = extractor_fn
        self.history: List[Dict[str, Any]] = []

        self.save_top_k = max(int(save_top_k), 0)
        self.candidates_dir = Path(candidates_dir) if candidates_dir is not None else None
        self.tokenizer = tokenizer
        self.metric_name = metric_name
        self.min_global_step = int(min_global_step)

        self.topk_candidates: List[Dict[str, Any]] = []

        if self.candidates_dir is not None:
            self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def _manifest_columns(self) -> List[str]:
        return ["step", "local_accuracy", "candidate_dir"]

    def _persist_topk_manifest(self) -> None:
        manifest_path = Path(cfg.topk_candidate_manifest_path)
        if self.topk_candidates:
            out = pd.DataFrame(self.topk_candidates).sort_values(
                ["local_accuracy", "step"],
                ascending=[False, False],
            )
        else:
            out = pd.DataFrame(columns=self._manifest_columns())
        out.to_csv(manifest_path, index=False)

    def _cleanup_non_topk_dirs(self) -> None:
        if self.candidates_dir is None or not self.candidates_dir.exists():
            return

        keep_dirs = {str(Path(item["candidate_dir"])) for item in self.topk_candidates}
        for path in self.candidates_dir.iterdir():
            if path.is_dir() and str(path) not in keep_dirs:
                shutil.rmtree(path, ignore_errors=True)

    def _should_save_candidate(self, step: int, score: float) -> bool:
        if self.save_top_k <= 0:
            return False
        if step < self.min_global_step:
            return False
        if len(self.topk_candidates) < self.save_top_k:
            return True

        kth_score = min(float(item["local_accuracy"]) for item in self.topk_candidates)
        return float(score) > kth_score + cfg.best_score_epsilon

    def _save_topk_candidate(self, model: torch.nn.Module, step: int, score: float) -> None:
        if self.candidates_dir is None or self.tokenizer is None:
            return

        candidate_dir = self.candidates_dir / f"step_{int(step):07d}_acc_{float(score):.6f}"
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir, ignore_errors=True)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(candidate_dir)
        self.tokenizer.save_pretrained(candidate_dir)

        self.topk_candidates = [
            item for item in self.topk_candidates
            if int(item["step"]) != int(step)
        ]
        self.topk_candidates.append(
            {
                "step": int(step),
                "local_accuracy": float(score),
                "candidate_dir": str(candidate_dir),
            }
        )

        self.topk_candidates = sorted(
            self.topk_candidates,
            key=lambda x: (-float(x["local_accuracy"]), -int(x["step"])),
        )[: self.save_top_k]

        self._cleanup_non_topk_dirs()
        self._persist_topk_manifest()

        print(
            f"[top-k saved] metric={self.metric_name} "
            f"step={step} score={score:.4f} "
            f"kept={len(self.topk_candidates)}/{self.save_top_k}"
        )

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        acc, pred_df = evaluate_accuracy(
            model,
            self.eval_df,
            extractor_fn=self.extractor_fn,
            router_top_k=self.router_top_k,
            max_new_tokens_grid=self.max_new_tokens_grid,
        )
        record = {
            "step": int(state.global_step),
            "local_accuracy": float(acc),
        }

        self.history.append(record)

        pd.DataFrame(self.history).to_csv(self.save_path, index=False)
        pred_df.to_csv(Path(self.save_path).with_suffix(".preds.csv"), index=False)

        summaries = summarize_eval_metrics(pred_df)
        for key, value in summaries.items():
            value.to_csv(
                Path(self.save_path).with_name(
                    f"{Path(self.save_path).stem}_{key}_step_{state.global_step}.csv"
                ),
                index=False,
            )

        print(f"\n[local-accuracy] step={state.global_step} acc={acc:.4f}")

        if model is not None and self._should_save_candidate(int(state.global_step), float(acc)):
            self._save_topk_candidate(model=model, step=int(state.global_step), score=float(acc))

        return control


# %% [markdown]
# ## Cell 17 — Data collator + 训练参数工厂

# %%
train_columns = ["input_ids", "attention_mask", "labels"]
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt",
)


def build_training_args(stage_name: str, lr: float, epochs: float) -> TrainingArguments:
    is_smoke_pipeline = cfg.smoke_test_mode and cfg.smoke_profile == "pipeline"

    max_steps = None
    if is_smoke_pipeline:
        if stage_name == "stage1":
            max_steps = cfg.smoke_train_max_steps_stage1
        elif stage_name.startswith("stage2"):
            max_steps = cfg.smoke_train_max_steps_stage2

    # ===== 用真实 warmup_steps，不再用 warmup_ratio =====
    if stage_name.startswith("stage1"):
        train_rows = max(len(stage1_ds), 1)
        warmup_frac = 0.05
    else:
        train_rows = max(len(stage2_ds), 1)
        warmup_frac = 0.08

    effective_batch = max(cfg.micro_batch_size * cfg.grad_accum, 1)
    steps_per_epoch = max(math.ceil(train_rows / effective_batch), 1)
    est_total_steps = max(int(math.ceil(epochs * steps_per_epoch)), 1)
    auto_warmup_steps = max(int(round(est_total_steps * warmup_frac)), 1)

    kwargs = dict(
        output_dir=str(Path(cfg.output_dir) / stage_name),
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(cfg.warmup_steps_override) if cfg.warmup_steps_override is not None else auto_warmup_steps,
        per_device_train_batch_size=cfg.micro_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=cfg.enable_gradient_checkpointing,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        bf16=cfg.use_bf16,
        fp16=not cfg.use_bf16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    if max_steps is not None and int(max_steps) > 0:
        kwargs["max_steps"] = int(max_steps)

    return TrainingArguments(**kwargs)

def maybe_extend_stage2_with_pseudolabels(
    model: torch.nn.Module,
    stage2_records_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
) -> pd.DataFrame:
    if not cfg.enable_consensus_pseudolabel_refresh or unlabeled_df.empty:
        return stage2_records_df

    pseudo_df = build_consensus_pseudolabels(
        model,
        unlabeled_df,
        template_ids=cfg.consensus_template_ids,
        max_rows=cfg.consensus_pseudolabel_max_rows,
        weak_families=CURRENT_WEAK_FAMILIES,
    )
    if pseudo_df.empty:
        print("no consensus pseudolabels were added to stage2")
        return stage2_records_df

    pseudo_df = pseudo_df.assign(
        prompt_family=pseudo_df["prompt"].map(infer_prompt_family),
        template_group=pseudo_df["prompt"].map(infer_template_group),
        answer_type=pseudo_df["answer"].map(infer_answer_shape_from_gold),
        prompt_chars=pseudo_df["prompt"].str.len(),
        answer_chars=pseudo_df["answer"].astype(str).str.len(),
        example_len_bucket=pd.cut(
            pseudo_df["prompt"].str.len(),
            bins=[0, 400, 800, 1400, 4000, 20000],
            labels=["xs", "s", "m", "l", "xl"],
            include_lowest=True,
        ),
    )
    pseudo_records = build_records_frame(pseudo_df)
    out = pd.concat([stage2_records_df, pseudo_records], ignore_index=True)
    out = out.drop_duplicates(subset=["prompt_text", "answer_text", "source"]).reset_index(drop=True)
    pseudo_records.to_csv(Path(cfg.work_dir) / "consensus_pseudolabel_records.csv", index=False)
    print(f"added consensus pseudolabels to stage2: {len(pseudo_records)} rows")
    return out


def refresh_stage2_training_assets(
    train_fold_df: pd.DataFrame,
    valid_fold_df: pd.DataFrame,
    stage1_records_df: pd.DataFrame,
    current_model: Optional[torch.nn.Module] = None,
    unlabeled_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dataset], Dict[str, Dataset], Dataset, Dataset, np.ndarray, pd.DataFrame]:
    refreshed_train_records = build_records_frame(
        train_fold_df,
        augment_templates=cfg.train_time_use_template_augmentation,
    )
    refreshed_valid_records = build_records_frame(
        valid_fold_df,
        augment_templates=False,
    )
    _stage1_records_ref, stage2_records_ref, _ = split_stage_records(refreshed_train_records)
    stage2_records_ref = maybe_extend_stage2_with_pseudolabels(current_model, stage2_records_ref, unlabeled_df if unlabeled_df is not None else pd.DataFrame()) if current_model is not None else stage2_records_ref
    _, stage2_variant_ref, valid_variant_ref = refresh_variant_datasets(
        stage1_records_df,
        stage2_records_ref,
        refreshed_valid_records,
    )

    stage2_ds_ref = stage2_variant_ref[cfg.stage2_supervision_variant]
    valid_ds_ref = valid_variant_ref[cfg.valid_supervision_variant]

    stage2_weights_ref = compute_sample_weights(stage2_records_ref)
    return (
        refreshed_train_records,
        refreshed_valid_records,
        stage2_variant_ref,
        valid_variant_ref,
        stage2_ds_ref,
        valid_ds_ref,
        stage2_weights_ref,
        stage2_records_ref,
    )

def perform_conservative_template_refresh(
    model: torch.nn.Module,
    eval_df_primary: pd.DataFrame,
    eval_df_secondary: pd.DataFrame,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    template_score_df, family_template_map_df, family_template_full_df = run_template_ablation(
        model,
        eval_df_primary,
        len(eval_df_primary),
    )
    _secondary_score_df, _secondary_best_df, family_template_secondary_df = run_template_ablation(
        model,
        eval_df_secondary,
        len(eval_df_secondary),
    )
    template_update_decisions_df = apply_template_ablation_updates(
        dict(BEST_TEMPLATE_BY_FAMILY),
        family_template_map_df,
        family_template_full_df,
        secondary_mapping_df=family_template_secondary_df,
    )
    BEST_TEMPLATE_ROUTER_BY_FAMILY.clear()
    BEST_TEMPLATE_ROUTER_BY_FAMILY.update(
        build_family_template_router_from_mapping(
            family_template_full_df,
            top_k=cfg.template_router_top_k,
        )
    )
    template_score_df.to_csv(Path(cfg.work_dir) / f"{prefix}_template_ablation_scores.csv", index=False)
    family_template_map_df.to_csv(Path(cfg.work_dir) / f"{prefix}_family_template_map.csv", index=False)
    family_template_full_df.to_csv(Path(cfg.work_dir) / f"{prefix}_family_template_full.csv", index=False)
    pd.DataFrame(
        [
            {"prompt_family": family, "template_ids": ",".join(template_ids)}
            for family, template_ids in sorted(BEST_TEMPLATE_ROUTER_BY_FAMILY.items())
        ]
    ).to_csv(Path(cfg.work_dir) / f"{prefix}_family_template_router.csv", index=False)
    template_update_decisions_df.to_csv(Path(cfg.work_dir) / f"{prefix}_family_template_update_decisions.csv", index=False)
    return template_score_df, family_template_map_df, family_template_full_df, template_update_decisions_df


def replay_family_concentration(train_records_df: pd.DataFrame, replay_buffer: Dict[str, int]) -> Tuple[float, Optional[str]]:
    if not replay_buffer:
        return 0.0, None
    replay_df = train_records_df.loc[train_records_df["id"].isin(replay_buffer.keys()), ["id", "prompt_family"]].copy()
    if replay_df.empty:
        return 0.0, None
    family_share = replay_df["prompt_family"].value_counts(normalize=True)
    return float(family_share.iloc[0]), str(family_share.index[0])


def should_refresh_stage2_assets(
    round_idx: int,
    pred_df: pd.DataFrame,
    train_records_df: pd.DataFrame,
    replay_buffer: Dict[str, int],
    refresh_state: Dict[str, Any],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    round_number = round_idx + 1
    if round_number >= cfg.stage2_rounds:
        return False, [], refresh_state

    reasons: List[str] = []
    if cfg.stage2_asset_refresh_interval_rounds > 0 and round_number % cfg.stage2_asset_refresh_interval_rounds == 0:
        reasons.append("interval")

    family_acc = pred_df.groupby("prompt_family")["hit"].mean().sort_values()
    weak_family_k = min(max(cfg.stage2_refresh_weak_family_top_k, 1), len(family_acc))
    weak_family_mean = float(family_acc.head(weak_family_k).mean()) if weak_family_k > 0 else 0.0
    prev_weak_family_mean = refresh_state.get("weak_family_mean")
    if prev_weak_family_mean is not None and weak_family_mean <= prev_weak_family_mean + cfg.stage2_refresh_min_weak_family_gain:
        reasons.append("weak_family_stalled")
    refresh_state["weak_family_mean"] = weak_family_mean

    concentration, dominant_family = replay_family_concentration(train_records_df, replay_buffer)
    refresh_state["replay_concentration"] = concentration
    if concentration >= cfg.stage2_refresh_replay_family_error_threshold:
        reasons.append(f"replay_concentrated::{dominant_family}")

    return bool(reasons), reasons, refresh_state


def maybe_refresh_stage2_assets(
    model: torch.nn.Module,
    round_idx: int,
    replay_buffer: Dict[str, int],
    hard_profile: Dict[str, Dict[str, float]],
    pred_df: pd.DataFrame,
    train_records_df: pd.DataFrame,
    refresh_state: Dict[str, Any],
    refresh_assets_fn: Optional[Callable[[torch.nn.Module, int], Optional[Dict[str, Any]]]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], List[str]]:
    if refresh_assets_fn is None:
        return None, refresh_state, []
    need_refresh, reasons, refresh_state = should_refresh_stage2_assets(
        round_idx,
        pred_df,
        train_records_df,
        replay_buffer,
        refresh_state,
    )
    if not need_refresh:
        return None, refresh_state, []
    refreshed = refresh_assets_fn(model, round_idx + 1)
    if refreshed is None:
        return None, refresh_state, reasons
    refreshed["sample_weights"] = compute_sample_weights(
        refreshed["train_records_df"],
        hard_profile=hard_profile,
        replay_buffer=replay_buffer,
    )
    return refreshed, refresh_state, reasons


def train_stage2_with_reweight(
    model: torch.nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    train_records_df: pd.DataFrame,
    initial_weights: np.ndarray,
    replay_buffer: Optional[Dict[str, int]] = None,
    refresh_assets_fn: Optional[Callable[[torch.nn.Module, int], Optional[Dict[str, Any]]]] = None,
) -> Tuple[torch.nn.Module, pd.DataFrame, Dict[str, int]]:
    current_weights = initial_weights
    last_pred_df = pd.DataFrame()
    replay_buffer = dict(replay_buffer or {})
    refresh_state: Dict[str, Any] = {}
    per_round_epochs = cfg.stage2_epochs / max(cfg.stage2_rounds, 1)

    for round_idx in range(cfg.stage2_rounds):
        round_name = f"stage2_round_{round_idx + 1}"
        trainer = WeightedTrainer(
            model=model,
            args=build_training_args(round_name, cfg.stage2_lr, per_round_epochs),
            train_dataset=train_dataset.remove_columns([c for c in train_dataset.column_names if c not in train_columns]),
            eval_dataset=eval_dataset.remove_columns([c for c in eval_dataset.column_names if c not in train_columns]),
            data_collator=collator,
            sample_weights=current_weights,
            callbacks=[local_callback],
        )
        trainer.train()

        round_acc, last_pred_df, round_multi_seed_df = run_serious_eval_suite(
            model,
            serious_eval_df,
            serious_eval_views,
            prefix=f"stage2_round_{round_idx + 1}",
        )

        hard_profile = build_hard_profile(last_pred_df)
        CURRENT_WEAK_FAMILIES[:] = get_weak_families_from_pred_df(last_pred_df, top_k=cfg.stage2_refresh_weak_family_top_k)
        probe_acc, replay_probe_pred_df = evaluate_accuracy(model, train_replay_probe_df, extractor_fn=fast_extract_prediction)
        replay_probe_pred_df.to_csv(Path(cfg.work_dir) / f"stage2_round_{round_idx + 1}_train_probe_predictions.csv", index=False)
        replay_buffer = update_replay_buffer(replay_buffer, replay_probe_pred_df)
        current_weights = compute_sample_weights(train_records_df, hard_profile=hard_profile, replay_buffer=replay_buffer)
        print(f"replay probe accuracy round {round_idx + 1}: {probe_acc:.4f}; replay buffer size={len(replay_buffer)}")
        refreshed_assets, refresh_state, refresh_reasons = maybe_refresh_stage2_assets(
            model,
            round_idx,
            replay_buffer,
            hard_profile,
            last_pred_df,
            train_records_df,
            refresh_state,
            refresh_assets_fn=refresh_assets_fn,
        )
        if refreshed_assets is not None:
            train_dataset = refreshed_assets["train_dataset"]
            eval_dataset = refreshed_assets["eval_dataset"]
            train_records_df = refreshed_assets["train_records_df"]
            current_weights = refreshed_assets["sample_weights"]
            print(f"stage2 assets refreshed after round {round_idx + 1}; reasons={','.join(refresh_reasons)}; new train rows={len(train_records_df)}")
        print(
            f"refreshed weights after round {round_idx + 1}:",
            np.quantile(current_weights, [0, 0.25, 0.5, 0.75, 1]),
        )

    return model, last_pred_df, replay_buffer


def run_supervision_variant_experiment(variant: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """对 answer_only 与 short_reasoning 做小规模正面对照。"""
    assert variant in stage1_variant_ds and variant in stage2_variant_ds and variant in valid_variant_ds

    exp_model, exp_lora_targets = load_training_model()
    print(f"\n[supervision-ablation] variant={variant} lora_targets={exp_lora_targets}")

    exp_callback = LocalAccuracyCallback(
        eval_df=fast_eval_df,
        save_path=str(Path(cfg.work_dir) / f"supervision_{variant}_local_accuracy.csv"),
    )
    exp_train_columns = ["input_ids", "attention_mask", "labels"]

    if len(stage1_variant_ds[variant]) > 0:
        trainer_stage1 = WeightedTrainer(
            model=exp_model,
            args=build_training_args(f"supervision_{variant}_stage1", cfg.stage1_lr, cfg.supervision_ablation_stage1_epochs),
            train_dataset=stage1_variant_ds[variant].remove_columns([c for c in stage1_variant_ds[variant].column_names if c not in exp_train_columns]),
            eval_dataset=valid_variant_ds[variant].remove_columns([c for c in valid_variant_ds[variant].column_names if c not in exp_train_columns]),
            data_collator=collator,
            sample_weights=stage1_weights,
            callbacks=[exp_callback],
        )
        trainer_stage1.train()

    variant_rounds = max(cfg.supervision_ablation_stage2_rounds, 1)
    variant_round_epochs = cfg.supervision_ablation_stage2_epochs / variant_rounds
    variant_weights = compute_sample_weights(stage2_records)
    variant_pred_df = pd.DataFrame()

    for round_idx in range(variant_rounds):
        trainer_stage2 = WeightedTrainer(
            model=exp_model,
            args=build_training_args(f"supervision_{variant}_stage2_round_{round_idx + 1}", cfg.stage2_lr, variant_round_epochs),
            train_dataset=stage2_variant_ds[variant].remove_columns([c for c in stage2_variant_ds[variant].column_names if c not in exp_train_columns]),
            eval_dataset=valid_variant_ds[variant].remove_columns([c for c in valid_variant_ds[variant].column_names if c not in exp_train_columns]),
            data_collator=collator,
            sample_weights=variant_weights,
            callbacks=[exp_callback],
        )
        trainer_stage2.train()
        _acc, variant_pred_df = evaluate_accuracy(exp_model, serious_eval_df)
        variant_weights = compute_sample_weights(stage2_records, hard_profile=build_hard_profile(variant_pred_df))

    overall_df = summarize_eval_metrics(variant_pred_df)["overall"].copy()
    overall_df.insert(0, "variant", variant)
    overall_df.insert(1, "lora_targets", ",".join(exp_lora_targets))
    grouped = summarize_eval_metrics(variant_pred_df)
    return overall_df, grouped


# %% [markdown]
# ## Cell 18 — A/B 监督形式预检 + 训练前模板先验
# 
# 这里保留 supervision 形式对照，但**不再**用未训练模型做 template ablation。
# 训练前只启用 family-aware 的启发式模板先验；真正的模型驱动 template ablation
# 放到 Stage 1 之后，再刷新 Stage 2 训练集和数据集对象。

# %%
print("supervision variant sizes:")
for variant, ds in stage2_variant_ds.items():
    print(variant, len(ds))
    sample_idx = 0
    sample_row = make_supervision_frame(stage2_records.head(1), variant).iloc[sample_idx]
    preview_text = sample_row["full_text"] if sample_row["supervision_variant"] == "answer_only" else sample_row["reasoning_full_text"]
    print(f"\n[{variant}] sample target preview:\n{preview_text[:500]}\n")

variant_panel_df = build_supervision_variant_panel(stage2_records.head(min(len(stage2_records), 256)))
display(variant_panel_df)
variant_panel_df.to_csv(Path(cfg.work_dir) / "supervision_variant_panel.csv", index=False)
print("stage1 supervision variant:", cfg.stage1_supervision_variant)
print("stage2 supervision variant:", cfg.stage2_supervision_variant)
print("valid supervision variant :", cfg.valid_supervision_variant)
print("legacy primary supervision variant (compat only):", cfg.primary_supervision_variant)
print("initial family template priors:")
display(pd.DataFrame(sorted(BEST_TEMPLATE_BY_FAMILY.items()), columns=["prompt_family", "template_id"]))

def can_run_supervision_ablation() -> Tuple[bool, str]:
    if not cfg.run_supervision_ablation:
        return False, "disabled by cfg"

    if not torch.cuda.is_available():
        return False, "CUDA unavailable"

    # 当前环境没有 bitsandbytes 时，ablation 会尝试再加载一个非量化 30B 模型，
    # 很容易走到 auto offload / meta tensors，导致 load_training_model() 报错。
    if not package_available("bitsandbytes"):
        return False, "bitsandbytes unavailable; supervision ablation would require a second full-size non-quantized 30B model"

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / 1024**3
        total_gib = total_bytes / 1024**3
    except Exception:
        return False, "unable to query CUDA free memory"

    # supervision ablation 会额外再起一个训练模型；留一个保守门槛，避免 meta/offload
    if free_gib < 55:
        return False, f"insufficient free CUDA memory for a second model ({free_gib:.2f} / {total_gib:.2f} GiB free)"

    return True, "ok"


ablation_ok, ablation_reason = can_run_supervision_ablation()

if ablation_ok:
    ablation_overall = []
    for variant in cfg.supervision_ablation_variants:
        overall_df, grouped = run_supervision_variant_experiment(variant)
        ablation_overall.append(overall_df)
        for key, value in grouped.items():
            value.to_csv(Path(cfg.work_dir) / f"supervision_{variant}_{key}.csv", index=False)

    ablation_overall_df = pd.concat(ablation_overall, ignore_index=True).sort_values(
        ["value", "boxed_parse_rate"], ascending=False
    )
    display(ablation_overall_df)
    ablation_overall_df.to_csv(Path(cfg.work_dir) / "supervision_ablation_overall.csv", index=False)
else:
    print(f"skip full supervision ablation: {ablation_reason}")

template_preview_df = build_seeded_family_balanced_subset(valid_fold, cfg.reasoning_template_eval_rows, seed=cfg.seed)
print("template preview rows (reasoning_template_eval_rows):", template_preview_df.shape)
template_score_df = pd.DataFrame()
family_template_map_df = pd.DataFrame(columns=["prompt_family", "best_template_id", "best_accuracy", "family_rows"])
template_update_decisions_df = pd.DataFrame()



# Cell 18.5
def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_adapter_snapshot(model: torch.nn.Module, tokenizer, save_dir: str, tag: str = "") -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[snapshot saved] {save_path}" + (f" ({tag})" if tag else ""))



# Cell 18.9
if not cfg.smoke_test_mode:
    topk_dir = Path(cfg.topk_candidate_dir)
    manifest_path = Path(cfg.topk_candidate_manifest_path)
    rerank_path = Path(cfg.final_rerank_report_path)
    export_info_path = Path(cfg.final_export_info_path)

    if topk_dir.exists():
        shutil.rmtree(topk_dir, ignore_errors=True)
    topk_dir.mkdir(parents=True, exist_ok=True)

    for p in [manifest_path, rerank_path, export_info_path]:
        if p.exists():
            p.unlink()

# Cell 18.95 — Stage1 best checkpoint -> Stage2 inheritance (OOM-safe, in-place adapter reload)

from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict


def pick_best_stage1_candidate_dir() -> Optional[str]:
    manifest_path = Path(cfg.topk_candidate_manifest_path)
    if not manifest_path.exists():
        print(f"[stage2 inherit] manifest not found: {manifest_path}")
        return None

    candidate_df = pd.read_csv(manifest_path)
    if candidate_df.empty:
        print("[stage2 inherit] manifest is empty")
        return None

    candidate_df = candidate_df.loc[
        candidate_df["candidate_dir"].map(lambda p: Path(p).exists())
    ].copy()

    if candidate_df.empty:
        print("[stage2 inherit] no candidate_dir exists on disk")
        return None

    candidate_df = candidate_df.sort_values(
        ["local_accuracy", "step"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best_dir = str(candidate_df.iloc[0]["candidate_dir"])
    best_step = int(candidate_df.iloc[0]["step"])
    best_acc = float(candidate_df.iloc[0]["local_accuracy"])
    print(
        f"[stage2 inherit] pick best Stage1 candidate: "
        f"step={best_step}, local_acc={best_acc:.4f}, dir={best_dir}"
    )
    return best_dir


def _get_active_adapter_name(model: torch.nn.Module) -> str:
    adapter_name = "default"
    try:
        active_adapter = getattr(model, "active_adapter", None)
        if isinstance(active_adapter, str) and active_adapter:
            adapter_name = active_adapter
        elif isinstance(active_adapter, (list, tuple)) and len(active_adapter) > 0:
            adapter_name = str(active_adapter[0])
    except Exception:
        pass
    return adapter_name


def _cleanup_cuda_after_inherit() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def load_adapter_weights_into_existing_model(model: torch.nn.Module, adapter_dir: str) -> torch.nn.Module:
    adapter_name = _get_active_adapter_name(model)

    print(f"[stage2 inherit] loading adapter weights into existing model from: {adapter_dir}")
    print(f"[stage2 inherit] target adapter_name: {adapter_name}")

    # 只加载 LoRA / PEFT 权重，不重新创建 base model
    adapter_state_dict = load_peft_weights(adapter_dir, device="cpu")

    load_result = set_peft_model_state_dict(
        model,
        adapter_state_dict,
        adapter_name=adapter_name,
    )

    if load_result is not None:
        try:
            print(f"[stage2 inherit] set_peft_model_state_dict result: {load_result}")
        except Exception:
            pass

    # 保持训练态
    try:
        model.config.use_cache = False
    except Exception:
        pass

    try:
        if cfg.enable_gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
    except Exception as exc:
        print(f"[stage2 inherit] gradient_checkpointing_enable skipped: {exc}")

    model.train()
    return model


def maybe_reload_best_stage1_candidate_for_stage2(model: torch.nn.Module) -> torch.nn.Module:
    if not cfg.stage2_inherit_best_stage1_candidate:
        print("[stage2 inherit] disabled by cfg")
        return model

    best_dir = pick_best_stage1_candidate_dir()
    if best_dir is None:
        print("[stage2 inherit] no best candidate found")
        if cfg.stage2_inherit_fallback_to_final_stage1_state:
            print("[stage2 inherit] fallback to final Stage1 in-memory model")
            return model
        raise RuntimeError("Stage2 inheritance failed: no reloadable Stage1 candidate")

    # 尽量释放 trainer / collator 对旧 stage1 训练对象的引用，但不要删 model 本体
    global trainer_stage1, collator

    if "trainer_stage1" in globals():
        try:
            if hasattr(trainer_stage1, "model"):
                trainer_stage1.model = None
        except Exception:
            pass
        try:
            del trainer_stage1
        except Exception:
            pass

    if "collator" in globals() and collator is not None:
        try:
            if hasattr(collator, "model"):
                collator.model = None
        except Exception:
            pass

    _cleanup_cuda_after_inherit()

    # ===== 关键修复：原地覆盖 adapter 权重，不重新加载一整份 base model =====
    model = load_adapter_weights_into_existing_model(model, best_dir)

    _cleanup_cuda_after_inherit()

    print(f"[stage2 inherit] loaded best Stage1 adapter into existing model from: {best_dir}")
    return model

# %% [markdown]
# ## Cell 19 — Stage 1 训练（短 / 规则明显 family 先收敛）

# %%
leaderboard_proxy_df = serious_eval_df.copy()

local_callback = LocalAccuracyCallback(
    eval_df=leaderboard_proxy_df,
    save_path=str(Path(cfg.work_dir) / "local_accuracy_history.csv"),
    extractor_fn=metric_extract_prediction,
    save_top_k=cfg.topk_candidate_keep,
    candidates_dir=None if cfg.smoke_test_mode else cfg.topk_candidate_dir,
    tokenizer=None if cfg.smoke_test_mode else tokenizer,
    metric_name="local_accuracy",
    min_global_step=cfg.topk_min_global_step,
    router_top_k=cfg.test_time_router_top_k,
    max_new_tokens_grid=(64, 96, 128),
)
if len(stage1_ds) > 0:
    trainer_stage1 = WeightedTrainer(
        model=model,
        args=build_training_args("stage1", cfg.stage1_lr, cfg.stage1_epochs),
        train_dataset=stage1_ds.remove_columns([c for c in stage1_ds.column_names if c not in train_columns]),
        eval_dataset=valid_ds.remove_columns([c for c in valid_ds.column_names if c not in train_columns]),
        data_collator=collator,
        sample_weights=stage1_weights,
        callbacks=[local_callback],
    )

    if cfg.smoke_test_mode and cfg.smoke_profile == "fast":
        print("[smoke-fast] skip stage1 trainer lifecycle; run direct eval only")
        smoke_stage1_acc, smoke_stage1_pred_df = evaluate_accuracy(
            model,
            fast_eval_df,
            extractor_fn=fast_extract_prediction,
        )
        print(f"[smoke-fast] stage1 direct local accuracy: {smoke_stage1_acc:.4f}")
        smoke_stage1_pred_df.to_csv(
            Path(cfg.work_dir) / "smoke_fast_stage1_direct_eval.csv",
            index=False,
        )

    else:
        print("[smoke] run stage1 trainer lifecycle" if cfg.smoke_test_mode else "run stage1 training")
        trainer_stage1.train()

        if cfg.smoke_test_mode:
            print("[smoke] skip trainer_stage1.evaluate() after train; eval already ran during train()")

            # 如果你还想做一次“训练后手动检查”，不要再走 trainer.evaluate()
            smoke_stage1_acc, smoke_stage1_pred_df = evaluate_accuracy(
                model,
                fast_eval_df,
                extractor_fn=fast_extract_prediction,
            )
            print(f"[smoke] post-train direct local accuracy: {smoke_stage1_acc:.4f}")
            smoke_stage1_pred_df.to_csv(
                Path(cfg.work_dir) / "smoke_stage1_post_train_eval.csv",
                index=False,
            )

        if cfg.save_stage1_snapshot:
            save_adapter_snapshot(
                model,
                tokenizer,
                str(Path(cfg.snapshot_root_dir) / "after_stage1_train"),
                tag="after_stage1_train",
            )
            cleanup_cuda()
else:
    print("skip stage1: no samples after curriculum filter")
# %% [markdown]
# ## Cell 20 — Stage 1 后刷新模板映射 / serious eval / Stage 2 权重
# 
# 关键修复点：
# - 模板消融改到 Stage 1 之后执行，避免未训练模型噪声；
# - 一旦更新 family-specific best template map，就立刻重建 train/valid records 与 Stage 2 datasets；
# - Serious eval 使用 family-balanced 主视图 + 多 seed 视图；
# - Sample-level replay 在进入 Stage 2 前先用训练 probe 错题做一次 bootstrap。

# %%
# ===== NEW: Stage2 改成继承 Stage1 最优 checkpoint，而不是最后状态 =====
model = maybe_reload_best_stage1_candidate_for_stage2(model)

# 重新绑定 collator，避免它还引用旧模型
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt",
)

stage1_acc, stage1_pred_df, stage1_multi_seed_df = run_serious_eval_suite(
    model,
    serious_eval_df,
    serious_eval_views,
    prefix="stage1_valid",
    run_multi_seed=cfg.run_stage1_multi_seed_eval,
)

if cfg.run_stage1_submission_style_eval:
    stage1_submission_style_acc, stage1_submission_style_df = offline_submission_style_eval(
        model,
        serious_eval_df.head(min(len(serious_eval_df), 128)),
        top_k=cfg.test_time_router_top_k,
    )
    print(f"stage1 submission-style accuracy: {stage1_submission_style_acc:.4f}")
    stage1_submission_style_df.to_csv(Path(cfg.work_dir) / "stage1_submission_style_eval.csv", index=False)
else:
    stage1_submission_style_acc = None
    stage1_submission_style_df = pd.DataFrame()
    print("skip stage1 submission-style eval (disabled by cfg)")

stage1_train_probe_acc, stage1_train_probe_pred_df = evaluate_accuracy(
    model,
    train_replay_probe_df,
    extractor_fn=fast_extract_prediction,
)
print(f"stage1 train replay probe accuracy: {stage1_train_probe_acc:.4f}")
stage1_train_probe_pred_df.to_csv(Path(cfg.work_dir) / "stage1_train_probe_predictions.csv", index=False)

replay_buffer = update_replay_buffer({}, stage1_train_probe_pred_df)
print("bootstrapped replay buffer size:", len(replay_buffer))

CURRENT_WEAK_FAMILIES[:] = get_weak_families_from_pred_df(
    stage1_pred_df,
    top_k=cfg.stage2_refresh_weak_family_top_k,
)
print("current weak families after stage1:", CURRENT_WEAK_FAMILIES)

if cfg.enable_prompt_template_ablation:
    template_eval_rows = max(cfg.reasoning_template_eval_rows, cfg.template_ablation_after_stage1_rows)
    template_eval_df = build_seeded_family_balanced_subset(
        valid_fold,
        template_eval_rows,
        seed=cfg.seed,
    )
    template_secondary_eval_df = build_seeded_family_balanced_subset(
        valid_fold,
        template_eval_rows,
        seed=cfg.template_ablation_secondary_seed,
    )
    refreshed_best_by_family, refreshed_router, family_template_full_df, template_update_decisions_df = perform_conservative_template_refresh(
        model,
        template_eval_df,
        template_secondary_eval_df,
        prefix="stage1",
    )

    BEST_TEMPLATE_BY_FAMILY.clear()
    BEST_TEMPLATE_BY_FAMILY.update(refreshed_best_by_family)

    BEST_TEMPLATE_ROUTER_BY_FAMILY.clear()
    BEST_TEMPLATE_ROUTER_BY_FAMILY.update(
        {k: list(v) for k, v in refreshed_router.items()}
    )

    display(family_template_full_df.head(20))
    display(template_update_decisions_df.head(20))

train_records, valid_records, stage2_variant_ds, valid_variant_ds, stage2_ds, valid_ds, stage2_weights, stage2_records = refresh_stage2_training_assets(
    train_fold,
    valid_fold,
    stage1_records,
    current_model=model,
    unlabeled_df=unlabeled_external_df,
)
print("refreshed stage2 records:", stage2_records.shape)
print("refreshed valid records:", valid_records.shape)
print("updated stage2 weight summary:", np.quantile(stage2_weights, [0, 0.25, 0.5, 0.75, 1]))

hard_profile = build_hard_profile(stage1_pred_df)
stage2_weights = compute_sample_weights(
    stage2_records,
    hard_profile=hard_profile,
    replay_buffer=replay_buffer,
)
print("stage2 weight summary after hard profile + replay bootstrap:", np.quantile(stage2_weights, [0, 0.25, 0.5, 0.75, 1]))

# %% [markdown]
# ## Cell 21 — Stage 2 训练（多轮重加权，而非轮间重建数据资产）

# %%
def refresh_stage2_assets_for_round(current_model: torch.nn.Module, completed_round: int) -> Optional[Dict[str, Any]]:
    if cfg.enable_prompt_template_ablation:
        round_eval_df = build_seeded_family_balanced_subset(
            valid_fold,
            cfg.stage2_refresh_template_eval_rows,
            seed=cfg.seed + completed_round,
        )
        round_secondary_eval_df = build_seeded_family_balanced_subset(
            valid_fold,
            cfg.stage2_refresh_template_eval_rows,
            seed=cfg.template_ablation_secondary_seed + completed_round,
        )
        refreshed_best_by_family, refreshed_router, _full_df, round_decisions_df = perform_conservative_template_refresh(
            current_model,
            round_eval_df,
            round_secondary_eval_df,
            prefix=f"stage2_round_{completed_round}",
        )

        BEST_TEMPLATE_BY_FAMILY.clear()
        BEST_TEMPLATE_BY_FAMILY.update(refreshed_best_by_family)

        BEST_TEMPLATE_ROUTER_BY_FAMILY.clear()
        BEST_TEMPLATE_ROUTER_BY_FAMILY.update(
            {k: list(v) for k, v in refreshed_router.items()}
        )

        display(round_decisions_df.head(20))

    refreshed_train_records, refreshed_valid_records, refreshed_stage2_variant_ds, refreshed_valid_variant_ds, refreshed_stage2_ds, refreshed_valid_ds, refreshed_stage2_weights, refreshed_stage2_records = refresh_stage2_training_assets(
        train_fold,
        valid_fold,
        stage1_records,
        current_model=current_model,
        unlabeled_df=unlabeled_external_df,
    )
    return {
        "train_records_df": refreshed_stage2_records,
        "train_dataset": refreshed_stage2_variant_ds[cfg.stage2_supervision_variant],
        "eval_dataset": refreshed_valid_variant_ds[cfg.valid_supervision_variant],
        "valid_records_df": refreshed_valid_records,
        "sample_weights": refreshed_stage2_weights,
    }


if cfg.smoke_test_mode and cfg.smoke_profile == "fast":
    print("[smoke-fast] skip stage2 training with reweight")
    stage2_last_pred_df = pd.DataFrame()
else:
    print("[smoke] run stage2 main loop" if cfg.smoke_test_mode else "run stage2 training with reweight")
    model, stage2_last_pred_df, replay_buffer = train_stage2_with_reweight(
        model=model,
        train_dataset=stage2_ds,
        eval_dataset=valid_ds,
        train_records_df=stage2_records,
        initial_weights=stage2_weights,
        replay_buffer=replay_buffer,
        refresh_assets_fn=refresh_stage2_assets_for_round,
    )

# Cell 22A
def safe_post_eval_block(name: str, fn, fail_open: bool = True):
    print(f"\n[post-eval:start] {name}")
    try:
        result = fn()
        print(f"[post-eval:done] {name}")
        cleanup_cuda()
        return result
    except Exception as e:
        cleanup_cuda()
        if fail_open:
            print(f"[post-eval:skip] {name} failed: {type(e).__name__}: {e}")
            return None
        raise

# Cell 22B
if cfg.run_post_train_heavy_eval:
    serious_result = safe_post_eval_block(
        "post_train_serious_eval",
        lambda: run_serious_eval_suite(
            model,
            serious_eval_df,
            serious_eval_views,
            prefix="post_train_valid",
            run_multi_seed=True,
        ),
        fail_open=cfg.post_eval_fail_open,
    )

    if serious_result is not None:
        post_acc, post_pred_df, post_multi_seed_df = serious_result
        display(post_pred_df.head(10))
    else:
        post_acc, post_pred_df, post_multi_seed_df = None, pd.DataFrame(), pd.DataFrame()
else:
    print("skip post-train serious eval (disabled by cfg)")


# Cell 22C

if cfg.run_post_train_heavy_eval:
    submission_style_result = safe_post_eval_block(
        "post_train_submission_style_eval",
        lambda: offline_submission_style_eval(
            model,
            serious_eval_df.head(min(len(serious_eval_df), 128)),
            top_k=cfg.test_time_router_top_k,
        ),
        fail_open=cfg.post_eval_fail_open,
    )

    if submission_style_result is not None:
        submission_style_acc, submission_style_df = submission_style_result
        print(f"post-train submission-style accuracy: {submission_style_acc:.4f}")
        display(submission_style_df.head(10))
        submission_style_df.to_csv(Path(cfg.work_dir) / "post_train_submission_style_eval.csv", index=False)
    else:
        submission_style_acc, submission_style_df = None, pd.DataFrame()
else:
    print("skip post-train submission-style eval (disabled by cfg)")


# Cell 22 D
if cfg.run_post_train_heavy_eval:
    completion_sanity_result = safe_post_eval_block(
        "post_train_completion_sanity",
        lambda: inspect_generation_completion_sanity(
            model,
            serious_eval_df,
            rows=min(len(serious_eval_df), 3),
        ),
        fail_open=cfg.post_eval_fail_open,
    )

    if completion_sanity_result is not None:
        completion_sanity_df = completion_sanity_result
        display(completion_sanity_df)
        completion_sanity_df.to_csv(Path(cfg.work_dir) / "post_train_generation_completion_sanity.csv", index=False)
    else:
        completion_sanity_df = pd.DataFrame()
else:
    print("skip post-train completion sanity (disabled by cfg)")


# Cell 22E

if cfg.run_post_train_heavy_eval:
    consensus_result = safe_post_eval_block(
        "post_train_consensus_eval",
        lambda: evaluate_with_consensus(
            model,
            serious_eval_df.head(min(len(serious_eval_df), 128)),
            cfg.test_time_template_candidates,
        ),
        fail_open=cfg.post_eval_fail_open,
    )

    if consensus_result is not None:
        consensus_acc, consensus_df = consensus_result
        print(f"post-train consensus accuracy: {consensus_acc:.4f}")
        display(consensus_df.head(10))
        consensus_df.to_csv(Path(cfg.work_dir) / "post_train_consensus_predictions.csv", index=False)
    else:
        consensus_acc, consensus_df = None, pd.DataFrame()
else:
    print("skip post-train consensus eval (disabled by cfg)")

#Cell 22F
if cfg.run_post_train_heavy_eval:
    template_disagreement_result = safe_post_eval_block(
        "post_train_template_disagreement",
        lambda: summarize_template_disagreement(
            model,
            serious_eval_df,
            cfg.test_time_template_candidates,
            max_rows=128,
        ),
        fail_open=cfg.post_eval_fail_open,
    )

    if template_disagreement_result is not None:
        template_disagreement_df = template_disagreement_result
        display(template_disagreement_df.head(20))
        template_disagreement_df.to_csv(Path(cfg.work_dir) / "template_disagreement_by_family.csv", index=False)
    else:
        template_disagreement_df = pd.DataFrame()
else:
    print("skip post-train template disagreement (disabled by cfg)")



# Cell 22.9
def _candidate_eval_target_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _cleanup_after_candidate_eval() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _strip_hf_device_map(model: torch.nn.Module) -> None:
    if hasattr(model, "hf_device_map"):
        try:
            delattr(model, "hf_device_map")
        except Exception:
            try:
                model.hf_device_map = None
            except Exception:
                pass


def load_backbone_model_for_adapter_eval() -> torch.nn.Module:
    cleanup_cuda_before_model_load()

    quant_config = build_quant_config()
    attn_implementation = resolve_attn_implementation()
    target_device = _candidate_eval_target_device()

    attempt_specs: List[Dict[str, Any]] = []
    if quant_config is not None:
        attempt_specs.append(
            {
                "label": "candidate_eval_quantized_single_gpu",
                "quant_config": quant_config,
                "attn_implementation": attn_implementation,
            }
        )

    attempt_specs.append(
        {
            "label": "candidate_eval_nonquant_single_device",
            "quant_config": None,
            "attn_implementation": attn_implementation,
        }
    )

    errors: List[str] = []
    base_model: Optional[torch.nn.Module] = None

    for spec in attempt_specs:
        cleanup_cuda_before_model_load(verbose=True)
        try:
            print(f"[candidate-reload] trying load mode: {spec['label']} on {target_device}")

            load_kwargs = build_model_load_kwargs(
                quant_config=spec["quant_config"],
                attn_implementation=spec["attn_implementation"],
            )

            # ===== 关键修复 =====
            # 训练时可以自动 device map，但 candidate rerank 这里不要再让 accelerate 介入自动平衡
            # 否则在 PEFT 挂 adapter 时容易触发 get_balanced_memory -> unhashable set
            load_kwargs.pop("max_memory", None)
            load_kwargs.pop("offload_folder", None)
            load_kwargs.pop("offload_state_dict", None)

            if spec["quant_config"] is not None:
                # 量化时显式单卡加载；后面会剥离 hf_device_map，避免 PEFT 再触发 accelerate 自动逻辑
                load_kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else None
            else:
                # 非量化 fallback 明确不走 auto device map，后面手动 to(device)
                load_kwargs["device_map"] = None
                load_kwargs["torch_dtype"] = torch.bfloat16 if cfg.use_bf16 else torch.float16

            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model,
                **load_kwargs,
            )

            # 非量化 fallback：手动放到单设备
            if spec["quant_config"] is None:
                base_model = base_model.to(target_device)

            # 非常关键：去掉 hf_device_map，防止后续 PeftModel.from_pretrained 再走 accelerate 平衡内存逻辑
            _strip_hf_device_map(base_model)

            force_nemotron_torch_fallback(base_model, tag="candidate_eval_backbone")
            base_model.config.use_cache = False
            base_model.eval()
            return base_model

        except Exception as exc:
            errors.append(f"{spec['label']}: {type(exc).__name__}: {exc}")
            if base_model is not None:
                try:
                    del base_model
                except Exception:
                    pass
                base_model = None
            _cleanup_after_candidate_eval()

    raise RuntimeError(
        "Unable to reload base model for candidate evaluation.\n"
        "Attempts:\n" + "\n".join(errors)
    )


def load_candidate_adapter_for_eval(adapter_dir: str) -> torch.nn.Module:
    backbone = load_backbone_model_for_adapter_eval()
    target_device = _candidate_eval_target_device()

    # 再保险：清掉底座模型上的 hf_device_map
    _strip_hf_device_map(backbone)

    candidate_model = PeftModel.from_pretrained(
        backbone,
        adapter_dir,
        is_trainable=False,
        torch_device=target_device,
        device_map=None,
        max_memory=None,
        low_cpu_mem_usage=False,
    )

    # 再保险：清掉 candidate 模型上的 hf_device_map
    _strip_hf_device_map(candidate_model)

    # 非量化模型可安全 to(device)；量化模型不要强行 to，避免 bitsandbytes 报错
    is_quantized = bool(
        getattr(candidate_model, "is_loaded_in_4bit", False)
        or getattr(candidate_model, "is_loaded_in_8bit", False)
        or getattr(backbone, "is_loaded_in_4bit", False)
        or getattr(backbone, "is_loaded_in_8bit", False)
    )
    if not is_quantized:
        try:
            candidate_model = candidate_model.to(target_device)
        except Exception:
            pass

    force_nemotron_torch_fallback(
        candidate_model,
        tag=f"candidate_eval::{Path(adapter_dir).name}",
    )
    candidate_model.config.use_cache = False
    candidate_model.eval()
    return candidate_model


def summarize_multi_seed_result(multi_seed_df: pd.DataFrame) -> Dict[str, float]:
    if multi_seed_df is None or multi_seed_df.empty:
        return {
            "multi_seed_mean_accuracy": float("nan"),
            "multi_seed_accuracy_std": float("nan"),
        }

    summary_row = multi_seed_df.loc[multi_seed_df["seed"].astype(str) == "mean±std"]
    if len(summary_row):
        row = summary_row.iloc[0]
        return {
            "multi_seed_mean_accuracy": float(row.get("accuracy", np.nan)),
            "multi_seed_accuracy_std": float(row.get("accuracy_std", np.nan)),
        }

    return {
        "multi_seed_mean_accuracy": float(multi_seed_df["accuracy"].mean()),
        "multi_seed_accuracy_std": float(multi_seed_df["accuracy"].std(ddof=0)),
    }


def load_topk_candidate_manifest() -> pd.DataFrame:
    manifest_path = Path(cfg.topk_candidate_manifest_path)
    if not manifest_path.exists():
        return pd.DataFrame(columns=["step", "local_accuracy", "candidate_dir"])

    out = pd.read_csv(manifest_path)
    if out.empty:
        return out

    out = out.loc[out["candidate_dir"].map(lambda p: Path(p).exists())].copy()
    return out.sort_values(["local_accuracy", "step"], ascending=[False, False]).reset_index(drop=True)


def rerank_topk_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()

    rows = []
    rerank_eval_df = serious_eval_df.copy()

    if cfg.final_rerank_run_submission_style:
        rerank_submission_df = rerank_eval_df.head(min(len(rerank_eval_df), cfg.final_rerank_submission_rows))
    else:
        rerank_submission_df = pd.DataFrame()

    total = len(candidate_df)
    for idx, cand in enumerate(candidate_df.itertuples(index=False), start=1):
        candidate_dir = str(cand.candidate_dir)
        print(
            f"\n[final-rerank] candidate {idx}/{total} | "
            f"step={cand.step} | local_acc={cand.local_accuracy:.4f} | dir={candidate_dir}"
        )

        candidate_model = None
        try:
            candidate_model = load_candidate_adapter_for_eval(candidate_dir)

            serious_acc, serious_pred_df, serious_multi_seed_df = run_serious_eval_suite(
                candidate_model,
                rerank_eval_df,
                serious_eval_views,
                prefix=f"final_rerank_{Path(candidate_dir).name}",
                run_multi_seed=True,
            )

            if "answer_type" in serious_pred_df.columns and "hit" in serious_pred_df.columns:
                short_text_mask = serious_pred_df["answer_type"].eq("short_text")
                short_text_accuracy = float(
                    serious_pred_df.loc[short_text_mask, "hit"].mean()
                ) if short_text_mask.any() else float("nan")
            else:
                short_text_accuracy = float("nan")

            multi_seed_summary = summarize_multi_seed_result(serious_multi_seed_df)

            if cfg.final_rerank_run_submission_style and not rerank_submission_df.empty:
                submission_style_acc, _ = offline_submission_style_eval(
                    candidate_model,
                    rerank_submission_df,
                    top_k=cfg.test_time_router_top_k,
                )
            else:
                submission_style_acc = float("nan")

            rows.append(
                {
                    "step": int(cand.step),
                    "local_accuracy": float(cand.local_accuracy),
                    "candidate_dir": candidate_dir,
                    "serious_accuracy": float(serious_acc),
                    "short_text_accuracy": float(short_text_accuracy),
                    "multi_seed_mean_accuracy": float(multi_seed_summary["multi_seed_mean_accuracy"]),
                    "multi_seed_accuracy_std": float(multi_seed_summary["multi_seed_accuracy_std"]),
                    "submission_style_accuracy": float(submission_style_acc),
                    "rerank_status": "ok",
                    "rerank_error": "",
                }
            )

        except Exception as exc:
            print(f"[final-rerank][warning] failed for {candidate_dir}: {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "step": int(cand.step),
                    "local_accuracy": float(cand.local_accuracy),
                    "candidate_dir": candidate_dir,
                    "serious_accuracy": float("nan"),
                    "short_text_accuracy": float("nan"),
                    "multi_seed_mean_accuracy": float("nan"),
                    "multi_seed_accuracy_std": float("nan"),
                    "submission_style_accuracy": float("nan"),
                    "rerank_status": "failed",
                    "rerank_error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if candidate_model is not None:
                try:
                    del candidate_model
                except Exception:
                    pass
            _cleanup_after_candidate_eval()

    report_df = pd.DataFrame(rows)

    # 如果全部 candidate 复选失败，则回退到 local_accuracy 排序，避免整条导出链路断掉
    valid_mask = report_df["rerank_status"].eq("ok") & report_df["serious_accuracy"].notna()
    if not valid_mask.any():
        print("[final-rerank][warning] all candidates failed during rerank; fallback to local_accuracy ordering.")
        report_df = report_df.sort_values(
            ["local_accuracy", "step"],
            ascending=[False, False],
        ).reset_index(drop=True)
        report_df["final_pick_rank"] = np.arange(1, len(report_df) + 1)
        return report_df

    # ===== 修复：开了 submission-style rerank 时，必须把 submission_style_accuracy 纳入排序 =====
    if cfg.final_rerank_run_submission_style:
        report_df = report_df.sort_values(
            by=[
                "rerank_status",
                "serious_accuracy",
                "submission_style_accuracy",
                "short_text_accuracy",
                "multi_seed_mean_accuracy",
                "multi_seed_accuracy_std",
                "local_accuracy",
                "step",
            ],
            ascending=[True, False, False, False, False, True, False, False],
            na_position="last",
        ).reset_index(drop=True)
    else:
        report_df = report_df.sort_values(
            by=[
                "rerank_status",
                "serious_accuracy",
                "short_text_accuracy",
                "multi_seed_mean_accuracy",
                "multi_seed_accuracy_std",
                "local_accuracy",
                "step",
            ],
            ascending=[True, False, False, False, True, False, False],
            na_position="last",
        ).reset_index(drop=True)

    report_df["final_pick_rank"] = np.arange(1, len(report_df) + 1)
    return report_df
# %% [markdown]
# ## Cell 23 — 保存 LoRA adapter，并检查 rank/配置是否合法
# 
# 比赛最终提交只需要兼容 Nemotron 基座的 LoRA adapter。
# 这里显式校验 `adapter_config.json`，避免训练完才发现不能交。

# %%
def release_training_objects_before_final_rerank() -> None:
    global model, trainer_stage1, trainer_stage2, collator

    # 断开 collator 对旧模型的引用
    if "collator" in globals() and collator is not None:
        try:
            if hasattr(collator, "model"):
                collator.model = None
        except Exception:
            pass
        try:
            del collator
        except Exception:
            pass

    # 清 trainer
    for trainer_name in ("trainer_stage1", "trainer_stage2"):
        if trainer_name in globals():
            try:
                trainer_obj = globals()[trainer_name]
                try:
                    if hasattr(trainer_obj, "model"):
                        trainer_obj.model = None
                except Exception:
                    pass
                try:
                    if hasattr(trainer_obj, "model_wrapped"):
                        trainer_obj.model_wrapped = None
                except Exception:
                    pass
                try:
                    if hasattr(trainer_obj, "optimizer"):
                        trainer_obj.optimizer = None
                except Exception:
                    pass
                try:
                    if hasattr(trainer_obj, "lr_scheduler"):
                        trainer_obj.lr_scheduler = None
                except Exception:
                    pass
                del globals()[trainer_name]
            except Exception:
                pass

    # 清主模型
    if "model" in globals():
        try:
            old_model = model
            del model
            del old_model
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def release_before_final_proxy_eval() -> None:
    gc.collect()

    try:
        _cleanup_after_candidate_eval()
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

final_adapter_dir = Path(cfg.adapter_dir)
if final_adapter_dir.exists():
    shutil.rmtree(final_adapter_dir, ignore_errors=True)

if cfg.smoke_test_mode:
    print("[smoke] skip top-k rerank; export current in-memory model")
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

else:
    topk_candidate_df = load_topk_candidate_manifest()

    if not topk_candidate_df.empty:
        release_training_objects_before_final_rerank()

        rerank_report_df = rerank_topk_candidates(topk_candidate_df)
        rerank_report_df.to_csv(Path(cfg.final_rerank_report_path), index=False)

        winner = rerank_report_df.iloc[0].to_dict()
        winner_dir = Path(winner["candidate_dir"])

        shutil.copytree(winner_dir, final_adapter_dir)

        Path(cfg.final_export_info_path).write_text(
            json.dumps(winner, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("[final export] selected winner from top-k candidates:")
        print(json.dumps(winner, ensure_ascii=False, indent=2))
    else:
        print("[final export] no top-k candidates found; fallback to current in-memory model")
        final_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)

        fallback_info = {
            "candidate_dir": str(final_adapter_dir),
            "local_accuracy": None,
            "step": None,
            "selection_mode": "fallback_current_in_memory_model",
        }

        Path(cfg.final_export_info_path).write_text(
            json.dumps(fallback_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

adapter_cfg_path = final_adapter_dir / "adapter_config.json"
assert adapter_cfg_path.exists(), "缺少 adapter_config.json，无法提交"

with open(adapter_cfg_path, "r", encoding="utf-8") as f:
    adapter_cfg = json.load(f)

assert int(adapter_cfg.get("r", cfg.lora_r)) <= cfg.max_lora_rank, "导出的 LoRA rank 超过比赛限制"
print(json.dumps(adapter_cfg, ensure_ascii=False, indent=2)[:1200])
print("adapter files:", sorted(p.name for p in final_adapter_dir.iterdir()))
# ===== 最终导出 adapter 的 proxy eval =====
if cfg.smoke_test_mode:
    print("[smoke] skip final exported adapter proxy eval in Cell 23")
else:
    release_before_final_proxy_eval()

    reloaded_final_model = None
    try:
        reloaded_final_model = load_candidate_adapter_for_eval(str(final_adapter_dir))

        final_export_proxy_acc, final_export_proxy_df = evaluate_accuracy(
            reloaded_final_model,
            serious_eval_df,
            extractor_fn=metric_extract_prediction,
            router_top_k=cfg.test_time_router_top_k,
            max_new_tokens_grid=(64, 96, 128),
        )

        print(f"[final exported adapter proxy accuracy] {final_export_proxy_acc:.4f}")
        final_export_proxy_df.to_csv(
            Path(cfg.work_dir) / "final_exported_adapter_proxy_eval.csv",
            index=False,
        )

    except RuntimeError as e:
        err_text = str(e)
        if "Unable to reload base model for candidate evaluation" in err_text or "CUDA out of memory" in err_text:
            print("[final proxy eval] skipped due to reload/OOM after rerank")
            print(err_text)
        else:
            raise

    finally:
        if reloaded_final_model is not None:
            try:
                del reloaded_final_model
            except Exception:
                pass
        try:
            _cleanup_after_candidate_eval()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
# %% [markdown]
# ## Cell 24 — 打包 submission.zip

# %%
zip_path = Path(cfg.submission_zip)
if zip_path.exists():
    zip_path.unlink()

allowed_submit_files = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
}

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    found = set()
    for path in sorted(Path(cfg.adapter_dir).iterdir()):
        if path.is_file() and path.name in allowed_submit_files:
            zf.write(path, arcname=path.name)
            found.add(path.name)

assert "adapter_config.json" in found, "submission.zip 缺少 adapter_config.json"
assert ("adapter_model.safetensors" in found) or ("adapter_model.bin" in found), "submission.zip 缺少 adapter 权重文件"

print("submission zip:", zip_path)
print("files packed:", sorted(found))
print("zip size (MB):", round(zip_path.stat().st_size / 1024 / 1024, 3))

# Cell 24.1
def release_training_objects_before_reload() -> None:
    global model, trainer_stage1, trainer_stage2, collator

    print("[smoke] releasing training objects before adapter reload")

    # 先断开 collator 对旧 model 的隐藏引用
    if "collator" in globals() and collator is not None:
        try:
            if hasattr(collator, "model"):
                collator.model = None
        except Exception:
            pass
        try:
            del collator
        except Exception:
            pass

    # 删除可能还在的 trainer
    for trainer_name in ("trainer_stage1", "trainer_stage2"):
        if trainer_name in globals():
            try:
                trainer_obj = globals()[trainer_name]

                # 尽量断开 trainer 内部对 model 的引用
                try:
                    if hasattr(trainer_obj, "model"):
                        trainer_obj.model = None
                except Exception:
                    pass
                try:
                    if hasattr(trainer_obj, "model_wrapped"):
                        trainer_obj.model_wrapped = None
                except Exception:
                    pass
                try:
                    if hasattr(trainer_obj, "optimizer"):
                        trainer_obj.optimizer = None
                except Exception:
                    pass
                try:
                    if hasattr(trainer_obj, "lr_scheduler"):
                        trainer_obj.lr_scheduler = None
                except Exception:
                    pass

                del globals()[trainer_name]
            except Exception:
                pass

    # 删除主模型
    if "model" in globals():
        try:
            old_model = model
            del model
            del old_model
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(
            f"[smoke] CUDA free after release: "
            f"{free_bytes / (1024 ** 3):.2f} / {total_bytes / (1024 ** 3):.2f} GiB"
        )

# Cell 24.5
def smoke_reload_exported_adapter_check(
    adapter_dir: Path,
    probe_df: pd.DataFrame,
    rows: int = 2,
) -> pd.DataFrame:
    print(f"[smoke] reload exported adapter from: {adapter_dir}")
    reload_model = None
    try:
        reload_model = load_candidate_adapter_for_eval(str(adapter_dir))
        probe_subset = probe_df.head(min(rows, len(probe_df))).copy()

        reload_acc, reload_pred_df = evaluate_accuracy(
            reload_model,
            probe_subset,
            extractor_fn=fast_extract_prediction,
        )
        print(f"[smoke] reloaded-adapter probe accuracy: {reload_acc:.4f}")
        return reload_pred_df

    finally:
        if reload_model is not None:
            try:
                del reload_model
            except Exception:
                pass
        _cleanup_after_candidate_eval()

# Cell 24.9
if cfg.smoke_test_mode and cfg.smoke_profile == "pipeline" and cfg.smoke_run_export_reload_check:
    # 关键：先释放训练期那份大模型，不然 reload 会在同进程里挤两份 30B
    release_training_objects_before_reload()

    print(f"[smoke] reload exported adapter from: {final_adapter_dir}")

    # 直接把 reload 后的模型重新赋给 model，后面 Cell 25 继续复用它
    model = load_candidate_adapter_for_eval(str(final_adapter_dir))

    probe_subset = fast_eval_df.head(min(cfg.smoke_export_reload_rows, len(fast_eval_df))).copy()
    smoke_reload_acc, smoke_reload_df = evaluate_accuracy(
        model,
        probe_subset,
        extractor_fn=fast_extract_prediction,
    )

    print(f"[smoke] reloaded-adapter probe accuracy: {smoke_reload_acc:.4f}")
    display(smoke_reload_df.head(10))
    smoke_reload_df.to_csv(
        Path(cfg.work_dir) / "smoke_reloaded_adapter_probe.csv",
        index=False,
    )
else:
    print("skip exported-adapter reload smoke check")


# %% [markdown]
# ## Cell 25 — 提交前 smoke test
# 
# 这一步的目标不是拿高分，而是确认：
# - 模型能正常加载 adapter；
# - 生成格式稳定输出 boxed；
# - 不会出现空输出或格式崩坏。

# %%
smoke_df = test_df.head(5).copy()
smoke_rows = []
for row in smoke_df.itertuples(index=False):
    result = submission_style_predict_row(
        model,
        row,
        template_ids=router_get_templates(infer_prompt_family(row.prompt), top_k=cfg.test_time_router_top_k),
    )
    smoke_rows.append(
        {
            "id": row.id,
            "pred": result["pred"],
            "decision_type": result["decision_type"],
            "fallback_used": result["fallback_used"],
            "template_ids": ",".join(result["template_ids"]),
            "raw": result["candidates"][0]["raw"][:1000] if result["candidates"] else "",
        }
    )
smoke_pred_df = pd.DataFrame(smoke_rows)
display(smoke_pred_df)
smoke_pred_df.to_csv(Path(cfg.work_dir) / "smoke_predictions.csv", index=False)


# %% [markdown]
# ## Cell 26 — 已落地能力 vs 下一步冲奖路线
# 
# **这一版已经落地：**
# 1. `family_aware_mix` 主监督路径。
# 2. Stage 1 后 template ablation，并强制刷新 Stage 2 records / datasets。
# 3. group hard mining + sample-level replay buffer 联合重加权。
# 4. serious eval（family-balanced 主视图 + 多 seed 视图）。
# 5. external mixture + consensus pseudolabel refresh 主路径接口。
# 
# **下一步仍然值得继续做：**
# 1. 更高质量的 family-specific synthetic data + verifier 过滤。
# 2. rejection sampling / self-distill，进一步提高伪标签质量。
# 3. 训练后做 prompt ensemble，按 family 或 answer_type 选择推理模板。
# 4. 只对 hardest families 做局部 RL / DPO / preference optimization。
# 5. 多 fold / 多 seed bagging，提升 private leaderboard 稳定性。

# %% [markdown]
# ## Cell 27 — 释放显存 / 导出配置快照

# %%
with open(Path(cfg.work_dir) / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("done")
