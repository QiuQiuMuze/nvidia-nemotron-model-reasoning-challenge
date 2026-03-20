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
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers>=4.49.0",
            "datasets>=3.2.0",
            "peft>=0.14.0",
            "trl>=0.15.0",
            "accelerate>=1.2.1",
            "bitsandbytes>=0.45.0",
            "scikit-learn>=1.6.0",
        ],
        check=True,
    )

# %% [markdown]
# ## Cell 2 — 导入依赖与全局配置
# 这里把所有“竞赛硬约束”显式写进配置，避免训练流程偏离提交规则。

# %%
import gc
import hashlib
import io
import json
import math
import random
import re
import shutil
import textwrap
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    base_model: str = "nvidia/Nemotron-3-Nano-30B-Base-8K"
    max_lora_rank: int = 32
    official_max_new_tokens: int = 7680
    official_temperature: float = 0.0
    official_top_p: float = 1.0
    official_max_model_len: int = 8192

    # ===== paths =====
    competition_path: str = "/kaggle/input/nvidia-nemotron-model-reasoning-challenge"
    extra_data_dir: str = "/kaggle/input"
    work_dir: str = "/kaggle/working/nemotron_advanced"
    output_dir: str = "/kaggle/working/nemotron_advanced/checkpoints"
    adapter_dir: str = "/kaggle/working/nemotron_advanced/adapter"
    submission_zip: str = "/kaggle/working/submission.zip"

    # ===== reproducibility =====
    seed: int = 3407

    # ===== training =====
    use_bf16: bool = True
    use_4bit: bool = True
    max_seq_len: int = 2048
    micro_batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    max_grad_norm: float = 0.3

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

    # ===== curriculum =====
    stage1_epochs: float = 0.6
    stage2_epochs: float = 1.2
    stage2_rounds: int = 3
    stage1_max_prompt_chars: int = 900
    stage1_lr: float = 1.6e-4
    stage2_lr: float = 9e-5

    # ===== optional leaderboard tricks =====
    enable_external_mixture: bool = True
    enable_prompt_template_ablation: bool = True
    enable_family_reweight: bool = True
    enable_length_bucket_bonus: bool = True
    run_supervision_ablation: bool = False
    primary_supervision_variant: str = "family_aware_mix"
    supervision_ablation_variants: Tuple[str, ...] = ("answer_only", "short_reasoning", "family_aware_mix")
    supervision_ablation_stage1_epochs: float = 0.20
    supervision_ablation_stage2_epochs: float = 0.20
    supervision_ablation_stage2_rounds: int = 1
    reasoning_template_eval_rows: int = 48
    enable_consensus_pseudolabel_refresh: bool = True
    consensus_pseudolabel_max_rows: int = 192
    consensus_template_ids: Tuple[str, ...] = (
        "T1_ultra_compact",
        "T3_hidden_reasoning",
        "T4_numeric_specialized",
        "T5_text_specialized",
    )
    template_ablation_after_stage1_rows: int = 128
    template_ablation_stage2_min_gain: float = 0.0
    fixed_sanity_rows: int = 64
    stage1_family_frequency_quantile: float = 0.55
    hard_mining_family_boost: float = 0.35
    hard_mining_template_group_boost: float = 0.20
    hard_mining_answer_type_boost: float = 0.20
    hard_mining_bucket_boost: float = 0.15
    hard_mining_sample_boost: float = 0.60
    replay_probe_rows: int = 128
    allow_target_auto_discovery: bool = False
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

cfg = CFG()

Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.adapter_dir).mkdir(parents=True, exist_ok=True)

assert cfg.lora_r <= cfg.max_lora_rank, "LoRA rank 超过比赛上限 32"

# %% [markdown]
# ## Cell 3 — 固定随机种子 / 环境检查

# %%
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

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
train_path = Path(cfg.competition_path) / "train.csv"
test_path = Path(cfg.competition_path) / "test.csv"

if not train_path.exists():
    # 兼容仓库本地调试
    train_path = Path("train.csv")
    test_path = Path("test.csv")

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


def infer_answer_type(answer: str) -> str:
    answer = str(answer).strip()
    if BINARY_RE.fullmatch(answer):
        return "binary"
    if ANSWER_NUMBER_RE.fullmatch(answer.replace(",", "")):
        return "numeric"
    if " " in answer:
        return "multi_token_text"
    return "short_text"

full_train_df["prompt_norm"] = full_train_df["prompt"].map(normalize_whitespace)
full_train_df["answer_norm"] = full_train_df["answer"].astype(str).str.strip()
full_train_df["prompt_family"] = full_train_df["prompt"].map(infer_prompt_family)
full_train_df["template_group"] = full_train_df["prompt"].map(infer_template_group)
full_train_df["answer_type"] = full_train_df["answer"].map(infer_answer_type)
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

splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.valid_size, random_state=cfg.seed)
train_idx, valid_idx = next(splitter.split(full_train_df, groups=full_train_df["template_group"]))

train_fold = full_train_df.iloc[train_idx].reset_index(drop=True)
valid_fold = full_train_df.iloc[valid_idx].reset_index(drop=True)

print("train fold:", train_fold.shape)
print("valid fold:", valid_fold.shape)
print("train families:", train_fold['prompt_family'].nunique())
print("valid families:", valid_fold['prompt_family'].nunique())
print("train template groups:", train_fold['template_group'].nunique())
print("valid template groups:", valid_fold['template_group'].nunique())

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
]


def canonicalize_answer(ans: Any) -> str:
    ans = str(ans).strip()
    ans = ans.replace("，", ",")
    if ans.startswith("$") and ans.endswith("$"):
        ans = ans[1:-1].strip()
    if ANSWER_NUMBER_RE.fullmatch(ans.replace(",", "")):
        numeric = float(ans.replace(",", ""))
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".12g")
    return normalize_whitespace(ans)


def boxed(ans: Any) -> str:
    clean = canonicalize_answer(ans)
    return clean if clean.startswith("\\boxed{") else f"\\boxed{{{clean}}}"


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
            candidate = candidate.rstrip(".。 ")
            candidate = re.sub(r"^\$|\$$", "", candidate)
            candidate = re.sub(r"^[=:：-]\s*", "", candidate)
            if candidate:
                return canonicalize_answer(candidate)
    return None


def extract_other_heuristics(text: str) -> Optional[str]:
    text = str(text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = re.sub(r"^\$|\$$", "", lines[-1]).strip()
        if last_line:
            if LAST_NUMBER_RE.fullmatch(last_line.replace(",", "")):
                return canonicalize_answer(last_line)
            if len(last_line) <= 128:
                return canonicalize_answer(last_line)
    return None


def extract_last_numeric(text: str) -> Optional[str]:
    text = str(text).strip()
    last_num_hits = LAST_NUMBER_RE.findall(text.replace(",", ""))
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


def approx_equal(pred: str, gold: str, rel_tol: float = cfg.numeric_rel_tol) -> bool:
    pred_c = canonicalize_answer(pred)
    gold_c = canonicalize_answer(gold)
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
        Infer the hidden rule from the examples. Reason briefly and return only the final text answer in \\boxed{{}}.
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
    "cipher_decrypt": "T5_text_specialized",
    "fewshot_pattern": "T3_hidden_reasoning",
    "matrix_reasoning": "T3_hidden_reasoning",
    "sequence": "T4_numeric_specialized",
    "open_template": "T5_text_specialized",
}
BEST_TEMPLATE_BY_FAMILY: Dict[str, str] = dict(FAMILY_TEMPLATE_PRIORS)


def choose_template_id(answer_type: str, prompt_family: str) -> str:
    if prompt_family in BEST_TEMPLATE_BY_FAMILY:
        return BEST_TEMPLATE_BY_FAMILY[prompt_family]
    if not cfg.enable_prompt_template_ablation:
        return "T2_compact"
    if answer_type in {"numeric", "binary"}:
        return "T4_numeric_specialized"
    if prompt_family in {"cipher_decrypt", "open_template"} or answer_type == "multi_token_text":
        return "T5_text_specialized"
    return "T3_hidden_reasoning"


def choose_template(problem: str, answer_type: str, prompt_family: str) -> Tuple[str, str]:
    template_id = choose_template_id(answer_type, prompt_family)
    return template_id, TEMPLATE_POOL[template_id](problem)


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

def build_training_record(row: pd.Series) -> Dict[str, Any]:
    template_id, prompt_text = choose_template(row.prompt, row.answer_type, row.prompt_family)
    answer_text = boxed(row.answer)
    short_reasoning_text = build_short_reasoning_scaffold(row.answer, row.answer_type, row.prompt_family)
    full_text = prompt_text + answer_text
    reasoning_full_text = prompt_text + short_reasoning_text

    difficulty = 1.0
    difficulty += min(row.prompt_chars / 1800.0, 1.5)
    difficulty += 0.2 if row.answer_type == "multi_token_text" else 0.0
    difficulty += 0.15 if row.example_len_bucket in {"l", "xl"} else 0.0
    difficulty += 0.2 if row.prompt_family == "open_template" else 0.0

    return {
        "id": row.id,
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
        "difficulty": difficulty,
        "prompt_chars": row.prompt_chars,
        "len_bucket": str(row.example_len_bucket),
    }


def build_records_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "id",
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
                "difficulty",
                "prompt_chars",
                "len_bucket",
            ]
        )
    return df.apply(build_training_record, axis=1, result_type="expand")


def split_stage_records(records_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    family_freq = records_df["prompt_family"].value_counts(normalize=True)
    stable_families = set(
        family_freq[family_freq >= family_freq.quantile(cfg.stage1_family_frequency_quantile)].index.tolist()
    )
    stage1_mask = (
        records_df["prompt_chars"].le(cfg.stage1_max_prompt_chars)
        & records_df["prompt_family"].isin(stable_families)
        & records_df["answer_type"].isin(["numeric", "binary", "short_text"])
    )
    stage1_records_local = records_df.loc[stage1_mask].reset_index(drop=True)
    stage2_records_local = records_df.reset_index(drop=True)
    return stage1_records_local, stage2_records_local, stable_families


train_records = build_records_frame(train_fold)
valid_records = build_records_frame(valid_fold)
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
print("stage2 records:", stage2_records.shape)
print("stable families used in stage1:", sorted(stable_families)[:20])

# %% [markdown]
# ## Cell 11 — 加载 tokenizer
# 
# Nemotron 系列 tokenizer 若带 chat template，可自行替换成 `apply_chat_template` 版本。
# 这里保守起见直接使用手写模板，确保迁移到 Kaggle 更稳定。

# %%
tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
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
    prompt_enc = tokenizer(
        example["prompt_text"],
        add_special_tokens=False,
        truncation=True,
        max_length=cfg.max_seq_len,
    )
    full_enc = tokenizer(
        example[target_key],
        add_special_tokens=False,
        truncation=True,
        max_length=cfg.max_seq_len,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))
    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

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
        short_reasoning_families = {"open_template", "cipher_decrypt", "matrix_reasoning"}
        tmp["supervision_variant"] = np.where(
            tmp["prompt_family"].isin(short_reasoning_families) | tmp["answer_type"].eq("multi_token_text"),
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

stage1_ds = stage1_variant_ds[cfg.primary_supervision_variant]
stage2_ds = stage2_variant_ds[cfg.primary_supervision_variant]
valid_ds = valid_variant_ds[cfg.primary_supervision_variant]

print(stage2_ds[0].keys())
print("non-masked labels:", sum(x != -100 for x in stage2_ds[0]["labels"]))

# %% [markdown]
# ## Cell 13 — 加载 4-bit 基座模型 + 自动发现 LoRA target modules
# 
# 高级版不把 target modules 写死在 `q_proj/k_proj/v_proj/o_proj`：
# Nemotron / Llama 系模型中，MLP 的 `gate/up/down_proj` 往往也很重要。
# 在 rank 固定 <= 32 的前提下，覆盖 attention + MLP 往往比只训 attention 更强。

# %%
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

def build_quant_config() -> Optional[BitsAndBytesConfig]:
    if not cfg.use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_training_model() -> Tuple[torch.nn.Module, List[str]]:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=build_quant_config(),
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    discovered_targets = discover_lora_targets(model.named_modules())
    if cfg.allow_target_auto_discovery:
        lora_targets = discovered_targets
    else:
        # 默认固定最终 LoRA target modules，避免实验噪声来自挂层漂移。
        lora_targets = [module_name for module_name in cfg.target_modules_final if module_name in discovered_targets]

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
    return model, lora_targets


model, lora_targets = load_training_model()
print("LoRA targets:", lora_targets)
model.print_trainable_parameters()

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


fast_eval_df = stratified_valid_subset(valid_fold, cfg.fast_eval_examples)
serious_eval_df = build_seeded_family_balanced_subset(valid_fold, cfg.serious_eval_examples, seed=cfg.seed)
serious_eval_views = {
    seed: build_seeded_family_balanced_subset(valid_fold, cfg.serious_eval_examples, seed=seed)
    for seed in cfg.serious_eval_seeds
}
fixed_sanity_df = build_fixed_sanity_subset(valid_fold, cfg.fixed_sanity_rows)
train_replay_probe_df = build_fixed_sanity_subset(train_fold, cfg.replay_probe_rows)
print("fast eval subset:", fast_eval_df.shape)
print("serious eval subset:", serious_eval_df.shape)
print("serious eval view seeds:", {seed: df.shape for seed, df in serious_eval_views.items()})
print("fixed sanity subset:", fixed_sanity_df.shape)
print("train replay probe subset:", train_replay_probe_df.shape)


@torch.no_grad()
def generate_answer_text(model: torch.nn.Module, prompt: str, max_new_tokens: int = 96) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=cfg.official_temperature,
        top_p=cfg.official_top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    df: pd.DataFrame,
    template_override: Optional[str] = None,
    max_new_tokens_grid: Sequence[int] = (64, 96),
    extractor_fn=official_like_extract_prediction,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for row in df.itertuples(index=False):
        answer_type = infer_answer_type(row.answer)
        prompt_family = infer_prompt_family(row.prompt)
        template_id, prompt = choose_template(row.prompt, answer_type, prompt_family)
        if template_override is not None:
            template_id = template_override
            prompt = TEMPLATE_POOL[template_id](row.prompt)

        candidate_raw = None
        candidate_pred = None
        boxed_hit = False
        boxed_parsed_success = False
        for max_new_tokens in max_new_tokens_grid:
            raw = generate_answer_text(model, prompt, max_new_tokens=max_new_tokens)
            boxed_matches = BOXED_RE.findall(raw)
            pred = extractor_fn(raw)
            if "\\boxed{" in raw:
                candidate_raw = raw
                candidate_pred = pred
                boxed_hit = True
                boxed_parsed_success = bool(boxed_matches)
                break
            if candidate_raw is None:
                candidate_raw = raw
                candidate_pred = pred
                boxed_parsed_success = bool(boxed_matches)

        gold = canonicalize_answer(row.answer)
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
                "template_id": template_id,
                "gold": gold,
                "pred": candidate_pred,
                "hit": approx_equal(candidate_pred, gold),
                "has_boxed": boxed_hit,
                "boxed_parsed_success": boxed_parsed_success,
                "failure_type": failure_type,
                "raw": candidate_raw[:1200],
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
    extractor_fn=official_like_extract_prediction,
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
    return {
        "family": (1.0 - pred_df.groupby("prompt_family")["hit"].mean()).to_dict(),
        "template_group": (1.0 - pred_df.groupby("template_group")["hit"].mean()).to_dict(),
        "answer_type": (1.0 - pred_df.groupby("answer_type")["hit"].mean()).to_dict(),
        "len_bucket": (1.0 - pred_df.groupby("len_bucket")["hit"].mean()).to_dict(),
    }


def run_serious_eval_suite(
    model: torch.nn.Module,
    primary_df: pd.DataFrame,
    eval_views: Dict[int, pd.DataFrame],
    prefix: str,
    extractor_fn=official_like_extract_prediction,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    primary_acc, primary_pred_df = evaluate_accuracy(model, primary_df, extractor_fn=extractor_fn)
    print(f"{prefix} serious accuracy: {primary_acc:.4f}")
    print_eval_summaries(primary_pred_df, prefix=prefix)
    multi_seed_df = evaluate_multi_seed_views(model, eval_views, extractor_fn=extractor_fn)
    display(multi_seed_df)
    primary_pred_df.to_csv(Path(cfg.work_dir) / f"{prefix}_predictions.csv", index=False)
    multi_seed_df.to_csv(Path(cfg.work_dir) / f"{prefix}_multi_seed_eval.csv", index=False)
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


def run_template_ablation(model: torch.nn.Module, df: pd.DataFrame, max_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            pred_df.groupby("prompt_family")["hit"].mean().reset_index().rename(columns={"hit": "accuracy"})
        )
        family_scores["template_id"] = template_id
        mapping_rows.append(family_scores)
    score_df = pd.DataFrame(rows).sort_values(["accuracy", "boxed_parse_rate", "has_boxed_rate"], ascending=False).reset_index(drop=True)
    mapping_df = pd.concat(mapping_rows, ignore_index=True)
    best_mapping = (
        mapping_df.sort_values(["prompt_family", "accuracy"], ascending=[True, False])
        .groupby("prompt_family", as_index=False)
        .first()
        .rename(columns={"template_id": "best_template_id", "accuracy": "best_accuracy"})
    )
    return score_df, best_mapping


@torch.no_grad()
def ensemble_predict(model: torch.nn.Module, prompt_text: str, template_ids: Sequence[str]) -> Dict[str, Any]:
    votes = []
    raw_candidates = []
    for template_id in template_ids:
        raw = generate_answer_text(model, TEMPLATE_POOL[template_id](prompt_text))
        pred = extract_prediction(raw)
        votes.append(pred)
        raw_candidates.append({"template_id": template_id, "pred": pred, "raw": raw[:800]})

    vote_counter = Counter(votes)
    best_pred, best_votes = sorted(vote_counter.items(), key=lambda x: (-x[1], x[0]))[0]
    return {
        "pred": best_pred,
        "votes": best_votes,
        "num_templates": len(template_ids),
        "candidates": raw_candidates,
    }


@torch.no_grad()
def evaluate_with_consensus(model: torch.nn.Module, df: pd.DataFrame, template_ids: Sequence[str]) -> Tuple[float, pd.DataFrame]:
    rows = []
    for row in df.itertuples(index=False):
        result = ensemble_predict(model, row.prompt, template_ids=template_ids)
        gold = canonicalize_answer(row.answer)
        rows.append(
            {
                "id": row.id,
                "prompt_family": infer_prompt_family(row.prompt),
                "answer_type": infer_answer_type(row.answer),
                "gold": gold,
                "pred": result["pred"],
                "hit": approx_equal(result["pred"], gold),
                "consensus_votes": result["votes"],
                "num_templates": result["num_templates"],
            }
        )
    out = pd.DataFrame(rows)
    return float(out["hit"].mean()) if len(out) else 0.0, out


@torch.no_grad()
def build_consensus_pseudolabels(model: torch.nn.Module, unlabeled_df: pd.DataFrame, template_ids: Sequence[str], max_rows: int = 128) -> pd.DataFrame:
    rows = []
    for row in unlabeled_df.head(max_rows).itertuples(index=False):
        result = ensemble_predict(model, row.prompt, template_ids=template_ids)
        if result["votes"] == result["num_templates"]:
            rows.append(
                {
                    "id": getattr(row, "id", f"pseudo_{len(rows)}"),
                    "prompt": row.prompt,
                    "answer": result["pred"],
                    "source": "consensus_pseudolabel",
                    "confidence": result["votes"] / result["num_templates"],
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def summarize_template_disagreement(model: torch.nn.Module, df: pd.DataFrame, template_ids: Sequence[str], max_rows: int = 128) -> pd.DataFrame:
    rows = []
    for row in df.head(max_rows).itertuples(index=False):
        result = ensemble_predict(model, row.prompt, template_ids=template_ids)
        disagreement = 1.0 - (result["votes"] / result["num_templates"])
        rows.append(
            {
                "id": row.id,
                "prompt_family": infer_prompt_family(row.prompt),
                "answer_type": infer_answer_type(getattr(row, "answer", "")),
                "disagreement": disagreement,
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
    def __init__(self, eval_df: pd.DataFrame, save_path: str, extractor_fn=fast_extract_prediction):
        self.eval_df = eval_df
        self.save_path = save_path
        self.extractor_fn = extractor_fn
        self.history: List[Dict[str, Any]] = []

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        acc, pred_df = evaluate_accuracy(model, self.eval_df, extractor_fn=self.extractor_fn)
        record = {"step": int(state.global_step), "local_accuracy": acc}
        self.history.append(record)
        pd.DataFrame(self.history).to_csv(self.save_path, index=False)
        pred_df.to_csv(Path(self.save_path).with_suffix(".preds.csv"), index=False)
        summaries = summarize_eval_metrics(pred_df)
        for key, value in summaries.items():
            value.to_csv(Path(self.save_path).with_name(f"{Path(self.save_path).stem}_{key}_step_{state.global_step}.csv"), index=False)
        print(f"\n[local-accuracy] step={state.global_step} acc={acc:.4f}")
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
    return TrainingArguments(
        output_dir=str(Path(cfg.output_dir) / stage_name),
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.micro_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        bf16=cfg.use_bf16,
        fp16=not cfg.use_bf16,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=False,
        label_names=["labels"],
    )


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
    )
    if pseudo_df.empty:
        print("no consensus pseudolabels were added to stage2")
        return stage2_records_df

    pseudo_df = pseudo_df.assign(
        prompt_family=pseudo_df["prompt"].map(infer_prompt_family),
        template_group=pseudo_df["prompt"].map(infer_template_group),
        answer_type=pseudo_df["answer"].map(infer_answer_type),
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
    refreshed_train_records = build_records_frame(train_fold_df)
    refreshed_valid_records = build_records_frame(valid_fold_df)
    _stage1_records_ref, stage2_records_ref, _ = split_stage_records(refreshed_train_records)
    stage2_records_ref = maybe_extend_stage2_with_pseudolabels(current_model, stage2_records_ref, unlabeled_df if unlabeled_df is not None else pd.DataFrame()) if current_model is not None else stage2_records_ref
    _, stage2_variant_ref, valid_variant_ref = refresh_variant_datasets(stage1_records_df, stage2_records_ref, refreshed_valid_records)
    stage2_ds_ref = stage2_variant_ref[cfg.primary_supervision_variant]
    valid_ds_ref = valid_variant_ref[cfg.primary_supervision_variant]
    stage2_weights_ref = compute_sample_weights(stage2_records_ref)
    return refreshed_train_records, refreshed_valid_records, stage2_variant_ref, valid_variant_ref, stage2_ds_ref, valid_ds_ref, stage2_weights_ref, stage2_records_ref


def train_stage2_with_refresh(
    model: torch.nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    train_records_df: pd.DataFrame,
    initial_weights: np.ndarray,
    replay_buffer: Optional[Dict[str, int]] = None,
) -> Tuple[torch.nn.Module, pd.DataFrame, Dict[str, int]]:
    current_weights = initial_weights
    last_pred_df = pd.DataFrame()
    replay_buffer = dict(replay_buffer or {})
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
        probe_acc, replay_probe_pred_df = evaluate_accuracy(model, train_replay_probe_df, extractor_fn=fast_extract_prediction)
        replay_probe_pred_df.to_csv(Path(cfg.work_dir) / f"stage2_round_{round_idx + 1}_train_probe_predictions.csv", index=False)
        replay_buffer = update_replay_buffer(replay_buffer, replay_probe_pred_df)
        current_weights = compute_sample_weights(train_records_df, hard_profile=hard_profile, replay_buffer=replay_buffer)
        print(f"replay probe accuracy round {round_idx + 1}: {probe_acc:.4f}; replay buffer size={len(replay_buffer)}")
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
print("current primary supervision variant:", cfg.primary_supervision_variant)
print("initial family template priors:")
display(pd.DataFrame(sorted(BEST_TEMPLATE_BY_FAMILY.items()), columns=["prompt_family", "template_id"]))

if cfg.run_supervision_ablation:
    ablation_overall = []
    for variant in cfg.supervision_ablation_variants:
        overall_df, grouped = run_supervision_variant_experiment(variant)
        ablation_overall.append(overall_df)
        for key, value in grouped.items():
            value.to_csv(Path(cfg.work_dir) / f"supervision_{variant}_{key}.csv", index=False)
    ablation_overall_df = pd.concat(ablation_overall, ignore_index=True).sort_values(["value", "boxed_parse_rate"], ascending=False)
    display(ablation_overall_df)
    ablation_overall_df.to_csv(Path(cfg.work_dir) / "supervision_ablation_overall.csv", index=False)
else:
    print("skip full supervision ablation: set cfg.run_supervision_ablation=True to compare answer_only / short_reasoning / family_aware_mix side-by-side.")

template_score_df = pd.DataFrame()
family_template_map_df = pd.DataFrame(columns=["prompt_family", "best_template_id", "best_accuracy"])

# %% [markdown]
# ## Cell 19 — Stage 1 训练（短 / 规则明显 family 先收敛）

# %%
local_callback = LocalAccuracyCallback(
    eval_df=fast_eval_df,
    save_path=str(Path(cfg.work_dir) / "local_accuracy_history.csv"),
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
    trainer_stage1.train()
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
stage1_acc, stage1_pred_df, stage1_multi_seed_df = run_serious_eval_suite(
    model,
    serious_eval_df,
    serious_eval_views,
    prefix="stage1_valid",
)

stage1_train_probe_acc, stage1_train_probe_pred_df = evaluate_accuracy(
    model,
    train_replay_probe_df,
    extractor_fn=fast_extract_prediction,
)
print(f"stage1 train replay probe accuracy: {stage1_train_probe_acc:.4f}")
stage1_train_probe_pred_df.to_csv(Path(cfg.work_dir) / "stage1_train_probe_predictions.csv", index=False)
replay_buffer = update_replay_buffer({}, stage1_train_probe_pred_df)
print("bootstrapped replay buffer size:", len(replay_buffer))

if cfg.enable_prompt_template_ablation:
    template_eval_df = build_seeded_family_balanced_subset(
        valid_fold,
        cfg.template_ablation_after_stage1_rows,
        seed=cfg.seed,
    )
    template_score_df, family_template_map_df = run_template_ablation(model, template_eval_df, cfg.template_ablation_after_stage1_rows)
    candidate_template_map = dict(zip(family_template_map_df["prompt_family"], family_template_map_df["best_template_id"]))
    updated_template_map = dict(BEST_TEMPLATE_BY_FAMILY)
    for family, template_id in candidate_template_map.items():
        updated_template_map[family] = template_id
    BEST_TEMPLATE_BY_FAMILY.clear()
    BEST_TEMPLATE_BY_FAMILY.update(updated_template_map)
    display(template_score_df)
    display(family_template_map_df.head(20))
    template_score_df.to_csv(Path(cfg.work_dir) / "template_ablation_scores_stage1.csv", index=False)
    family_template_map_df.to_csv(Path(cfg.work_dir) / "family_template_map_stage1.csv", index=False)

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
stage2_weights = compute_sample_weights(stage2_records, hard_profile=hard_profile, replay_buffer=replay_buffer)
print("stage2 weight summary after hard profile + replay bootstrap:", np.quantile(stage2_weights, [0, 0.25, 0.5, 0.75, 1]))
# %% [markdown]
# ## Cell 21 — Stage 2 训练（多轮错题驱动重加权）

# %%
model, stage2_last_pred_df, replay_buffer = train_stage2_with_refresh(
    model=model,
    train_dataset=stage2_ds,
    eval_dataset=valid_ds,
    train_records_df=stage2_records,
    initial_weights=stage2_weights,
    replay_buffer=replay_buffer,
)

# %% [markdown]
# ## Cell 22 — 训练后本地近似评估 + 分组报表

# %%
post_acc, post_pred_df, post_multi_seed_df = run_serious_eval_suite(
    model,
    serious_eval_df,
    serious_eval_views,
    prefix="post_train_valid",
)
display(post_pred_df.head(10))

consensus_acc, consensus_df = evaluate_with_consensus(model, serious_eval_df.head(min(len(serious_eval_df), 128)), cfg.test_time_template_candidates)
print(f"post-train consensus accuracy: {consensus_acc:.4f}")
display(consensus_df.head(10))
consensus_df.to_csv(Path(cfg.work_dir) / "post_train_consensus_predictions.csv", index=False)
template_disagreement_df = summarize_template_disagreement(model, serious_eval_df, cfg.test_time_template_candidates, max_rows=128)
display(template_disagreement_df.head(20))
template_disagreement_df.to_csv(Path(cfg.work_dir) / "template_disagreement_by_family.csv", index=False)

# %% [markdown]
# ## Cell 23 — 保存 LoRA adapter，并检查 rank/配置是否合法
# 
# 比赛最终提交只需要兼容 Nemotron 基座的 LoRA adapter。
# 这里显式校验 `adapter_config.json`，避免训练完才发现不能交。

# %%
model.save_pretrained(cfg.adapter_dir)
tokenizer.save_pretrained(cfg.adapter_dir)

adapter_cfg_path = Path(cfg.adapter_dir) / "adapter_config.json"
assert adapter_cfg_path.exists(), "缺少 adapter_config.json，无法提交"
with open(adapter_cfg_path, "r", encoding="utf-8") as f:
    adapter_cfg = json.load(f)

assert int(adapter_cfg.get("r", cfg.lora_r)) <= cfg.max_lora_rank, "导出的 LoRA rank 超过比赛限制"
print(json.dumps(adapter_cfg, ensure_ascii=False, indent=2)[:1200])
print("adapter files:", sorted(p.name for p in Path(cfg.adapter_dir).iterdir()))

# %% [markdown]
# ## Cell 24 — 打包 submission.zip

# %%
zip_path = Path(cfg.submission_zip)
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(Path(cfg.adapter_dir).rglob("*")):
        if path.is_file():
            zf.write(path, arcname=path.relative_to(cfg.adapter_dir))

print("submission zip:", zip_path)
print("zip size (MB):", round(zip_path.stat().st_size / 1024 / 1024, 3))

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
    _template_id, prompt_text = choose_template(row.prompt, "numeric", infer_prompt_family(row.prompt))
    raw = generate_answer_text(model, prompt_text)
    pred = extract_prediction(raw)
    smoke_rows.append({"id": row.id, "pred": pred, "raw": raw[:1000]})
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
