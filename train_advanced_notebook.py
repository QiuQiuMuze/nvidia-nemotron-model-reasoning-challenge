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
    approx_eval_examples: int = 96
    numeric_rel_tol: float = 1e-4

    # ===== curriculum =====
    stage1_epochs: float = 0.6
    stage2_epochs: float = 1.2
    stage1_max_prompt_chars: int = 900
    stage1_lr: float = 1.6e-4
    stage2_lr: float = 9e-5

    # ===== optional leaderboard tricks =====
    enable_external_mixture: bool = False
    enable_prompt_template_ablation: bool = True
    enable_family_reweight: bool = True
    enable_length_bucket_bonus: bool = True

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

external_df = load_optional_external_data(cfg.extra_data_dir) if cfg.enable_external_mixture else pd.DataFrame(columns=["id", "prompt", "answer", "source"])
full_train_df = pd.concat([train_df, external_df], ignore_index=True)

print("official train:", train_df.shape)
print("external train:", external_df.shape)
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
    return re.sub(r"[^a-z0-9]+", "_", prompt_low[:80]).strip("_") or "unknown"


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
train_idx, valid_idx = next(splitter.split(full_train_df, groups=full_train_df["prompt_family"]))

train_fold = full_train_df.iloc[train_idx].reset_index(drop=True)
valid_fold = full_train_df.iloc[valid_idx].reset_index(drop=True)

print("train fold:", train_fold.shape)
print("valid fold:", valid_fold.shape)
print("train families:", train_fold['prompt_family'].nunique())
print("valid families:", valid_fold['prompt_family'].nunique())

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


def extract_prediction(text: str) -> str:
    text = str(text).strip()
    boxed_hits = BOXED_RE.findall(text)
    if boxed_hits:
        return canonicalize_answer(boxed_hits[-1])

    last_num_hits = LAST_NUMBER_RE.findall(text.replace(",", ""))
    if last_num_hits:
        return canonicalize_answer(last_num_hits[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return canonicalize_answer(lines[-1] if lines else text)


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


def choose_template(problem: str, answer_type: str) -> str:
    if not cfg.enable_prompt_template_ablation:
        return template_compact(problem)
    if answer_type in {"numeric", "binary"}:
        return template_compact(problem)
    return template_reasoning(problem)

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
    prompt_text = choose_template(row.prompt, row.answer_type)
    answer_text = boxed(row.answer)
    full_text = prompt_text + answer_text

    difficulty = 1.0
    difficulty += min(row.prompt_chars / 1800.0, 1.5)
    difficulty += 0.2 if row.answer_type == "multi_token_text" else 0.0
    difficulty += 0.15 if row.example_len_bucket in {"l", "xl"} else 0.0

    return {
        "id": row.id,
        "prompt_family": row.prompt_family,
        "answer_type": row.answer_type,
        "prompt_text": prompt_text,
        "answer_text": answer_text,
        "full_text": full_text,
        "gold_answer": canonicalize_answer(row.answer),
        "source": row.source,
        "difficulty": difficulty,
        "prompt_chars": row.prompt_chars,
        "len_bucket": str(row.example_len_bucket),
    }

train_records = train_fold.apply(build_training_record, axis=1, result_type="expand")
valid_records = valid_fold.apply(build_training_record, axis=1, result_type="expand")

print(train_records.shape, valid_records.shape)
display(train_records.head(2))

# %% [markdown]
# ## Cell 10 — Stage 1 / Stage 2 curriculum 切分
# 
# 两阶段训练比“一把梭全量训练”更稳定：
# - Stage 1 先学短 prompt、规则明显的题；
# - Stage 2 再纳入长文本和复杂 family。

# %%
stage1_records = train_records.query("prompt_chars <= @cfg.stage1_max_prompt_chars").reset_index(drop=True)
stage2_records = train_records.reset_index(drop=True)

print("stage1 records:", stage1_records.shape)
print("stage2 records:", stage2_records.shape)

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
    prompt_enc = tokenizer(
        example["prompt_text"],
        add_special_tokens=False,
        truncation=True,
        max_length=cfg.max_seq_len,
    )
    full_enc = tokenizer(
        example["full_text"],
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
        "len_bucket": example["len_bucket"],
    }

stage1_ds = Dataset.from_pandas(stage1_records).map(tokenize_answer_only)
stage2_ds = Dataset.from_pandas(stage2_records).map(tokenize_answer_only)
valid_ds = Dataset.from_pandas(valid_records).map(tokenize_answer_only)

remove_cols = [
    c
    for c in stage2_ds.column_names
    if c not in {"input_ids", "attention_mask", "labels", "gold_answer", "prompt_text", "difficulty", "prompt_family", "len_bucket"}
]
stage1_ds = stage1_ds.remove_columns(remove_cols)
stage2_ds = stage2_ds.remove_columns(remove_cols)
valid_ds = valid_ds.remove_columns(remove_cols)

print(stage2_ds[0].keys())
print("non-masked labels:", sum(x != -100 for x in stage2_ds[0]["labels"]))

# %% [markdown]
# ## Cell 13 — 加载 4-bit 基座模型 + 自动发现 LoRA target modules
# 
# 高级版不把 target modules 写死在 `q_proj/k_proj/v_proj/o_proj`：
# Nemotron / Llama 系模型中，MLP 的 `gate/up/down_proj` 往往也很重要。
# 在 rank 固定 <= 32 的前提下，覆盖 attention + MLP 往往比只训 attention 更强。

# %%
quant_config = None
if cfg.use_4bit:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    cfg.base_model,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()


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

lora_targets = discover_lora_targets(model.named_modules())
print("LoRA targets:", lora_targets)

lora_config = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=lora_targets,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% [markdown]
# ## Cell 14 — 自定义 sampler：family reweight + length bonus + difficulty curriculum
# 
# 这一步是真正的“高级训练味道”：
# - 稀有 family 给予更高采样权重，避免被头部模板淹没；
# - 中长样本给予适度 bonus，提高真实测试稳健性；
# - difficulty 进入 curriculum 权重，而不是只做硬过滤。

# %%

def compute_sample_weights(df_like: pd.DataFrame) -> np.ndarray:
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

valid_eval_df = stratified_valid_subset(valid_fold, cfg.approx_eval_examples)
print("approx eval subset:", valid_eval_df.shape)


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
def evaluate_accuracy(model: torch.nn.Module, df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    rows = []
    for row in df.itertuples(index=False):
        prompt = choose_template(row.prompt, infer_answer_type(row.answer))
        raw = generate_answer_text(model, prompt)
        pred = extract_prediction(raw)
        gold = canonicalize_answer(row.answer)
        rows.append(
            {
                "id": row.id,
                "prompt_family": infer_prompt_family(row.prompt),
                "gold": gold,
                "pred": pred,
                "hit": approx_equal(pred, gold),
                "raw": raw[:1200],
            }
        )
    out = pd.DataFrame(rows)
    acc = float(out["hit"].mean()) if len(out) else 0.0
    return acc, out


class LocalAccuracyCallback(TrainerCallback):
    def __init__(self, eval_df: pd.DataFrame, save_path: str):
        self.eval_df = eval_df
        self.save_path = save_path
        self.history: List[Dict[str, Any]] = []

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        acc, pred_df = evaluate_accuracy(model, self.eval_df)
        record = {"step": int(state.global_step), "local_accuracy": acc}
        self.history.append(record)
        pd.DataFrame(self.history).to_csv(self.save_path, index=False)
        pred_df.to_csv(Path(self.save_path).with_suffix(".preds.csv"), index=False)
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

# %% [markdown]
# ## Cell 18 — Stage 1 训练（短 / 规则明显样本先收敛）

# %%
local_callback = LocalAccuracyCallback(
    eval_df=valid_eval_df,
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
# ## Cell 19 — Stage 2 训练（全量混合 + 稀有 family 重加权）

# %%
trainer_stage2 = WeightedTrainer(
    model=model,
    args=build_training_args("stage2", cfg.stage2_lr, cfg.stage2_epochs),
    train_dataset=stage2_ds.remove_columns([c for c in stage2_ds.column_names if c not in train_columns]),
    eval_dataset=valid_ds.remove_columns([c for c in valid_ds.column_names if c not in train_columns]),
    data_collator=collator,
    sample_weights=stage2_weights,
    callbacks=[local_callback],
)
trainer_stage2.train()

# %% [markdown]
# ## Cell 20 — 训练后本地近似评估

# %%
post_acc, post_pred_df = evaluate_accuracy(model, valid_eval_df)
print(f"post-train approx accuracy: {post_acc:.4f}")
display(post_pred_df.head(10))
post_pred_df.to_csv(Path(cfg.work_dir) / "post_train_valid_predictions.csv", index=False)

# %% [markdown]
# ## Cell 21 — 保存 LoRA adapter，并检查 rank/配置是否合法
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
# ## Cell 22 — 打包 submission.zip

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
# ## Cell 23 — 提交前 smoke test
# 
# 这一步的目标不是拿高分，而是确认：
# - 模型能正常加载 adapter；
# - 生成格式稳定输出 boxed；
# - 不会出现空输出或格式崩坏。

# %%
smoke_df = test_df.head(5).copy()
smoke_rows = []
for row in smoke_df.itertuples(index=False):
    prompt_text = choose_template(row.prompt, "numeric")
    raw = generate_answer_text(model, prompt_text)
    pred = extract_prediction(raw)
    smoke_rows.append({"id": row.id, "pred": pred, "raw": raw[:1000]})
smoke_pred_df = pd.DataFrame(smoke_rows)
display(smoke_pred_df)
smoke_pred_df.to_csv(Path(cfg.work_dir) / "smoke_predictions.csv", index=False)

# %% [markdown]
# ## Cell 24 — 冲奖建议：你接下来最值得继续做的 6 件事
# 
# 1. **外挂高质量 synthetic data**：对每个 prompt family 做 teacher 生成 / programmatic augmentation。
# 2. **做 family-level ablation**：分家族统计本地 acc，针对弱项单独补数据。
# 3. **引入 rejection sampling**：teacher 多采样，只保留 exact-match 或可验证正确的样本。
# 4. **训练后做 prompt ensemble**：同一个 adapter 用 2~4 个模板离线 A/B，选最优模板提交。
# 5. **局部 RL / DPO**：只对 hardest families 做偏好优化，避免全局噪声放大。
# 6. **多次 seed + CV bagging**：不同 family split 训练多个 rank-32 adapter，再选最佳单模提交。

# %% [markdown]
# ## Cell 25 — 释放显存 / 导出配置快照

# %%
with open(Path(cfg.work_dir) / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("done")
