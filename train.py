# =========================
# Cell 1
# =========================
# 这个 Python 3 环境已经预装了很多有用的数据分析库
# 它基于 kaggle/python Docker 镜像构建：https://github.com/kaggle/docker-python
# 例如，这里加载了一些常用的库

import numpy as np  # 线性代数
import pandas as pd  # 数据处理、CSV 文件读取（例如 pd.read_csv）

# 输入数据文件位于只读目录 "../input/" 中
# 例如，运行下面这段代码（点击运行或按 Shift+Enter）可以列出 input 目录下的所有文件

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 你可以向当前目录 (/kaggle/working/) 写入最多 20GB 数据，
# 当你点击 "Save & Run All" 时，这些数据会被保存为输出
# 你也可以写入临时文件到 /kaggle/temp/，但这些文件不会在会话结束后保存


# =========================
# Cell 2
# =========================
# 安装本 notebook 需要的库
!pip install -q transformers datasets peft accelerate trl


# =========================
# Cell 3
# =========================
# 导入训练、LoRA、数据处理需要的库
import os
import re
import gc
import json
import math
import time
import random
import shutil
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)


# =========================
# Cell 4
# =========================
# 全局配置
SEED = 42

# 这里一定要替换成比赛官方 demo 里实际使用的 Nemotron 基座模型名
BASE_MODEL = "REPLACE_WITH_OFFICIAL_NEMOTRON_BASE_MODEL"

MAX_LENGTH = 768
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 1
LORA_RANK = 32

OUTPUT_DIR = "/kaggle/working/nemotron_lora_output"
ADAPTER_DIR = "/kaggle/working/nemotron_lora_adapter"
ZIP_BASE = "/kaggle/working/submission"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)


# =========================
# Cell 5
# =========================
# 固定随机种子，保证结果更稳定
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================
# Cell 6
# =========================
# 检查当前是否启用了 GPU
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU detected")


# =========================
# Cell 7
# =========================
# 读取比赛训练集与测试集
train = pd.read_csv("/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv")
test = pd.read_csv("/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/test.csv")

print(train.shape, test.shape)
display(train.head())
display(test.head())


# =========================
# Cell 8
# =========================
# 划分训练集 / 验证集
train_df, valid_df = train_test_split(
    train,
    test_size=0.1,
    random_state=SEED
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print(train_df.shape, valid_df.shape)


# =========================
# Cell 9
# =========================
# 构造答案标准化函数、boxed 包装函数、以及统一的 prompt 模板
def canonicalize_answer(ans):
    ans = str(ans).strip()

    # 如果是纯数字，尽量标准化
    try:
        val = float(ans)
        if val.is_integer():
            ans = str(int(val))
        else:
            ans = str(val)
    except:
        pass

    return ans

def boxed_answer(ans):
    ans = canonicalize_answer(ans)
    if ans.startswith("\\boxed{") and ans.endswith("}"):
        return ans
    return f"\\boxed{{{ans}}}"

def build_prompt(problem):
    return f"""You are a precise reasoning assistant.

Solve the following task carefully.
Output only the final answer wrapped in \\boxed{{}}.

Task:
{problem}

Answer:
"""


# =========================
# Cell 10
# =========================
# 将原始样本转换成训练所需格式：
# prompt_text: 只包含问题模板
# answer_text: 只包含 boxed 后的答案
# full_text: prompt + answer
def format_example(row):
    prompt_text = build_prompt(row["prompt"])
    answer_text = boxed_answer(row["answer"])
    full_text = prompt_text + answer_text

    return {
        "prompt_text": prompt_text,
        "answer_text": answer_text,
        "full_text": full_text,
        "raw_answer": canonicalize_answer(row["answer"])
    }

train_fmt = train_df.apply(format_example, axis=1, result_type="expand")
valid_fmt = valid_df.apply(format_example, axis=1, result_type="expand")

display(train_fmt.head())
print(train_fmt["full_text"].iloc[0])


# =========================
# Cell 11
# =========================
# 转成 Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_fmt)
valid_dataset = Dataset.from_pandas(valid_fmt)

print(train_dataset[0])


# =========================
# Cell 12
# =========================
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("pad_token:", tokenizer.pad_token)
print("eos_token:", tokenizer.eos_token)


# =========================
# Cell 13
# =========================
# 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)


# =========================
# Cell 14
# =========================
# 可选检查：看目标模块名字里是否包含 q_proj / k_proj / v_proj / o_proj
sample_names = []
for name, module in model.named_modules():
    if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        sample_names.append(name)

print("matched module count:", len(sample_names))
print(sample_names[:30])


# =========================
# Cell 15
# =========================
# 配置 LoRA，并将 LoRA 挂到模型上
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =========================
# Cell 16
# =========================
# 分词函数：
# 1. prompt_text 单独 tokenize，得到 prompt 长度
# 2. full_text tokenize，作为完整输入
# 3. 只让 answer 部分参与 loss，prompt 部分 labels 设为 -100
# 4. padding 部分 labels 也设为 -100
def tokenize_function(example):
    prompt_enc = tokenizer(
        example["prompt_text"],
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False
    )

    full_enc = tokenizer(
        example["full_text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        add_special_tokens=False
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    prompt_len = len(prompt_enc["input_ids"])
    labels = input_ids.copy()

    # prompt 部分不参与 loss
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    # padding 部分不参与 loss
    labels = [
        token_id if mask == 1 else -100
        for token_id, mask in zip(labels, attention_mask)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# =========================
# Cell 17
# =========================
# 对训练集和验证集进行 tokenize
tokenized_train = train_dataset.map(tokenize_function, batched=False)
tokenized_valid = valid_dataset.map(tokenize_function, batched=False)

keep_cols = [
    "input_ids",
    "attention_mask",
    "labels",
    "raw_answer",
    "prompt_text",
    "answer_text",
    "full_text"
]

tokenized_train = tokenized_train.remove_columns(
    [c for c in tokenized_train.column_names if c not in keep_cols]
)

tokenized_valid = tokenized_valid.remove_columns(
    [c for c in tokenized_valid.column_names if c not in keep_cols]
)

print(tokenized_train[0].keys())


# =========================
# Cell 18
# =========================
# 检查一个样本的 prompt / answer / full_text / labels 是否合理
sample = tokenized_train[0]

print("prompt_text:")
print(train_dataset[0]["prompt_text"])
print("\nanswer_text:")
print(train_dataset[0]["answer_text"])
print("\nfull_text:")
print(train_dataset[0]["full_text"])

valid_label_count = sum(1 for x in sample["labels"] if x != -100)
print("\nnon-masked label count:", valid_label_count)


# =========================
# Cell 19
# =========================
# 本地评估用的答案提取与比较函数
# 优先提取 \boxed{} 中内容；否则回退到最后一行
def extract_boxed_answer(text):
    text = str(text)
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return None

def try_parse_float(x):
    try:
        return float(str(x).replace(",", "").strip())
    except:
        return None

def normalize_extracted_answer(text):
    text = str(text).strip()

    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return canonicalize_answer(boxed)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return canonicalize_answer(lines[-1])

    return canonicalize_answer(text)

def approx_equal(pred, gold, rel_tol=1e-4):
    pred = canonicalize_answer(pred)
    gold = canonicalize_answer(gold)

    if pred == gold:
        return True

    p = try_parse_float(pred)
    g = try_parse_float(gold)

    if p is not None and g is not None:
        if g == 0:
            return abs(p - g) <= rel_tol
        return abs(p - g) / max(abs(g), 1e-12) <= rel_tol

    return False


# =========================
# Cell 20
# =========================
# 生成原始输出，并在验证子集上做本地近似评估
@torch.no_grad()
def generate_raw_output(model, tokenizer, prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

@torch.no_grad()
def evaluate_subset(model, tokenizer, df, n=20):
    n = min(n, len(df))
    sub = df.iloc[:n].copy()

    rows = []

    for _, row in sub.iterrows():
        prompt = build_prompt(row["prompt"])
        raw_output = generate_raw_output(model, tokenizer, prompt)
        pred = normalize_extracted_answer(raw_output)
        gold = canonicalize_answer(row["answer"])
        hit = approx_equal(pred, gold)

        rows.append({
            "prompt": row["prompt"],
            "gold": gold,
            "pred": pred,
            "hit": hit,
            "raw_output": raw_output
        })

    result_df = pd.DataFrame(rows)
    acc = result_df["hit"].mean() if len(result_df) > 0 else 0.0
    return acc, result_df


# =========================
# Cell 21
# =========================
# 训练前先看一下模型在验证子集上的大致表现
pre_acc, pre_df = evaluate_subset(model, tokenizer, valid_df, n=10)
print("pre-train approx acc:", pre_acc)
display(pre_df.head())


# =========================
# Cell 22
# =========================
# 配置 Trainer 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=torch.cuda.is_available()
)


# =========================
# Cell 23
# =========================
# 给 Trainer 的数据集里只保留训练必需字段
train_for_trainer = tokenized_train.remove_columns(
    [c for c in ["raw_answer", "prompt_text", "answer_text", "full_text"] if c in tokenized_train.column_names]
)

valid_for_trainer = tokenized_valid.remove_columns(
    [c for c in ["raw_answer", "prompt_text", "answer_text", "full_text"] if c in tokenized_valid.column_names]
)

print(train_for_trainer[0].keys())


# =========================
# Cell 24
# =========================
# 构造 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_for_trainer,
    eval_dataset=valid_for_trainer,
    data_collator=default_data_collator
)


# =========================
# Cell 25
# =========================
# 开始训练
trainer.train()


# =========================
# Cell 26
# =========================
# 训练后再次在验证子集上做本地近似评估
post_acc, post_df = evaluate_subset(model, tokenizer, valid_df, n=20)
print("post-train approx acc:", post_acc)
display(post_df.head(10))


# =========================
# Cell 27
# =========================
# 保存 LoRA adapter 和 tokenizer
os.makedirs(ADAPTER_DIR, exist_ok=True)

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

print("Saved files:")
print(os.listdir(ADAPTER_DIR))


# =========================
# Cell 28
# =========================
# 将 adapter 目录打包成 zip
if os.path.exists(ZIP_BASE + ".zip"):
    os.remove(ZIP_BASE + ".zip")

shutil.make_archive(ZIP_BASE, "zip", ADAPTER_DIR)
print("Created zip:", ZIP_BASE + ".zip")


# =========================
# Cell 29
# =========================
# 检查 zip 是否存在以及大小
print(os.path.exists(ZIP_BASE + ".zip"))
print(os.path.getsize(ZIP_BASE + ".zip") if os.path.exists(ZIP_BASE + ".zip") else "zip not found")


# =========================
# Cell 30
# =========================
# 仅用于自己检查几条测试集推理输出，不是最终提交格式
sample_prompts = test["prompt"].iloc[:5].tolist()

for i, p in enumerate(sample_prompts):
    raw = generate_raw_output(model, tokenizer, build_prompt(p))
    pred = normalize_extracted_answer(raw)

    print("=" * 100)
    print("sample:", i)
    print("pred:", pred)
    print("raw:")
    print(raw[:1500])


# =========================
# Cell 31
# =========================
# 可选：保存本地验证结果，便于后面分析
post_df.to_csv("/kaggle/working/local_valid_eval.csv", index=False)
print("saved:", "/kaggle/working/local_valid_eval.csv")


# =========================
# Cell 32
# =========================
# 可选：释放内存 / 显存
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()