from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _lazy_imports():
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "Stage 1 training requires optional dependencies. Install with: pip install -e .[stage1]"
        ) from exc
    return load_dataset, LoraConfig, SFTConfig, SFTTrainer, AutoTokenizer, AutoModelForCausalLM


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage-1 DSTA SFT model with HF/TRL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--eval-file", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    load_dataset, LoraConfig, SFTConfig, SFTTrainer, AutoTokenizer, AutoModelForCausalLM = _lazy_imports()

    train_file = args.train_file or cfg["train_file"]
    eval_file = args.eval_file
    data_files = {"train": train_file}
    if eval_file:
        data_files["validation"] = eval_file
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        torch_dtype="auto",
    )

    peft_config = None
    if cfg.get("use_lora", False):
        lora = cfg.get("lora", {})
        peft_config = LoraConfig(
            r=lora.get("r", 16),
            lora_alpha=lora.get("alpha", 32),
            lora_dropout=lora.get("dropout", 0.05),
            target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            task_type="CAUSAL_LM",
        )

    sft_args = SFTConfig(
        output_dir=cfg["output_dir"],
        learning_rate=cfg.get("learning_rate", 2e-5),
        num_train_epochs=cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        bf16=cfg.get("bf16", True),
        max_seq_length=cfg.get("max_seq_length", 4096),
        chat_template_path=cfg.get("chat_template_path"),
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
