from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from dsta_rag.prompts import STAGE1_SYSTEM_PROMPT
from dsta_rag.protocol import parse_protocol, validate_protocol
from dsta_rag.utils import dump_json, read_jsonl, token_f1, write_jsonl


def _lazy_imports():
    try:
        from peft import PeftModel
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Stage 1 evaluation requires optional dependencies. Install with: pip install -e .[stage1]"
        ) from exc
    return PeftModel, AutoModelForCausalLM, AutoTokenizer, torch


def _load_model(model_path: str, base_model: Optional[str]):
    PeftModel, AutoModelForCausalLM, AutoTokenizer, torch = _lazy_imports()
    model_kwargs = {"trust_remote_code": True, "torch_dtype": "auto"}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model_dir = Path(model_path)
    adapter_config_path = model_dir / "adapter_config.json"

    if adapter_config_path.exists():
        with adapter_config_path.open("r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_name = base_model or adapter_cfg.get("base_model_name_or_path")
        if not base_name:
            raise SystemExit("Detected LoRA adapter checkpoint, but base model is unknown. Pass --base-model.")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return tokenizer, model


def _build_prompt(tokenizer: Any, question: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"[SYSTEM]\n{system_prompt}\n[USER]\n{question}\n[ASSISTANT]\n"


def _normalize_example(row: Dict[str, Any]) -> Dict[str, str]:
    if "question" in row:
        return {
            "qid": str(row.get("qid", "")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
        }
    if "messages" in row:
        question = ""
        for msg in row["messages"]:
            if msg.get("role") == "user":
                question = msg.get("content", "")
                break
        metadata = row.get("metadata", {})
        return {
            "qid": str(metadata.get("qid", row.get("id", ""))),
            "question": str(question),
            "answer": str(metadata.get("answer", "")),
        }
    raise ValueError("Unsupported input row format: expected `question` or `messages`.")


def main() -> None:
    _, _, _, torch = _lazy_imports()
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 checkpoint instruction-following behavior.")
    parser.add_argument("--model-path", required=True, help="Model checkpoint path (full model or LoRA adapter dir).")
    parser.add_argument("--base-model", default=None, help="Required only when --model-path is LoRA adapter without base info.")
    parser.add_argument("--input", required=True, help="JSONL with `question/answer` rows or stage1 SFT rows.")
    parser.add_argument("--output", default=None, help="Optional JSONL to save per-example generations and metrics.")
    parser.add_argument("--summary", default=None, help="Optional JSON file for aggregate metrics.")
    parser.add_argument("--system-prompt", default=STAGE1_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer, model = _load_model(args.model_path, args.base_model)

    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(args.input)):
        if args.limit is not None and idx >= args.limit:
            break
        rows.append(_normalize_example(row))

    if not rows:
        raise SystemExit("No rows found for Stage-1 evaluation.")

    console = Console()
    console.print(f"Loaded model from [bold]{args.model_path}[/bold]. Evaluating [bold]{len(rows)}[/bold] examples...")

    results: List[Dict[str, Any]] = []
    with console.status("Running generation..."):
        for row in rows:
            prompt_text = _build_prompt(tokenizer, row["question"], args.system_prompt)
            model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.temperature <= 0:
                generate_kwargs["do_sample"] = False
            else:
                generate_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})

            with torch.inference_mode():
                output = model.generate(**model_inputs, **generate_kwargs)
            generated_ids = output[0][model_inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            parsed = parse_protocol(response)
            errors = validate_protocol(parsed)
            pred_answer = parsed.get("answer") or ""
            has_answer = bool(pred_answer)
            has_search = bool(parsed.get("query"))

            metrics = {
                "valid": int(len(errors) == 0),
                "has_assess": int(parsed.get("assess_label") is not None),
                "has_refine": int(len(parsed.get("refine_items") or []) > 0),
                "has_rectify": int(bool(parsed.get("next_search_target") or parsed.get("missing_slots") or parsed.get("why_insufficient"))),
                "has_think": int(bool(parsed.get("think"))),
                "has_terminal_action": int(has_answer or has_search),
                "answer_em": int(has_answer and pred_answer.strip().lower() == row["answer"].strip().lower()),
                "answer_f1": token_f1(pred_answer, row["answer"]) if has_answer else 0.0,
            }

            results.append(
                {
                    "qid": row["qid"],
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "response": response,
                    "parsed": parsed,
                    "errors": errors,
                    "metrics": metrics,
                }
            )

    summary = {
        "count": len(results),
        "valid_rate": mean(r["metrics"]["valid"] for r in results),
        "assess_rate": mean(r["metrics"]["has_assess"] for r in results),
        "refine_rate": mean(r["metrics"]["has_refine"] for r in results),
        "rectify_rate": mean(r["metrics"]["has_rectify"] for r in results),
        "think_rate": mean(r["metrics"]["has_think"] for r in results),
        "terminal_action_rate": mean(r["metrics"]["has_terminal_action"] for r in results),
        "answer_em": mean(r["metrics"]["answer_em"] for r in results),
        "answer_f1": mean(r["metrics"]["answer_f1"] for r in results),
    }

    table = Table(title="Stage-1 Instruction Following Evaluation")
    table.add_column("metric")
    table.add_column("value")
    for key, value in summary.items():
        if key == "count":
            table.add_row(key, str(value))
        else:
            table.add_row(key, f"{value:.4f}")
    console.print(table)

    if args.output:
        write_jsonl(results, args.output)
    if args.summary:
        dump_json(summary, args.summary)


if __name__ == "__main__":
    main()
