from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _bootstrap_src() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256_upper_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().upper()


def _sha256_upper_text(text: str) -> str:
    return _sha256_upper_bytes(text.encode("utf-8"))


def _sha256_upper_json(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_upper_text(s)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    data = yaml.safe_load(_read_text(path))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping/dict.")
    return data


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _json_dumps_stable(obj: Any) -> str:
    def _default(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return sorted([str(x) for x in o])
        if isinstance(o, tuple):
            return list(o)
        return str(o)

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_default)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ln in _read_text(path).splitlines():
        s = ln.strip()
        if not s:
            continue
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("Each JSONL line must be an object.")
        items.append(obj)
    return items


def _resolve_path(project_root: Path, raw: str) -> Path:
    if os.path.isabs(raw):
        return Path(raw).resolve()
    p1 = (project_root / raw).resolve()
    if p1.exists():
        return p1
    p2 = (project_root.parent / raw).resolve()
    if p2.exists():
        return p2
    return p1


def _render_message(m: Mapping[str, Any]) -> str:
    role = str(m.get("role") or "")
    name = str(m.get("name") or "")
    if role == "assistant" and "tool_call" in m:
        return f"ASSISTANT({name}) TOOL_CALL: {_json_dumps_stable(m.get('tool_call'))}\n"
    if role == "tool":
        return f"TOOL({name}): {_json_dumps_stable(m.get('content'))}\n"
    if role == "user":
        return f"USER({name}): {_json_dumps_stable(m.get('content'))}\n"
    if role == "assistant":
        return f"ASSISTANT({name}): {str(m.get('content') or '')}\n"
    if role == "system":
        return f"SYSTEM: {str(m.get('content') or '')}\n"
    return f"{role.upper()}({name}): {_json_dumps_stable(dict(m))}\n"


def _render_with_supervision_spans(messages: Sequence[Mapping[str, Any]]) -> Tuple[str, List[Tuple[int, int]]]:
    chunks: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for m in messages:
        role = str(m.get("role") or "")
        s = _render_message(m)
        chunks.append(s)
        if role == "assistant":
            spans.append((cursor, cursor + len(s)))
        cursor += len(s)
    return "".join(chunks), spans


def _spans_intersect(a0: int, a1: int, b0: int, b1: int) -> bool:
    return (a0 < b1) and (b0 < a1)


def _mask_labels_by_spans(
    input_ids: List[int],
    offsets: Sequence[Tuple[int, int]],
    supervise_spans: Sequence[Tuple[int, int]],
    ignore_id: int = -100,
) -> List[int]:
    labels = [ignore_id] * len(input_ids)
    for i, (s0, s1) in enumerate(offsets):
        if s0 == 0 and s1 == 0:
            continue
        for (a0, a1) in supervise_spans:
            if _spans_intersect(s0, s1, a0, a1):
                labels[i] = input_ids[i]
                break
    return labels


def _mask_labels_fallback_last_assistant(
    tokenizer: Any,
    text: str,
    spans: Sequence[Tuple[int, int]],
    input_ids: List[int],
    ignore_id: int = -100,
    max_length: int = 2048,
) -> Tuple[List[int], bool]:
    if not spans:
        return list(input_ids), True

    a0, a1 = spans[-1]
    if not (0 <= a0 <= a1 <= len(text)):
        return list(input_ids), True

    try:
        enc_prefix = tokenizer(text[:a0], return_tensors=None, truncation=True, max_length=max_length)
        enc_upto_end = tokenizer(text[:a1], return_tensors=None, truncation=True, max_length=max_length)
        prefix_len = int(len(enc_prefix.get("input_ids") or []))
        end_len = int(len(enc_upto_end.get("input_ids") or []))
    except Exception:
        return list(input_ids), True

    start = min(max(prefix_len, 0), len(input_ids))
    end = min(max(end_len, 0), len(input_ids))
    if end <= start:
        return list(input_ids), True

    labels = [ignore_id] * len(input_ids)
    for i in range(start, end):
        labels[i] = input_ids[i]
    return labels, False


def _infer_target_modules(model: Any) -> List[str]:
    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "Wqkv",
        "wo",
        "wq",
        "wk",
        "wv",
        "c_attn",
        "c_proj",
    ]
    found: List[str] = []
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            if leaf not in found:
                found.append(leaf)
    if found:
        return found
    return ["q_proj", "v_proj"]


def _hash_dir_files(path: Path) -> str:
    parts: List[bytes] = []
    for p in sorted([x for x in path.rglob("*") if x.is_file()], key=lambda x: str(x).replace("\\", "/")):
        rel = str(p.relative_to(path)).replace("\\", "/")
        parts.append(rel.encode("utf-8") + b"\n")
        parts.append(p.read_bytes())
        parts.append(b"\n")
    return _sha256_upper_bytes(b"".join(parts))


@dataclass(frozen=True)
class TrainArgs:
    base_model: str
    train_jsonl: str
    seed: int
    max_steps: int
    lr: float
    batch_size: int
    grad_accum: int
    output_dir: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: Optional[List[str]]


def main() -> int:
    project_root = _bootstrap_src()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_modules", type=str, default=None)
    parser.add_argument("--allow_schema_mismatch", action="store_true")
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    cfg_path: Optional[Path] = None
    config_loaded = False
    if args.config:
        cfg_path = _resolve_path(project_root, args.config)
        cfg = _load_yaml(cfg_path)
        config_loaded = True

    def _pick(cli_val: Any, yaml_key: str, default: Any) -> Any:
        if cli_val is not None:
            return cli_val
        if isinstance(cfg, dict) and yaml_key in cfg and cfg.get(yaml_key) is not None:
            return cfg.get(yaml_key)
        return default

    seed_val = _pick(args.seed, "seed", None)
    max_steps_val = _pick(args.max_steps, "max_steps", None)
    lr_val = _pick(args.lr, "lr", None)
    batch_size_val = _pick(args.batch_size, "batch_size", None)
    grad_accum_val = _pick(args.grad_accum, "grad_accum", None)
    output_dir_val = _pick(args.output_dir, "output_dir", None)
    train_jsonl_val = _pick(args.train_jsonl, "train_jsonl", None)

    missing: List[str] = []
    if train_jsonl_val is None:
        missing.append("train_jsonl")
    if seed_val is None:
        missing.append("seed")
    if max_steps_val is None:
        missing.append("max_steps")
    if lr_val is None:
        missing.append("lr")
    if batch_size_val is None:
        missing.append("batch_size")
    if grad_accum_val is None:
        missing.append("grad_accum")
    if output_dir_val is None:
        missing.append("output_dir")
    if missing:
        print(f"config_loaded={bool(config_loaded)}", file=sys.stderr)
        print(f"missing_required={','.join(missing)}", file=sys.stderr)
        return 2

    base_model = str(args.base_model or cfg.get("base_model_id") or "sshleifer/tiny-gpt2")
    train_jsonl_path = _resolve_path(project_root, str(train_jsonl_val))
    out_dir = _resolve_path(project_root, str(output_dir_val))
    out_dir.mkdir(parents=True, exist_ok=True)

    target_modules: Optional[List[str]] = None
    tm = str(_pick(args.target_modules, "target_modules", "") or "").strip()
    if tm:
        target_modules = [x.strip() for x in tm.split(",") if x.strip()]

    lora_r_val = _pick(args.lora_r, "lora_r", 8)
    lora_alpha_val = _pick(args.lora_alpha, "lora_alpha", 16)
    lora_dropout_val = _pick(args.lora_dropout, "lora_dropout", 0.05)

    train_args = TrainArgs(
        base_model=base_model,
        train_jsonl=str(train_jsonl_path),
        seed=int(seed_val),
        max_steps=int(max_steps_val),
        lr=float(lr_val),
        batch_size=int(batch_size_val),
        grad_accum=int(grad_accum_val),
        output_dir=str(out_dir),
        lora_r=int(lora_r_val),
        lora_alpha=int(lora_alpha_val),
        lora_dropout=float(lora_dropout_val),
        target_modules=target_modules,
    )

    config_obj: Dict[str, Any] = {
        "script": "dist/scripts/09_train_lora_sft_toy.py",
        "config_path": str(cfg_path) if cfg_path else "",
        "train_args": json.loads(_json_dumps_stable(train_args.__dict__)),
        "raw_config": cfg,
    }
    config_hash = _sha256_upper_json(config_obj)
    data_bytes = train_jsonl_path.read_bytes()
    data_hash = _sha256_upper_bytes(data_bytes)

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        print(
            "Missing dependencies for LoRA SFT.\n"
            "Install (CPU):\n"
            "  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  python -m pip install --upgrade transformers peft accelerate\n",
            file=sys.stderr,
        )
        return 2

    _set_global_seed(int(train_args.seed))

    tokenizer = AutoTokenizer.from_pretrained(train_args.base_model, use_fast=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(train_args.base_model)
    model.to(device)
    model.train()

    if target_modules is None:
        target_modules = _infer_target_modules(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(train_args.lora_r),
        lora_alpha=int(train_args.lora_alpha),
        lora_dropout=float(train_args.lora_dropout),
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    items = _read_jsonl(train_jsonl_path)
    if not items:
        print("Empty train_jsonl.", file=sys.stderr)
        return 1
    schema_bad_count = 0
    for it in items:
        if str(it.get("schema_version") or "") != "sft_trajectory_v1":
            schema_bad_count += 1
    schema_ok = schema_bad_count == 0
    print(f"config_loaded={bool(config_loaded)}")
    print(f"schema_ok={bool(schema_ok)}")
    print(f"schema_bad_count={int(schema_bad_count)}")
    if (not schema_ok) and (not bool(args.allow_schema_mismatch)):
        return 1

    examples: List[Dict[str, Any]] = []
    skipped_no_messages = 0
    fallback_count = 0
    offset_mapping_printed = False
    warned_supervise_all = False
    for it in items:
        msgs = it.get("messages")
        if not isinstance(msgs, list) or not msgs:
            skipped_no_messages += 1
            continue
        text, spans = _render_with_supervision_spans([m for m in msgs if isinstance(m, dict)])
        try:
            enc = tokenizer(
                text,
                return_tensors=None,
                return_offsets_mapping=True,
                truncation=True,
                max_length=2048,
            )
        except Exception:
            enc = tokenizer(
                text,
                return_tensors=None,
                truncation=True,
                max_length=2048,
            )

        input_ids = list(enc["input_ids"])
        offsets_any = enc.get("offset_mapping") if hasattr(enc, "get") else None
        offsets = offsets_any if isinstance(offsets_any, list) else []

        offset_ok = bool(offsets)
        if not offset_mapping_printed:
            print(f"offset_mapping_available={bool(offset_ok)}")
            offset_mapping_printed = True

        if offset_ok:
            labels = _mask_labels_by_spans(input_ids, offsets, spans)
        else:
            fallback_count += 1
            labels, supervised_all = _mask_labels_fallback_last_assistant(tokenizer, text, spans, input_ids)
            if supervised_all and (not warned_supervise_all):
                print("warning=offset_mapping_missing_supervise_all")
                warned_supervise_all = True
        attn_any = enc.get("attention_mask") if hasattr(enc, "get") else None
        examples.append({"input_ids": input_ids, "labels": labels, "attention_mask": list(attn_any or [1] * len(input_ids))})

    print(f"fallback_count={int(fallback_count)}")

    if not examples:
        print("No usable examples after preprocessing.", file=sys.stderr)
        return 1

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids_t = torch.full((len(batch), max_len), fill_value=int(tokenizer.pad_token_id), dtype=torch.long)
        labels_t = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
        attn_t = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            input_ids_t[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
            labels_t[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
            attn_t[i, :n] = torch.tensor(ex["attention_mask"], dtype=torch.long)
        return {"input_ids": input_ids_t.to(device), "labels": labels_t.to(device), "attention_mask": attn_t.to(device)}

    optim = torch.optim.AdamW(model.parameters(), lr=float(train_args.lr))

    def _lr_at(step: int) -> float:
        if train_args.max_steps <= 0:
            return float(train_args.lr)
        t = min(max(step, 0), train_args.max_steps)
        frac = 1.0 - (float(t) / float(train_args.max_steps))
        return float(train_args.lr) * max(frac, 0.0)

    step = 0
    micro = 0
    running_loss = 0.0
    rng = random.Random(int(train_args.seed))

    while step < int(train_args.max_steps):
        batch = [examples[rng.randrange(0, len(examples))] for _ in range(int(train_args.batch_size))]
        batch_t = _collate(batch)
        out = model(**batch_t)
        loss = out.loss
        (loss / float(train_args.grad_accum)).backward()
        running_loss += float(loss.detach().cpu().item())
        micro += 1

        if micro % int(train_args.grad_accum) == 0:
            lr_eff = _lr_at(step)
            for pg in optim.param_groups:
                pg["lr"] = lr_eff
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 1 == 0:
                print(f"step={step} loss={running_loss/float(micro):.6f} lr={lr_eff:.6g}")

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    adapter_hash = _hash_dir_files(adapter_dir)

    snapshot = {
        "timestamp_utc": None,
        "seed": int(train_args.seed),
        "device": str(device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "base_model": str(train_args.base_model),
        "config_hash": str(config_hash),
        "data_hash": str(data_hash),
        "adapter_hash": str(adapter_hash),
        "train_args": json.loads(_json_dumps_stable(train_args.__dict__)),
        "lora_config": json.loads(_json_dumps_stable(lora_cfg.to_dict())),
        "n_items": int(len(items)),
        "n_examples": int(len(examples)),
        "skipped_no_messages": int(skipped_no_messages),
    }
    try:
        from agentiad_repro.utils import utc_now_iso

        snapshot["timestamp_utc"] = utc_now_iso()
    except Exception:
        snapshot["timestamp_utc"] = None
    (out_dir / "train_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"output_dir={str(out_dir)}")
    print(f"adapter_dir={str(adapter_dir)}")
    print(f"seed={int(train_args.seed)}")
    print(f"config_hash={config_hash}")
    print(f"data_hash={data_hash}")
    print(f"adapter_hash={adapter_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
