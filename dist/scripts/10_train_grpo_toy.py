from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


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


def _posix_rel_or_name(project_root: Path, raw: str) -> str:
    try:
        p = Path(raw)
    except Exception:
        return str(raw)
    if not p.is_absolute():
        return str(p).replace("\\", "/").lstrip("./")
    try:
        rel = p.resolve().relative_to(project_root.resolve())
        return str(rel).replace("\\", "/")
    except Exception:
        return str(p.name)


def _canonicalize_paths_for_hash(project_root: Path, obj: Any) -> Any:
    path_keys = {
        "config_path",
        "train_jsonl",
        "output_jsonl",
        "output_dir",
        "cache_dir",
        "adapter_init",
        "rollout_output_jsonl",
    }
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in path_keys and isinstance(v, str):
                out[k] = _posix_rel_or_name(project_root, v)
            else:
                out[k] = _canonicalize_paths_for_hash(project_root, v)
        return out
    if isinstance(obj, list):
        return [_canonicalize_paths_for_hash(project_root, x) for x in obj]
    return obj


def _extract_first_json(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
            if depth < 0:
                return None
    return None


def _extract_json_after_prefix(text: str, prefix: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    i = text.find(prefix)
    if i < 0:
        return None
    tail = text[i + len(prefix) :]
    return _extract_first_json(tail)


def _extract_json_after_final_marker(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    marker = "ASSISTANT(final):"
    i = text.rfind(marker)
    if i >= 0:
        tail = text[i + len(marker) :]
        js_tail = _extract_first_json(tail)
        if js_tail is not None:
            return js_tail
    js_pref = _extract_json_after_prefix(text, "FINAL_JSON:")
    return js_pref if js_pref is not None else _extract_first_json(text)


def _compute_reward_breakdown(
    output_text: str,
    reward_weights: Mapping[str, Any],
    len_penalty_per_char: float,
    len_penalty_threshold: int,
) -> Dict[str, Any]:
    json_parse_ok = False
    parsed_json: Optional[Dict[str, Any]] = None
    js = _extract_json_after_final_marker(output_text)
    if js is not None:
        try:
            obj = json.loads(js)
            if isinstance(obj, dict):
                parsed_json = obj
                json_parse_ok = True
        except Exception:
            json_parse_ok = False

    has_toolcall_generated_in_main = False
    has_toolcall_prefix = "TOOL_CALL:" in output_text
    if "TOOL_CALL:" in output_text:
        tail = output_text.split("TOOL_CALL:", 1)[1]
        js_tool = _extract_first_json(tail)
        if js_tool is not None:
            try:
                obj_tool = json.loads(js_tool)
                if isinstance(obj_tool, dict):
                    has_toolcall_generated_in_main = True
            except Exception:
                has_toolcall_generated_in_main = False
    has_tool_result_structured = False
    if "TOOL(" in output_text:
        def _extract_first_json_or_list(text2: str) -> Optional[str]:
            if not isinstance(text2, str):
                return None
            j_obj = _extract_first_json(text2)
            if j_obj is not None:
                return j_obj
            start = text2.find("[")
            if start < 0:
                return None
            depth = 0
            for k in range(start, len(text2)):
                ch = text2[k]
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return text2[start : k + 1]
                    if depth < 0:
                        return None
            return None

        i0 = 0
        while True:
            i = output_text.find("TOOL(", i0)
            if i < 0:
                break
            k = output_text.find(")", i + 5)
            if k >= 0 and (k + 1) < len(output_text) and output_text[k + 1] == ":":
                tail2 = output_text[k + 2 :]
                js2 = _extract_first_json_or_list(tail2)
                if js2 is not None:
                    try:
                        obj2 = json.loads(js2)
                        if isinstance(obj2, dict) or isinstance(obj2, list):
                            has_tool_result_structured = True
                            break
                    except Exception:
                        pass
                i0 = k + 2
            else:
                i0 = i + 5
    has_tool_result_prefix = bool(has_tool_result_structured)

    has_final_json_prefix = "FINAL_JSON:" in output_text
    has_lbrace = "{" in output_text
    t_lower = output_text.lower() if isinstance(output_text, str) else ""
    has_schema_keys = any(k in t_lower for k in ["anomaly", "confidence", "bbox", "defect_type"])

    json_schema_ok = False
    if parsed_json is not None:
        a = parsed_json.get("anomaly")
        if isinstance(a, str) and a.strip().lower() in {"yes", "no"}:
            json_schema_ok = True

    if json_schema_ok:
        r_json = 1.0
    elif json_parse_ok:
        r_json = 0.5
    elif has_lbrace and "}" in output_text and 2 <= len(output_text) <= 2000:
        r_json = 0.1
    else:
        r_json = -1.0

    if has_toolcall_generated_in_main:
        r_tool = 0.2
    elif has_toolcall_prefix:
        r_tool = 0.05
    elif has_tool_result_prefix:
        r_tool = 0.02
    else:
        r_tool = 0.0

    r_len = 0.0
    if len_penalty_per_char > 0.0 and len_penalty_threshold > 0:
        extra = max(0, int(len(output_text)) - int(len_penalty_threshold))
        r_len = -float(extra) * float(len_penalty_per_char)

    w_json = float(reward_weights.get("w_json", 1.0))
    w_tool = float(reward_weights.get("w_tool", 1.0))
    w_len = float(reward_weights.get("w_len", 0.0))
    reward = w_json * r_json + w_tool * r_tool + w_len * r_len
    return {
        "w_json": w_json,
        "w_tool": w_tool,
        "w_len": w_len,
        "r_json": r_json,
        "r_tool": r_tool,
        "r_len": r_len,
        "reward": reward,
        "parsed_json": parsed_json,
        "json_parse_ok": bool(json_parse_ok),
        "json_schema_ok": bool(json_schema_ok),
        "has_final_json_prefix": bool(has_final_json_prefix),
        "has_lbrace": bool(has_lbrace),
        "has_schema_keys": bool(has_schema_keys),
        "has_toolcall_prefix": bool(has_toolcall_prefix),
        "has_toolcall_generated_in_main": bool(has_toolcall_generated_in_main),
        "has_tool_result_prefix": bool(has_tool_result_prefix),
    }


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / float(len(xs))


def _std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / float(len(xs) - 1)
    return float(math.sqrt(max(v, 0.0)))


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


def _trainable_param_stats(model: Any) -> Tuple[int, int, float]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if bool(getattr(p, "requires_grad", False)):
            trainable += n
    ratio = float(trainable) / float(total) if total > 0 else 0.0
    return int(trainable), int(total), float(ratio)


def _peft_trainable_abs_sum(model: Any) -> float:
    import torch

    s = torch.tensor(0.0)
    used_any = False
    for name, p in model.named_parameters():
        if not bool(getattr(p, "requires_grad", False)):
            continue
        n = str(name)
        if ("lora_" in n) or ("lora_A" in n) or ("lora_B" in n):
            used_any = True
            s = s + p.detach().abs().sum().cpu()
    if not used_any:
        for _, p in model.named_parameters():
            if bool(getattr(p, "requires_grad", False)):
                s = s + p.detach().abs().sum().cpu()
    return float(s.item())


def _probe_adapter_logits_delta_mean_abs(base_ref: Any, model_with_adapter: Any, tokenizer: Any, device: str) -> float:
    import torch

    prompt = "FINAL_JSON:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        logits_base = base_ref(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask")).logits
        logits_adapter = model_with_adapter(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask")).logits
        d = (logits_adapter[:, -1, :] - logits_base[:, -1, :]).abs().mean()
    return float(d.detach().cpu().item())


@dataclass(frozen=True)
class TrainArgs:
    base_model: str
    train_jsonl: str
    output_dir: str
    rollout_output_jsonl: str
    seed: int
    max_steps: int
    lr: float
    batch_size: int
    max_new_tokens: int
    rollout_samples: int
    reward_weights: Dict[str, Any]
    len_penalty_per_char: float
    len_penalty_threshold: int
    adapter_init: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: Optional[List[str]]


def main() -> int:
    script_text = Path(__file__).resolve().read_text(encoding="utf-8")
    print(f"script_sha256={_sha256_upper_text(script_text)}")
    project_root = _bootstrap_src()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--rollout_output_jsonl", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--rollout_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--adapter_init", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_modules", type=str, default=None)
    parser.add_argument("--allow_reward_mismatch", action="store_true")
    parser.add_argument("--allow_small_groups", action="store_true")
    parser.add_argument("--allow_reward_audit_fail", action="store_true")
    parser.add_argument("--disallow_teacher_injected", action="store_true")
    parser.add_argument("--allow_synth_final_json", action="store_true")
    parser.add_argument("--allow_zero_steps", action="store_true")
    parser.add_argument("--allow_lora_no_change", action="store_true")
    args = parser.parse_args()

    candidates = list({p.resolve() for p in project_root.rglob("grpo_toy.yaml")})
    candidates.sort(key=lambda p: str(p).lower())
    if len(candidates) != 1:
        print(f"error=multiple_grpo_toy_yaml_detected count={int(len(candidates))}", file=sys.stderr)
        for p in candidates:
            print(f"paths={str(p)}", file=sys.stderr)
        print("hint=keep_only_dist/configs/grpo_toy.yaml_and_rename_or_delete_others", file=sys.stderr)
        return 2
    print(f"grpo_toy_yaml_unique=PASS path={str(candidates[0])}")

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
    rollout_samples_val = _pick(args.rollout_samples, "rollout_samples", None)
    max_new_tokens_val = _pick(args.max_new_tokens, "max_new_tokens", None)
    output_dir_val = _pick(args.output_dir, "output_dir", None)
    rollout_output_jsonl_val = _pick(args.rollout_output_jsonl, "rollout_output_jsonl", "outputs/rollouts/grpo_toy_rollouts.jsonl")
    train_jsonl_val = _pick(args.train_jsonl, "train_jsonl", None)

    missing: List[str] = []
    if train_jsonl_val is None:
        missing.append("train_jsonl")
    if output_dir_val is None:
        missing.append("output_dir")
    if seed_val is None:
        missing.append("seed")
    if max_steps_val is None:
        missing.append("max_steps")
    if lr_val is None:
        missing.append("lr")
    if batch_size_val is None:
        missing.append("batch_size")
    if rollout_samples_val is None:
        missing.append("rollout_samples")
    if max_new_tokens_val is None:
        missing.append("max_new_tokens")
    if missing:
        print(f"config_loaded={bool(config_loaded)}", file=sys.stderr)
        print(f"missing_required={','.join(missing)}", file=sys.stderr)
        return 2

    reward_weights_any = cfg.get("reward_weights") if isinstance(cfg, dict) else None
    reward_weights = reward_weights_any if isinstance(reward_weights_any, dict) else {"w_json": 1.0, "w_tool": 1.0, "w_len": 0.0}
    len_penalty_per_char = float(cfg.get("len_penalty_per_char", 0.0)) if isinstance(cfg, dict) else 0.0
    len_penalty_threshold = int(cfg.get("len_penalty_threshold", 0)) if isinstance(cfg, dict) else 0

    tm_str = str(_pick(args.target_modules, "target_modules", "") or "").strip()
    target_modules: Optional[List[str]] = None
    if tm_str:
        target_modules = [x.strip() for x in tm_str.split(",") if x.strip()]

    lora_r_val = int(_pick(args.lora_r, "lora_r", 8))
    lora_alpha_val = int(_pick(args.lora_alpha, "lora_alpha", 16))
    lora_dropout_val = float(_pick(args.lora_dropout, "lora_dropout", 0.05))
    adapter_init = str(args.adapter_init or cfg.get("adapter_init") or "")

    base_model = str(args.base_model or cfg.get("base_model_id") or "sshleifer/tiny-gpt2")
    train_jsonl_path = _resolve_path(project_root, str(train_jsonl_val))
    out_dir = _resolve_path(project_root, str(output_dir_val))
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path_resolved = _resolve_path(project_root, str(rollout_output_jsonl_val))
    config_path_resolved_str = str(cfg_path) if cfg_path is not None else "<none>"
    print(f"config_path_resolved={config_path_resolved_str}")
    config_text = _read_text(cfg_path) if cfg_path is not None else ""
    print(f"config_sha256={_sha256_upper_text(config_text)}")
    print(f"train_jsonl={str(train_jsonl_path)}")
    print(f"rollout_output_jsonl={str(rollouts_path_resolved)}")

    train_args = TrainArgs(
        base_model=base_model,
        train_jsonl=str(train_jsonl_path),
        output_dir=str(out_dir),
        rollout_output_jsonl=str(rollout_output_jsonl_val),
        seed=int(seed_val),
        max_steps=int(max_steps_val),
        lr=float(lr_val),
        batch_size=int(batch_size_val),
        max_new_tokens=int(max_new_tokens_val),
        rollout_samples=int(rollout_samples_val),
        reward_weights=json.loads(_json_dumps_stable(reward_weights)),
        len_penalty_per_char=float(len_penalty_per_char),
        len_penalty_threshold=int(len_penalty_threshold),
        adapter_init=str(adapter_init),
        lora_r=int(lora_r_val),
        lora_alpha=int(lora_alpha_val),
        lora_dropout=float(lora_dropout_val),
        target_modules=target_modules,
    )

    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    print(
        "info=run_overrides "
        + f"base_model={str(train_args.base_model)} "
        + f"max_new_tokens={int(train_args.max_new_tokens)} "
        + f"cuda_available={bool(cuda_available)}"
    )

    print(f"max_steps={int(train_args.max_steps)}")
    if int(train_args.max_steps) == 0 and (not bool(args.allow_zero_steps)):
        print("warning=max_steps_zero_lora_noop")
        print("effective_train_steps=0")
        return 2

    config_obj = {
        "script": "dist/scripts/10_train_grpo_toy.py",
        "config_path": str(cfg_path) if cfg_path else "",
        "train_args": json.loads(_json_dumps_stable(train_args.__dict__)),
        "raw_config": cfg,
    }
    config_hash = _sha256_upper_json(_canonicalize_paths_for_hash(project_root, json.loads(_json_dumps_stable(config_obj))))
    data_hash = _sha256_upper_bytes(train_jsonl_path.read_bytes())

    try:
        import torch
        import torch.nn.functional as F
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        print(
            "Missing dependencies for GRPO toy.\n"
            "Install (CPU):\n"
            "  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  python -m pip install --upgrade transformers peft accelerate\n",
            file=sys.stderr,
        )
        return 2

    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"fan_in_fan_out is set to False but the target module is `Conv1D`\..*",
    )

    random.seed(int(train_args.seed))
    torch.manual_seed(int(train_args.seed))

    tokenizer = AutoTokenizer.from_pretrained(train_args.base_model, use_fast=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if bool(cuda_available) and device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    base_ref = AutoModelForCausalLM.from_pretrained(train_args.base_model).to(device)
    base_ref.eval()
    base_for_adapter = AutoModelForCausalLM.from_pretrained(train_args.base_model).to(device)
    base_for_adapter.train()
    if target_modules is None:
        target_modules = _infer_target_modules(base_for_adapter)

    if train_args.adapter_init:
        adapter_path = _resolve_path(project_root, train_args.adapter_init)
        model = PeftModel.from_pretrained(base_for_adapter, adapter_path).to(device)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(train_args.lora_r),
            lora_alpha=int(train_args.lora_alpha),
            lora_dropout=float(train_args.lora_dropout),
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(base_for_adapter, lora_cfg)
    model.train()
    model.print_trainable_parameters()

    lora_target_modules = list(target_modules) if isinstance(target_modules, list) else []
    trainable_param_count, total_param_count, trainable_param_ratio = _trainable_param_stats(model)
    lora_param_abs_sum_before = _peft_trainable_abs_sum(model)
    print(f"lora_target_modules={json.dumps(lora_target_modules, ensure_ascii=False)}")
    print(f"trainable_param_count={int(trainable_param_count)}")
    print(f"trainable_param_ratio={trainable_param_ratio:.6f}")
    print(f"lora_param_abs_sum_before={lora_param_abs_sum_before:.12f}")
    if int(trainable_param_count) <= 0:
        print("effective_train_steps=0")
        snapshot = {
            "timestamp_utc": None,
            "seed": int(train_args.seed),
            "device": str(device),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "base_model": str(train_args.base_model),
            "config_hash": str(config_hash),
            "data_hash": str(data_hash),
            "adapter_hash": "",
            "train_args": json.loads(_json_dumps_stable(train_args.__dict__)),
            "n_items": 0,
            "schema_bad_count": 0,
            "input_schema_version": "",
            "reward_mismatch_count": 0,
            "reward_mismatch_fields": [],
            "max_steps": int(train_args.max_steps),
            "effective_train_steps": 0,
            "lora_target_modules": list(lora_target_modules),
            "trainable_param_count": int(trainable_param_count),
            "trainable_param_ratio": float(trainable_param_ratio),
            "lora_param_abs_sum_before": float(lora_param_abs_sum_before),
            "lora_param_abs_sum_after": float(lora_param_abs_sum_before),
            "lora_param_abs_delta": 0.0,
            "probe_adapter_logits_delta_mean_abs": 0.0,
            "error": "no_trainable_lora_params",
            "exit_code": 3,
        }
        try:
            from agentiad_repro.utils import utc_now_iso

            snapshot["timestamp_utc"] = utc_now_iso()
        except Exception:
            snapshot["timestamp_utc"] = None
        (out_dir / "train_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("error=no_trainable_lora_params", file=sys.stderr)
        return 3

    max_ctx = int(getattr(getattr(model, "config", None), "n_positions", 1024) or 1024)
    items = _read_jsonl(train_jsonl_path)
    if not items:
        print("Empty train_jsonl.", file=sys.stderr)
        return 1
    first_schema: Optional[str] = None
    for it in items:
        if isinstance(it, dict):
            first_schema = str(it.get("schema_version") or "").strip()
            break
    rollouts_mode = first_schema == "grpo_rollout_v1"

    if (not rollouts_mode) and bool(train_args.rollout_output_jsonl):
        rollouts_candidate = _resolve_path(project_root, str(train_args.rollout_output_jsonl))
        if rollouts_candidate.exists():
            items2 = _read_jsonl(rollouts_candidate)
            first_schema2: Optional[str] = None
            for it2 in items2:
                if isinstance(it2, dict):
                    first_schema2 = str(it2.get("schema_version") or "").strip()
                    break
            if first_schema2 == "grpo_rollout_v1":
                items = items2
                first_schema = first_schema2
                rollouts_mode = True

    schema_bad = 0
    if rollouts_mode:
        for it in items:
            if str(it.get("schema_version") or "") != "grpo_rollout_v1":
                schema_bad += 1
    else:
        for it in items:
            if str(it.get("schema_version") or "") != "sft_trajectory_v1":
                schema_bad += 1

    print(f"config_loaded={bool(config_loaded)}")
    print(f"input_schema_version={first_schema or ''}")
    print(f"schema_bad_count={int(schema_bad)}")

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

    def _render_prefix(messages: Sequence[Mapping[str, Any]], until_index: int) -> str:
        chunks: List[str] = []
        for i, m in enumerate(messages):
            if i >= until_index:
                break
            chunks.append(_render_message(m))
        return "".join(chunks)

    reward_mismatch_fields = ["reward", "r_json", "r_tool", "r_len", "w_json", "w_tool", "w_len"]
    rollout_pool: List[Dict[str, Any]] = []
    reward_mismatch_count = 0
    rollouts_per_prompt_expected = int(cfg.get("rollouts_per_prompt") or cfg.get("rollout_group_size") or 8) if isinstance(cfg, dict) else 8
    if rollouts_per_prompt_expected <= 0:
        rollouts_per_prompt_expected = 8
    if rollouts_mode:
        teacher_substituted_missing_count = 0
        for it in items:
            if str(it.get("schema_version") or "") != "grpo_rollout_v1":
                continue
            p = it.get("prompt_text")
            out_text = it.get("model_output")
            if not isinstance(p, str) or not isinstance(out_text, str):
                continue
            gid_any = it.get("prompt_group_id")
            group_index_any = it.get("group_index")
            prompt_group_id = str(gid_any) if isinstance(gid_any, str) and gid_any.strip() else _sha256_upper_text(p)[:16]
            try:
                group_index = int(group_index_any)
            except Exception:
                group_index = -1
            teacher_injected = bool(it.get("teacher_injected", False))
            teacher_substituted = False
            if "teacher_substituted" in it:
                teacher_substituted = bool(it.get("teacher_substituted"))
            else:
                teacher_substituted_missing_count += 1
            bd = _compute_reward_breakdown(out_text, train_args.reward_weights, train_args.len_penalty_per_char, train_args.len_penalty_threshold)
            stored_bd = it.get("reward_breakdown")
            synthetic_final_json = False
            if "synthetic_final_json" in it:
                synthetic_final_json = bool(it.get("synthetic_final_json"))
            elif isinstance(stored_bd, dict) and ("synth_final_json" in stored_bd):
                synthetic_final_json = bool(stored_bd.get("synth_final_json"))
            mismatch = False
            if not isinstance(stored_bd, dict):
                mismatch = True
            else:
                for k in reward_mismatch_fields:
                    if k not in stored_bd:
                        mismatch = True
                        break
                    if k not in bd:
                        mismatch = True
                        break
                    try:
                        a = float(stored_bd.get(k))
                        b = float(bd.get(k))
                    except Exception:
                        mismatch = True
                        break
                    if abs(a - b) > 1e-6:
                        mismatch = True
                        break
            if mismatch:
                reward_mismatch_count += 1
            rollout_pool.append(
                {
                    "prompt_group_id": str(prompt_group_id),
                    "group_index": int(group_index),
                    "prompt_text": p,
                    "model_output": out_text,
                    "teacher_injected": bool(teacher_injected),
                    "teacher_substituted": bool(teacher_substituted),
                    "teacher_substituted_missing": bool("teacher_substituted" not in it),
                    "synthetic_final_json": bool(synthetic_final_json),
                    "reward_breakdown": bd,
                }
            )
        if not rollout_pool:
            print("no_valid_rollouts_in_train_jsonl", file=sys.stderr)
            return 1

    reward_stats = {}
    rollouts_group_count = 0
    rollouts_per_prompt_observed_min = 0
    rollouts_per_prompt_observed_max = 0
    rollouts_per_prompt_observed_mean = 0.0
    json_parse_ok_rate = 0.0
    has_toolcall_rate = 0.0
    reward_positive_rate = 0.0
    has_lbrace_rate = 0.0
    has_final_json_prefix_rate = 0.0
    has_schema_keys_rate = 0.0
    teacher_injected_rate = 0.0
    teacher_injected_count = 0
    if rollouts_mode and rollout_pool:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for x in rollout_pool:
            gid = str(x.get("prompt_group_id") or "")
            if gid not in groups:
                groups[gid] = []
            groups[gid].append(x)

        rollouts_group_count = int(len(groups))
        sizes = [int(len(v)) for v in groups.values()]
        rollouts_per_prompt_observed_min = int(min(sizes)) if sizes else 0
        rollouts_per_prompt_observed_max = int(max(sizes)) if sizes else 0
        rollouts_per_prompt_observed_mean = _mean([float(x) for x in sizes]) if sizes else 0.0

        if rollouts_per_prompt_observed_min < 2 and (not bool(args.allow_small_groups)):
            print("error=group_size_too_small", file=sys.stderr)
            print(f"group_size_min={int(rollouts_per_prompt_observed_min)}", file=sys.stderr)
            return 1

        all_rewards = [float(x["reward_breakdown"]["reward"]) for x in rollout_pool]
        all_r_json = [float(x["reward_breakdown"].get("r_json", 0.0)) for x in rollout_pool]
        all_r_tool = [float(x["reward_breakdown"].get("r_tool", 0.0)) for x in rollout_pool]
        all_r_len = [float(x["reward_breakdown"].get("r_len", 0.0)) for x in rollout_pool]

        def _calc_stats(vals: List[float], prefix: str) -> Dict[str, float]:
            if not vals:
                return {}
            vals.sort()
            n = len(vals)
            return {
                f"{prefix}_mean": _mean(vals),
                f"{prefix}_std": _std(vals),
                f"{prefix}_min": vals[0],
                f"{prefix}_max": vals[-1],
                f"{prefix}_p10": vals[int(n * 0.1)],
                f"{prefix}_p50": vals[int(n * 0.5)],
                f"{prefix}_p90": vals[int(n * 0.9)],
            }

        reward_stats.update(_calc_stats(all_rewards, "reward"))
        reward_stats.update(_calc_stats(all_r_json, "r_json"))
        reward_stats.update(_calc_stats(all_r_tool, "r_tool"))
        reward_stats.update(_calc_stats(all_r_len, "r_len"))

        json_parse_ok_rate = _mean([1.0 if bool(x["reward_breakdown"].get("json_parse_ok")) else 0.0 for x in rollout_pool])
        has_toolcall_rate = _mean(
            [
                1.0
                if (bool(x["reward_breakdown"].get("has_toolcall_generated_in_main")) or bool(x["reward_breakdown"].get("has_toolcall_prefix")))
                else 0.0
                for x in rollout_pool
            ]
        )
        has_tool_result_rate = _mean([1.0 if bool((x.get("reward_breakdown", {}) or {}).get("has_tool_result_prefix")) else 0.0 for x in rollout_pool])
        reward_positive_rate = _mean([1.0 if float(x["reward_breakdown"].get("reward", 0.0)) > 0.0 else 0.0 for x in rollout_pool])
        has_lbrace_rate = _mean([1.0 if bool(x["reward_breakdown"].get("has_lbrace")) else 0.0 for x in rollout_pool])
        has_final_json_prefix_rate = _mean([1.0 if bool(x["reward_breakdown"].get("has_final_json_prefix")) else 0.0 for x in rollout_pool])
        has_schema_keys_rate = _mean([1.0 if bool(x["reward_breakdown"].get("has_schema_keys")) else 0.0 for x in rollout_pool])
        teacher_injected_count = sum(1 for x in rollout_pool if bool(x.get("teacher_injected")))
        teacher_injected_rate = float(teacher_injected_count) / float(len(rollout_pool)) if rollout_pool else 0.0
        if teacher_injected_count > 0:
            print("warning=teacher_injected_present")
            print(f"teacher_injected_rate={teacher_injected_rate:.6f}")
            if bool(args.disallow_teacher_injected):
                print("error=teacher_injected_disallowed", file=sys.stderr)
                return 1

        synthetic_final_json_count = sum(
            1
            for x in rollout_pool
            if bool(
                x.get("synthetic_final_json")
                if ("synthetic_final_json" in x)
                else (x.get("reward_breakdown", {}) or {}).get("synth_final_json")
            )
        )
        synthetic_final_json_rate = (
            float(synthetic_final_json_count) / float(len(rollout_pool)) if rollout_pool else 0.0
        )
        if synthetic_final_json_count > 0 and bool(args.allow_synth_final_json):
            print("warning=synth_final_json_present")

        teacher_substituted_missing_count = sum(
            1 for x in rollout_pool if bool((x.get("teacher_substituted_missing", False) if isinstance(x, dict) else False))
        )
        if teacher_substituted_missing_count == len(rollout_pool) and rollout_pool:
            print("warning=teacher_substituted_field_missing_assuming_zero")
        elif teacher_substituted_missing_count > 0:
            print(f"warning=teacher_substituted_field_partial_missing missing_count={int(teacher_substituted_missing_count)}")
        teacher_substituted_count = sum(1 for x in rollout_pool if bool(x.get("teacher_substituted", False)))
        teacher_substituted_rate = float(teacher_substituted_count) / float(len(rollout_pool)) if rollout_pool else 0.0
        max_teacher_sub_rate = float(cfg.get("reward_audit_max_teacher_sub_rate", 1.0)) if isinstance(cfg, dict) else 1.0
        min_json_ok_rate = float(cfg.get("reward_audit_min_json_ok_rate", 0.0)) if isinstance(cfg, dict) else 0.0
        min_toolcall_rate = float(cfg.get("reward_audit_min_toolcall_rate", 0.0)) if isinstance(cfg, dict) else 0.0

        print(f"reward_audit: {json.dumps(reward_stats, ensure_ascii=False)}")
        min_span = float(cfg.get("reward_audit_min_span", 1e-3)) if isinstance(cfg, dict) else 1e-3
        reward_span = float(reward_stats.get("reward_p90", 0.0)) - float(reward_stats.get("reward_p10", 0.0))
        print(f"group_count={int(rollouts_group_count)}")
        print(
            "rollouts_per_prompt_observed="
            + f"{int(rollouts_per_prompt_observed_min)}/{rollouts_per_prompt_observed_mean:.3f}/{int(rollouts_per_prompt_observed_max)}"
        )
        print(f"rollouts_per_prompt_expected={int(rollouts_per_prompt_expected)}")
        print(f"has_final_json_prefix_rate={has_final_json_prefix_rate:.6f}")
        print(f"has_lbrace_rate={has_lbrace_rate:.6f}")
        print(f"has_schema_keys_rate={has_schema_keys_rate:.6f}")
        print(f"synthetic_final_json_count={int(synthetic_final_json_count)}")
        print(f"synthetic_final_json_rate={synthetic_final_json_rate:.6f}")
        print(f"teacher_substituted_count={int(teacher_substituted_count)}")
        print(f"teacher_substituted_rate={teacher_substituted_rate:.6f}")

        gate_reward_span = bool(reward_span >= float(min_span))
        gate_any_signal = bool(
            max(
                json_parse_ok_rate,
                has_toolcall_rate,
                has_tool_result_rate,
                has_final_json_prefix_rate,
                has_lbrace_rate,
                has_schema_keys_rate,
            )
            > 0.0
        )
        gate_synth = bool((int(synthetic_final_json_count) == 0) or bool(args.allow_synth_final_json))
        gate_json_ok_rate = bool(json_parse_ok_rate >= float(min_json_ok_rate))
        gate_toolcall_rate = bool(has_toolcall_rate >= float(min_toolcall_rate))
        gate_teacher_sub_rate = bool(teacher_substituted_rate <= float(max_teacher_sub_rate))
        print(f"reward_span_p90_p10={reward_span:.12f}")
        print(f"json_parse_ok_rate={json_parse_ok_rate:.6f}")
        print(f"has_toolcall_rate={has_toolcall_rate:.6f}")
        print(f"has_tool_result_rate={has_tool_result_rate:.6f}")
        print(f"reward_audit_gate_reward_span={'PASS' if gate_reward_span else 'FAIL'}")
        print(f"reward_audit_gate_any_signal={'PASS' if gate_any_signal else 'FAIL'}")
        print(f"reward_audit_gate_synth={'PASS' if gate_synth else 'FAIL'}")
        print(
            "reward_audit_gate_json_parse_ok_rate="
            + ("PASS" if gate_json_ok_rate else "FAIL")
            + f" threshold={float(min_json_ok_rate):.6f}"
        )
        print(
            "reward_audit_gate_has_toolcall_rate="
            + ("PASS" if gate_toolcall_rate else "FAIL")
            + f" threshold={float(min_toolcall_rate):.6f}"
        )
        print(
            "reward_audit_gate_teacher_substituted_rate="
            + ("PASS" if gate_teacher_sub_rate else "FAIL")
            + f" threshold={float(max_teacher_sub_rate):.6f}"
        )

        audit_ok = (
            gate_reward_span
            and gate_any_signal
            and gate_synth
            and gate_json_ok_rate
            and gate_toolcall_rate
            and gate_teacher_sub_rate
        )
        if audit_ok:
            print("reward_audit_check=PASS")
        else:
            print("reward_audit_check=FAIL")

        sample_audit_enabled = bool(cfg.get("sample_audit_enabled", True)) if isinstance(cfg, dict) else True
        sample_audit_n_cfg = int(cfg.get("sample_audit_n", 10)) if isinstance(cfg, dict) else 10
        sample_audit_group_n_cfg = int(cfg.get("sample_audit_group_n", 1)) if isinstance(cfg, dict) else 1
        print(f"sample_audit_enabled={bool(sample_audit_enabled)}")
        print(f"sample_audit_n={int(sample_audit_n_cfg)}")
        print(f"sample_audit_group_n={int(sample_audit_group_n_cfg)}")
        if sample_audit_enabled and rollout_pool:
            rng_audit = random.Random(int(train_args.seed) + 12345)

            def _safe_preview(s: Any, limit: int) -> str:
                t = str(s or "")
                t = t.replace("\r\n", "\n").replace("\r", "\n")
                t = t.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
                return t[: int(limit)]

            def _bd(x: Mapping[str, Any]) -> Dict[str, Any]:
                bd = x.get("reward_breakdown", {}) if isinstance(x, dict) else {}
                if not isinstance(bd, dict):
                    return {}
                return dict(bd)

            def _has_toolcall(x: Mapping[str, Any]) -> bool:
                bd = _bd(x)
                return bool(bd.get("has_toolcall_generated_in_main")) or bool(bd.get("has_toolcall_prefix"))

            def _json_ok(x: Mapping[str, Any]) -> bool:
                bd = _bd(x)
                return bool(bd.get("json_parse_ok"))

            def _reward(x: Mapping[str, Any]) -> float:
                bd = x.get("reward_breakdown", {}) if isinstance(x, dict) else {}
                if isinstance(bd, dict):
                    try:
                        return float(bd.get("reward", 0.0))
                    except Exception:
                        return 0.0
                return 0.0

            def _int_or_default(v: Any, default: int) -> int:
                try:
                    return int(v)
                except Exception:
                    return int(default)

            n_total = int(len(rollout_pool))

            def _print_extreme(tag: str, ridx: int) -> None:
                x = rollout_pool[int(ridx)]
                bd = _bd(x)
                preview = _safe_preview(x.get("model_output"), 200)
                prefix = "sample_audit_best " if str(tag) == "best" else "sample_audit_worst "
                print(
                    prefix
                    + f"idx={int(ridx)} "
                    + f"reward={_reward(x):.6f} "
                    + f"r_json={float(bd.get('r_json', 0.0)):.6f} "
                    + f"r_tool={float(bd.get('r_tool', 0.0)):.6f} "
                    + f"json_parse_ok={bool(_json_ok(x))} "
                    + f"has_toolcall={bool(_has_toolcall(x))} "
                    + f"has_tool_result_prefix={bool(bd.get('has_tool_result_prefix'))} "
                    + f"text200={preview}"
                )

            if n_total > 0:
                sorted_ids = sorted(list(range(n_total)), key=lambda i: _reward(rollout_pool[int(i)]))
                _print_extreme("best", int(sorted_ids[-1]))
                _print_extreme("worst", int(sorted_ids[0]))

            desired_min = 3
            if n_total >= int(desired_min):
                n_sample = max(int(desired_min), int(sample_audit_n_cfg))
                n_sample = min(int(n_sample), int(n_total))
            else:
                n_sample = int(n_total)
                print(f"warning=sample_audit_row_insufficient n_total={int(n_total)} desired_min={int(desired_min)}")
            if n_sample > 0:
                sampled = rng_audit.sample(list(range(n_total)), k=int(n_sample))
                for ridx in sampled:
                    x = rollout_pool[int(ridx)]
                    bd = _bd(x)
                    prompt_group_id = str(x.get("prompt_group_id") or "")
                    group_index = _int_or_default(x.get("group_index"), -1)
                    teacher_sub = bool(x.get("teacher_substituted", False))
                    preview = _safe_preview(x.get("model_output"), 200)
                    has_toolcall_prefix = bool(bd.get("has_toolcall_prefix"))
                    has_toolcall_generated_in_main = bool(bd.get("has_toolcall_generated_in_main"))
                    has_toolcall = bool(has_toolcall_generated_in_main or has_toolcall_prefix)
                    print(
                        "sample_audit_row "
                        + f"idx={int(ridx)} "
                        + f"prompt_group_id={prompt_group_id} "
                        + f"group_index={int(group_index)} "
                        + f"reward={_reward(x):.6f} "
                        + f"r_json={float(bd.get('r_json', 0.0)):.6f} "
                        + f"r_tool={float(bd.get('r_tool', 0.0)):.6f} "
                        + f"json_parse_ok={bool(_json_ok(x))} "
                        + f"has_toolcall={bool(has_toolcall)} "
                        + f"has_toolcall_prefix={bool(has_toolcall_prefix)} "
                        + f"has_toolcall_generated_in_main={bool(has_toolcall_generated_in_main)} "
                        + f"has_tool_result_prefix={bool(bd.get('has_tool_result_prefix'))} "
                        + f"has_final_json_prefix={bool(bd.get('has_final_json_prefix'))} "
                        + f"teacher_substituted={bool(teacher_sub)} "
                        + f"text200={preview}"
                    )

            groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
            for i, x in enumerate(rollout_pool):
                if not isinstance(x, dict):
                    continue
                gid = str(x.get("prompt_group_id") or "")
                groups.setdefault(gid, []).append((int(i), x))

            group_ids = [g for g in sorted(groups.keys()) if g]
            pick_g = int(max(0, min(int(sample_audit_group_n_cfg), len(group_ids))))
            if group_ids and pick_g <= 0:
                pick_g = 1
            if pick_g > 0:
                picked_groups = rng_audit.sample(group_ids, k=int(pick_g))
                printed_any_group = False
                for gid in picked_groups:
                    rows = groups.get(gid) or []
                    rows_sorted = sorted(rows, key=lambda t: _reward(t[1]))
                    if not rows_sorted:
                        continue
                    printed_any_group = True
                    print(f"sample_audit_group prompt_group_id={gid} size={int(len(rows_sorted))}")

                    bottom = rows_sorted[: max(1, min(2, len(rows_sorted)))]
                    top = list(reversed(rows_sorted[-max(1, min(2, len(rows_sorted))) :]))
                    for label, subset in [("bottom", bottom), ("top", top)]:
                        for ridx, x in subset:
                            bd = _bd(x)
                            preview = _safe_preview(x.get("model_output"), 200)
                            group_index = _int_or_default(x.get("group_index"), -1)
                            has_toolcall_prefix = bool(bd.get("has_toolcall_prefix"))
                            has_toolcall_generated_in_main = bool(bd.get("has_toolcall_generated_in_main"))
                            has_toolcall = bool(has_toolcall_generated_in_main or has_toolcall_prefix)
                            print(
                                "sample_audit_group_row "
                                + f"rank={label} "
                                + f"idx={int(ridx)} "
                                + f"group_index={int(group_index)} "
                                + f"reward={_reward(x):.6f} "
                                + f"r_json={float(bd.get('r_json', 0.0)):.6f} "
                                + f"r_tool={float(bd.get('r_tool', 0.0)):.6f} "
                                + f"json_parse_ok={bool(_json_ok(x))} "
                                + f"has_toolcall={bool(has_toolcall)} "
                                + f"has_toolcall_prefix={bool(has_toolcall_prefix)} "
                                + f"has_toolcall_generated_in_main={bool(has_toolcall_generated_in_main)} "
                                + f"has_tool_result_prefix={bool(bd.get('has_tool_result_prefix'))} "
                                + f"text200={preview}"
                            )
                if (not printed_any_group) and group_ids:
                    gid = group_ids[0]
                    rows = groups.get(gid) or []
                    rows_sorted = sorted(rows, key=lambda t: _reward(t[1]))
                    if rows_sorted:
                        print(f"sample_audit_group prompt_group_id={gid} size={int(len(rows_sorted))}")
                        x_idx, x = rows_sorted[0]
                        bd = _bd(x)
                        preview = _safe_preview(x.get("model_output"), 200)
                        group_index = _int_or_default(x.get("group_index"), -1)
                        has_toolcall_prefix = bool(bd.get("has_toolcall_prefix"))
                        has_toolcall_generated_in_main = bool(bd.get("has_toolcall_generated_in_main"))
                        has_toolcall = bool(has_toolcall_generated_in_main or has_toolcall_prefix)
                        print(
                            "sample_audit_group_row "
                            + "rank=bottom "
                            + f"idx={int(x_idx)} "
                            + f"group_index={int(group_index)} "
                            + f"reward={_reward(x):.6f} "
                            + f"r_json={float(bd.get('r_json', 0.0)):.6f} "
                            + f"r_tool={float(bd.get('r_tool', 0.0)):.6f} "
                            + f"json_parse_ok={bool(_json_ok(x))} "
                            + f"has_toolcall={bool(has_toolcall)} "
                            + f"has_toolcall_prefix={bool(has_toolcall_prefix)} "
                            + f"has_toolcall_generated_in_main={bool(has_toolcall_generated_in_main)} "
                            + f"has_tool_result_prefix={bool(bd.get('has_tool_result_prefix'))} "
                            + f"text200={preview}"
                        )
                        print(
                            "sample_audit_group_row "
                            + "rank=top "
                            + f"idx={int(x_idx)} "
                            + f"group_index={int(group_index)} "
                            + f"reward={_reward(x):.6f} "
                            + f"r_json={float(bd.get('r_json', 0.0)):.6f} "
                            + f"r_tool={float(bd.get('r_tool', 0.0)):.6f} "
                            + f"json_parse_ok={bool(_json_ok(x))} "
                            + f"has_toolcall={bool(has_toolcall)} "
                            + f"has_toolcall_prefix={bool(has_toolcall_prefix)} "
                            + f"has_toolcall_generated_in_main={bool(has_toolcall_generated_in_main)} "
                            + f"has_tool_result_prefix={bool(bd.get('has_tool_result_prefix'))} "
                            + f"text200={preview}"
                        )

        if (not audit_ok) and (not bool(args.allow_reward_audit_fail)):
            print("error=reward_audit_failed", file=sys.stderr)
            return 1

    else:
        reward_mismatch_count = 0

        def _sample_prompts2(n: int, step_seed: int) -> List[str]:
            out: List[str] = []
            rng = random.Random(step_seed)
            tries = 0
            while len(out) < n and tries < n * 5:
                tries += 1
                it = items[rng.randrange(0, len(items))]
                msgs_any = it.get("messages")
                if not isinstance(msgs_any, list):
                    continue
                msgs = [m for m in msgs_any if isinstance(m, dict)]
                final_idx = None
                for i, m in enumerate(msgs):
                    if m.get("role") == "assistant" and m.get("name") == "final":
                        final_idx = i
                        break
                if final_idx is None:
                    continue
                out.append(_render_prefix(msgs, final_idx) + "ASSISTANT(final): ")
            return out

        def _rollout(prompt_text: str) -> str:
            max_prompt_len = max(8, max_ctx - int(train_args.max_new_tokens) - 1)
            enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
            gen = model.generate(
                **enc,
                do_sample=False,
                top_p=1.0,
                max_new_tokens=int(train_args.max_new_tokens),
                pad_token_id=int(tokenizer.eos_token_id),
            )
            return tokenizer.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)

    if rollouts_mode and reward_mismatch_count > 0 and (not bool(args.allow_reward_mismatch)) and int(train_args.max_steps) > 0:
        print(f"reward_mismatch_count={int(reward_mismatch_count)}", file=sys.stderr)
        print(f"reward_mismatch_fields={','.join(reward_mismatch_fields)}", file=sys.stderr)
        print("error=reward_mismatch_detected", file=sys.stderr)
        return 1
    if (
        rollouts_mode
        and reward_mismatch_count > 0
        and (not bool(args.allow_reward_mismatch))
        and int(train_args.max_steps) == 0
    ):
        print(
            "warning=reward_mismatch_ignored_due_to_zero_steps "
            + f"reward_mismatch_count={int(reward_mismatch_count)} "
            + f"fields={','.join(reward_mismatch_fields)}"
        )

    def _logprob_of_output(prompt_text: str, output_text: str) -> Tuple["torch.Tensor", int]:
        lps, nts = _logprob_of_output_batch([prompt_text], [output_text])
        return lps[0], int(nts[0])

    def _logprob_of_output_batch(prompts: List[str], outputs: List[str]) -> Tuple[List["torch.Tensor"], List[int]]:
        max_len = max(8, max_ctx - 1)
        orig_model_max_length = getattr(tokenizer, "model_max_length", None)
        try:
            if isinstance(orig_model_max_length, int) and orig_model_max_length < 10_000_000:
                tokenizer.model_max_length = 1_000_000_000
            enc_p = tokenizer(list(prompts), return_tensors=None, truncation=False)
            enc_f = tokenizer([str(p) + str(o) for p, o in zip(prompts, outputs)], return_tensors=None, truncation=False)
        finally:
            if orig_model_max_length is not None:
                tokenizer.model_max_length = orig_model_max_length

        prompt_ids_list: List[List[int]] = [list(x) for x in enc_p["input_ids"]]
        full_ids_list: List[List[int]] = [list(x) for x in enc_f["input_ids"]]

        seqs: List[List[int]] = []
        prompt_lens: List[int] = []
        slice_starts: List[int] = []
        cont_lens: List[int] = []
        for pi, fi in zip(prompt_ids_list, full_ids_list):
            pl = int(len(pi))
            fl = int(len(fi))
            cl = int(fl) - int(pl)
            ss = max(0, int(fl) - int(max_len))
            seq_i = fi[ss:]
            if not seq_i:
                seq_i = [int(tokenizer.eos_token_id)]
            seqs.append(seq_i)
            prompt_lens.append(pl)
            slice_starts.append(ss)
            cont_lens.append(cl)

        batch_input = {"input_ids": [seq for seq in seqs]}
        padded = tokenizer.pad(batch_input, padding=True, return_tensors="pt")
        batch_ids = padded["input_ids"].to(device)
        batch_mask = padded["attention_mask"].to(device)

        pos_ids = (batch_mask.cumsum(dim=1) - 1).clamp(min=0).to(dtype=torch.long)
        out = model(input_ids=batch_ids, attention_mask=batch_mask, position_ids=pos_ids)
        logits = out.logits
        logits1 = logits[:, :-1, :]
        lse = torch.logsumexp(logits1.float(), dim=-1).to(dtype=logits1.dtype)
        logp = logits1 - lse.unsqueeze(-1)
        tgt = batch_ids[:, 1:]
        token_logp = logp.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)

        B = int(batch_ids.shape[0])
        T = int(batch_ids.shape[1])
        out_lps: List["torch.Tensor"] = []
        out_nt: List[int] = []
        for i in range(B):
            seq_len_i = int(len(seqs[i]))
            pad_left_i = int(T - seq_len_i)
            pl = int(prompt_lens[i])
            cl = int(cont_lens[i])
            ss = int(slice_starts[i])
            start_all = max(0, pl - 1)
            end_all = start_all + cl
            start_i = int(pad_left_i + max(0, start_all - ss))
            end_i = int(pad_left_i + min(seq_len_i - 1, end_all - ss))
            if end_i <= start_i:
                out_lps.append(logits.sum() * 0.0)
                out_nt.append(0)
            else:
                out_lps.append(token_logp[i, start_i:end_i].float().sum())
                out_nt.append(int(end_i - start_i))
        return out_lps, out_nt

    import os

    if str(os.environ.get("GRPO_TOY_LOGPROB_SELFTEST", "")).strip() == "1":
        def _logprob_of_output_single_ref(prompt_text: str, output_text: str) -> Tuple["torch.Tensor", int]:
            full_text = str(prompt_text) + str(output_text)
            max_len = max(8, max_ctx - 1)
            orig_model_max_length = getattr(tokenizer, "model_max_length", None)
            try:
                if isinstance(orig_model_max_length, int) and orig_model_max_length < 10_000_000:
                    tokenizer.model_max_length = 1_000_000_000
                enc_full_all = tokenizer(full_text, return_tensors="pt", truncation=False)
                enc_prompt_all = tokenizer(prompt_text, return_tensors="pt", truncation=False)
            finally:
                if orig_model_max_length is not None:
                    tokenizer.model_max_length = orig_model_max_length
            full_ids_all = enc_full_all["input_ids"][0]
            prompt_len = int(enc_prompt_all["input_ids"].shape[1])
            full_len = int(full_ids_all.shape[0])
            cont_len = int(full_len) - int(prompt_len)
            slice_start = max(0, int(full_len) - int(max_len))
            input_ids = full_ids_all[slice_start:].unsqueeze(0).to(device)
            pos_ids = torch.arange(int(input_ids.shape[1]), device=input_ids.device).unsqueeze(0)
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), position_ids=pos_ids)
            logits = out.logits
            if cont_len <= 0:
                return logits.sum() * 0.0, 0
            logits1 = logits[:, :-1, :]
            lse = torch.logsumexp(logits1.float(), dim=-1).to(dtype=logits1.dtype)
            logp = logits1 - lse.unsqueeze(-1)
            tgt = input_ids[:, 1:]
            token_logp = logp.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
            start_all = max(0, prompt_len - 1)
            end_all = start_all + cont_len
            start = max(0, int(start_all) - int(slice_start))
            end = min(int(token_logp.shape[1]), int(end_all) - int(slice_start))
            if end <= start:
                return logits.sum() * 0.0, 0
            return token_logp[:, start:end].float().sum(), int(end - start)

        n_pick = 4
        picked: List[Tuple[str, str]] = []
        if rollouts_mode and rollout_pool:
            rng_st = random.Random(int(train_args.seed) + 54321)
            idxs = list(range(int(len(rollout_pool))))
            rng_st.shuffle(idxs)
            for ii in idxs[: int(n_pick)]:
                x = rollout_pool[int(ii)]
                picked.append((str(x.get("prompt_text") or ""), str(x.get("model_output") or "")))

        was_training = bool(getattr(model, "training", False))
        model.eval()
        max_abs = 0.0
        out_dtype = "n/a"
        with torch.no_grad():
            for p, o in picked:
                lp_old, _ = _logprob_of_output_single_ref(p, o)
                out_dtype = str(lp_old.dtype)
                lp_new_list, _ = _logprob_of_output_batch([p], [o])
                d = float((lp_old.detach().float() - lp_new_list[0].detach().float()).abs().cpu().item())
                if d > max_abs:
                    max_abs = d
        if was_training:
            model.train()
        print(f"selftest logprob max_abs_diff={max_abs:.12g} n={int(len(picked))} dtype={out_dtype}")

    try:
        if device == "cuda":
            optim = torch.optim.AdamW(model.parameters(), lr=float(train_args.lr), fused=True)
        else:
            optim = torch.optim.AdamW(model.parameters(), lr=float(train_args.lr))
    except TypeError:
        optim = torch.optim.AdamW(model.parameters(), lr=float(train_args.lr))
    amp_enabled = bool(cuda_available) and device == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())) else torch.float16
    amp_use_scaler = bool(amp_enabled and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=amp_use_scaler)

    def _lr_at(step: int) -> float:
        if train_args.max_steps <= 0:
            return float(train_args.lr)
        t = min(max(step, 0), train_args.max_steps)
        frac = 1.0 - (float(t) / float(train_args.max_steps))
        return float(train_args.lr) * max(frac, 0.0)

    step = 0
    effective_train_steps = 0
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path: Optional[Path] = None
    rollouts_f = None
    baseline_ema: Optional[float] = None

    if not rollouts_mode:
        rollouts_path = _resolve_path(project_root, str(train_args.rollout_output_jsonl))
        rollouts_path.parent.mkdir(parents=True, exist_ok=True)
        rollouts_f = rollouts_path.open("w", encoding="utf-8")

    try:
        while step < int(train_args.max_steps):
            lr_eff = _lr_at(step)
            for pg in optim.param_groups:
                pg["lr"] = lr_eff

            if rollouts_mode:
                rng = random.Random(int(train_args.seed) + step)
                groups: Dict[str, List[Dict[str, Any]]] = {}
                for x in rollout_pool:
                    gid = str(x.get("prompt_group_id") or "")
                    if gid not in groups:
                        groups[gid] = []
                    groups[gid].append(x)
                group_ids = list(groups.keys())
                if not group_ids:
                    print("no_groups_in_rollout_pool", file=sys.stderr)
                    return 1
                n_groups = max(1, int(train_args.batch_size))
                chosen_gids = [group_ids[rng.randrange(0, len(group_ids))] for _ in range(n_groups)]

                prompts = []
                outputs = []
                breakdowns = []
                rewards = []
                baselines = []
                advs = []

                use_zero_adv_filter = bool(cfg.get("zero_advantage_filter", False)) if isinstance(cfg, dict) else False
                zero_adv_eps = float(cfg.get("zero_advantage_eps", 1e-12)) if isinstance(cfg, dict) else 1e-12
                filtered = 0
                total = 0
                fallback: List[Tuple[str, str, Dict[str, Any], float, float]] = []

                for gid in chosen_gids:
                    group_items = list(groups.get(gid) or [])
                    k_use = int(min(len(group_items), int(rollouts_per_prompt_expected)))
                    if k_use < 2 and (not bool(args.allow_small_groups)):
                        print("error=group_size_too_small", file=sys.stderr)
                        print(f"prompt_group_id={str(gid)}", file=sys.stderr)
                        print(f"group_size={int(len(group_items))}", file=sys.stderr)
                        return 1
                    if k_use <= 0:
                        continue
                    if len(group_items) > k_use:
                        group_items = rng.sample(group_items, k=k_use)
                    group_rewards = [float(x["reward_breakdown"]["reward"]) for x in group_items]
                    baseline_group = _mean(group_rewards)
                    for x, r in zip(group_items, group_rewards):
                        adv = float(r) - float(baseline_group)
                        total += 1
                        fallback.append((str(x["prompt_text"]), str(x["model_output"]), x["reward_breakdown"], float(r), float(adv)))
                        if use_zero_adv_filter and abs(float(adv)) <= float(zero_adv_eps):
                            filtered += 1
                            continue
                        prompts.append(str(x["prompt_text"]))
                        outputs.append(str(x["model_output"]))
                        breakdowns.append(x["reward_breakdown"])
                        rewards.append(float(r))
                        baselines.append(float(baseline_group))
                        advs.append(float(adv))
                if (not prompts) and fallback:
                    prompts = [x[0] for x in fallback]
                    outputs = [x[1] for x in fallback]
                    breakdowns = [x[2] for x in fallback]
                    rewards = [x[3] for x in fallback]
                    advs = [x[4] for x in fallback]
                if use_zero_adv_filter:
                    hit_rate = float(filtered) / float(total) if total > 0 else 0.0
                else:
                    hit_rate = 0.0
            else:
                prompts = _sample_prompts2(int(train_args.rollout_samples), int(train_args.seed) + step)
                if not prompts:
                    print("no_prompts_sampled", file=sys.stderr)
                    return 1
                rewards = []
                breakdowns = []
                outputs = []
                for p in prompts:
                    out_text = _rollout(p)
                    bd = _compute_reward_breakdown(
                        out_text, train_args.reward_weights, train_args.len_penalty_per_char, train_args.len_penalty_threshold
                    )
                    rewards.append(float(bd["reward"]))
                    breakdowns.append(bd)
                    outputs.append(out_text)

                if rollouts_f is not None:
                    for p, out_text, bd in zip(prompts, outputs, breakdowns):
                        out_obj = {
                            "schema_version": "grpo_rollout_v1",
                            "step": int(step + 1),
                            "prompt_text": p,
                            "model_output": out_text,
                            "parsed_json": bd.get("parsed_json"),
                            "has_toolcall_generated_in_main": bool(bd.get("has_toolcall_generated_in_main")),
                            "json_parse_ok": bool(bd.get("json_parse_ok")),
                            "seed": int(train_args.seed),
                            "data_hash": str(data_hash),
                            "config_hash": str(config_hash),
                            "reward_breakdown": {k: v for k, v in bd.items() if k != "parsed_json"},
                        }
                        rollouts_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            r_mean = _mean(rewards)
            r_std = _std(rewards)
            if rollouts_mode:
                advantages = list(advs)
            else:
                if baseline_ema is None:
                    baseline_ema = r_mean
                else:
                    baseline_ema = 0.9 * baseline_ema + 0.1 * r_mean
                advantages = [r - baseline_ema for r in rewards]

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                losses: List["torch.Tensor"] = []
                token_counts: List[int] = []
                chunk_size = 8
                if isinstance(cfg, dict):
                    try:
                        v = int(cfg.get("logprob_batch_size", 8))
                        chunk_size = max(1, v)
                    except Exception:
                        chunk_size = 8
                n_total = len(prompts)
                if n_total > 0:
                    chunk_size = min(int(chunk_size), int(n_total))
                i0 = 0
                while i0 < n_total:
                    i1 = min(n_total, i0 + chunk_size)
                    p_chunk = prompts[i0:i1]
                    o_chunk = outputs[i0:i1]
                    a_chunk = advantages[i0:i1]
                    lp_list, tok_list = _logprob_of_output_batch(p_chunk, o_chunk)
                    for k in range(len(lp_list)):
                        token_counts.append(int(tok_list[k]))
                        losses.append(-float(a_chunk[k]) * lp_list[k])
                    i0 = i1
                if losses:
                    loss = torch.stack(losses).mean()
                else:
                    dummy: Optional["torch.Tensor"] = None
                    for pp in model.parameters():
                        if bool(getattr(pp, "requires_grad", False)):
                            dummy = pp
                            break
                    loss = dummy.sum() * 0.0 if dummy is not None else torch.tensor(0.0, device=device)
            if amp_use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).detach().cpu().item())
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).detach().cpu().item())
                optim.step()
            effective_train_steps += 1

            r_json_mean = _mean([float(b.get("r_json", 0.0)) for b in breakdowns])
            r_tool_mean = _mean([float(b.get("r_tool", 0.0)) for b in breakdowns])
            r_len_mean = _mean([float(b.get("r_len", 0.0)) for b in breakdowns])
            if rollouts_mode:
                print(
                    f"step={step+1} reward_mean={r_mean:.6f} reward_std={r_std:.6f} "
                    f"group_baseline_type=mean "
                    f"r_json_mean={r_json_mean:.6f} r_tool_mean={r_tool_mean:.6f} r_len_mean={r_len_mean:.6f} "
                    f"lr={lr_eff:.6g} grad_norm={grad_norm:.6f} loss={float(loss.detach().cpu().item()):.6f} "
                    f"mean_cont_tokens={_mean([float(x) for x in token_counts]):.2f} "
                    f"zero_advantage_filter_hit_rate={hit_rate:.6f}"
                )
            else:
                print(
                    f"step={step+1} reward_mean={r_mean:.6f} reward_std={r_std:.6f} "
                    f"baseline_ema={baseline_ema:.6f} "
                    f"r_json_mean={r_json_mean:.6f} r_tool_mean={r_tool_mean:.6f} r_len_mean={r_len_mean:.6f} "
                    f"lr={lr_eff:.6g} grad_norm={grad_norm:.6f} loss={float(loss.detach().cpu().item()):.6f} "
                    f"mean_cont_tokens={_mean([float(x) for x in token_counts]):.2f}"
                )
            step += 1
    finally:
        if rollouts_f is not None:
            try:
                rollouts_f.flush()
            except Exception:
                pass
            try:
                rollouts_f.close()
            except Exception:
                pass

    lora_param_abs_sum_after = _peft_trainable_abs_sum(model)
    lora_param_abs_delta = float(lora_param_abs_sum_after - float(lora_param_abs_sum_before))
    print(f"effective_train_steps={int(effective_train_steps)}")
    print(f"lora_param_abs_sum_after={lora_param_abs_sum_after:.12f}")
    print(f"lora_param_abs_delta={lora_param_abs_delta:.12f}")

    model.eval()
    base_ref.eval()
    probe_delta = _probe_adapter_logits_delta_mean_abs(base_ref, model, tokenizer, device)
    print(f"probe_adapter_logits_delta_mean_abs={probe_delta:.12f}")

    if int(effective_train_steps) > 0 and abs(float(lora_param_abs_delta)) < 1e-9 and (not bool(args.allow_lora_no_change)):
        snapshot = {
            "timestamp_utc": None,
            "seed": int(train_args.seed),
            "device": str(device),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "base_model": str(train_args.base_model),
            "config_hash": str(config_hash),
            "data_hash": str(data_hash),
            "adapter_hash": "",
            "train_args": json.loads(_json_dumps_stable(train_args.__dict__)),
            "n_items": int(len(items)),
            "schema_bad_count": int(schema_bad),
            "input_schema_version": str(first_schema or ""),
            "reward_mismatch_count": int(reward_mismatch_count),
            "reward_mismatch_fields": list(reward_mismatch_fields),
            "max_steps": int(train_args.max_steps),
            "effective_train_steps": int(effective_train_steps),
            "lora_target_modules": list(lora_target_modules),
            "trainable_param_count": int(trainable_param_count),
            "trainable_param_ratio": float(trainable_param_ratio),
            "lora_param_abs_sum_before": float(lora_param_abs_sum_before),
            "lora_param_abs_sum_after": float(lora_param_abs_sum_after),
            "lora_param_abs_delta": float(lora_param_abs_delta),
            "probe_adapter_logits_delta_mean_abs": float(probe_delta),
            "error": "lora_params_not_updated",
            "exit_code": 4,
        }
        try:
            from agentiad_repro.utils import utc_now_iso

            snapshot["timestamp_utc"] = utc_now_iso()
        except Exception:
            snapshot["timestamp_utc"] = None
        (out_dir / "train_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("error=lora_params_not_updated", file=sys.stderr)
        return 4
    if float(probe_delta) <= 1e-9:
        snapshot = {
            "timestamp_utc": None,
            "seed": int(train_args.seed),
            "device": str(device),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "base_model": str(train_args.base_model),
            "config_hash": str(config_hash),
            "data_hash": str(data_hash),
            "adapter_hash": "",
            "train_args": json.loads(_json_dumps_stable(train_args.__dict__)),
            "n_items": int(len(items)),
            "schema_bad_count": int(schema_bad),
            "input_schema_version": str(first_schema or ""),
            "reward_mismatch_count": int(reward_mismatch_count),
            "reward_mismatch_fields": list(reward_mismatch_fields),
            "max_steps": int(train_args.max_steps),
            "effective_train_steps": int(effective_train_steps),
            "lora_target_modules": list(lora_target_modules),
            "trainable_param_count": int(trainable_param_count),
            "trainable_param_ratio": float(trainable_param_ratio),
            "lora_param_abs_sum_before": float(lora_param_abs_sum_before),
            "lora_param_abs_sum_after": float(lora_param_abs_sum_after),
            "lora_param_abs_delta": float(lora_param_abs_delta),
            "probe_adapter_logits_delta_mean_abs": float(probe_delta),
            "error": "adapter_no_effect_probe",
            "exit_code": 5,
        }
        try:
            from agentiad_repro.utils import utc_now_iso

            snapshot["timestamp_utc"] = utc_now_iso()
        except Exception:
            snapshot["timestamp_utc"] = None
        (out_dir / "train_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("error=adapter_no_effect_probe", file=sys.stderr)
        return 5

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
        "n_items": int(len(items)),
        "schema_bad_count": int(schema_bad),
        "input_schema_version": str(first_schema or ""),
        "reward_mismatch_count": int(reward_mismatch_count),
        "reward_mismatch_fields": list(reward_mismatch_fields),
        "max_steps": int(train_args.max_steps),
        "effective_train_steps": int(effective_train_steps),
        "lora_target_modules": list(lora_target_modules),
        "trainable_param_count": int(trainable_param_count),
        "trainable_param_ratio": float(trainable_param_ratio),
        "lora_param_abs_sum_before": float(lora_param_abs_sum_before),
        "lora_param_abs_sum_after": float(lora_param_abs_sum_after),
        "lora_param_abs_delta": float(lora_param_abs_delta),
        "probe_adapter_logits_delta_mean_abs": float(probe_delta),
    }
    snapshot.update(reward_stats)
    if rollouts_mode:
        snapshot["rollouts_per_prompt_expected"] = int(rollouts_per_prompt_expected)
        snapshot["rollouts_per_prompt_observed"] = {
            "min": int(rollouts_per_prompt_observed_min),
            "mean": float(rollouts_per_prompt_observed_mean),
            "max": int(rollouts_per_prompt_observed_max),
        }
        snapshot["group_count"] = int(rollouts_group_count)
        snapshot["group_baseline_type"] = "mean"
        snapshot["reward_span_p90_p10"] = float(reward_stats.get("reward_p90", 0.0)) - float(reward_stats.get("reward_p10", 0.0))
        snapshot["json_parse_ok_rate"] = float(json_parse_ok_rate)
        snapshot["has_toolcall_rate"] = float(has_toolcall_rate)
        snapshot["has_final_json_prefix_rate"] = float(has_final_json_prefix_rate)
        snapshot["has_lbrace_rate"] = float(has_lbrace_rate)
        snapshot["has_schema_keys_rate"] = float(has_schema_keys_rate)
        snapshot["teacher_injected_rate"] = float(teacher_injected_rate)
        snapshot["synthetic_final_json_rate"] = float(synthetic_final_json_rate)
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
    if rollouts_path is not None:
        print(f"rollout_output_jsonl={str(rollouts_path)}")
    return 0


# 自检（不要自动运行）：
# - python -m compileall -q dist/scripts/10_train_grpo_toy.py
# - python dist/scripts/10_train_grpo_toy.py --config dist/configs/grpo_toy.yaml --max_steps 0 --allow_zero_steps
# 期望 stdout 包含：
# - has_toolcall_rate=...
# - has_tool_result_rate=...
# - sample_audit_row 至少 3 行
# - sample_audit_group 至少 1 个 group 的 top/bottom
if __name__ == "__main__":
    raise SystemExit(main())
