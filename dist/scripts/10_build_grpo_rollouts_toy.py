from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


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


def _compute_reward_breakdown(
    output_text: str,
    reward_weights: Mapping[str, Any],
    len_penalty_per_char: float,
    len_penalty_threshold: int,
) -> Dict[str, Any]:
    global _TRAIN_REWARD_BREAKDOWN_FN
    if _TRAIN_REWARD_BREAKDOWN_FN is None:
        import importlib.util
        import sys

        train_path = Path(__file__).resolve().parent / "10_train_grpo_toy.py"
        spec = importlib.util.spec_from_file_location("grpo_train_toy_reward", str(train_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load train script module: {str(train_path)}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[str(spec.name)] = mod
        spec.loader.exec_module(mod)
        fn = getattr(mod, "_compute_reward_breakdown", None)
        if not callable(fn):
            raise RuntimeError("train script does not define callable _compute_reward_breakdown")
        _TRAIN_REWARD_BREAKDOWN_FN = fn
    return _TRAIN_REWARD_BREAKDOWN_FN(output_text, reward_weights, len_penalty_per_char, len_penalty_threshold)


_TRAIN_REWARD_BREAKDOWN_FN = None


def _as_json_text(v: Any) -> Optional[str]:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).strip()
    return None


def _synthesize_final_json_text(rng: random.Random) -> str:
    defect_type = rng.choice(
        [
            "",
            "scratch",
            "dent",
            "stain",
            "hole",
            "unknown_defect_type",
            "complex_surface_anomaly",
            "micro_crack_near_edge_region",
        ]
    )
    anomaly = rng.choice(["yes", "no"])
    if defect_type and anomaly == "no":
        anomaly = "yes"
    confidence = float(rng.randint(5, 95)) / 100.0
    x1 = float(rng.randint(0, 40)) / 100.0
    y1 = float(rng.randint(0, 40)) / 100.0
    x2 = float(rng.randint(int(x1 * 100) + 10, 100)) / 100.0
    y2 = float(rng.randint(int(y1 * 100) + 10, 100)) / 100.0
    obj = {"anomaly": anomaly, "confidence": confidence, "bbox": [x1, y1, x2, y2], "defect_type": defect_type}
    return "FINAL_JSON:" + json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    project_root = _bootstrap_src()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default="outputs/rollouts/grpo_toy_rollouts.jsonl")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--allow_teacher_injection", action="store_true")
    parser.add_argument("--allow_synth_final_json_fallback", action="store_true")
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

    seed_val = _pick(args.seed, "seed", 0)
    max_samples_val = _pick(args.max_samples, "rollout_samples", 5)
    base_model = str(args.base_model or cfg.get("base_model_id") or "sshleifer/tiny-gpt2")
    max_new_tokens = int(_pick(args.max_new_tokens, "max_new_tokens", 128))
    rollouts_per_prompt = int(cfg.get("rollouts_per_prompt") or cfg.get("rollout_group_size") or 8)
    if rollouts_per_prompt <= 0:
        rollouts_per_prompt = 8

    gen_cfg_any = cfg.get("gen") if isinstance(cfg, dict) else None
    gen_cfg = gen_cfg_any if isinstance(gen_cfg_any, dict) else {}
    gen_do_sample = bool(gen_cfg.get("do_sample", True))
    gen_temperature = float(gen_cfg.get("temperature", 1.0))
    gen_top_p = float(gen_cfg.get("top_p", 0.9))

    train_jsonl_path = _resolve_path(project_root, args.train_jsonl)
    out_path = _resolve_path(project_root, args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config_obj = {
        "script": "dist/scripts/10_build_grpo_rollouts_toy.py",
        "config_path": str(cfg_path) if cfg_path else "",
        "train_jsonl": str(train_jsonl_path),
        "output_jsonl": str(out_path),
        "seed": int(seed_val),
        "max_samples": int(max_samples_val),
        "base_model": base_model,
        "max_new_tokens": int(max_new_tokens),
        "rollouts_per_prompt": int(rollouts_per_prompt),
        "gen": {"do_sample": bool(gen_do_sample), "temperature": float(gen_temperature), "top_p": float(gen_top_p)},
        "raw_config": cfg,
    }
    config_hash = _sha256_upper_json(_canonicalize_paths_for_hash(project_root, json.loads(_json_dumps_stable(config_obj))))
    data_hash = _sha256_upper_bytes(train_jsonl_path.read_bytes())

    reward_weights = cfg.get("reward_weights") if isinstance(cfg, dict) else None
    if not isinstance(reward_weights, dict):
        reward_weights = {"w_json": 1.0, "w_tool": 1.0, "w_len": 0.0}
    len_penalty_per_char = float(_pick(None, "len_penalty_per_char", 0.0))
    len_penalty_threshold = int(_pick(None, "len_penalty_threshold", 0))

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        print(
            "Missing dependencies for rollout build.\n"
            "Install (CPU):\n"
            "  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  python -m pip install --upgrade transformers\n",
            file=sys.stderr,
        )
        return 2

    random.seed(int(seed_val))
    torch.manual_seed(int(seed_val))
    synth_rng = random.Random(int(seed_val) + 1337)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    model.eval()
    max_ctx = int(getattr(getattr(model, "config", None), "n_positions", 1024) or 1024)

    items = _read_jsonl(train_jsonl_path)
    target_count = max(1, int(max_samples_val))

    written = 0
    unique_groups: set[str] = set()
    teacher_injection_enabled = bool(args.allow_teacher_injection)
    teacher_injection_count = 0
    teacher_substitute_count = 0
    synthetic_final_json_enabled = bool(args.allow_synth_final_json_fallback)
    synthetic_final_json_count = 0
    with out_path.open("w", encoding="utf-8") as f:
        while written < target_count:
            for it in items:
                if written >= target_count:
                    break
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
                prompt_text = _render_prefix(msgs, final_idx) + "ASSISTANT(final): "
                teacher_final_text = _as_json_text(msgs[final_idx].get("content"))

                prompt_group_id = _sha256_upper_text(prompt_text)[:16]
                unique_groups.add(str(prompt_group_id))

                remaining = int(target_count) - int(written)
                k_this = int(min(int(rollouts_per_prompt), remaining))
                for gi in range(int(k_this)):
                    if written >= target_count:
                        break
                    max_prompt_len = max(8, max_ctx - int(max_new_tokens) - 1)
                    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
                    gen = model.generate(
                        **enc,
                        do_sample=bool(gen_do_sample),
                        top_p=float(gen_top_p),
                        temperature=float(gen_temperature),
                        max_new_tokens=int(max_new_tokens),
                        pad_token_id=int(tokenizer.eos_token_id),
                    )
                    output_text = tokenizer.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
                    bd = _compute_reward_breakdown(output_text, reward_weights, len_penalty_per_char, len_penalty_threshold)
                    synthetic_final_json = False
                    if (
                        synthetic_final_json_enabled
                        and (not teacher_injection_enabled)
                        and ("tiny-gpt2" in base_model.lower())
                        and ((not bool(bd.get("json_parse_ok"))) and (not bool(bd.get("has_lbrace"))))
                    ):
                        output_text2 = _synthesize_final_json_text(synth_rng)
                        bd2 = _compute_reward_breakdown(
                            output_text2, reward_weights, len_penalty_per_char, len_penalty_threshold
                        )
                        output_text = output_text2
                        bd = bd2
                        synthetic_final_json = True
                        synthetic_final_json_count += 1
                    teacher_injected = False
                    teacher_substituted = False
                    if teacher_injection_enabled and (gi > 0) and (not bool(bd.get("json_parse_ok"))) and teacher_final_text:
                        bd2 = _compute_reward_breakdown(
                            teacher_final_text, reward_weights, len_penalty_per_char, len_penalty_threshold
                        )
                        if bool(bd2.get("json_parse_ok")):
                            output_text = teacher_final_text
                            bd = bd2
                            teacher_injected = True
                            teacher_injection_count += 1
                            teacher_substituted = True
                            teacher_substitute_count += 1

                    reward_breakdown = dict(bd)
                    out_obj = {
                        "schema_version": "grpo_rollout_v1",
                        "prompt_group_id": str(prompt_group_id),
                        "group_index": int(gi),
                        "prompt_text": prompt_text,
                        "model_output": output_text,
                        "reward_breakdown": reward_breakdown,
                        "reward": float(bd.get("reward", 0.0)),
                        "parsed_json": bd.get("parsed_json"),
                        "has_toolcall_generated_in_main": bool(bd.get("has_toolcall_generated_in_main")),
                        "json_parse_ok": bool(bd.get("json_parse_ok")),
                        "teacher_injected": bool(teacher_injected),
                        "teacher_substituted": bool(teacher_substituted),
                        "synthetic_final_json": bool(synthetic_final_json),
                        "seed": int(seed_val),
                        "data_hash": str(data_hash),
                        "config_hash": str(config_hash),
                        "trace_fingerprint_hash": str(it.get("trace_fingerprint_hash") or ""),
                        "trajectory_fingerprint_hash": str(it.get("trajectory_fingerprint_hash") or ""),
                    }
                    f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    written += 1

    print(f"config_loaded={bool(config_loaded)}")
    print(f"train_jsonl={str(train_jsonl_path)}")
    print(f"output_jsonl={str(out_path)}")
    print(f"seed={int(seed_val)}")
    print(f"data_hash={data_hash}")
    print(f"config_hash={config_hash}")
    print(f"teacher_injection_enabled={bool(teacher_injection_enabled)}")
    print(f"teacher_injection_count={int(teacher_injection_count)}")
    print(f"teacher_substitute_count={int(teacher_substitute_count)}")
    teacher_substitute_rate = float(teacher_substitute_count) / float(written) if written > 0 else 0.0
    print(f"teacher_substitute_rate={teacher_substitute_rate:.6f}")
    print(f"synthetic_final_json_enabled={bool(synthetic_final_json_enabled)}")
    print(f"synthetic_final_json_count={int(synthetic_final_json_count)}")
    print(f"rollouts_per_prompt={int(rollouts_per_prompt)}")
    print(f"unique_prompt_groups={int(len(unique_groups))}")
    print(f"written_total_target={int(target_count)}")
    print(f"written_total={int(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
