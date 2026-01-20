from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _bootstrap_src() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256_upper_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _sha256_upper_json(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_upper_text(s)


def _safe_rel(project_root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except Exception:
        return str(p.resolve()).replace("\\", "/")


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    data = yaml.safe_load(_read_text(path))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping/dict.")
    return data


def _merge_cfg(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k == "base_config":
            continue
        out[k] = v
    return out


def _load_l2_infer_module(project_root: Path) -> Any:
    import importlib.util

    l2_path = (project_root / "scripts" / "06_run_agentiad_infer.py").resolve()
    spec = importlib.util.spec_from_file_location("agentiad_repro_l2_infer", str(l2_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import: {l2_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_l2_infer(
    project_root: Path,
    config_path: Path,
    max_samples: int,
    seed: int,
    run_name: str,
    out_split: Optional[str],
    max_new_tokens: Optional[int],
    dry_run: bool,
) -> int:
    mod = _load_l2_infer_module(project_root)
    argv0 = list(sys.argv)
    try:
        argv = [
            "06_run_agentiad_infer.py",
            "--config",
            str(config_path),
            "--max_samples",
            str(int(max_samples)),
            "--seed",
            str(int(seed)),
            "--run_name",
            str(run_name),
        ]
        if out_split:
            argv += ["--split", str(out_split)]
        if max_new_tokens is not None:
            argv += ["--max_new_tokens", str(int(max_new_tokens))]
        if dry_run:
            argv += ["--dry_run"]
        sys.argv = argv
        rc = int(mod.main())
        return rc
    finally:
        sys.argv = argv0


def _canon_final_json(final_obj: Any) -> Tuple[Dict[str, Any], str]:
    if not isinstance(final_obj, dict):
        return {}, json.dumps({}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    s = json.dumps(final_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    try:
        obj2 = json.loads(s)
        return obj2 if isinstance(obj2, dict) else {}, s
    except Exception:
        return dict(final_obj), s


def _write_jsonl(path: Path, items: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(dict(it), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    project_root = _bootstrap_src()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_jsonl", type=str, default="outputs/traces/trajectories_sft.jsonl")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--require_tool", action="store_true")
    parser.add_argument("--allow_skip", action="store_true")
    args = parser.parse_args()

    from agentiad_repro.utils import get_env_snapshot, load_paths, sha256_text

    cfg_path = Path(args.config).resolve()
    cfg_text = _read_text(cfg_path)
    cfg = _load_yaml(cfg_path)
    base_cfg_text = ""
    if "base_config" in cfg and cfg["base_config"]:
        base_path = (project_root / str(cfg["base_config"])).resolve()
        base_cfg_text = _read_text(base_path)
        base_cfg = _load_yaml(base_path)
        cfg = _merge_cfg(base_cfg, cfg)
    config_hash = sha256_text(base_cfg_text + "\n" + cfg_text)

    vlm_model_id = str(cfg.get("vlm_model_id", "")).strip()
    if not vlm_model_id:
        vlm_model_id = str(cfg.get("model_id", "")).strip()
    model_short = "".join(ch if ch.isalnum() else "_" for ch in (vlm_model_id or "model"))[-40:].strip("_") or "model"

    split = str(args.split) if args.split is not None else str(cfg.get("split", "train"))
    run_name = str(args.run_name) if args.run_name is not None else str(cfg.get("run_name", "")).strip()
    if not run_name:
        run_name = f"L3_SFT_{model_short}_{split}_seed{int(args.seed)}"

    paths = load_paths(project_root)
    out_jsonl_path = (project_root / args.out_jsonl).resolve() if not os.path.isabs(args.out_jsonl) else Path(args.out_jsonl).resolve()

    rc = _run_l2_infer(
        project_root=project_root,
        config_path=cfg_path,
        max_samples=int(args.max_samples),
        seed=int(args.seed),
        run_name=run_name,
        out_split=split,
        max_new_tokens=args.max_new_tokens,
        dry_run=bool(args.dry_run),
    )
    if rc != 0:
        return int(rc)

    trace_root = (paths.traces_dir / run_name).resolve()
    if not trace_root.exists():
        print(f"Missing trace dir: {trace_root}", file=sys.stderr)
        return 2

    samples: List[Tuple[str, Path]] = []
    for child in trace_root.iterdir():
        if child.is_dir() and (child / "trace.json").exists():
            samples.append((child.name, child))
    samples.sort(key=lambda x: x[0])

    env = get_env_snapshot()
    env_hash = _sha256_upper_json(env)

    items: List[Dict[str, Any]] = []
    require_tool = bool(args.require_tool)
    allow_skip = bool(args.allow_skip)
    max_samples = int(args.max_samples)

    skipped_trace = 0
    skipped_final = 0

    candidates: List[Tuple[str, Path, Dict[str, Any]]] = []
    for sample_id, sample_dir in samples:
        trace_path = sample_dir / "trace.json"
        try:
            trace_obj = json.loads(_read_text(trace_path))
        except Exception:
            skipped_trace += 1
            continue
        if not isinstance(trace_obj, dict):
            skipped_trace += 1
            continue
        turns_any = trace_obj.get("turns")
        turns: List[Any] = turns_any if isinstance(turns_any, list) else []
        has_tool_turn = any(
            isinstance(t, dict) and str(t.get("name") or "") in {"pz", "cr"}
            for t in turns
        )
        if require_tool and not has_tool_turn:
            continue
        candidates.append((sample_id, sample_dir, trace_obj))

    n_total_candidates = int(len(candidates))

    for sample_id, sample_dir, trace in candidates:
        if len(items) >= max_samples:
            break

        trace_path = sample_dir / "trace.json"
        final_path = sample_dir / "final.json"
        crop_path = sample_dir / "crop.png"
        ref_path = sample_dir / "ref.png"

        try:
            final_obj = json.loads(_read_text(final_path))
        except Exception:
            skipped_final += 1
            continue

        final_obj2, final_text = _canon_final_json(final_obj)
        turns_any = trace.get("turns") if isinstance(trace, dict) else None
        turns = turns_any if isinstance(turns_any, list) else []

        input_obj = trace.get("input") if isinstance(trace, dict) else None
        if not isinstance(input_obj, dict):
            input_obj = {}
        query_image_src = input_obj.get("query_image")

        messages: List[Dict[str, Any]] = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            name = str(t.get("name") or "")
            prompt = t.get("prompt")
            raw = t.get("raw_output")
            tool_call = t.get("tool_call")
            tool_result = t.get("tool_result")

            if name == "global":
                messages.append(
                    {
                        "role": "user",
                        "name": "global",
                        "content": {"prompt": prompt, "images": [{"kind": "query", "source": query_image_src}]},
                    }
                )
                messages.append({"role": "assistant", "name": "global", "content": raw})
                continue

            if name == "pz":
                messages.append({"role": "assistant", "name": "pz", "tool_call": tool_call})
                messages.append({"role": "tool", "name": "pz.crop_image_normalized", "content": {"call": tool_call, "result": tool_result}})
                messages.append(
                    {
                        "role": "user",
                        "name": "pz",
                        "content": {"prompt": prompt, "images": [{"kind": "crop", "path": _safe_rel(project_root, crop_path)}]},
                    }
                )
                messages.append({"role": "assistant", "name": "pz", "content": raw})
                continue

            if name == "cr":
                cr_imgs = [{"kind": "crop", "path": _safe_rel(project_root, crop_path)}]
                if ref_path.exists():
                    cr_imgs.append({"kind": "ref", "path": _safe_rel(project_root, ref_path)})
                messages.append({"role": "assistant", "name": "cr", "tool_call": tool_call})
                messages.append({"role": "tool", "name": "cr.query_image", "content": {"call": tool_call, "result": tool_result}})
                messages.append({"role": "user", "name": "cr", "content": {"prompt": prompt, "images": cr_imgs}})
                messages.append({"role": "assistant", "name": "cr", "content": raw})
                continue

        messages.append({"role": "assistant", "name": "final", "content": final_text})

        item = {
            "schema_version": "sft_trajectory_v1",
            "run_name": str(run_name),
            "sample_id": str(sample_id),
            "trace_fingerprint_hash": (trace.get("trace_fingerprint_hash") if isinstance(trace, dict) else None),
            "fingerprint": (trace.get("fingerprint") if isinstance(trace, dict) else None),
            "config_hash": str(config_hash),
            "env_hash": str(env_hash),
            "paths": {
                "sample_dir": _safe_rel(project_root, sample_dir),
                "trace_json": _safe_rel(project_root, trace_path),
                "final_json": _safe_rel(project_root, final_path),
                "crop_png": _safe_rel(project_root, crop_path) if crop_path.exists() else "",
                "ref_png": _safe_rel(project_root, ref_path) if ref_path.exists() else "",
            },
            "messages": messages,
            "final": final_obj2,
        }
        item_fingerprint = json.loads(json.dumps(item, ensure_ascii=False))
        if isinstance(item_fingerprint, dict):
            item_fingerprint.pop("env_hash", None)
            item_fingerprint.pop("trajectory_fingerprint_hash", None)
        item["trajectory_fingerprint_hash"] = _sha256_upper_json(item_fingerprint)
        items.append(item)

    _write_jsonl(out_jsonl_path, items)

    print(f"run_name={run_name}")
    print(f"trace_dir={_safe_rel(project_root, trace_root)}")
    print(f"out_jsonl={_safe_rel(project_root, out_jsonl_path)}")
    print(f"N_total_candidates={n_total_candidates}")
    print(f"N_written={len(items)}")
    print(f"skipped_trace={int(skipped_trace)}")
    print(f"skipped_final={int(skipped_final)}")
    if items:
        print(f"first_sample_id={items[0].get('sample_id')}")
        print(f"first_trace_fingerprint_hash={items[0].get('trace_fingerprint_hash')}")
        print(f"first_trajectory_fingerprint_hash={items[0].get('trajectory_fingerprint_hash')}")
        msgs0 = items[0].get("messages") if isinstance(items[0], dict) else None
        has_tool_pair = False
        if isinstance(msgs0, list):
            for i in range(len(msgs0) - 1):
                m0 = msgs0[i]
                m1 = msgs0[i + 1]
                if not isinstance(m0, dict) or not isinstance(m1, dict):
                    continue
                if m0.get("role") == "assistant" and "tool_call" in m0 and m1.get("role") == "tool":
                    has_tool_pair = True
                    break
        print(f"first_has_tool={bool(has_tool_pair)}")

    if require_tool and len(items) == 0:
        return 1
    if (skipped_trace > 0 or skipped_final > 0) and (not allow_skip):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
