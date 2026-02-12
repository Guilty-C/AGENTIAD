# [REMOTE EXECUTION DETECTED? CHECK GUIDELINES]
# This project enforces a strict "Zero-Pollution" remote execution protocol for shared lab servers.
# See REMOTE_EXECUTION_GUIDE.txt for the mandatory "Upload -> Tmp Run -> Cleanup" workflow.

from __future__ import annotations

import argparse
import hashlib
import json
import random
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


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _load_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _rel(project_root: Path, p: Path) -> str:
    try:
        return Path(str(p)).resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        return str(p)


def _iter_sample_dirs(trace_root: Path) -> Iterable[Path]:
    if not trace_root.exists() or not trace_root.is_dir():
        return []
    return [p for p in trace_root.iterdir() if p.is_dir()]


def _extract_fingerprints(trace: Mapping[str, Any]) -> Tuple[Optional[int], str, str, Optional[str]]:
    fp = trace.get("fingerprint") if isinstance(trace, dict) else None
    if not isinstance(fp, dict):
        return None, "", "", None
    seed_v = fp.get("seed")
    seed: Optional[int]
    try:
        seed = int(seed_v) if seed_v is not None else None
    except Exception:
        seed = None
    prompt_hash = _safe_str(fp.get("prompt_hash")).strip()
    config_hash = _safe_str(fp.get("config_hash")).strip()
    git_commit = fp.get("git_commit")
    git_commit_s = _safe_str(git_commit).strip() if git_commit is not None else None
    return seed, config_hash, prompt_hash, git_commit_s


def _turns_to_conversation(trace: Mapping[str, Any], input_obj: Mapping[str, Any]) -> List[Dict[str, Any]]:
    turns = trace.get("turns") if isinstance(trace, dict) else None
    conv: List[Dict[str, Any]] = []
    conv.append(
        {
            "role": "user",
            "content": json.dumps(dict(input_obj), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        }
    )
    if not isinstance(turns, list):
        return conv

    for t in turns:
        if not isinstance(t, dict):
            continue
        rnd = t.get("round")
        name = t.get("name")
        prompt = t.get("prompt")
        raw_output = t.get("raw_output")
        parsed = t.get("parsed")
        meta = t.get("meta")
        tool_call = t.get("tool_call")
        tool_result = t.get("tool_result")

        conv.append(
            {
                "role": "system",
                "round": rnd,
                "name": name,
                "content": _safe_str(prompt),
            }
        )
        conv.append(
            {
                "role": "assistant",
                "round": rnd,
                "name": name,
                "content": _safe_str(raw_output),
                "parsed": parsed,
                "meta": meta,
            }
        )
        if isinstance(tool_call, dict) and _safe_str(tool_call.get("name")).strip():
            conv.append(
                {
                    "role": "tool",
                    "round": rnd,
                    "name": _safe_str(tool_call.get("name")).strip(),
                    "args": tool_call.get("args"),
                    "result": tool_result,
                }
            )
    return conv


def main() -> int:
    project_root = _bootstrap_src()

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_root", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--schema_version", type=str, default="sft_trace_v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from agentiad_repro.utils import ensure_dir, utc_now_iso, write_json

    trace_root = Path(args.trace_root).resolve()
    out_jsonl = Path(args.out_jsonl).resolve()
    ensure_dir(out_jsonl.parent)

    run_name = trace_root.name
    rng = random.Random(int(args.seed))

    n_total = 0
    n_written = 0
    n_skipped_missing_final = 0
    n_skipped_parse_error = 0

    first_config_hash = ""
    first_prompt_hash = ""

    tmp_path = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    with tmp_path.open("w", encoding="utf-8") as f:
        sample_dirs = list(_iter_sample_dirs(trace_root))
        rng.shuffle(sample_dirs)
        for sd in sample_dirs:
            n_total += 1
            trace_path = sd / "trace.json"
            final_path = sd / "final.json"
            if not trace_path.exists() or not final_path.exists():
                n_skipped_missing_final += 1
                continue

            try:
                trace = _load_json(trace_path)
                final_obj = _load_json(final_path)
            except Exception:
                n_skipped_parse_error += 1
                continue

            if not isinstance(trace, dict) or not isinstance(final_obj, dict):
                n_skipped_parse_error += 1
                continue

            seed_v, config_hash, prompt_hash, git_commit = _extract_fingerprints(trace)
            if not first_config_hash and config_hash:
                first_config_hash = config_hash
            if not first_prompt_hash and prompt_hash:
                first_prompt_hash = prompt_hash

            trace_fingerprint_hash = _safe_str(trace.get("trace_fingerprint_hash")).strip()
            if not trace_fingerprint_hash:
                n_skipped_parse_error += 1
                continue

            input_obj = {
                "sample_id": _safe_str(trace.get("sample_id")).strip() or sd.name,
                "class_name": _safe_str(trace.get("class_name")).strip(),
                "query_image_ref": _safe_str((trace.get("input") or {}).get("query_image")).strip()
                if isinstance(trace.get("input"), dict)
                else "",
                "gt_label": _safe_str(trace.get("gt_label")).strip(),
            }

            conv_turns = _turns_to_conversation(trace, input_obj)

            traj = {
                "schema_version": str(args.schema_version),
                "input": input_obj,
                "conversation": {"turns": conv_turns},
                "target": final_obj,
                "audit": {
                    "timestamp_utc": utc_now_iso(),
                    "run_name": _safe_str(trace.get("run_name")).strip() or run_name,
                    "seed": int(seed_v) if seed_v is not None else None,
                    "config_hash": config_hash,
                    "prompt_hash": prompt_hash,
                    "git_commit": git_commit,
                    "trace_fingerprint_hash": trace_fingerprint_hash,
                    "source_trace_path": str(trace_path),
                },
            }

            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            n_written += 1

    tmp_path.replace(out_jsonl)

    sha256_jsonl = hashlib.sha256(out_jsonl.read_bytes()).hexdigest().upper() if out_jsonl.exists() else ""
    summary = {
        "timestamp_utc": utc_now_iso(),
        "schema_version": str(args.schema_version),
        "run_name": run_name,
        "config_hash": first_config_hash,
        "prompt_hash": first_prompt_hash,
        "N_total_traces": int(n_total),
        "N_written": int(n_written),
        "N_skipped_missing_final": int(n_skipped_missing_final),
        "N_skipped_parse_error": int(n_skipped_parse_error),
        "sha256_jsonl": sha256_jsonl,
        "out_jsonl": _rel(project_root, out_jsonl),
    }
    out_summary = out_jsonl.with_suffix(".summary.json")
    write_json(out_summary, summary)
    sys.stderr.write(
        "DONE run_name="
        + run_name
        + " N_total_traces="
        + str(int(n_total))
        + " N_written="
        + str(int(n_written))
        + " N_skipped_missing_final="
        + str(int(n_skipped_missing_final))
        + " N_skipped_parse_error="
        + str(int(n_skipped_parse_error))
        + "\n"
    )
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
