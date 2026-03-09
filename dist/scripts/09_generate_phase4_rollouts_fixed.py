#!/usr/bin/env python3
"""
Generate Phase2 rollout JSONL files with shard-level sanity checks and logging.
"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Phase2 source root containing seed_*/ev_06_gpu*/")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Phase2 output root for mirrored seed/shard rollouts")
    parser.add_argument("--rollouts_per_prompt", type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sanity_log_path = output_root / "phase2_sanity.log"

    def _log(msg: str) -> None:
        print(msg)
        with sanity_log_path.open("a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    if not input_path.exists() or not input_path.is_dir():
        _log(f"Phase2 failed: input_dir not found or not a directory: {input_path}")
        return 1

    seed_dirs = sorted([p for p in input_path.glob("seed_*") if p.is_dir()], key=lambda p: p.name)
    if not seed_dirs:
        _log(f"warning=no_seed_dirs_found path={str(input_path)}")
        _log("Phase2 failed: no rollouts found")
        return 1

    seed_dirs_processed = 0
    shard_dirs_processed = 0
    jsonl_files_found = 0
    jsonl_files_written = 0

    for seed_dir in seed_dirs:
        seed_dirs_processed += 1
        seed_name = seed_dir.name
        shard_dirs = sorted([p for p in seed_dir.glob("ev_06_gpu*") if p.is_dir()], key=lambda p: p.name)
        if not shard_dirs:
            _log(f"warning=missing_shards seed={seed_name} path={str(seed_dir.resolve())}")
            continue

        for shard_dir in shard_dirs:
            shard_dirs_processed += 1
            shard_dir_abs = shard_dir.resolve()
            shard_jsonls = sorted([p for p in shard_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
            if not shard_jsonls:
                _log(f"warning=empty_gpu_shard path={str(shard_dir_abs)}")
                continue
            jsonl_files_found += len(shard_jsonls)

            out_shard_dir = output_root / seed_name / shard_dir.name
            out_shard_dir.mkdir(parents=True, exist_ok=True)

            for in_file in shard_jsonls:
                in_abs = in_file.resolve()
                out_file = out_shard_dir / in_file.name
                line_count = 0
                written_lines = 0
                with in_abs.open("r", encoding="utf-8") as fin, out_file.open("w", encoding="utf-8") as fout:
                    for line in fin:
                        line_count += 1
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            rollout = json.loads(s)
                        except Exception as e:
                            _log(f"warning=invalid_json file={str(in_abs)} line={line_count} err={type(e).__name__}:{e}")
                            continue
                        fout.write(json.dumps(rollout, ensure_ascii=False) + "\n")
                        written_lines += 1
                jsonl_files_written += 1
                _log(
                    f"info=written_jsonl input={str(in_abs)} output={str(out_file.resolve())} "
                    f"lines_read={line_count} lines_written={written_lines}"
                )

    _log(
        "summary="
        + f"seed_dirs_processed={seed_dirs_processed} "
        + f"gpu_shard_dirs_processed={shard_dirs_processed} "
        + f"jsonl_files_found={jsonl_files_found} "
        + f"jsonl_files_written={jsonl_files_written}"
    )

    if jsonl_files_found == 0:
        _log("Phase2 failed: no rollouts found")
        return 1

    _log(f"phase2_sanity_log={str(sanity_log_path.resolve())}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
