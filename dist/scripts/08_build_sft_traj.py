#!/usr/bin/env python3

import os
import glob
import json
import sys


USAGE = (
    "Usage:\n"
    "  python dist/scripts/08_build_sft_traj.py --input_dir <phase2_root> --output_dir <phase3_root>\n"
    "\n"
    "Example:\n"
    "  python dist/scripts/08_build_sft_traj.py "
    "--input_dir dist/outputs/phase2_full_sharded_20260303_140050 "
    "--output_dir dist/outputs/phase3_traj_real_raw_mb4_fix_strict_20260306_171524"
)


def _parse_args(argv):
    input_dir = ""
    output_dir = ""

    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok in ("--input_dir", "--input-dir"):
            if i + 1 >= len(argv):
                print("error=missing_input_dir_value", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return None
            input_dir = argv[i + 1]
            i += 2
            continue
        if tok in ("--output_dir", "--output-dir"):
            if i + 1 >= len(argv):
                print("error=missing_output_dir_value", file=sys.stderr)
                print(USAGE, file=sys.stderr)
                return None
            output_dir = argv[i + 1]
            i += 2
            continue
        if tok in ("-h", "--help"):
            print(USAGE)
            return None

        print(f"error=unknown_arg arg={tok}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return None

    input_dir = os.path.abspath(input_dir.strip())
    output_dir = os.path.abspath(output_dir.strip())

    if not input_dir:
        print("error=missing_input_dir", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return None
    if not output_dir:
        print("error=missing_output_dir", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return None

    return input_dir, output_dir


def _iter_jsonl_files(seed_dir):
    pattern = os.path.join(seed_dir, "*.jsonl")
    paths = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    paths.sort()
    return paths


def main():
    parsed = _parse_args(sys.argv)
    if parsed is None:
        return 0 if any(x in ("-h", "--help") for x in sys.argv[1:]) else 1

    input_dir, output_dir = parsed

    if not os.path.isdir(input_dir):
        print(f"error=input_dir_not_found path={input_dir}", file=sys.stderr)
        return 1

    seed_glob = os.path.join(input_dir, "seed_*")
    seed_dirs = [p for p in glob.glob(seed_glob) if os.path.isdir(p)]
    seed_dirs.sort()

    if not seed_dirs:
        print(f"error=no_seed_folders_found input_dir={input_dir}", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)

    total_seeds = len(seed_dirs)
    total_files = 0
    total_rollouts = 0
    generated_output_paths = []

    required_fields = ("prompt", "output", "reward")

    for seed_dir in seed_dirs:
        seed_dir_abs = os.path.abspath(seed_dir)
        seed_name = os.path.basename(seed_dir_abs.rstrip(os.sep))
        out_seed_dir = os.path.join(output_dir, seed_name)
        os.makedirs(out_seed_dir, exist_ok=True)

        seed_files = _iter_jsonl_files(seed_dir_abs)
        if not seed_files:
            print(f"warning=empty_seed_folder path={seed_dir_abs}")
            continue

        for in_file in seed_files:
            in_file_abs = os.path.abspath(in_file)
            total_files += 1

            try:
                with open(in_file_abs, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"warning=file_read_failed path={in_file_abs} err={type(e).__name__}:{e}")
                continue

            print(f"info=processed_file path={in_file_abs} lines_read={len(lines)}")

            out_items = []
            for idx, raw in enumerate(lines, start=1):
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception as e:
                    print(
                        f"warning=invalid_json path={in_file_abs} line_index={idx} "
                        f"err={type(e).__name__}:{e}"
                    )
                    continue

                if not isinstance(obj, dict):
                    print(f"warning=invalid_rollout_type path={in_file_abs} line_index={idx}")
                    continue

                missing = [k for k in required_fields if k not in obj]
                if missing:
                    print(
                        f"warning=missing_phase2_fields path={in_file_abs} line_index={idx} "
                        f"missing={','.join(missing)}"
                    )
                    continue

                out_obj = dict(obj)
                out_obj["phase"] = 3
                out_items.append(out_obj)

            out_file = os.path.join(out_seed_dir, os.path.basename(in_file_abs))
            out_file_abs = os.path.abspath(out_file)
            try:
                with open(out_file_abs, "w", encoding="utf-8") as fw:
                    for item in out_items:
                        fw.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"warning=file_write_failed path={out_file_abs} err={type(e).__name__}:{e}")
                continue

            total_rollouts += len(out_items)
            generated_output_paths.append(out_file_abs)

    print(
        f"summary=phase3_sft_traj total_seeds={total_seeds} "
        f"total_files={total_files} total_rollouts={total_rollouts}"
    )
    print(f"summary=total_files_processed total_files={total_files}")

    if generated_output_paths:
        print("generated_output_paths=" + json.dumps(generated_output_paths, ensure_ascii=False))
    else:
        print("generated_output_paths=[]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
