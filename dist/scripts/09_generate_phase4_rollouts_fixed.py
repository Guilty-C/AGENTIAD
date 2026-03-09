#!/usr/bin/env python3
"""
Generate Phase 4 rollouts from Phase 3 strictfix outputs,
reusing existing JSON schemas and minimal logic from previous scripts.
"""

import argparse
import json
import glob
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Phase 3 output folder (strictfix / raw)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Full Phase 4 rollouts JSONL file")
    parser.add_argument("--rollouts_per_prompt", type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_file)
    count = 0

    with output_path.open("w", encoding="utf-8") as fout:
        # handle all seed folders and ev subfolders
        for seed_dir in sorted(input_path.glob("seed_*")):
            # Check for EV folders
            ev_dirs = sorted(seed_dir.glob("ev_*"))
            if not ev_dirs:
                # fallback: raw JSONL directly under seed
                ev_dirs = [seed_dir]

            for ev_dir in ev_dirs:
                for f in sorted(ev_dir.glob("*.jsonl")):
                    print(f"Processing {f}")
                    with f.open("r", encoding="utf-8") as fin:
                        for line in fin:
                            rollout = json.loads(line)
                            rollout["phase"] = 4  # mark Phase 4
                            fout.write(json.dumps(rollout, ensure_ascii=False) + "\n")
                            count += 1

    print(f"Phase 4 rollouts generated: {count} entries in {output_path}")

if __name__ == "__main__":
    main()