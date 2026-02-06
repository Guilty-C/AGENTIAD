from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import zipfile
import io
import platform
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

ACCEPT_V21_SPEC = {
    "total_score": 11,
    "gates": {
        "T0": {"score": 1, "desc": "compileall returncode == 0"},
        "G0": {"score": 1, "desc": "Invariant check: pre_csv_sha == post_csv_sha AND pre_snap_sha == post_snap_sha"},
        "G1": {"score": 2, "desc": "Missing L1 case -> compare_suite=FAIL & evidence exists"},
        "G2": {"score": 2, "desc": "3-way script SHA lock"},
        "G3": {"score": 1, "desc": "Manifest completeness & MISSING status"},
        "G4": {"score": 1, "desc": "index_body_sha256 verification"},
        "G5": {"score": 2, "desc": "PASS case checks (match_rate, space, output fields, zip content)"},
        "G6": {"score": 1, "desc": "Baseline diagnostics keys present"},
    },
    "thresholds": {
        "answer_match_rate": 0.99,
    },
    "required_diagnostics": [
        "gt_source_counts_json", 
        "gt_space", 
        "anomaly_rate", 
        "l2_answer_space_detected"
    ]
}

def _bootstrap_src() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


def _sha256_upper_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().upper()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    import json
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_gt_label(item: Dict[str, Any], allow_anomaly: bool = True) -> tuple[str, str]:
    # 1. Try item["answer"]
    gt = str(item.get("answer", "")).strip().upper()
    if gt and gt in {"A", "B", "C", "D", "0", "1"}:
        return gt, "item_answer"
        
    # 2. Try final["answer"] or final["label"]
    if "final" in item and isinstance(item["final"], dict):
        final = item["final"]
        for key in ["answer", "label"]:
            val = str(final.get(key, "")).strip().upper()
            if val and val in {"A", "B", "C", "D", "0", "1"}:
                return val, "final_answer_or_label"
                
    # 3. Try final["anomaly"]
    if allow_anomaly and "final" in item and isinstance(item["final"], dict):
        anomaly = str(item["final"].get("anomaly", "")).strip().lower()
        if anomaly == "yes":
            return "1", "final_anomaly"
        elif anomaly == "no":
            return "0", "final_anomaly"
            
    return "MISSING", "MISSING"


def _verify_index_body_sha256(index_path: Path) -> bool:
    if not index_path.exists():
        return False
    
    lines = index_path.read_text(encoding="utf-8").splitlines()
    filtered_lines = []
    recorded_sha = None
    
    for line in lines:
        if line.startswith("index_body_sha256="):
            recorded_sha = line.split("=", 1)[1].strip()
            continue
        if line.strip() == "# Self-Check Results" or line.startswith("zip_selfcheck=") or \
           line.startswith("script_sha_match=") or line.startswith("index_in_zip_matches_disk=") or \
           line.startswith("error=") and "zip_manifest_missing_entries" in line or \
           line.startswith("error=") and "script_sha_mismatch_in_zip" in line:
           # We stop at self-check block start usually, or exclude specific self-check lines if appended
           pass
        
        # Actually, let's be strict based on how we generated it.
        # We generated it by taking lines BEFORE self-check.
        # So we should truncate at "# Self-Check Results"
        if line.strip() == "# Self-Check Results":
            break
            
        filtered_lines.append(line)
        
    # Remove trailing empty lines that might have been added by the V2 update process (separation)
    while filtered_lines and not filtered_lines[-1].strip():
        filtered_lines.pop()
        
    if not recorded_sha:
        return False
        
    body_content = "\n".join(filtered_lines) + "\n"
    calc_sha = _sha256_upper_bytes(body_content.encode("utf-8"))
    
    return calc_sha == recorded_sha


def _run_baseline_logic(input_path: Path, output_csv: Path, snapshot_path: Path, seed: int = 42, allow_gt_cache_from_l2: bool = False) -> None:
    items = _read_jsonl(input_path)
    rows = []
    rng = random.Random(seed)
    
    # Determine L2 Space to align semantics
    project_root = _bootstrap_src()
    l2_csv_path = project_root / "dist/outputs/tables/L2_agent.csv"
    allow_anomaly = True
    l2_space_str = "unknown"
    required_space = set()
    l2_idx_map = {}
    
    if l2_csv_path.exists():
        try:
            import csv
            with l2_csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                l2_rows = list(reader)
            l2_answers = {r.get("answer", "").strip().upper() for r in l2_rows if r.get("answer")}
            l2_answers.discard("")
            l2_answers.discard("MISSING")
            
            # Map idx -> answer for fallback
            if allow_gt_cache_from_l2:
                for r in l2_rows:
                    try:
                        ridx = int(r.get("idx", -1))
                        rans = r.get("answer", "").strip().upper()
                        if ridx >= 0 and rans and rans != "MISSING":
                            l2_idx_map[ridx] = rans
                    except ValueError:
                        pass
            
            if l2_answers.issubset({"A", "B", "C", "D"}) and l2_answers:
                # L2 is MCQ, so L1 must be MCQ. Disable anomaly fallback.
                allow_anomaly = False
                l2_space_str = "MCQ(A-D)"
                required_space = {"A", "B", "C", "D"}
            elif l2_answers.issubset({"0", "1"}) and l2_answers:
                allow_anomaly = True
                l2_space_str = "Binary(0/1)"
                required_space = {"0", "1"}
        except Exception:
            l2_idx_map = {}
            pass
    else:
        l2_idx_map = {}
            
    print(f"Aligning Baseline: L2 Space seems {l2_space_str}. allow_anomaly={allow_anomaly}", file=sys.stderr)
    
    # Pre-scan for gt_space
    all_gts = []
    gt_source_counts = {"item_answer": 0, "final_answer_or_label": 0, "final_anomaly": 0, "l2_fallback": 0, "MISSING": 0}
    
    for idx, item in enumerate(items):
        gt, src = _extract_gt_label(item, allow_anomaly=allow_anomaly)
        if gt == "MISSING" and idx in l2_idx_map:
            gt = l2_idx_map[idx]
            src = "l2_fallback"
            
        gt_source_counts[src] = gt_source_counts.get(src, 0) + 1
        if gt != "MISSING":
            all_gts.append(gt)
            
    gt_space = set(all_gts)

    # Calculate anomaly_rate early
    total_valid = gt_source_counts["item_answer"] + gt_source_counts["final_answer_or_label"] + gt_source_counts["final_anomaly"] + gt_source_counts["l2_fallback"]
    anomaly_rate = 0.0
    if total_valid > 0:
        anomaly_rate = gt_source_counts["final_anomaly"] / total_valid

    # B) Set metadata early
    global _BASELINE_METADATA
    
    gt_fallback_used = (gt_source_counts["l2_fallback"] > 0)
    
    _BASELINE_METADATA = {
        "gt_source_counts": gt_source_counts,
        "l2_space_str": l2_space_str,
        "anomaly_rate": anomaly_rate,
        "gt_space": sorted(list(gt_space)),
        "gt_fallback_used": gt_fallback_used,
        "gt_fallback_count": gt_source_counts["l2_fallback"],
        "gt_direct_count": total_valid - gt_source_counts["l2_fallback"],
        "gt_missing_count": gt_source_counts["MISSING"],
        "gt_direct_rate": (total_valid - gt_source_counts["l2_fallback"]) / len(items) if items else 0.0
    }
    
    # Strict missing check BEFORE writing
    if not allow_gt_cache_from_l2 and gt_source_counts["MISSING"] > 0:
        raise ValueError("gt_missing_in_input_no_fallback")
    
    # Hard Gates for required_space alignment
    if not gt_space:
        if not allow_gt_cache_from_l2 and gt_source_counts["l2_fallback"] == 0:
             # Strict mode failure
             raise ValueError("gt_missing_in_input_no_fallback")
        raise ValueError("l1_all_missing_gt")
        
    if required_space:
        if not gt_space.issubset(required_space):
            raise ValueError(f"baseline_gt_not_aligned_with_l2_task: gt_space={sorted(list(gt_space))} required={sorted(list(required_space))}")

    # Determine pred space
    pred_space_type = "unknown"
    if gt_space.issubset({"0", "1"}):
        pred_space_type = "binary"
        pred_options = ["0", "1"]
    elif gt_space.issubset({"A", "B", "C", "D"}):
        pred_space_type = "mcq"
        pred_options = ["A", "B", "C", "D"]
    else:
        raise ValueError(f"gt_space_mixed_or_unknown: {sorted(list(gt_space))}")

    for idx, item in enumerate(items):
        gt, src = _extract_gt_label(item, allow_anomaly=allow_anomaly)
        if gt == "MISSING" and idx in l2_idx_map:
            gt = l2_idx_map[idx]
            src = "l2_fallback"
        
        # Prediction
        pred = rng.choice(pred_options)
            
        correct = 1 if pred == gt else 0
        
        rows.append({
            "idx": idx,
            "split": item.get("split", "test"),
            "answer": gt,
            "pred": pred,
            "correct": correct,
            "method": "random_baseline",
            "triggered": 0,  # L1 baseline does not use tools
        })
        
    # Stats Logging
    non_null_answers = sum(1 for r in rows if r["answer"] and r["answer"] != "MISSING")
    positive_corrects = sum(1 for r in rows if r["correct"] == 1)
    label_space_set = set(r["pred"] for r in rows) | set(r["answer"] for r in rows)
    # Filter out MISSING/empty
    label_space_set = {x for x in label_space_set if x and x != "MISSING"}
    
    # Calculate answer_nunique (excluding MISSING/empty)
    valid_answers = [r["answer"] for r in rows if r["answer"] and r["answer"] != "MISSING"]
    answer_nunique = len(set(valid_answers))
    
    print(f"L1 Baseline Stats:", file=sys.stderr)
    print(f"gt_source_counts={gt_source_counts}", file=sys.stderr)
    print(f"l1_answer_valid_rate={non_null_answers/len(rows):.4f}", file=sys.stderr)
    print(f"l1_correct_positive_rate={positive_corrects/len(rows):.4f}", file=sys.stderr)
    print(f"l1_label_space={sorted(list(label_space_set))}", file=sys.stderr)
    print(f"l1_answer_nunique={answer_nunique}", file=sys.stderr)
    print(f"l1_gt_space={sorted(list(gt_space))}", file=sys.stderr)
    
    if answer_nunique <= 1 and non_null_answers > 0:
        print("WARNING: label seems degenerate (answer_nunique <= 1)", file=sys.stderr)
        
    # Check GT Source Health
    if anomaly_rate > 0.9:
        raise ValueError(f"gt_source_final_anomaly_dominant: {anomaly_rate:.2%}")

    # Write CSV
    import csv
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "split", "answer", "pred", "correct", "method", "triggered"])
        writer.writeheader()
        writer.writerows(rows)
        
    # Write Snapshot
    snapshot = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_file": str(input_path.name),
        "input_sha256": _sha256_upper_bytes(input_path.read_bytes()),
        "row_count": len(rows),
        "acc": sum(r["correct"] for r in rows) / len(rows) if rows else 0.0,
        "seed": seed,
        "method": "random_baseline"
    }
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def _run_baseline_suite(allow_gt_cache_from_l2: bool = False) -> int:
    project_root = _bootstrap_src()
    
    # allow_gt_cache_from_l2 is now passed directly
    allow_gt = allow_gt_cache_from_l2
    
    evidence_dir = project_root / "dist/outputs/evidence_baseline"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir, ignore_errors=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = Path(__file__).resolve()
    script_bytes = script_path.read_bytes()
    script_sha256 = _sha256_upper_bytes(script_bytes)
    
    index_lines = []
    index_lines.append(f"index_body_sha256_definition=sha256(LF-joined lines, excluding index_body_sha256 line and excluding self-check block)")
    index_lines.append(f"repo_script_sha256={script_sha256}")
    index_lines.append(f"executing_file={str(script_path)}")
    index_lines.append("suite=baseline_l1")
    index_lines.append("")
    
    files_to_zip = [] # (abs_path, arcname)
    files_to_zip.append((script_path, "dist/scripts/04_eval_baseline.py"))
    
    # Input
    input_path = project_root / "dist/data/mmad/mmad_minireal_512.jsonl"
    if not input_path.exists():
        print(f"FAIL: Input {input_path} not found", file=sys.stderr)
        return 1
    input_bytes = input_path.read_bytes()
    input_sha = _sha256_upper_bytes(input_bytes)
    index_lines.append(f"input_mmad_minireal_512_sha256={input_sha}")
    files_to_zip.append((input_path, "dist/data/mmad/mmad_minireal_512.jsonl"))
    
    # Run Baseline
    output_dir = project_root / "dist/outputs/tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "L1_baseline.csv"
    snap_path = output_dir / "L1_baseline_snapshot.json"
    
    # A) Cleanup stale
    removed_csv = 0
    removed_snap = 0
    if csv_path.exists():
        csv_path.unlink()
        removed_csv = 1
    if snap_path.exists():
        snap_path.unlink()
        removed_snap = 1
        
    print(f"stale_cleanup=removed_csv={removed_csv} removed_snapshot={removed_snap}", file=sys.stderr)
    print(f"baseline_suite_is_destructive=true", file=sys.stderr)

    error_reason = None
    print(f"Running Baseline Logic... (allow_gt_cache_from_l2={allow_gt})", file=sys.stderr)
    try:
        _run_baseline_logic(input_path, csv_path, snap_path, allow_gt_cache_from_l2=allow_gt)
        status = "PASS"
    except Exception as e:
        print(f"Error running baseline: {e}", file=sys.stderr)
        status = "FAIL"
        error_reason = str(e)
        # A) Stale check on failure
        if csv_path.exists() or snap_path.exists():
            print("ERROR: stale baseline artifact detected", file=sys.stderr)
            return 1
        
    # Meta from logic
    if "_BASELINE_METADATA" in globals():
        meta = globals()["_BASELINE_METADATA"]
        index_lines.append(f"gt_source_counts_json={json.dumps(meta['gt_source_counts'])}")
        index_lines.append(f"l2_answer_space_detected={meta['l2_space_str']}")
        index_lines.append(f"anomaly_rate={meta['anomaly_rate']:.4f}")
        index_lines.append(f"gt_space={json.dumps(meta['gt_space'])}")
        index_lines.append(f"gt_fallback_used={str(meta.get('gt_fallback_used', False)).lower()}")
        index_lines.append(f"gt_fallback_count={meta.get('gt_fallback_count', 0)}")
        index_lines.append(f"gt_direct_count={meta.get('gt_direct_count', 0)}")
        index_lines.append(f"gt_missing_count={meta.get('gt_missing_count', 0)}")
        index_lines.append(f"gt_direct_rate={meta.get('gt_direct_rate', 0.0):.4f}")
        
        # Check L2 snapshot integrity if fallback used
        if meta.get("gt_fallback_used"):
            l2_snap_path = project_root / "dist/outputs/tables/L2_agent_snapshot.json"
            if not l2_snap_path.exists():
                status = "FAIL"
                error_reason = "l2_gt_cache_provenance_missing_no_snapshot"
            else:
                try:
                    l2_snap = json.loads(l2_snap_path.read_text(encoding="utf-8"))
                    if l2_snap.get("input_sha256") != input_sha:
                         status = "FAIL"
                         error_reason = "l2_gt_cache_provenance_input_mismatch"
                except Exception:
                     status = "FAIL"
                     error_reason = "l2_gt_cache_provenance_snapshot_corrupt"
    
    # Evidence Collection
    if csv_path.exists():
        files_to_zip.append((csv_path, "dist/outputs/tables/L1_baseline.csv"))
        csv_sha = _sha256_upper_bytes(csv_path.read_bytes())
        index_lines.append(f"baseline_csv_sha256={csv_sha}")
        
    if snap_path.exists():
        files_to_zip.append((snap_path, "dist/outputs/tables/L1_baseline_snapshot.json"))
        snap_sha = _sha256_upper_bytes(snap_path.read_bytes())
        index_lines.append(f"baseline_snapshot_sha256={snap_sha}")
        
    # Baseline Gates
    if status == "PASS" and csv_path.exists():
        import csv
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        non_null = sum(1 for r in rows if r["answer"] and r["answer"] != "MISSING")
        l1_answer_valid_rate = non_null/len(rows) if rows else 0.0
        
        valid_answers = [r["answer"] for r in rows if r["answer"] and r["answer"] != "MISSING"]
        answer_nunique = len(set(valid_answers))
        
        if l1_answer_valid_rate < 0.99:
            status = "FAIL"
            error_reason = "l1_answer_missing_or_blank"
            print(f"FAIL: L1 Answer Valid Rate {l1_answer_valid_rate:.4f} < 0.99", file=sys.stderr)
        elif answer_nunique <= 1:
            status = "FAIL"
            error_reason = "l1_label_degenerate"
            print(f"FAIL: L1 Answer N-Unique {answer_nunique} <= 1", file=sys.stderr)
            
    if status == "FAIL" and 'error_reason' in locals():
        index_lines.append(f"error={error_reason}")
        
    index_lines.append(f"status={status}")
    index_lines.append("")
    
    # Evidence Package Manifest
    index_lines.append("# Evidence Package Manifest")
    # G3: Explicitly list INDEX as it is required in manifest
    index_lines.append("file=evidence_baseline/INDEX.txt status=GENERATED")
    
    manifest_map = {}
    for abs_p, arcname in files_to_zip:
        if abs_p.exists():
            data = abs_p.read_bytes()
            sha = _sha256_upper_bytes(data)
            sz = len(data)
            index_lines.append(f"file={arcname} bytes={sz} sha256={sha}")
            manifest_map[arcname] = sha
        else:
            index_lines.append(f"file={arcname} status=MISSING")
            
    # A) Enforce "zip truth" + version lock
    disk_script_sha256 = _sha256_upper_bytes(script_path.read_bytes())
    index_lines.append(f"repo_script_sha256={disk_script_sha256}")

    # Write INDEX (v1)
    index_path = evidence_dir / "INDEX.txt"
    index_content = "\n".join(index_lines) + "\n"
    index_path.write_text(index_content, encoding="utf-8")
    
    # Calculate index_body_sha256
    placeholder = "index_body_sha256=__PLACEHOLDER__"
    insert_idx = 1
    index_lines.insert(insert_idx, placeholder)
    
    lines_for_hash = index_lines[:insert_idx] + index_lines[insert_idx+1:]
    body_content = "\n".join(lines_for_hash) + "\n"
    body_sha = _sha256_upper_bytes(body_content.encode("utf-8"))
    
    index_lines[insert_idx] = f"index_body_sha256={body_sha}"
    final_content = "\n".join(index_lines) + "\n"
    index_path.write_text(final_content, encoding="utf-8")
    
    # Zip (v1)
    files_to_zip.insert(0, (index_path, "evidence_baseline/INDEX.txt"))
    zip_path = project_root / "dist/outputs/evidence_baseline_package.zip"
    if zip_path.exists():
        zip_path.unlink()
        
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_p, arcname in files_to_zip:
            if abs_p.exists():
                zf.write(abs_p, arcname)
                
    # Self-Check
    zip_check_pass = True
    with zipfile.ZipFile(zip_path, "r") as zf:
        for arcname, expected_sha in manifest_map.items():
            try:
                actual_sha = _sha256_upper_bytes(zf.read(arcname))
                if actual_sha != expected_sha:
                    zip_check_pass = False
            except KeyError:
                zip_check_pass = False
                
    script_sha_match = (script_sha256 == _sha256_upper_bytes(script_path.read_bytes()))
    
    # Real Index Verification
    index_match = False
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "evidence_baseline/INDEX.txt" in zf.namelist():
            if zf.read("evidence_baseline/INDEX.txt") == index_path.read_bytes():
                index_match = True

    # Update INDEX (v2)
    final_lines = []
    final_lines.append("")
    final_lines.append("# Self-Check Results")
    final_lines.append(f"zip_selfcheck={'PASS' if zip_check_pass else 'FAIL'}")
    final_lines.append(f"script_sha_match={'PASS' if script_sha_match else 'FAIL'}")
    final_lines.append(f"index_in_zip_matches_disk={'PASS' if index_match else 'FAIL'}")
    
    with open(index_path, "a", encoding="utf-8") as f:
        f.write("\n".join(final_lines) + "\n")
        
    # Re-Zip (v2)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_p, arcname in files_to_zip:
            if abs_p.exists():
                zf.write(abs_p, arcname)
                
    # Final Verify
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.read("evidence_baseline/INDEX.txt") != index_path.read_bytes():
            print("FAIL: Index consistency check failed", file=sys.stderr)
            return 1
            
    print(f"evidence_dir={evidence_dir}")
    print(f"evidence_zip={zip_path}")
    
    if not zip_check_pass or not script_sha_match or not index_match or status != "PASS":
        print("baseline_suite=FAIL")
        return 1
        
    print("baseline_suite=PASS")
    return 0


def _run_compare_suite() -> int:
    project_root = _bootstrap_src()
    evidence_dir = project_root / "dist/outputs/evidence_compare"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir, ignore_errors=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = Path(__file__).resolve()
    script_bytes = script_path.read_bytes()
    script_sha256 = _sha256_upper_bytes(script_bytes)
    
    index_lines = []
    index_lines.append(f"index_body_sha256_definition=sha256(LF-joined lines, excluding index_body_sha256 line and excluding self-check block)")
    index_lines.append(f"executing_file={str(script_path)}")
    index_lines.append("suite=compare_l1_l2")
    
    files_to_zip = [] # (abs_path, arcname)
    files_to_zip.append((script_path, "dist/scripts/04_eval_baseline.py"))
    
    # Inputs
    l1_csv_path = project_root / "dist/outputs/tables/L1_baseline.csv"
    l2_csv_path = project_root / "dist/outputs/tables/L2_agent.csv"
    l1_snap_path = project_root / "dist/outputs/tables/L1_baseline_snapshot.json"
    
    # A) Make FAIL manifest explicit
    expected_inputs = [
        (l1_csv_path, "dist/outputs/tables/L1_baseline.csv"),
        (l2_csv_path, "dist/outputs/tables/L2_agent.csv"),
        (l1_snap_path, "dist/outputs/tables/L1_baseline_snapshot.json")
    ]
    for p, arc in expected_inputs:
        files_to_zip.append((p, arc))
    
    # Initialize state
    status = "FAIL"
    error_reason = None
    compare_csv_path = None
    summary_path = None
    
    print("Running Compare Logic...", file=sys.stderr)
    
    try:
        if not l1_csv_path.exists():
            raise ValueError(f"missing_l1_baseline_csv")
        if not l2_csv_path.exists():
            raise ValueError(f"missing_l2_agent_csv")
            
        # C) Verify L1 snapshot sanity
        if not l1_snap_path.exists():
            raise ValueError("missing_l1_baseline_snapshot")
    
        # Verify input sha in snapshot
        snap_data = json.loads(l1_snap_path.read_text(encoding="utf-8"))
        
        # Need to read input to check sha
        input_path = project_root / "dist/data/mmad/mmad_minireal_512.jsonl"
        if not input_path.exists():
             raise ValueError(f"missing_original_input_for_verification")
             
        input_bytes = input_path.read_bytes()
        input_sha = _sha256_upper_bytes(input_bytes)
        
        if snap_data.get("input_sha256") != input_sha:
            raise ValueError(f"l1_snapshot_input_sha_mismatch")
            
        l1_bytes = l1_csv_path.read_bytes()
        l2_bytes = l2_csv_path.read_bytes()
        l1_sha = _sha256_upper_bytes(l1_bytes)
        l2_sha = _sha256_upper_bytes(l2_bytes)
        l1_snap_sha = _sha256_upper_bytes(l1_snap_path.read_bytes())
        
        index_lines.append(f"compare_inputs=L1:{l1_sha}({len(l1_bytes)}b)|L2:{l2_sha}({len(l2_bytes)}b)")
        index_lines.append(f"baseline_snapshot_sha256={l1_snap_sha}")
        
        # Logic
        import pandas as pd
        df1 = pd.read_csv(l1_csv_path)
        df2 = pd.read_csv(l2_csv_path)
        
        if "triggered" not in df2.columns:
            raise ValueError("missing_triggered_column_in_agent_csv")
            
        df1_renamed = df1.rename(columns={
            "pred": "baseline_pred", 
            "correct": "baseline_correct",
            "triggered": "baseline_triggered"
        })
        df2_renamed = df2.rename(columns={
            "pred": "agent_pred", 
            "correct": "agent_correct",
            "triggered": "agent_triggered",
            "answer": "agent_answer"
        })
        
        cols1 = ["idx", "split", "answer", "baseline_pred", "baseline_correct", "baseline_triggered"]
        cols2 = ["idx", "agent_pred", "agent_correct", "agent_triggered", "agent_answer"]
        
        extra_cols = ["margin", "top1", "top2", "n_rois"]
        agent_rename_map = {}
        for c in extra_cols:
            if c in df2_renamed.columns:
                agent_rename_map[c] = f"agent_{c}"
                
        if agent_rename_map:
            df2_renamed = df2_renamed.rename(columns=agent_rename_map)
            cols2.extend(agent_rename_map.values())
                
        merged = pd.merge(
            df1_renamed[cols1], 
            df2_renamed[cols2], 
            on="idx", 
            how="inner"
        )
        
        acc_baseline = merged["baseline_correct"].mean()
        acc_agent = merged["agent_correct"].mean()
        trigger_baseline = merged["baseline_triggered"].mean()
        trigger_agent = merged["agent_triggered"].mean()
        
        def normalize_ans(x):
            return str(x).strip().upper() if pd.notna(x) else ""
            
        merged["l1_answer_norm"] = merged["answer"].apply(normalize_ans)
        merged["agent_answer_norm"] = merged["agent_answer"].apply(normalize_ans)
        answer_match_rate = (merged["l1_answer_norm"] == merged["agent_answer_norm"]).mean()
        
        baseline_correct_recomputed = (merged["baseline_pred"].apply(normalize_ans) == merged["l1_answer_norm"]).astype(int)
        agent_correct_recomputed = (merged["agent_pred"].apply(normalize_ans) == merged["l1_answer_norm"]).astype(int)
        
        recomp_acc_baseline = baseline_correct_recomputed.mean()
        recomp_acc_agent = agent_correct_recomputed.mean()
        
        print(f"Compare Suite Semantic Stats:", file=sys.stderr)
        print(f"answer_match_rate={answer_match_rate:.4f}", file=sys.stderr)
        print(f"acc_baseline_orig={acc_baseline:.4f} recomp={recomp_acc_baseline:.4f}", file=sys.stderr)
        print(f"acc_agent_orig={acc_agent:.4f} recomp={recomp_acc_agent:.4f}", file=sys.stderr)
        
        pred_space = set(df1_renamed["baseline_pred"].dropna().unique())
        ans_space = set(df1_renamed["answer"].dropna().unique())
        agent_pred_space = set(merged["agent_pred"].dropna().unique())
        
        pred_space = {str(x) for x in pred_space if str(x).strip()}
        ans_space = {str(x) for x in ans_space if str(x).strip()}
        agent_pred_space = {str(x) for x in agent_pred_space if str(x).strip()}
        
        print(f"answer_space={sorted(list(ans_space))}", file=sys.stderr)
        print(f"baseline_pred_space={sorted(list(pred_space))}", file=sys.stderr)
        print(f"agent_pred_space={sorted(list(agent_pred_space))}", file=sys.stderr)

        index_lines.append(f"answer_match_rate={answer_match_rate:.4f}")
        index_lines.append(f"ans_space={json.dumps(sorted(list(ans_space)))}")
        index_lines.append(f"agent_pred_space={json.dumps(sorted(list(agent_pred_space)))}")
        index_lines.append(f"baseline_pred_space={json.dumps(sorted(list(pred_space)))}")
        
        baseline_diff = abs(acc_baseline - recomp_acc_baseline)
        agent_diff = abs(acc_agent - recomp_acc_agent)
        
        l1_answer_valid_rate = df1_renamed["answer"].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != "" and str(x) != "MISSING" else 0).mean()
        if l1_answer_valid_rate < 0.99:
             print(f"FAIL: L1 Answer Valid Rate {l1_answer_valid_rate:.4f} < 0.99", file=sys.stderr)
             raise ValueError("l1_answer_missing_or_blank")

        if answer_match_rate < 0.99:
             print(f"FAIL: Answer Match Rate {answer_match_rate:.4f} < 0.99", file=sys.stderr)
             raise ValueError("answer_mismatch_l1_l2")

        if baseline_diff > 1e-9:
             print(f"FAIL: Baseline Correctness Inconsistent Diff={baseline_diff}", file=sys.stderr)
             raise ValueError("baseline_correct_inconsistent")

        if agent_diff > 1e-9:
             print(f"FAIL: Agent Correctness Inconsistent Diff={agent_diff}", file=sys.stderr)
             raise ValueError("agent_correct_inconsistent")

        if df1_renamed["baseline_correct"].isin([0, 1]).all() == False:
             print("FAIL: Baseline Correct contains non-boolean values", file=sys.stderr)
             raise ValueError("baseline_correct_non_boolean")
             
        if df1_renamed["baseline_correct"].nunique() <= 1 and len(df1_renamed) > 1:
             print("FAIL: Baseline Correct is degenerate (all same value)", file=sys.stderr)
             raise ValueError("baseline_correct_degenerate_all_same")

        if pred_space and ans_space and pred_space.isdisjoint(ans_space):
             print(f"FAIL: Baseline Label Space Mismatch. Pred={pred_space} Ans={ans_space}", file=sys.stderr)
             raise ValueError("label_space_mismatch")
             
        if agent_pred_space and ans_space and agent_pred_space.isdisjoint(ans_space):
             print(f"FAIL: Agent Label Space Mismatch. Pred={agent_pred_space} Ans={ans_space}", file=sys.stderr)
             raise ValueError("agent_label_space_mismatch")
        
        missing_in_l2 = set(df1["idx"]) - set(df2["idx"])
        missing_in_l1 = set(df2["idx"]) - set(df1["idx"])
        union_len = len(set(df1["idx"]) | set(df2["idx"]))
        coverage = len(merged) / union_len if union_len > 0 else 0.0
        
        if coverage != 1.0:
            print(f"WARNING: Coverage is {coverage:.4f} (not 1.0). Matched={len(merged)} Union={union_len}", file=sys.stderr)
            print(f"Missing in Agent: {len(missing_in_l2)}", file=sys.stderr)
            print(f"Missing in Baseline: {len(missing_in_l1)}", file=sys.stderr)
        
        out_dir = project_root / "dist/outputs/tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        compare_csv_path = out_dir / "L1L2_compare.csv"
        merged.to_csv(compare_csv_path, index=False)
        
        log_dir = project_root / "dist/outputs/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        summary_path = log_dir / "L1L2_compare_summary.json"
        
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_matched": len(merged),
            "acc_baseline": float(acc_baseline) if not pd.isna(acc_baseline) else None,
            "acc_agent": float(acc_agent) if not pd.isna(acc_agent) else None,
            "trigger_rate_baseline": float(trigger_baseline) if not pd.isna(trigger_baseline) else None,
            "trigger_rate_agent": float(trigger_agent) if not pd.isna(trigger_agent) else None,
            "coverage": float(coverage),
            "missing_in_l2_count": len(missing_in_l2),
            "missing_in_l1_count": len(missing_in_l1),
            "l1_sha256": l1_sha,
            "l2_sha256": l2_sha,
            "answer_match_rate": float(answer_match_rate),
            "ans_space": sorted(list(ans_space)),
            "baseline_pred_space": sorted(list(pred_space)),
            "agent_pred_space": sorted(list(agent_pred_space))
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        
        status = "PASS"
        error_reason = None
        
    except Exception as e:
        print(f"Error running compare logic: {e}", file=sys.stderr)
        status = "FAIL"
        if 'error_reason' not in locals() or error_reason is None:
            error_reason = str(e)
        if 'compare_csv_path' not in locals(): compare_csv_path = None
        if 'summary_path' not in locals(): summary_path = None

    outputs_meta = []
    if compare_csv_path and compare_csv_path.exists():
        files_to_zip.append((compare_csv_path, "dist/outputs/tables/L1L2_compare.csv"))
        sha = _sha256_upper_bytes(compare_csv_path.read_bytes())
        outputs_meta.append(f"compare_csv={sha}")
        
    if summary_path and summary_path.exists():
        files_to_zip.append((summary_path, "dist/outputs/logs/L1L2_compare_summary.json"))
        sha = _sha256_upper_bytes(summary_path.read_bytes())
        outputs_meta.append(f"summary={sha}")
        
    index_lines.append(f"outputs={'|'.join(outputs_meta)}")
    index_lines.append("trigger_semantics=TRIGGERED_MEANS_PZCR_BRANCH")
    if status == "FAIL" and error_reason:
        index_lines.append(f"error={error_reason}")
    index_lines.append(f"status={status}")
    index_lines.append("")
    
    index_lines.append("# Evidence Package Manifest")
    # G3: Explicitly list INDEX as it is required in manifest
    index_lines.append("file=evidence_compare/INDEX.txt status=GENERATED")
    
    manifest_map = {}
    for abs_p, arcname in files_to_zip:
        if abs_p.exists():
            data = abs_p.read_bytes()
            sha = _sha256_upper_bytes(data)
            sz = len(data)
            index_lines.append(f"file={arcname} bytes={sz} sha256={sha}")
            manifest_map[arcname] = sha
        else:
            index_lines.append(f"file={arcname} status=MISSING")
            
    disk_script_sha256 = _sha256_upper_bytes(script_path.read_bytes())
    index_lines.append(f"repo_script_sha256={disk_script_sha256}")

    index_path = evidence_dir / "INDEX.txt"
    index_content = "\n".join(index_lines) + "\n"
    index_path.write_text(index_content, encoding="utf-8")
    
    placeholder = "index_body_sha256=__PLACEHOLDER__"
    insert_idx = 1
    index_lines.insert(insert_idx, placeholder)
    
    lines_for_hash = index_lines[:insert_idx] + index_lines[insert_idx+1:]
    body_content = "\n".join(lines_for_hash) + "\n"
    body_sha = _sha256_upper_bytes(body_content.encode("utf-8"))
    
    index_lines[insert_idx] = f"index_body_sha256={body_sha}"
    final_content = "\n".join(index_lines) + "\n"
    index_path.write_text(final_content, encoding="utf-8")
    
    files_to_zip.insert(0, (index_path, "evidence_compare/INDEX.txt"))
    zip_path = project_root / "dist/outputs/evidence_compare_package.zip"
    if zip_path.exists():
        zip_path.unlink()
        
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_p, arcname in files_to_zip:
            if abs_p.exists():
                zf.write(abs_p, arcname)
                
    zip_check_pass = True
    missing_entries = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_files = set(zf.namelist())
        # FIX: expected_files should only contain existing files (which are in manifest_map)
        expected_files = set(manifest_map.keys())
        expected_files.add("evidence_compare/INDEX.txt")
        
        missing_entries = expected_files - zip_files
        if missing_entries:
            zip_check_pass = False
            print(f"FAIL: Zip missing entries: {missing_entries}", file=sys.stderr)
            status = "FAIL"
            
        for arcname, expected_sha in manifest_map.items():
            if arcname not in zip_files: continue
            try:
                actual_sha = _sha256_upper_bytes(zf.read(arcname))
                if actual_sha != expected_sha:
                    zip_check_pass = False
            except KeyError:
                zip_check_pass = False
                
        if "dist/scripts/04_eval_baseline.py" in zip_files:
             script_in_zip_sha = _sha256_upper_bytes(zf.read("dist/scripts/04_eval_baseline.py"))
             if script_in_zip_sha != disk_script_sha256:
                 zip_check_pass = False
                 print(f"FAIL: Script inside zip mismatch. Zip={script_in_zip_sha} Disk={disk_script_sha256}", file=sys.stderr)
                 status = "FAIL"
                 
    script_sha_match = (script_sha256 == disk_script_sha256)
    
    # Real Index Verification
    index_match = False
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "evidence_compare/INDEX.txt" in zf.namelist():
            if zf.read("evidence_compare/INDEX.txt") == index_path.read_bytes():
                index_match = True
    
    final_lines = []
    final_lines.append("")
    final_lines.append("# Self-Check Results")
    final_lines.append(f"zip_selfcheck={'PASS' if zip_check_pass else 'FAIL'}")
    if not zip_check_pass and missing_entries:
         final_lines.append(f"error=zip_manifest_missing_entries:{','.join(sorted(list(missing_entries)))}")
         final_lines.append("status=FAIL")
    
    if not zip_check_pass:
         with zipfile.ZipFile(zip_path, "r") as zf:
             if "dist/scripts/04_eval_baseline.py" in zf.namelist():
                 s_sha = _sha256_upper_bytes(zf.read("dist/scripts/04_eval_baseline.py"))
                 if s_sha != disk_script_sha256:
                     final_lines.append("error=script_sha_mismatch_in_zip")
                     final_lines.append("status=FAIL")

    final_lines.append(f"script_sha_match={'PASS' if script_sha_match else 'FAIL'}")
    final_lines.append(f"index_in_zip_matches_disk={'PASS' if index_match else 'FAIL'}")
    
    with open(index_path, "a", encoding="utf-8") as f:
        f.write("\n".join(final_lines) + "\n")
        
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_p, arcname in files_to_zip:
            if abs_p.exists():
                zf.write(abs_p, arcname)
                
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.read("evidence_compare/INDEX.txt") != index_path.read_bytes():
            print("FAIL: Index consistency check failed", file=sys.stderr)
            return 1
            
    print(f"evidence_dir={evidence_dir}")
    print(f"evidence_zip={zip_path}")
    
    if not zip_check_pass or not script_sha_match or not index_match or status != "PASS":
        print("compare_suite=FAIL")
        return 1
        
    print("compare_suite=PASS")
    return 0


def _parse_index(path: Path) -> Dict[str, str]:
    if not path.exists(): return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    res = {}
    for line in lines:
        if "=" in line and not line.startswith("#"):
            parts = line.split("=", 1)
            res[parts[0].strip()] = parts[1].strip()
    return res

def _backup_artifacts(project_root: Path) -> Dict[str, Optional[bytes]]:
    backup = {}
    csv_path = project_root / "dist/outputs/tables/L1_baseline.csv"
    snap_path = project_root / "dist/outputs/tables/L1_baseline_snapshot.json"
    
    backup["csv"] = csv_path.read_bytes() if csv_path.exists() else None
    backup["snap"] = snap_path.read_bytes() if snap_path.exists() else None
    return backup

def _restore_artifacts(project_root: Path, backup: Dict[str, Optional[bytes]]) -> None:
    csv_path = project_root / "dist/outputs/tables/L1_baseline.csv"
    snap_path = project_root / "dist/outputs/tables/L1_baseline_snapshot.json"
    
    if backup["csv"] is not None:
        csv_path.write_bytes(backup["csv"])
    elif csv_path.exists():
        csv_path.unlink()
        
    if backup["snap"] is not None:
        snap_path.write_bytes(backup["snap"])
    elif snap_path.exists():
        snap_path.unlink()

def _run_acceptance_audit() -> int:
    project_root = _bootstrap_src()
    script_path = Path(__file__).resolve()
    script_bytes = script_path.read_bytes()
    script_sha256 = _sha256_upper_bytes(script_bytes)
    
    report_lines = []
    report_lines.append(f"acceptance_run_utc={datetime.utcnow().isoformat()}Z")
    report_lines.append(f"executing_file={script_path}")
    report_lines.append(f"python_version={sys.version.split()[0]}")
    report_lines.append(f"platform={platform.platform()}")
    report_lines.append(f"script_sha256={script_sha256}")
    
    results = {
        "score": 0,
        "final_verdict": "FAIL",
        "failed_gates": [],
        "remediations": [],
        "gates": {},
        "measurements": {
            "script_sha256": script_sha256,
            "pre_csv_sha": None,
            "pre_snap_sha": None,
            "post_csv_sha": None,
            "post_snap_sha": None
        }
    }
    
    # Artifacts Found
    report_lines.append("artifacts_found:")
    check_paths = [
        project_root / "dist/scripts/04_eval_baseline.py",
        project_root / "dist/outputs/evidence_compare/INDEX.txt",
        project_root / "dist/outputs/evidence_compare_package.zip",
        project_root / "dist/outputs/tables/L1_baseline.csv",
        project_root / "dist/outputs/tables/L2_agent.csv",
        project_root / "dist/outputs/tables/L1L2_compare.csv",
        project_root / "dist/outputs/logs/L1L2_compare_summary.json"
    ]
    for p in check_paths:
        if p.exists():
            report_lines.append(f"  {p.name}: exists bytes={p.stat().st_size} sha256={_sha256_upper_bytes(p.read_bytes())}")
        else:
            report_lines.append(f"  {p.name}: MISSING")
            
    # Capture pre-state
    pre_backup = _backup_artifacts(project_root)
    if pre_backup["csv"] is not None: 
        results["measurements"]["pre_csv_sha"] = _sha256_upper_bytes(pre_backup["csv"])
    if pre_backup["snap"] is not None: 
        results["measurements"]["pre_snap_sha"] = _sha256_upper_bytes(pre_backup["snap"])
        
    try:
        # T0. Compile
        t0_pass = True
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "compileall", "-q", str(script_path)], 
                capture_output=True
            )
            if proc.returncode != 0:
                t0_pass = False
        except Exception:
            t0_pass = False
            
        results["gates"]["T0"] = t0_pass
        if t0_pass:
            report_lines.append("T0: compile=PASS")
        else:
            report_lines.append("T0: compile=FAIL reason=compile_error")
            results["failed_gates"].append("T0")

        # T1. Compare Suite FAIL (Missing L1)
        l1_csv = project_root / "dist/outputs/tables/L1_baseline.csv"
        if l1_csv.exists():
            l1_csv.unlink()
            
        # Run and capture
        f_stdout = io.StringIO()
        f_stderr = io.StringIO()
        try:
            with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                _run_compare_suite()
        except Exception:
            pass
            
        t1_out = f_stdout.getvalue()
        
        # Verify T1 Gates
        index_path = project_root / "dist/outputs/evidence_compare/INDEX.txt"
        zip_path = project_root / "dist/outputs/evidence_compare_package.zip"
        
        # G1
        last_line = t1_out.strip().splitlines()[-1] if t1_out.strip() else ""
        g1_pass = index_path.exists() and zip_path.exists() and last_line == "compare_suite=FAIL"
        results["gates"]["G1"] = g1_pass
        
        g2_pass = False
        g3_pass = False
        g4_pass = False
        
        if index_path.exists():
            idx = _parse_index(index_path)
            disk_sha = _sha256_upper_bytes(script_path.read_bytes())
            
            # G0: Invariant (Moved from end to gate)
            # Actually, G0 logic is pre vs post. We need to check it at the end.
            # But we can define the gate result variable here for reporting.
            
            # G2: 3-way lock
            repo_sha = idx.get("repo_script_sha256")
            zip_sha = None
            if zip_path.exists():
                 try:
                     with zipfile.ZipFile(zip_path, "r") as zf:
                         if "dist/scripts/04_eval_baseline.py" in zf.namelist():
                             zip_sha = _sha256_upper_bytes(zf.read("dist/scripts/04_eval_baseline.py"))
                 except Exception:
                     pass
            
            results["measurements"]["g2_disk_sha"] = disk_sha
            results["measurements"]["g2_repo_sha"] = repo_sha
            results["measurements"]["g2_zip_sha"] = zip_sha
            
            if disk_sha == repo_sha == zip_sha and disk_sha is not None:
                 g2_pass = True
            
            # G3: Manifest
            manifest_entries = [l for l in index_path.read_text(encoding="utf-8").splitlines() if "file=" in l]
            required_files = [
                "dist/outputs/tables/L1_baseline.csv",
                "dist/outputs/tables/L2_agent.csv",
                "dist/outputs/tables/L1_baseline_snapshot.json",
                "dist/scripts/04_eval_baseline.py",
                "evidence_compare/INDEX.txt"
            ]
            found_req = 0
            correct_status = True
            
            for rf in required_files:
                found_this = False
                for entry in manifest_entries:
                    if f"file={rf}" in entry:
                        found_this = True
                        if rf == "dist/outputs/tables/L1_baseline.csv":
                            if "status=MISSING" not in entry:
                                correct_status = False
                        break
                if found_this:
                    found_req += 1
                    
            if found_req >= len(required_files) and correct_status:
                g3_pass = True
                
            # G4: Index Body SHA
            if _verify_index_body_sha256(index_path):
                g4_pass = True
                
        results["gates"]["G2"] = g2_pass
        results["gates"]["G3"] = g3_pass
        results["gates"]["G4"] = g4_pass

        if g1_pass and g2_pass and g3_pass and g4_pass:
            report_lines.append("T1: PASS")
        else:
            report_lines.append(f"T1: FAIL G1={g1_pass} G2={g2_pass} G3={g3_pass} G4={g4_pass}")
            if not g1_pass: results["failed_gates"].append("G1"); results["remediations"].append("Check compare_suite=FAIL output and evidence existence")
            if not g2_pass: results["failed_gates"].append("G2"); results["remediations"].append("Check 3-way script SHA lock")
            if not g3_pass: results["failed_gates"].append("G3"); results["remediations"].append("Check manifest completeness/MISSING status")
            if not g4_pass: results["failed_gates"].append("G4"); results["remediations"].append("Check index_body_sha256 verification")

        # T2. Compare Suite PASS
        # Generate aligned L1
        l2_csv = project_root / "dist/outputs/tables/L2_agent.csv"
        g5_pass = False
        
        if not l2_csv.exists():
            report_lines.append("T2: FAIL reason=L2_agent_csv_missing_cannot_align")
            results["failed_gates"].append("G5")
        else:
            import csv
            l2_rows = []
            with l2_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                l2_rows = list(reader)
                
            l2_answers = {r.get("answer", "").strip().upper() for r in l2_rows if r.get("answer")}
            l2_answers.discard("")
            l2_answers.discard("MISSING")
            
            l2_space = set()
            l2_space_str = "UNKNOWN"
            if l2_answers.issubset({"A", "B", "C", "D"}) and l2_answers:
                 l2_space = {"A", "B", "C", "D"}
                 l2_space_str = "MCQ"
            elif l2_answers.issubset({"0", "1"}) and l2_answers:
                 l2_space = {"0", "1"}
                 l2_space_str = "Binary"
                 
            results["measurements"]["l2_space_str"] = l2_space_str
            
            if l2_space_str == "UNKNOWN":
                 report_lines.append("T2: FAIL reason=l2_answer_space_unknown")
                 results["failed_gates"].append("G5")
            else:
                # Generate L1
                l1_rows = []
                for i, r in enumerate(l2_rows):
                    ans = r.get("answer", "").strip().upper()
                    pred = ans
                    correct = 1
                    
                    if i == 0 and len(l2_rows) > 1:
                        options = ["A", "B", "C", "D", "0", "1"]
                        for opt in options:
                            if opt != ans:
                                pred = opt
                                correct = 0
                                break
                    
                    l1_rows.append({
                        "idx": r["idx"],
                        "split": "test",
                        "answer": ans,
                        "pred": pred,
                        "correct": correct,
                        "method": "aligned_baseline",
                        "triggered": 0
                    })
                    
                with l1_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["idx", "split", "answer", "pred", "correct", "method", "triggered"])
                    writer.writeheader()
                    writer.writerows(l1_rows)
                    
                snap_path = project_root / "dist/outputs/tables/L1_baseline_snapshot.json"
                input_path = project_root / "dist/data/mmad/mmad_minireal_512.jsonl"
                if input_path.exists():
                    input_sha = _sha256_upper_bytes(input_path.read_bytes())
                else:
                    input_sha = "MOCK"
                    
                snap = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "input_sha256": input_sha,
                    "row_count": len(l1_rows),
                    "acc": sum(r["correct"] for r in l1_rows) / len(l1_rows) if l1_rows else 0.0,
                    "method": "aligned_baseline"
                }
                snap_path.write_text(json.dumps(snap), encoding="utf-8")
                
                f_stdout = io.StringIO()
                f_stderr = io.StringIO()
                try:
                    with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                        _run_compare_suite()
                except Exception:
                    pass
                    
                t2_out = f_stdout.getvalue()
                
                # Verify G5
                last_line_t2 = t2_out.strip().splitlines()[-1] if t2_out.strip() else ""
                check_stdout = last_line_t2 == "compare_suite=PASS"
                
                idx = _parse_index(index_path)
                amr = float(idx.get("answer_match_rate", 0.0))
                check_amr = amr >= ACCEPT_V21_SPEC["thresholds"]["answer_match_rate"]
                
                ans_space_loaded = set(json.loads(idx.get("ans_space", "[]")))
                check_space = (ans_space_loaded == l2_space)
                
                check_outputs = True
                comp_csv = project_root / "dist/outputs/tables/L1L2_compare.csv"
                summ_json = project_root / "dist/outputs/logs/L1L2_compare_summary.json"
                if not comp_csv.exists() or not summ_json.exists(): check_outputs = False
                
                outputs_val = idx.get("outputs", "")
                if "compare_csv=" not in outputs_val: check_outputs = False
                if "summary=" not in outputs_val: check_outputs = False
                
                check_zip = True
                if zip_path.exists():
                     with zipfile.ZipFile(zip_path, "r") as zf:
                         if "dist/outputs/tables/L1L2_compare.csv" not in zf.namelist(): check_zip = False
                         if "dist/outputs/logs/L1L2_compare_summary.json" not in zf.namelist(): check_zip = False
                else:
                     check_zip = False
                
                check_selfcheck = idx.get("zip_selfcheck") == "PASS"
                
                results["measurements"]["g5_amr"] = amr
                results["measurements"]["g5_space_match"] = check_space
                
                if check_stdout and check_amr and check_space and check_outputs and check_zip and check_selfcheck:
                    g5_pass = True
                    report_lines.append("T2: PASS")
                else:
                    report_lines.append(f"T2: FAIL Stdout={check_stdout} AMR={check_amr} Space={check_space} Out={check_outputs} Zip={check_zip} Self={check_selfcheck}")
                    results["failed_gates"].append("G5")
                    results["remediations"].append("Check G5 requirements (PASS, AMR>=0.99, space match, outputs, zip)")

        results["gates"]["G5"] = g5_pass

        # Baseline Suite Checks
        f_stdout = io.StringIO()
        f_stderr = io.StringIO()
        try:
            with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                _run_baseline_suite(allow_gt_cache_from_l2=False)
        except Exception:
            pass
        
        base_idx_path = project_root / "dist/outputs/evidence_baseline/INDEX.txt"
        g6_pass = False
        
        if not base_idx_path.exists():
             report_lines.append("Baseline: FAIL (No INDEX)")
             results["failed_gates"].append("G6")
        else:
            b_idx = _parse_index(base_idx_path)
            req_fields = ACCEPT_V21_SPEC["required_diagnostics"]
            diag_ok = True
            for f in req_fields:
                if f not in b_idx:
                    diag_ok = False
                    break
            
            if diag_ok:
                g6_pass = True
                if b_idx.get("status") != "PASS":
                    # Check strict fail logic
                    g6_pass = False
                    reason = b_idx.get('error', 'unknown_error')
                    results["remediations"].append(f"Baseline suite status != PASS ({reason})")
                
                report_lines.append(f"Baseline: {'PASS' if g6_pass else 'FAIL'} (Status={b_idx.get('status')} Error={b_idx.get('error', 'None')})")
            else:
                report_lines.append("Baseline: FAIL (Missing Diagnostics)")
                results["remediations"].append("Missing baseline diagnostics keys")
                 
        results["gates"]["G6"] = g6_pass
        if not g6_pass and "G6" not in results["failed_gates"]:
            results["failed_gates"].append("G6")

    finally:
        # Restore pre-state
        _restore_artifacts(project_root, pre_backup)
        
        # Verify restoration
        post_backup = _backup_artifacts(project_root)
        post_csv_sha = _sha256_upper_bytes(post_backup["csv"]) if post_backup["csv"] is not None else None
        post_snap_sha = _sha256_upper_bytes(post_backup["snap"]) if post_backup["snap"] is not None else None
        
        results["measurements"]["post_csv_sha"] = post_csv_sha
        results["measurements"]["post_snap_sha"] = post_snap_sha
        
        # G0 Check
        invariant_ok = (results["measurements"].get("pre_csv_sha") == post_csv_sha) and \
                       (results["measurements"].get("pre_snap_sha") == post_snap_sha)
                       
        results["gates"]["G0"] = invariant_ok
        if not invariant_ok:
            results["failed_gates"].append("G0")
            results["remediations"].append("Side-effects detected on baseline artifacts")
            report_lines.append("G0: FAIL (Side-effects detected)")
        else:
            report_lines.insert(report_lines.index("T0: compile=PASS") + 1 if "T0: compile=PASS" in report_lines else len(report_lines), "G0: PASS")

    # Scoring
    score = 0
    if results["gates"].get("T0"): score += ACCEPT_V21_SPEC["gates"]["T0"]["score"]
    if results["gates"].get("G0"): score += ACCEPT_V21_SPEC["gates"]["G0"]["score"]
    if results["gates"].get("G1"): score += ACCEPT_V21_SPEC["gates"]["G1"]["score"]
    if results["gates"].get("G2"): score += ACCEPT_V21_SPEC["gates"]["G2"]["score"]
    if results["gates"].get("G3"): score += ACCEPT_V21_SPEC["gates"]["G3"]["score"]
    if results["gates"].get("G4"): score += ACCEPT_V21_SPEC["gates"]["G4"]["score"]
    if results["gates"].get("G5"): score += ACCEPT_V21_SPEC["gates"]["G5"]["score"]
    if results["gates"].get("G6"): score += ACCEPT_V21_SPEC["gates"]["G6"]["score"]
    
    if results["failed_gates"]:
        final_verdict = "FAIL"
    else:
        final_verdict = "PASS"
        
    results["score"] = score
    results["final_verdict"] = final_verdict
    
    # Extra Evidence
    if (project_root / "dist/outputs/evidence_compare_package.zip").exists():
        with zipfile.ZipFile(project_root / "dist/outputs/evidence_compare_package.zip", "r") as zf:
             sha = _sha256_upper_bytes((project_root / "dist/outputs/evidence_compare_package.zip").read_bytes())
             report_lines.append(f"zip_sha256={sha}")
             report_lines.append(f"zip_namelist={json.dumps(zf.namelist())}")
             
    # INDEX Dump (Compare)
    idx_p = project_root / "dist/outputs/evidence_compare/INDEX.txt"
    if idx_p.exists():
        report_lines.append("Compare INDEX Dump:")
        report_lines.append(idx_p.read_text(encoding="utf-8").strip())
        
    report_lines.append(f"FINAL_SCORE={score}/11")
    report_lines.append(f"FINAL_VERDICT={final_verdict}")
    report_lines.append(f"FAILED_GATES={results['failed_gates']}")
    report_lines.append(f"REMEDIATIONS={results['remediations']}")
    
    # Print Report
    report_content = "\n".join(report_lines)
    print(report_content)
    
    # JSON Output
    # Self-Check JSON
    final_json_str = ""
    try:
        final_json_str = json.dumps(results)
        # Double check it parses back
        json.loads(final_json_str)
    except Exception as e:
        # Emergency fail
        final_verdict = "FAIL"
        emergency = {
            "score": 0,
            "final_verdict": "FAIL",
            "failed_gates": ["JSON_CHECK"],
            "remediations": [f"ACCEPTANCE_JSON serialization failed: {e}"],
            "gates": {},
            "measurements": {
                "script_sha256": script_sha256,
                "pre_csv_sha": results["measurements"].get("pre_csv_sha"),
                "pre_snap_sha": results["measurements"].get("pre_snap_sha"),
                "post_csv_sha": results["measurements"].get("post_csv_sha"),
                "post_snap_sha": results["measurements"].get("post_snap_sha")
            }
        }
        final_json_str = json.dumps(emergency)
        
    print(f"ACCEPTANCE_JSON={final_json_str}")
    
    sys.stdout.flush()
    
    # Final Line
    print(f"acceptance_audit={final_verdict}")
    sys.stdout.flush()
    
    return 0 if final_verdict == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_suite", action="store_true")
    parser.add_argument("--compare_suite", action="store_true")
    parser.add_argument("--acceptance_audit", action="store_true")
    parser.add_argument("--allow_gt_cache_from_l2", action="store_true")
    args = parser.parse_args()
    
    if args.acceptance_audit:
        return _run_acceptance_audit()

    if args.baseline_suite:
        return _run_baseline_suite(allow_gt_cache_from_l2=args.allow_gt_cache_from_l2)
        
    if args.compare_suite:
        return _run_compare_suite()
        
    print("Usage: python 04_eval_baseline.py --baseline_suite | --compare_suite | --acceptance_audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
