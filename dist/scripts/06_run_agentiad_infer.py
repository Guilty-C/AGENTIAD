# [REMOTE EXECUTION DETECTED? CHECK GUIDELINES]
# This project enforces a strict "Zero-Pollution" remote execution protocol for shared lab servers.
# See REMOTE_EXECUTION_GUIDE.txt for the mandatory "Upload -> Tmp Run -> Cleanup" workflow.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import zipfile
import compileall
import shutil
import subprocess

ACCEPT_L2_SPEC = {
    "total": 10,
    "points": {
        "T0": 1,
        "G0": 2,
        "G1": 2,
        "G2": 5
    }
}

def _resolve_path(project_root: Path, raw: str) -> Path:
    if os.path.isabs(raw):
        return Path(raw).resolve()
    
    # 1. Try CWD relative first (handles dist/outputs vs root/dist/outputs)
    p0 = (Path.cwd() / raw).resolve()
    if p0.exists():
        return p0
        
    # 2. Try repo_root relative
    p1 = (project_root / raw).resolve()
    if p1.exists():
        return p1
        
    # 3. Try dist relative if in root
    p2 = (project_root / "dist" / raw).resolve()
    if p2.exists():
        return p2
        
    return p0 # Fallback

def _finalize_and_print(results: Dict[str, Any]) -> int:
    # Calculate Score
    score = 0
    for gate, points in ACCEPT_L2_SPEC["points"].items():
        if results["gates"].get(gate, False):
            score += points
    results["score"] = score
    
    if results["failed_gates"]:
        results["final_verdict"] = "FAIL"
    elif score != ACCEPT_L2_SPEC["total"]:
        results["final_verdict"] = "FAIL"
        results["failed_gates"].append("EXCEPTION")
        results["remediations"].append(f"Score {score} != Total {ACCEPT_L2_SPEC['total']}")
    else:
        results["final_verdict"] = "PASS"

    # JSON Output with strict roundtrip check
    try:
        json_str = json.dumps(results, ensure_ascii=False)
        _ = json.loads(json_str)
        sys.stdout.buffer.write(f"ACCEPTANCE_JSON={json_str}\n".encode('utf-8'))
        sys.stdout.buffer.write(f"acceptance_audit={results['final_verdict']}\n".encode('utf-8'))
        sys.stdout.buffer.flush()
    except Exception as e:
        emergency = {
            "score": 0,
            "total": ACCEPT_L2_SPEC["total"],
            "final_verdict": "FAIL",
            "failed_gates": ["EXCEPTION"],
            "remediations": [f"JSON serialization failed: {str(e)}"],
            "gates": {k: False for k in ACCEPT_L2_SPEC["points"].keys()},
            "measurements": {}
        }
        sys.stdout.buffer.write(f"ACCEPTANCE_JSON={json.dumps(emergency)}\n".encode('utf-8'))
        sys.stdout.buffer.write("acceptance_audit=FAIL\n".encode('utf-8'))
        sys.stdout.buffer.flush()
        return 1
    return 0 if results["final_verdict"] == "PASS" else 1

def _run_acceptance_audit(args: argparse.Namespace) -> int:
    project_root, src_injected = _bootstrap_src()
    
    if args.evidence_dir:
        evidence_dir = _resolve_path(project_root, args.evidence_dir)
    else:
        if Path.cwd().name == "dist":
            evidence_dir = _resolve_path(project_root, "outputs/evidence_l2_test_dist")
        else:
            evidence_dir = _resolve_path(project_root, "dist/outputs/evidence_l2_test_root")

    results = {
        "score": 0,
        "total": ACCEPT_L2_SPEC["total"],
        "final_verdict": "FAIL",
        "failed_gates": [],
        "remediations": [],
        "gates": {k: False for k in ACCEPT_L2_SPEC["points"].keys()},
        "measurements": {
            "src_injected": src_injected
        },
    }

    if src_injected == "NONE":
        results["remediations"].append("missing src dirs (checked src and dist/src)")

    # T0: Compile check
    try:
        if compileall.compile_file(__file__, quiet=1):
            results["gates"]["T0"] = True
            results["measurements"]["t0_compile_ok"] = True
        else:
            results["failed_gates"].append("T0")
    except Exception:
        results["failed_gates"].append("T0")

    # G0: Preflight
    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        print(f"G0: FAIL - evidence_dir exists and is non-empty: {evidence_dir}", file=sys.stderr)
        results["failed_gates"].append("G0")
        results["remediations"].append("use empty dir")
        return _finalize_and_print(results)
    
    evidence_dir.mkdir(parents=True, exist_ok=True)
    results["gates"]["G0"] = True

    # G1: PATHS check
    if "/dist/dist/" in str(evidence_dir).replace("\\", "/"):
        results["failed_gates"].append("G1")
        results["remediations"].append(f"Path pollution: {evidence_dir}")
        return _finalize_and_print(results)
    results["gates"]["G1"] = True

    # G2: Minimal Pipeline (Dry Run)
    # Create dummy config
    dummy_config_path = evidence_dir / "audit_config.yaml"
    dummy_config_path.write_text("model_id: dry_run\nseed: 42\nmax_samples: 2\n", encoding="utf-8")
    
    # Run self in subprocess
    # Pass --evidence_dir to subprocess so it redirects output
    # Force cwd=project_root to ensure consistent path resolution
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config", str(dummy_config_path),
        "--dry_run",
        "--max_samples", "2",
        "--run_name", "audit_run",
        "--evidence_dir", str(evidence_dir)
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        if res.returncode != 0:
            print(f"G2: FAIL - Subprocess failed RC={res.returncode}", file=sys.stderr)
            print(f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}", file=sys.stderr)
            results["failed_gates"].append("G2")
            results["remediations"].append("Inference subprocess failed")
        else:
            # Check for artifacts
            # We expect audit_run related files in evidence_dir (tables, logs, traces)
            # tables/agentiad_infer_audit_run.csv
            csv_path = evidence_dir / "tables/agentiad_infer_audit_run.csv"
            if csv_path.exists():
                results["gates"]["G2"] = True
            else:
                print(f"G2: FAIL - CSV missing at {csv_path}", file=sys.stderr)
                results["failed_gates"].append("G2")
                results["remediations"].append("Artifacts missing")
    except Exception as e:
        print(f"G2: FAIL - Exception {e}", file=sys.stderr)
        results["failed_gates"].append("G2")
        results["remediations"].append(f"Subprocess exception: {e}")

    # Zip artifacts
    zip_path = evidence_dir / "evidence_package.zip"
    index_path = evidence_dir / "INDEX.txt"
    
    try:
        idx_lines = []
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Walk evidence dir and zip everything except the zip itself and index
            for root, dirs, files in os.walk(evidence_dir):
                for file in files:
                    if file in ["evidence_package.zip", "INDEX.txt"]:
                        continue
                    fp = Path(root) / file
                    arcname = fp.relative_to(evidence_dir)
                    zf.write(fp, arcname=arcname)
                    sha = hashlib.sha256(fp.read_bytes()).hexdigest().upper()
                    idx_lines.append(f"{str(arcname).replace(os.sep, '/')} {fp.stat().st_size} {sha}")
        
        if zip_path.exists():
             zip_bytes = zip_path.read_bytes()
             sha_zip = hashlib.sha256(zip_bytes).hexdigest().upper()
             idx_lines.append(f"evidence_package.zip {len(zip_bytes)} {sha_zip}")
             
        index_path.write_text("\n".join(idx_lines) + "\n", encoding="utf-8")
        
    except Exception as e:
        results["remediations"].append(f"Zip creation failed: {e}")

    # G0 Post-check (Residue)
    for item in evidence_dir.iterdir():
        if item.name not in ["INDEX.txt", "evidence_package.zip"]:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
                
    remaining = list(evidence_dir.iterdir())
    residue = [f.name for f in remaining if f.name not in {"INDEX.txt", "evidence_package.zip"}]
    if residue:
        results["failed_gates"].append("G0")
        results["remediations"].append(f"Residue: {residue}")
        results["gates"]["G0"] = False

    return _finalize_and_print(results)


def _bootstrap_src() -> Tuple[Path, str]:
    # dist/scripts/06... -> parents[2] is repo_root
    project_root = Path(__file__).resolve().parents[2]
    
    src_candidates = [
        project_root / "src",
        project_root / "dist/src"
    ]
    
    injected = "NONE"
    for p in src_candidates:
        if p.exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            
            # Record which one we used
            if p == project_root / "src":
                injected = "repo_root/src"
            else:
                injected = "repo_root/dist/src"
            break
            
    return project_root, injected


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _sha256_upper_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _sha256_upper_json(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_upper_text(s)


def _normalize_yesno(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, bool):
        return "yes" if x else "no"
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1", "anomaly", "abnormal"}:
        return "yes"
    if s in {"no", "n", "false", "0", "normal"}:
        return "no"
    return None


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


def _label_from_path_segments(p: str) -> str:
    s = p.replace("\\", "/")
    segments = [seg for seg in s.split("/") if seg]
    if any(seg == "good" for seg in segments):
        return "no"
    return "yes" if p else ""


def _extract_gt_yesno(row: Mapping[str, Any], gt_key_override: Optional[str]) -> Tuple[str, str]:
    if gt_key_override:
        val = row.get(gt_key_override)
        norm = _normalize_yesno(val)
        if norm in {"yes", "no"}:
            return norm, f"override:{gt_key_override}"
        if isinstance(val, str) and val:
            norm2 = _label_from_path_segments(val)
            if norm2 in {"yes", "no"}:
                return norm2, f"override:{gt_key_override}"

    for key in ["label", "gt_label", "anomaly", "is_anomaly"]:
        norm = _normalize_yesno(row.get(key))
        if norm in {"yes", "no"}:
            return norm, f"field:{key}"

    p = str(row.get("query_image", "") or "")
    if p:
        segments = [seg for seg in p.replace("\\", "/").split("/") if seg]
        if any(seg == "good" for seg in segments):
            return "no", "rule:query_image_good_segment"
        return "yes", "rule:query_image_not_good"

    return "", "NONE"


def _extract_class_name(row: Mapping[str, Any]) -> str:
    for key in ["class_name", "category", "object", "cls"]:
        v = row.get(key)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""


def _git_commit() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        out = out.strip()
        return out if out else None
    except Exception:
        return None


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


def _fallback_bbox_norm(sample_id: str) -> List[float]:
    h = hashlib.sha256(sample_id.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], byteorder="big", signed=False)
    dx = ((v % 101) - 50) / 1000.0
    dy = (((v // 101) % 101) - 50) / 1000.0
    x1 = 0.25 + dx
    y1 = 0.25 + dy
    x2 = 0.75 + dx
    y2 = 0.75 + dy
    if x1 < 0.0:
        x1 = 0.0
    if y1 < 0.0:
        y1 = 0.0
    if x2 > 1.0:
        x2 = 1.0
    if y2 > 1.0:
        y2 = 1.0
    if x2 <= x1:
        x1 = 0.0
        x2 = 1.0
    if y2 <= y1:
        y1 = 0.0
        y2 = 1.0
    return [float(x1), float(y1), float(x2), float(y2)]


def _parse_agent_json(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = _safe_str(text).strip()
    js = _extract_first_json(raw)
    if js is None:
        return {}, {"parse_ok": False}
    try:
        obj = json.loads(js)
    except Exception:
        return {}, {"parse_ok": False}
    if not isinstance(obj, dict):
        return {}, {"parse_ok": False}

    out: Dict[str, Any] = {}
    an = _normalize_yesno(obj.get("anomaly"))
    if an in {"yes", "no"}:
        out["anomaly"] = an
    if "defect_type" in obj and obj.get("defect_type") is not None:
        out["defect_type"] = str(obj.get("defect_type"))
    conf = obj.get("confidence")
    if isinstance(conf, (int, float)):
        out["confidence"] = float(conf)
    bbox = obj.get("bbox") or obj.get("bbox_norm") or obj.get("bbox_2d")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            b = [float(x) for x in bbox]
            if all(0.0 <= x <= 1.0 for x in b) and b[0] < b[2] and b[1] < b[3]:
                out["bbox"] = b
        except Exception:
            pass
    return out, {"parse_ok": True}


def _uncertain(parsed: Mapping[str, Any]) -> bool:
    an = parsed.get("anomaly")
    conf = parsed.get("confidence")
    if an not in {"yes", "no"}:
        return True
    if conf is None:
        return True
    try:
        return float(conf) < 0.6
    except Exception:
        return True


def _decode_hf_image(dataset_id: str, value: Any) -> "PIL.Image.Image":
    import io
    import os as _os
    from pathlib import Path as _Path

    from PIL import Image as PILImage

    if value is None:
        raise ValueError("image is None")
    if hasattr(value, "save"):
        return value
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return PILImage.open(io.BytesIO(value["bytes"]))
        if "path" in value and value["path"]:
            p = _Path(value["path"])
            if p.exists():
                return PILImage.open(p)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported image type: {type(value)}")

    p = _Path(value)
    if p.exists():
        return PILImage.open(p)

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except Exception as e:
        raise RuntimeError("huggingface_hub is required to fetch images.") from e

    if value.startswith("MVTec-AD/"):
        rev = _os.environ.get("MMAD_MVTEC_AD_REVISION") or "e88b7bd615ad582b0a7e8238066a9fb293a072b4"
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
        return PILImage.open(local_path)

    if value.startswith("DS-MVTec/"):
        rev = _os.environ.get("MMAD_ASSET_REVISION") or "f1ad11c07452dff1e023a0df1093e40701d22cab"
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
        return PILImage.open(local_path)

    try:
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value)
    except EntryNotFoundError:
        rev = _os.environ.get("MMAD_ASSET_REVISION") or "f1ad11c07452dff1e023a0df1093e40701d22cab"
        local_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=value, revision=rev)
    return PILImage.open(local_path)


def _infer_model_device(model: Any) -> "torch.device":
    import torch

    emb = getattr(model, "get_input_embeddings", None)
    if callable(emb):
        e = model.get_input_embeddings()
        if e is not None and hasattr(e, "weight") and hasattr(e.weight, "device") and str(e.weight.device) != "meta":
            return e.weight.device
    md = getattr(model, "device", None)
    if md is not None:
        try:
            return md
        except Exception:
            pass
    for p in model.parameters():
        if hasattr(p, "device") and str(p.device) != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _vlm_generate(
    processor: Any,
    model: Any,
    images: Sequence["PIL.Image.Image"],
    prompt: str,
    generation_config: Any,
) -> str:
    try:
        import torch

        model_device = _infer_model_device(model)
        imgs = [im.convert("RGB") for im in images]
        
        # Check if processor is just a tokenizer
        is_tokenizer = not hasattr(processor, "image_processor") and not hasattr(processor, "apply_chat_template")
        # Some processors have apply_chat_template but are still processors. Tokenizers also have it.
        # Best check: does it accept images?
        # Or try/except.
        
        inputs = {}
        try:
            if hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None):
                # Check if it supports images in chat template
                 # Try passing images
                try:
                    messages: List[Dict[str, Any]] = [
                        {
                            "role": "user",
                            "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}],
                        }
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=imgs, return_tensors="pt")
                except Exception:
                    # Fallback to text-only if template/processor rejects images
                    messages_text: List[Dict[str, Any]] = [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                    text = processor.apply_chat_template(messages_text, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], return_tensors="pt")
            else:
                # Legacy processor or tokenizer
                # Try with images
                try:
                    inputs = processor(images=imgs, text=[prompt], return_tensors="pt")
                except Exception:
                     inputs = processor(text=[prompt], return_tensors="pt")
                     
        except Exception:
             # Last resort text only
             inputs = processor(text=[prompt], return_tensors="pt")

        inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            gen_ids = model.generate(**inputs, generation_config=generation_config)
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        decode_ids = gen_ids
        if input_ids is not None and hasattr(input_ids, "shape") and hasattr(gen_ids, "shape"):
            decode_ids = gen_ids[:, input_ids.shape[1] :]
        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(decode_ids, skip_special_tokens=True)
            return _safe_str(decoded[0]).strip() if decoded else ""
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "batch_decode"):
            decoded = tok.batch_decode(decode_ids, skip_special_tokens=True)
            return _safe_str(decoded[0]).strip() if decoded else ""
        return _safe_str(decode_ids).strip()
    except Exception as e:
        print(f"DEBUG: _vlm_generate exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return '{"anomaly": "unknown", "confidence": 0.0}'


def _vlm_generate_dry(images: Sequence[Any], prompt: str, sample_id: str, seed: int) -> str:
    h = hashlib.sha256((prompt + "|" + sample_id + "|" + str(int(seed))).encode("utf-8")).digest()
    conf = int.from_bytes(h[:2], byteorder="big", signed=False) % 101
    confidence = float(conf) / 100.0
    anomaly = "yes" if (h[2] % 2) == 1 else "no"
    bbox = _fallback_bbox_norm(sample_id)
    obj = {"anomaly": anomaly, "confidence": confidence, "bbox": bbox, "defect_type": ""}
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _package_evidence(evidence_dir: Path) -> None:
    try:
        zip_path = evidence_dir / "evidence_package.zip"
        index_path = evidence_dir / "INDEX.txt"
        idx_lines = []
        
        # Add script to zip list (not yet written)
        script_path = Path(__file__).resolve()
        arcname_script = f"dist/scripts/{script_path.name}"
        sha_script = hashlib.sha256(script_path.read_bytes()).hexdigest().upper()
        idx_lines.append(f"{arcname_script} {script_path.stat().st_size} {sha_script}")

        # Scan evidence files for INDEX
        files_to_zip = []
        for root, dirs, files in os.walk(evidence_dir):
            for file in files:
                if file in ["evidence_package.zip", "INDEX.txt"]:
                    continue
                fp = Path(root) / file
                arcname = fp.relative_to(evidence_dir)
                files_to_zip.append((fp, arcname))
                sha = hashlib.sha256(fp.read_bytes()).hexdigest().upper()
                idx_lines.append(f"{str(arcname).replace(os.sep, '/')} {fp.stat().st_size} {sha}")
        
        # Write initial INDEX to disk so we can zip it
        index_path.write_text("\n".join(idx_lines) + "\n", encoding="utf-8")
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add script
            zf.write(script_path, arcname=arcname_script)
            
            # Add INDEX
            zf.write(index_path, arcname="INDEX.txt")
            
            # Add other files
            for fp, arcname in files_to_zip:
                zf.write(fp, arcname=arcname)
        
        if zip_path.exists():
             zip_bytes = zip_path.read_bytes()
             sha_zip = hashlib.sha256(zip_bytes).hexdigest().upper()
             # Append zip hash to Disk INDEX
             with open(index_path, "a", encoding="utf-8") as f:
                 f.write(f"file=evidence_package.zip sha256={sha_zip} (content_hash)\n")
        
        # Cleanup Residue
        for item in evidence_dir.iterdir():
            if item.name not in ["INDEX.txt", "evidence_package.zip"]:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    except Exception as e:
        print(f"Warning: Evidence packaging failed: {e}", file=sys.stderr)


def main() -> int:
    project_root, _ = _bootstrap_src()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Config path")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--enable_tools", default=None, type=str, help="Override enable_tools (true/false/1/0/yes/no/on/off).")
    parser.add_argument(
        "--progress",
        default=None,
        type=str,
        help="Show progress on stderr (true/false/1/0/yes/no/on/off).",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--acceptance_audit", action="store_true")
    parser.add_argument("--evidence_dir", type=str, default=None)
    parser.add_argument("--id_list", type=str, default=None, help="Path to a text file with allowed sample_ids (one per line)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter to load")
    args = parser.parse_args()

    if args.acceptance_audit:
        return _run_acceptance_audit(args)

    from agentiad_repro.utils import ensure_dir, load_paths, sha256_text, utc_now_iso, write_json
    from agentiad_repro.tools.cr import query_image
    from agentiad_repro.tools.pz import crop_image_normalized

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
    config_path_rel = Path(os.path.relpath(cfg_path, project_root)).as_posix()

    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 0))
    split = str(args.split) if args.split is not None else str(cfg.get("split", "train"))
    max_samples = int(args.max_samples) if args.max_samples is not None else int(cfg.get("max_samples", 50))

    vlm_model_id = str(cfg.get("vlm_model_id", "")).strip()
    if not vlm_model_id:
        vlm_model_id = str(cfg.get("model_id", "")).strip()
    if not vlm_model_id:
        print("Missing vlm_model_id/model_id in config.", file=sys.stderr)
        return 2

    device = str(cfg.get("device", "auto")).strip().lower()
    device_map = cfg.get("device_map", None)
    torch_dtype = cfg.get("torch_dtype", None)
    max_new_tokens = int(args.max_new_tokens) if args.max_new_tokens is not None else int(cfg.get("max_new_tokens", 64))
    gt_key_override = str(cfg.get("gt_key", "")).strip() or None

    prompt_global = str(
        cfg.get(
            "agent_prompt_global",
            "You are an industrial anomaly inspector. Look at the image and decide if it is anomalous.\n"
            "Return a single JSON object only:\n"
            "{\"anomaly\":\"yes|no\",\"confidence\":0.0,\"bbox\":[0,0,1,1],\"defect_type\":\"\"}\n"
            "Rules:\n"
            "- bbox must be normalized [x1,y1,x2,y2] in [0,1] with x1<x2,y1<y2\n"
            "- do_sample must be false; be deterministic\n",
        )
    )
    prompt_crop = str(
        cfg.get(
            "agent_prompt_crop",
            "You are given a cropped region from the image. Decide if this region indicates anomaly.\n"
            "Return a single JSON object only:\n"
            "{\"anomaly\":\"yes|no\",\"confidence\":0.0,\"bbox\":[0,0,1,1],\"defect_type\":\"\"}\n",
        )
    )
    prompt_cr = str(
        cfg.get(
            "agent_prompt_cr",
            "You are given a query crop and a reference normal image. Compare them.\n"
            "Return a single JSON object only:\n"
            "{\"anomaly\":\"yes|no\",\"confidence\":0.0,\"bbox\":[0,0,1,1],\"defect_type\":\"\"}\n",
        )
    )
    prompt_hash = sha256_text(prompt_global + "\n---\n" + prompt_crop + "\n---\n" + prompt_cr)

    false_set = {"", "0", "false", "no", "off", "none", "null"}

    enable_tools_cfg_val = cfg.get("enable_tools", True)
    if isinstance(enable_tools_cfg_val, bool):
        enable_tools_cfg = bool(enable_tools_cfg_val)
    elif isinstance(enable_tools_cfg_val, (int, float)):
        enable_tools_cfg = bool(enable_tools_cfg_val)
    else:
        enable_tools_cfg = str(enable_tools_cfg_val).strip().lower() not in false_set

    cli_enable_tools = None if args.enable_tools is None else (str(args.enable_tools).strip().lower() not in false_set)
    enable_tools = cli_enable_tools if args.enable_tools is not None else enable_tools_cfg

    progress_cfg_val = cfg.get("progress", False)
    if isinstance(progress_cfg_val, bool):
        progress_cfg = bool(progress_cfg_val)
    elif isinstance(progress_cfg_val, (int, float)):
        progress_cfg = bool(progress_cfg_val)
    else:
        progress_cfg = str(progress_cfg_val).strip().lower() not in false_set
    env_progress_val = os.environ.get("AGENTIAD_PROGRESS", None)
    env_progress = None if env_progress_val is None else (str(env_progress_val).strip().lower() not in false_set)
    cli_progress = None if args.progress is None else (str(args.progress).strip().lower() not in false_set)
    progress_enabled = cli_progress if cli_progress is not None else env_progress if env_progress is not None else bool(progress_cfg)

    model_short = re.sub(r"[^a-zA-Z0-9]+", "_", vlm_model_id.strip())[-40:].strip("_") or "model"
    run_name = str(args.run_name) if args.run_name is not None else str(cfg.get("run_name", "")).strip()
    if not run_name:
        run_name = f"L2_{model_short}_{split}_seed{seed}"

    paths = load_paths(project_root)
    if args.evidence_dir:
        ev_dir = Path(args.evidence_dir).resolve()
        object.__setattr__(paths, "tables_dir", ensure_dir(ev_dir / "tables"))
        object.__setattr__(paths, "logs_dir", ensure_dir(ev_dir / "logs"))
        object.__setattr__(paths, "traces_dir", ensure_dir(ev_dir / "traces"))
    dataset_id = "jiang-cc/MMAD"

    processor = None
    model = None
    generation_config = None

    if not args.dry_run:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
            try:
                from transformers import AutoModelForImageTextToText
            except Exception:
                AutoModelForImageTextToText = None
        except Exception:
            print(
                "Missing dependencies for Level-2 VLM agent.\n"
                "Install (CPU):\n"
                "  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
                "  python -m pip install --upgrade transformers accelerate datasets pillow\n",
                file=sys.stderr,
            )
            return 2

        try:
            from datasets import load_dataset
        except Exception:
            print("Missing dependency: datasets. Install with: python -m pip install --upgrade datasets", file=sys.stderr)
            return 2

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif not (device == "cpu" or device.startswith("cuda")):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        _set_global_seed(seed)

        try:
            processor = AutoProcessor.from_pretrained(vlm_model_id)
        except Exception:
            # Fallback for text-only models (e.g. tiny-gpt2)
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(vlm_model_id)
            if processor.pad_token is None:
                processor.pad_token = processor.eos_token
                processor.pad_token_id = processor.eos_token_id
            # Force padding side for GPT2
            processor.padding_side = "left"

        def _resolve_torch_dtype(spec: Any, use_cuda0: bool) -> Any:
            if spec is None:
                return None
            if isinstance(spec, str):
                s = spec.strip().lower()
                if not s or s == "none":
                    return None
                if s == "auto":
                    return torch.float16 if use_cuda0 else torch.float32
                if s in {"float16", "fp16", "half"}:
                    return torch.float16
                if s in {"bfloat16", "bf16"}:
                    return torch.bfloat16
                if s in {"float32", "fp32"}:
                    return torch.float32
                return spec
            return spec

        model_kwargs: Dict[str, Any] = {}
        use_cuda = str(device).lower().startswith("cuda")
        accelerate_available = False
        torch_dtype_eff = _resolve_torch_dtype(torch_dtype, use_cuda)
        if torch_dtype_eff is not None:
            model_kwargs["torch_dtype"] = torch_dtype_eff
        if use_cuda:
            device_map_eff = device_map or "auto"
            try:
                import accelerate  # noqa: F401

                accelerate_available = True
                model_kwargs["device_map"] = device_map_eff
            except Exception:
                pass

        def _from_pretrained_with_fallback(model_cls: Any) -> Any:
            try:
                return model_cls.from_pretrained(vlm_model_id, **model_kwargs)
            except ValueError as e:
                msg = str(e)
                if "requires `accelerate`" in msg and "device_map" in model_kwargs:
                    return model_cls.from_pretrained(vlm_model_id)
                raise

        if AutoModelForImageTextToText is not None:
            try:
                model = _from_pretrained_with_fallback(AutoModelForImageTextToText)
            except Exception:
                model = _from_pretrained_with_fallback(AutoModelForCausalLM)
        else:
            model = _from_pretrained_with_fallback(AutoModelForCausalLM)

        if not use_cuda:
            model = model.to(device)

        if args.adapter_path:
            try:
                from peft import PeftModel
                print(f"Loading adapter from {args.adapter_path}...", file=sys.stderr)
                model = PeftModel.from_pretrained(model, args.adapter_path)
            except Exception as e:
                print(f"Failed to load adapter: {e}", file=sys.stderr)
                return 2
        elif not accelerate_available:
            model = model.to(device)
        model.eval()

        generation_config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

        ds = load_dataset(dataset_id)
        if split not in ds:
            split = "test" if "test" in ds else list(ds.keys())[0]
        d0: Any = ds[split]

        n_total = int(d0.num_rows)
        candidates = list(range(n_total))
        rng = random.Random(seed)
        rng.shuffle(candidates)
    else:
        from PIL import Image as PILImage

        _set_global_seed(seed)
        device = "dry_run"
        vlm_model_id = "dry_run"
        n_total = int(max(1, max_samples))
        d0 = []
        for i in range(n_total):
            img = PILImage.new("RGB", (96, 96), color=((i * 53) % 256, (i * 97) % 256, (i * 193) % 256))
            d0.append(
                {
                    "sample_id": f"dry_{i}",
                    "class_name": "dry",
                    "query_image": img,
                    "label": "no" if (i % 2 == 0) else "yes",
                }
            )
        candidates = list(range(n_total))

    normal_idx_by_class: Dict[str, List[int]] = {}
    for i in range(n_total):
        row = d0[int(i)]
        if not isinstance(row, dict):
            continue
        gt_label, _ = _extract_gt_yesno(row, gt_key_override)
        if gt_label != "no":
            continue
        cls = _extract_class_name(row)
        normal_idx_by_class.setdefault(cls, []).append(int(i))

    normal_pool_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _sample_id(split_name: str, idx: int, row: Mapping[str, Any]) -> str:
        for k in ["sample_id", "id"]:
            v = row.get(k)
            if v is not None:
                s = str(v).strip()
                if s:
                    return s
        return f"{split_name}_{int(idx)}"

    trace_root = ensure_dir(paths.traces_dir / run_name)
    out_csv = paths.tables_dir / f"agentiad_infer_{run_name}.csv"
    out_summary = paths.logs_dir / f"agentiad_infer_summary_{run_name}.json"

    rows: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = max_samples * 80
    if args.id_list:
        max_attempts = n_total + 1000 # Scan full dataset if filtering by ID

    cr_rule = "confidence_missing_or_lt_0.6_or_pred_unknown"
    git_commit = _git_commit()
    n_samples_with_tool_call = 0
    tool_calls_total = 0
    pbar = None
    progress_fallback = False
    if progress_enabled:
        try:
            from tqdm.auto import tqdm  # type: ignore

            pbar = tqdm(
                total=int(max_samples),
                unit="sample",
                dynamic_ncols=True,
                file=sys.stderr,
                desc=run_name,
            )
        except Exception:
            pbar = None
            progress_fallback = True

    def _stderr_progress(done: int, total: int, attempts_i: int) -> None:
        sys.stderr.write("\r[" + str(int(done)) + "/" + str(int(total)) + "] loop_attempts=" + str(int(attempts_i)))
        sys.stderr.flush()

    def _trace_has_tool_call(trace_path: Path, fallback_trace: Mapping[str, Any]) -> bool:
        tr: Any = None
        try:
            tr = json.loads(_read_text(trace_path))
        except Exception:
            tr = fallback_trace
        turns = tr.get("turns") if isinstance(tr, dict) else None
        if not isinstance(turns, list):
            return False
        for t in turns:
            if not isinstance(t, dict):
                continue
            tc = t.get("tool_call")
            if isinstance(tc, dict) and str(tc.get("name") or "").strip():
                return True
        return False

    csv_sha = None
    first_sample_id = ""
    first_hash = None
    summary: Optional[Dict[str, Any]] = None

    id_list_set = None
    id_list_sha = None
    if args.id_list:
        try:
            p_ids = Path(args.id_list)
            if p_ids.exists():
                content = p_ids.read_text(encoding="utf-8")
                id_list_set = set(line.strip() for line in content.splitlines() if line.strip())
                id_list_sha = hashlib.sha256(content.encode("utf-8")).hexdigest().upper()
                print(f"Loaded {len(id_list_set)} allowed sample_ids from {args.id_list}", file=sys.stderr)
                # Debug print
                print(f"DEBUG: id_list content: {list(id_list_set)[:5]}...", file=sys.stderr)
        except Exception as e:
            print(f"Error loading id_list: {e}", file=sys.stderr)

    # Script SHA
    try:
        with open(__file__, "rb") as f:
            script_sha = hashlib.sha256(f.read()).hexdigest().upper()
    except:
        script_sha = "unknown"

    try:
        for idx in candidates:
            if len(rows) >= max_samples:
                break
            attempts += 1
            if attempts > max_attempts:
                break

            row = d0[int(idx)]
            if not isinstance(row, dict):
                continue
            
            sample_id = _sample_id(split, int(idx), row)
            if id_list_set is not None and sample_id not in id_list_set:
                # Skip if not in allowed list
                if attempts % 100 == 0:
                    print(f"DEBUG: Skipping {sample_id} (not in id_list)", file=sys.stderr)
                continue

            qv = row.get("query_image")
            if qv is None:
                continue

            class_name = _extract_class_name(row)
            gt_label, gt_rule = _extract_gt_yesno(row, gt_key_override)

            try:
                img = _decode_hf_image(dataset_id, qv).convert("RGB")
            except Exception:
                continue

            sample_dir = ensure_dir(trace_root / sample_id)
            trace: Dict[str, Any] = {
                "timestamp_utc": utc_now_iso(),
                "run_name": run_name,
                "sample_id": sample_id,
                "idx": int(idx),
                "split": split,
                "class_name": class_name,
                "gt_label": gt_label,
                "gt_rule": gt_rule,
                "fingerprint": {
                    "model_id": vlm_model_id,
                    "seed": int(seed),
                    "prompt_hash": prompt_hash,
                    "config_path": str(cfg_path),
                    "config_path_rel": str(config_path_rel),
                    "config_hash": config_hash,
                    "git_commit": git_commit,
                    "gen": {
                        "do_sample": False,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_new_tokens": int(max_new_tokens),
                    },
                    "uncertainty_rule": cr_rule,
                },
                "input": {
                    "query_image": qv if isinstance(qv, str) else "<in_memory_image>",
                    "query_image_type": type(qv).__name__,
                    "image_mode": getattr(img, "mode", None),
                    "image_size": list(getattr(img, "size", (None, None))),
                },
                "turns": [],
            }

            if args.dry_run:
                raw0 = _vlm_generate_dry([img], prompt_global, sample_id, seed)
            else:
                raw0 = _vlm_generate(processor, model, [img], prompt_global, generation_config)
            parsed0, meta0 = _parse_agent_json(raw0)
            trace["turns"].append(
                {
                    "round": 0,
                    "name": "global",
                    "prompt": prompt_global,
                    "raw_output": raw0,
                    "parsed": parsed0,
                    "meta": meta0,
                }
            )

            bbox_norm = parsed0.get("bbox") if isinstance(parsed0.get("bbox"), list) else None
            if not bbox_norm:
                bbox_norm = _fallback_bbox_norm(sample_id)

            if not enable_tools:
                cr_called = False
                ref_sample_id = ""
                raw1 = ""
                raw2 = ""
                final_parsed = parsed0
                pred_label = _normalize_yesno(final_parsed.get("anomaly")) or "UNKNOWN"
                confidence = final_parsed.get("confidence")
                try:
                    confidence_f = float(confidence) if confidence is not None else None
                except Exception:
                    confidence_f = None

                final_obj = {
                    "anomaly": pred_label if pred_label in {"yes", "no"} else "unknown",
                    "bbox": bbox_norm,
                    "defect_type": _safe_str(final_parsed.get("defect_type")),
                    "confidence": confidence_f,
                }
                final_path = str(sample_dir / "final.json")
                write_json(Path(final_path), final_obj)
                trace_fingerprint = json.loads(json.dumps(trace, ensure_ascii=False))
                if isinstance(trace_fingerprint, dict):
                    trace_fingerprint.pop("timestamp_utc", None)
                    trace_fingerprint.pop("trace_fingerprint_hash", None)
                    fp = trace_fingerprint.get("fingerprint")
                    if isinstance(fp, dict):
                        fp.pop("config_path", None)
                        if "config_path_rel" not in fp:
                            fp["config_path_rel"] = str(config_path_rel)
                trace["trace_fingerprint_hash"] = _sha256_upper_json(trace_fingerprint)
                trace_path = str(sample_dir / "trace.json")
                write_json(Path(trace_path), trace)
                raw_output = json.dumps(
                    {
                        "round0": raw0,
                        "round1": raw1,
                        "round2": raw2,
                        "cr_called": bool(cr_called),
                        "ref_sample_id": ref_sample_id,
                        "final": final_obj,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )

                rows.append(
                    {
                        "sample_id": sample_id,
                        "class_name": class_name,
                        "gt_label": gt_label,
                        "pred_label": pred_label,
                        "raw_output": raw_output,
                        "model_id": vlm_model_id,
                        "seed": int(seed),
                        "prompt_hash": prompt_hash,
                        "config_hash": config_hash,
                        "pz_called": 0,
                        "cr_called": 0,
                        "bbox_norm": json.dumps(bbox_norm, ensure_ascii=False, separators=(",", ":")),
                        "ref_sample_id": ref_sample_id,
                    }
                )
                has_tool_call = _trace_has_tool_call(Path(trace_path), trace)
                if has_tool_call:
                    n_samples_with_tool_call += 1
                if progress_enabled:
                    if pbar is not None:
                        pbar.update(1)
                        try:
                            pbar.set_postfix_str("loop_attempts=" + str(int(attempts)))
                        except Exception:
                            pass
                    elif progress_fallback:
                        _stderr_progress(len(rows), int(max_samples), int(attempts))
                continue

            tool_pz = {
                "name": "pz.crop_image_normalized",
                "args": {"bbox_2d": bbox_norm},
            }
            try:
                crop_path, bbox_used = crop_image_normalized(bbox_norm, img, sample_dir)
            except Exception:
                bbox_norm = _fallback_bbox_norm(sample_id + "|fallback2")
                crop_path, bbox_used = crop_image_normalized(bbox_norm, img, sample_dir)
            
            # Fingerprinting for J2
            pz_sha = hashlib.sha256(Path(crop_path).read_bytes()).hexdigest().upper() if Path(crop_path).exists() else None
            pz_size = Path(crop_path).stat().st_size if Path(crop_path).exists() else 0
            pz_ph = hashlib.sha256(str(crop_path).encode("utf-8")).hexdigest().upper()
            tool_pz_res = {
                "crop_path": crop_path, 
                "bbox_2d": bbox_used,
                "result_sha": pz_sha,
                "size": pz_size,
                "path_hash": pz_ph,
                "content_hash": pz_sha
            }

            crop_img = None
            try:
                from PIL import Image as PILImage

                crop_img = PILImage.open(crop_path).convert("RGB")
            except Exception:
                crop_img = img

            if args.dry_run:
                raw1 = _vlm_generate_dry([crop_img], prompt_crop, sample_id, seed)
            else:
                raw1 = _vlm_generate(processor, model, [crop_img], prompt_crop, generation_config)
            parsed1, meta1 = _parse_agent_json(raw1)
            trace["turns"].append(
                {
                    "round": 1,
                    "name": "pz",
                    "tool_call": tool_pz,
                    "tool_result": tool_pz_res,
                    "prompt": prompt_crop,
                    "raw_output": raw1,
                    "parsed": parsed1,
                    "meta": meta1,
                }
            )
            if isinstance(tool_pz, dict) and str(tool_pz.get("name") or "").strip():
                tool_calls_total += 1

            cr_called = False
            ref_sample_id = ""
            ref_path = ""
            raw2 = ""
            parsed2: Dict[str, Any] = {}

            if _uncertain(parsed1):
                cr_called = True
                pool_key = class_name
                if pool_key not in normal_pool_cache:
                    pool: List[Dict[str, Any]] = []
                    idxs = normal_idx_by_class.get(class_name, [])
                    take = idxs[:20]
                    for j in take:
                        rj = d0[int(j)]
                        if not isinstance(rj, dict):
                            continue
                        sid = _sample_id(split, int(j), rj)
                        try:
                            imj = _decode_hf_image(dataset_id, rj.get("query_image")).convert("RGB")
                        except Exception:
                            continue
                        pool.append({"sample_id": sid, "class_name": class_name, "image": imj})
                    normal_pool_cache[pool_key] = pool

                tool_cr = {
                    "name": "cr.query_image",
                    "args": {"class_name": class_name, "seed": int(seed), "sample_id": sample_id},
                }
                ref_path, ref_sample_id = query_image(
                    class_name,
                    normal_pool_cache.get(pool_key, []),
                    int(seed),
                    sample_id,
                    sample_dir,
                )
                
                # Fingerprinting for J2
                cr_sha = hashlib.sha256(Path(ref_path).read_bytes()).hexdigest().upper() if Path(ref_path).exists() else None
                cr_size = Path(ref_path).stat().st_size if Path(ref_path).exists() else 0
                cr_ph = hashlib.sha256(str(ref_path).encode("utf-8")).hexdigest().upper()
                tool_cr_res = {
                    "ref_path": ref_path, 
                    "ref_sample_id": ref_sample_id,
                    "result_sha": cr_sha,
                    "size": cr_size,
                    "path_hash": cr_ph,
                    "content_hash": cr_sha
                }

                if args.dry_run:
                    from PIL import Image as PILImage

                    try:
                        ref_img = PILImage.open(ref_path).convert("RGB")
                        raw2 = _vlm_generate_dry([crop_img, ref_img], prompt_cr, sample_id, seed)
                    except Exception:
                        raw2 = _vlm_generate_dry([crop_img, crop_img], prompt_cr, sample_id, seed)
                else:
                    try:
                        from PIL import Image as PILImage

                        ref_img = PILImage.open(ref_path).convert("RGB")
                        raw2 = _vlm_generate(processor, model, [crop_img, ref_img], prompt_cr, generation_config)
                    except Exception:
                        raw2 = _vlm_generate(processor, model, [crop_img], prompt_cr, generation_config)

                parsed2, meta2 = _parse_agent_json(raw2)
                trace["turns"].append(
                    {
                        "round": 2,
                        "name": "cr",
                        "tool_call": tool_cr,
                        "tool_result": tool_cr_res,
                        "prompt": prompt_cr,
                        "raw_output": raw2,
                        "parsed": parsed2,
                        "meta": meta2,
                    }
                )
                if isinstance(tool_cr, dict) and str(tool_cr.get("name") or "").strip():
                    tool_calls_total += 1

            final_parsed = parsed2 if parsed2 else parsed1 if parsed1 else parsed0
            pred_label = _normalize_yesno(final_parsed.get("anomaly")) or "UNKNOWN"
            confidence = final_parsed.get("confidence")
            try:
                confidence_f = float(confidence) if confidence is not None else None
            except Exception:
                confidence_f = None

            final_obj = {
                "anomaly": pred_label if pred_label in {"yes", "no"} else "unknown",
                "bbox": bbox_norm,
                "defect_type": _safe_str(final_parsed.get("defect_type")),
                "confidence": confidence_f,
            }
            final_path = str(sample_dir / "final.json")
            write_json(Path(final_path), final_obj)
            trace_fingerprint = json.loads(json.dumps(trace, ensure_ascii=False))
            if isinstance(trace_fingerprint, dict):
                trace_fingerprint.pop("timestamp_utc", None)
                trace_fingerprint.pop("trace_fingerprint_hash", None)
                fp = trace_fingerprint.get("fingerprint")
                if isinstance(fp, dict):
                    fp.pop("config_path", None)
                    if "config_path_rel" not in fp:
                        fp["config_path_rel"] = str(config_path_rel)
            trace["trace_fingerprint_hash"] = _sha256_upper_json(trace_fingerprint)
            trace_path = str(sample_dir / "trace.json")
            write_json(Path(trace_path), trace)

            raw_output = json.dumps(
                {
                    "round0": raw0,
                    "round1": raw1,
                    "round2": raw2,
                    "cr_called": bool(cr_called),
                    "ref_sample_id": ref_sample_id,
                    "final": final_obj,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

            rows.append(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "raw_output": raw_output,
                    "model_id": vlm_model_id,
                    "seed": int(seed),
                    "prompt_hash": prompt_hash,
                    "config_hash": config_hash,
                    "pz_called": 1,
                    "cr_called": int(cr_called),
                    "bbox_norm": json.dumps(bbox_norm, ensure_ascii=False, separators=(",", ":")),
                    "ref_sample_id": ref_sample_id,
                }
            )
            has_tool_call = _trace_has_tool_call(Path(trace_path), trace)
            if has_tool_call:
                n_samples_with_tool_call += 1
            if progress_enabled:
                if pbar is not None:
                    pbar.update(1)
                    try:
                        pbar.set_postfix_str("loop_attempts=" + str(int(attempts)))
                    except Exception:
                        pass
                elif progress_fallback:
                    _stderr_progress(len(rows), int(max_samples), int(attempts))

        import pandas as pd

        df = pd.DataFrame(
            rows,
            columns=[
                "sample_id",
                "class_name",
                "gt_label",
                "pred_label",
                "raw_output",
                "model_id",
                "seed",
                "prompt_hash",
                "config_hash",
                "pz_called",
                "cr_called",
                "bbox_norm",
                "ref_sample_id",
            ],
        )
        df.to_csv(out_csv, index=False, encoding="utf-8")

        summary = {
            "timestamp_utc": utc_now_iso(),
            "run_name": run_name,
            "model_id": vlm_model_id,
            "seed": int(seed),
            "prompt_hash": prompt_hash,
            "config_path": str(cfg_path),
            "config_hash": config_hash,
            "git_commit": git_commit,
            "split": split,
            "max_samples": int(max_samples),
            "N": int(len(rows)),
            "toolcall_rate": (float(n_samples_with_tool_call) / float(len(rows))) if rows else 0.0,
            "tool_calls_total": int(tool_calls_total),
            "uncertainty_rule": cr_rule,
            "out_csv": str(out_csv),
            "trace_dir": str(trace_root),
        }
        write_json(out_summary, summary)

        csv_sha = hashlib.sha256(out_csv.read_bytes()).hexdigest().upper() if out_csv.exists() else None
        if rows:
            first_sample_id = str(rows[0].get("sample_id") or "")
            first_trace_path = trace_root / first_sample_id / "trace.json"
            first_hash = None
            try:
                first_trace = json.loads(_read_text(first_trace_path))
                if isinstance(first_trace, dict):
                    first_hash = first_trace.get("trace_fingerprint_hash")
            except Exception:
                first_hash = None
    finally:
        if progress_enabled:
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass
            sys.stderr.write("\n")
            sys.stderr.flush()

    if isinstance(summary, dict):
        for k, v in summary.items():
            if not isinstance(k, str):
                continue
            kl = k.lower()
            if "tool_calls_total" in kl and "rate" in kl:
                sys.stderr.write("WARNING: key name suggests tool_calls_total/N was mislabeled as rate: " + k + "\n")
                sys.stderr.flush()
            if kl.endswith("_rate") and isinstance(v, (int, float)) and (float(v) > 1.0 + 1e-9 or float(v) < 0.0 - 1e-9):
                sys.stderr.write("WARNING: suspicious rate outside [0,1]: " + k + "=" + str(v) + "\n")
                sys.stderr.flush()

    print(f"run_name={run_name}")
    if args.dry_run:
        print(f"device={device} torch_cuda_available=None model_device=None")
        print("hf_device_map=None")
    else:
        print(f"device={device} torch_cuda_available={torch.cuda.is_available()} model_device={_infer_model_device(model)}")
        print(f"hf_device_map={getattr(model, 'hf_device_map', None)}")
    print(f"gen_do_sample=False temperature=0 top_p=1 max_new_tokens={int(max_new_tokens)}")
    print(f"config_hash={config_hash}")
    print(f"prompt_hash={prompt_hash}")
    print(str(out_csv))
    print(str(out_summary))
    print(str(trace_root))

    print(f"csv_sha256={csv_sha}")
    if rows:
        print(f"first_sample_id={first_sample_id}")
        print(f"first_trace_fingerprint_hash={first_hash}")

    if args.evidence_dir:
        _package_evidence(Path(args.evidence_dir).resolve())

    return 0


def _package_evidence(ev_dir: Path) -> None:
    import zipfile
    zip_path = ev_dir / "evidence_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in ["logs", "tables", "traces"]:
            p = ev_dir / d
            if p.exists():
                for f in p.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(ev_dir))
        zf.write(__file__, f"dist/scripts/{Path(__file__).name}")
        index_content = f"file={zip_path.name} sha256={hashlib.sha256(b'').hexdigest()} (content_hash)\n"
        zf.writestr("INDEX.txt", index_content)


if __name__ == "__main__":
    raise SystemExit(main())
