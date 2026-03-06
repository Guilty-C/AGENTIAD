from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _bootstrap_repo() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for p in [repo_root / "dist" / "src", repo_root / "src", repo_root]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return repo_root


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ln in _read_text(path).splitlines():
        s = ln.strip()
        if not s:
            continue
        obj = json.loads(s)
        if isinstance(obj, dict):
            items.append(obj)
    return items


def _resolve_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    p1 = (Path.cwd() / p).resolve()
    if p1.exists():
        return p1
    return (repo_root / p).resolve()


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest().upper()


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(event)
    rec.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _replay_eval_events(events_path: Path) -> Tuple[Dict[str, int], List[str]]:
    allowed_failure_reasons = {
        "build_eval_example_failed",
        "missing_answer_tags",
        "answer_json_parse_error",
        "answer_schema_invalid",
        "gt_answer_missing",
        "gt_answer_json_parse_error",
        "gt_answer_schema_invalid",
    }
    counts: Dict[str, int] = {
        "samples_evaluated": 0,
        "samples_failed": 0,
        "samples_generation_ok": 0,
        "samples_schema_valid": 0,
        "samples_comparison_ready": 0,
        "samples_gt_not_comparable": 0,
        "samples_schema_valid_but_not_comparable": 0,
        "samples_with_audit_note": 0,
        "anomaly_present_comparable_count": 0,
        "anomaly_present_match_count": 0,
        "top_anomaly_comparable_count": 0,
        "top_anomaly_match_count": 0,
    }
    invariant_mismatches: List[str] = []
    sample_event_counts: Dict[str, int] = {}
    if not events_path.exists():
        return counts, invariant_mismatches

    for rec in _read_jsonl(events_path):
        ev = str(rec.get("event") or "")
        if ev not in {"sample_ok", "sample_failed"}:
            continue
        sample_id = str(rec.get("sample_id") or "")

        # Sample-event uniqueness audit: one sample_id must map to exactly one sample event.
        if not sample_id:
            invariant_mismatches.append("invariant:sample_event_missing_sample_id")
        else:
            sample_event_counts[sample_id] = int(sample_event_counts.get(sample_id, 0)) + 1
            if sample_event_counts[sample_id] > 1:
                invariant_mismatches.append(f"invariant:duplicate_sample_event:sample_id={sample_id}")

        counts["samples_evaluated"] += 1
        failure_reason = str(rec.get("failure_reason") or "").strip()
        if ev == "sample_failed" and failure_reason:
            counts["samples_failed"] += 1

        # Freeze replay definitions on sample-event domain only.
        generation_ok = _as_bool(rec.get("generation_ok"))
        answer_schema_valid = _as_bool(rec.get("answer_schema_valid"))
        ap_comparable = _as_bool(rec.get("ap_comparable"))
        top_comparable = _as_bool(rec.get("top_anomaly_comparable"))
        comparison_ready = _as_bool(rec.get("comparison_ready"))
        # Backward-compatible fallback if comparison_ready field is absent.
        if "comparison_ready" not in rec:
            comparison_ready = bool(ap_comparable or top_comparable)
        derived_comparison_ready = bool(ap_comparable or top_comparable)
        comparability_note = str(rec.get("comparability_note") or "").strip()
        audit_note = str(rec.get("audit_note") or "").strip()

        # Event-level invariants.
        if ev == "sample_ok" and failure_reason:
            invariant_mismatches.append(f"invariant:sample_ok_has_failure_reason:sample_id={sample_id}")
        if ev == "sample_failed" and not failure_reason:
            invariant_mismatches.append(f"invariant:sample_failed_missing_failure_reason:sample_id={sample_id}")
        if ev == "sample_failed" and failure_reason:
            allowed = failure_reason in allowed_failure_reasons or failure_reason.startswith("generation_failed:")
            if not allowed:
                invariant_mismatches.append(f"invariant:failure_reason_not_in_allowed_set:sample_id={sample_id}")
        if ev == "sample_ok" and (not generation_ok):
            invariant_mismatches.append(f"invariant:sample_ok_generation_not_ok:sample_id={sample_id}")
        if ev == "sample_ok" and (not answer_schema_valid):
            invariant_mismatches.append(f"invariant:sample_ok_schema_not_valid:sample_id={sample_id}")
        if comparison_ready != derived_comparison_ready:
            invariant_mismatches.append(f"invariant:comparison_ready_mismatch:sample_id={sample_id}")
        if comparability_note and (ev != "sample_ok" or comparison_ready):
            invariant_mismatches.append(f"invariant:comparability_note_invalid_context:sample_id={sample_id}")
        if ev == "sample_ok" and (not comparison_ready) and (not comparability_note):
            invariant_mismatches.append(f"invariant:sample_ok_not_comparable_missing_comparability_note:sample_id={sample_id}")
        if ev == "sample_failed" and comparison_ready:
            invariant_mismatches.append(f"invariant:sample_failed_should_not_be_comparison_ready:sample_id={sample_id}")
        if ev == "sample_failed" and (ap_comparable or top_comparable):
            invariant_mismatches.append(f"invariant:sample_failed_should_not_be_comparable:sample_id={sample_id}")
        # sample_failed events must never be interpreted as comparability outcomes.
        # Keep comparability_note empty on failed events to avoid semantic drift.
        if ev == "sample_failed" and comparability_note:
            invariant_mismatches.append(f"invariant:sample_failed_should_not_have_comparability_note:sample_id={sample_id}")
        if ev == "sample_failed" and failure_reason.startswith("generation_failed:") and generation_ok:
            invariant_mismatches.append(f"invariant:generation_failed_reason_but_generation_ok:sample_id={sample_id}")
        if ev == "sample_failed" and failure_reason in {"missing_answer_tags", "answer_json_parse_error", "answer_schema_invalid"}:
            if (not generation_ok) or answer_schema_valid:
                invariant_mismatches.append(
                    f"invariant:{failure_reason}_state_mismatch:sample_id={sample_id}"
                )
        if ev == "sample_ok" and (not comparison_ready) and ("comparability:" in audit_note):
            invariant_mismatches.append(f"invariant:audit_note_should_not_embed_comparability:sample_id={sample_id}")
        if audit_note in {
            "build_eval_example_failed",
            "missing_answer_tags",
            "answer_json_parse_error",
            "answer_schema_invalid",
            "gt_answer_missing",
            "gt_answer_json_parse_error",
            "gt_answer_schema_invalid",
        } or audit_note.startswith("generation_failed:"):
            invariant_mismatches.append(f"invariant:audit_note_looks_like_primary_failure:sample_id={sample_id}")
        if ev == "sample_failed" and failure_reason and audit_note == failure_reason:
            invariant_mismatches.append(f"invariant:audit_note_duplicates_failure_reason:sample_id={sample_id}")

        # Replay counts are defined strictly on sample-event rows only.
        if generation_ok:
            counts["samples_generation_ok"] += 1
        if answer_schema_valid:
            counts["samples_schema_valid"] += 1
        if ev == "sample_ok" and comparison_ready:
            counts["samples_comparison_ready"] += 1
        if ev == "sample_ok" and (not comparison_ready):
            counts["samples_gt_not_comparable"] += 1
        if ev == "sample_ok" and answer_schema_valid and (not comparison_ready):
            counts["samples_schema_valid_but_not_comparable"] += 1

        if audit_note:
            counts["samples_with_audit_note"] += 1

        if ap_comparable:
            counts["anomaly_present_comparable_count"] += 1
            if _as_bool(rec.get("anomaly_present_match")):
                counts["anomaly_present_match_count"] += 1

        if top_comparable:
            counts["top_anomaly_comparable_count"] += 1
            if _as_bool(rec.get("top_anomaly_match")):
                counts["top_anomaly_match_count"] += 1

    return counts, invariant_mismatches


def _normalize_model_ref(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return s
    p = Path(s).expanduser()
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"base_model absolute path not found: {p}")
        if not p.is_dir():
            raise FileNotFoundError(f"base_model absolute path is not a directory: {p}")
        return str(p.resolve())
    if p.exists() and p.is_dir():
        return str(p.resolve())
    return s


def _load_l2_module(repo_root: Path) -> Any:
    p = repo_root / "dist" / "scripts" / "06_run_agentiad_infer.py"
    spec = importlib.util.spec_from_file_location("agentiad_l2_infer_mod_eval", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _resolve_image_candidate(raw: str, sample_dir: Optional[Path], mmad_root: Optional[Path], repo_root: Path) -> Optional[Path]:
    s = str(raw or "").strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if p.exists():
        return p.resolve()
    if sample_dir is not None and not p.is_absolute():
        p2 = (sample_dir / p).resolve()
        if p2.exists():
            return p2
    if mmad_root is not None and not p.is_absolute():
        p3 = (mmad_root / p).resolve()
        if p3.exists():
            return p3
    if not p.is_absolute():
        p4 = (repo_root / p).resolve()
        if p4.exists():
            return p4
    return None


@dataclass
class EvalExample:
    sample_id: str
    prompt_text: str
    image_path: Optional[Path]


@dataclass
class GTAnswer:
    anomaly_present: Optional[bool]
    top_anomaly: Optional[str]
    reason: str


def _build_eval_example(
    item: Mapping[str, Any],
    paper_contract: Any,
    repo_root: Path,
    mmad_root: Optional[Path],
) -> Tuple[Optional[EvalExample], Optional[str]]:
    sample_id = str(item.get("sample_id") or "")
    messages_any = item.get("messages")
    messages = messages_any if isinstance(messages_any, list) else []

    paths_obj = item.get("paths")
    sample_dir: Optional[Path] = None
    if isinstance(paths_obj, Mapping):
        sample_dir_raw = str(paths_obj.get("sample_dir") or "").strip()
        if sample_dir_raw:
            sample_dir = Path(sample_dir_raw)

    context_lines: List[str] = []
    image_candidates: List[str] = []

    for m in messages:
        if not isinstance(m, Mapping):
            continue
        role = str(m.get("role") or "")
        name = str(m.get("name") or "")
        if role == "user":
            content = m.get("content")
            if isinstance(content, Mapping):
                prompt = str(content.get("prompt") or "").strip()
                if prompt:
                    context_lines.append(f"USER[{name}]: {prompt}")
                imgs = content.get("images")
                if isinstance(imgs, list):
                    for im in imgs:
                        if isinstance(im, Mapping):
                            p = str(im.get("path") or im.get("source") or "").strip()
                            if p:
                                image_candidates.append(p)
            continue
        if role == "tool":
            context_lines.append(f"TOOL[{name}]: {_safe_json(m.get('content'))}")
            continue

    resolved_image: Optional[Path] = None
    for raw in image_candidates:
        resolved_image = _resolve_image_candidate(raw, sample_dir, mmad_root, repo_root)
        if resolved_image is not None:
            break
    if resolved_image is None and sample_dir is not None:
        for fallback_name in ["crop.png", "ref.png", "query.png"]:
            p = (sample_dir / fallback_name).resolve()
            if p.exists():
                resolved_image = p
                break

    if not context_lines:
        return None, "missing_context"
    prompt_text = (
        f"{paper_contract.SYSTEM_PROMPT}\n\n"
        "Trajectory context:\n"
        + "\n".join(context_lines)
        + "\n\nGive the final contract-compliant answer."
    )
    return EvalExample(sample_id=sample_id, prompt_text=prompt_text, image_path=resolved_image), None


def _build_chat_prompt(processor: Any, prompt_text: str) -> str:
    if hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        return str(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"<image>\n{prompt_text}\nAssistant:"


def _extract_gt_answer(item: Mapping[str, Any], paper_contract: Any) -> GTAnswer:
    messages_any = item.get("messages")
    messages = messages_any if isinstance(messages_any, list) else []
    final_content = None
    for m in messages:
        if not isinstance(m, Mapping):
            continue
        if str(m.get("role") or "") == "assistant" and str(m.get("name") or "") == "final":
            c = m.get("content")
            if isinstance(c, str):
                final_content = c
    if not isinstance(final_content, str) or not final_content.strip():
        return GTAnswer(anomaly_present=None, top_anomaly=None, reason="gt_answer_missing")
    answer_xml = paper_contract.extract_answer_xml(final_content)
    if answer_xml is None:
        return GTAnswer(anomaly_present=None, top_anomaly=None, reason="gt_answer_missing")
    try:
        obj = json.loads(answer_xml)
    except Exception:
        return GTAnswer(anomaly_present=None, top_anomaly=None, reason="gt_answer_json_parse_error")
    if not isinstance(obj, dict):
        return GTAnswer(anomaly_present=None, top_anomaly=None, reason="gt_answer_json_parse_error")
    ok, _ = paper_contract.validate_schema(obj)
    if not ok:
        return GTAnswer(anomaly_present=None, top_anomaly=None, reason="gt_answer_schema_invalid")
    gt_ap = obj.get("anomaly_present")
    gt_ta = obj.get("top_anomaly")
    gt_ta_norm = str(gt_ta).strip() if isinstance(gt_ta, str) else None
    return GTAnswer(
        anomaly_present=bool(gt_ap) if isinstance(gt_ap, bool) else None,
        # Keep raw stripped string (including "" / "none") so comparability policy
        # can explicitly classify missing vs empty vs literal "none".
        top_anomaly=gt_ta_norm if gt_ta_norm is not None else None,
        reason="",
    )


def _classify_top_anomaly_value(value: Optional[str]) -> Tuple[Optional[str], str]:
    """
    Classify top_anomaly for comparability audit.
    Rules:
    - missing (None), empty string, and literal "none" are NOT comparable classes.
    - only non-empty strings except "none" are comparable defect categories.
    """
    if value is None:
        return None, "missing"
    s = str(value).strip()
    if not s:
        return None, "empty"
    if s.lower() == "none":
        return s, "none"
    return s, "category"


def main() -> int:
    repo_root = _bootstrap_repo()
    from agentiad_repro.paper_contract import PaperContract

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", "--base-model", dest="base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--processor_dir", type=str, default=None)
    parser.add_argument("--val_jsonl", "--val-jsonl", dest="val_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--local_files_only", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    output_dir = _resolve_path(repo_root, str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    events_path = logs_dir / "eval_events.jsonl"
    summary_path = output_dir / "phase4_1_eval_summary.json"

    summary: Dict[str, Any] = {
        "phase": "4.1",
        "mode": "vlm_lora_sft_eval",
        "status": "FAIL",
        "errors": [],
    }

    try:
        val_jsonl = _resolve_path(repo_root, str(args.val_jsonl))
        adapter_dir = _resolve_path(repo_root, str(args.adapter_dir))
        processor_dir = _resolve_path(repo_root, str(args.processor_dir)) if args.processor_dir else None
        base_model = _normalize_model_ref(str(args.base_model))
        if not val_jsonl.exists():
            raise FileNotFoundError(f"val_jsonl not found: {val_jsonl}")
        if not adapter_dir.exists():
            raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")
        if processor_dir is not None and not processor_dir.exists():
            raise FileNotFoundError(f"processor_dir not found: {processor_dir}")

        l2_mod = _load_l2_module(repo_root)

        local_only_env = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        if args.local_files_only is None:
            local_only = bool(local_only_env)
        else:
            local_only = str(args.local_files_only).strip().lower() in {"1", "true", "yes", "on"}
        if local_only and hasattr(l2_mod, "_ensure_local_only_model_ready_or_raise"):
            l2_mod._ensure_local_only_model_ready_or_raise(base_model, local_only=True)

        import torch
        from peft import PeftModel
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig

        try:
            from transformers import AutoModelForImageTextToText
        except Exception:
            AutoModelForImageTextToText = None

        processor_src = str(processor_dir) if processor_dir is not None else str(base_model)
        try:
            processor = AutoProcessor.from_pretrained(processor_src, local_files_only=local_only)
        except Exception:
            processor = AutoTokenizer.from_pretrained(processor_src, local_files_only=local_only)

        def _infer_device() -> str:
            d = str(args.device or "auto").strip().lower()
            if d == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
            if d.startswith("cuda") and not torch.cuda.is_available():
                return "cpu"
            return d

        device = _infer_device()
        model_kwargs: Dict[str, Any] = {}
        if str(device).startswith("cuda"):
            try:
                import accelerate  # noqa: F401

                model_kwargs["device_map"] = "auto"
            except Exception:
                pass

        def _load_model(model_cls: Any) -> Any:
            try:
                return model_cls.from_pretrained(base_model, local_files_only=local_only, **model_kwargs)
            except ValueError as e:
                msg = str(e)
                if "requires `accelerate`" in msg and "device_map" in model_kwargs:
                    return model_cls.from_pretrained(base_model, local_files_only=local_only)
                raise

        if AutoModelForImageTextToText is not None:
            try:
                model = _load_model(AutoModelForImageTextToText)
            except Exception:
                model = _load_model(AutoModelForCausalLM)
        else:
            model = _load_model(AutoModelForCausalLM)

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        if hasattr(l2_mod, "_ensure_decoder_only_left_padding"):
            l2_mod._ensure_decoder_only_left_padding(processor, model)
        if not str(device).startswith("cuda") or "device_map" not in model_kwargs:
            model = model.to(device)
        model.eval()

        infer_model_device = getattr(l2_mod, "_infer_model_device", None)
        model_device = infer_model_device(model) if callable(infer_model_device) else next(model.parameters()).device

        mmad_root_raw = str(os.environ.get("MMAD_ROOT") or "").strip()
        mmad_root = Path(mmad_root_raw).resolve() if mmad_root_raw else None

        items = _read_jsonl(val_jsonl)
        # Stable summary semantics:
        # - samples_requested: requested evaluation count after max_samples cap
        # - samples_evaluated: samples that wrote exactly one sample_ok/sample_failed event
        # - samples_generation_ok: generate(...) succeeded
        # - samples_schema_valid: predicted answer passed contract schema
        # Scheme A semantics (fixed):
        # - sample_ok: generation succeeded + predicted answer schema valid
        # - sample_ok is NOT equivalent to correctness comparable
        # - sample_ok is NOT equivalent to correctness matched
        # - comparability/correctness readiness is tracked by dedicated counters/notes
        # - samples_failed: sample_failed events with a single primary failure_reason
        # - samples_comparison_ready: sample_ok samples with >=1 comparable correctness axis
        # - samples_gt_not_comparable: sample_ok samples with 0 comparable correctness axes
        # - samples_schema_valid_but_not_comparable: schema-valid samples with 0 comparable axes
        # - samples_with_audit_note: samples that wrote non-empty audit_note
        # Event-rollup formulas (from logs/eval_events.jsonl):
        # - samples_evaluated = count(event in {"sample_ok","sample_failed"})
        # - samples_failed = count(event == "sample_failed" and failure_reason != "")
        # - samples_schema_valid = count(answer_schema_valid == true)
        # - samples_comparison_ready = count(event == "sample_ok" and comparison_ready == true)
        # - samples_gt_not_comparable = count(event == "sample_ok" and comparison_ready == false)
        # - samples_schema_valid_but_not_comparable = count(event == "sample_ok" and answer_schema_valid == true and comparison_ready == false)
        # - samples_with_audit_note = count(event in {"sample_ok","sample_failed"} and audit_note != "")
        samples_requested = int(min(len(items), max(0, int(args.max_samples))))
        samples_evaluated = 0
        samples_generation_ok = 0
        samples_schema_valid = 0
        samples_failed = 0
        samples_comparison_ready = 0
        samples_gt_not_comparable = 0
        samples_schema_valid_but_not_comparable = 0
        samples_with_audit_note = 0
        anomaly_present_comparable_count = 0
        anomaly_present_match_count = 0
        top_anomaly_comparable_count = 0
        top_anomaly_match_count = 0

        _append_event(
            events_path,
            {
                "event": "start",
                "base_model": str(base_model),
                "adapter_dir": str(adapter_dir),
                "processor_dir": str(processor_dir) if processor_dir is not None else "",
                "val_jsonl": str(val_jsonl),
                "samples_requested": int(samples_requested),
            },
        )

        gen_cfg = GenerationConfig(
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

        for idx, item in enumerate(items):
            if idx >= samples_requested:
                break
            sample_id = str(item.get("sample_id") or "")
            gt = _extract_gt_answer(item, PaperContract)
            gt_anomaly_present = gt.anomaly_present
            gt_top_anomaly = gt.top_anomaly
            ex, err = _build_eval_example(item, PaperContract, repo_root=repo_root, mmad_root=mmad_root)
            if ex is None:
                samples_failed += 1
                samples_evaluated += 1
                reason = "build_eval_example_failed"
                build_detail = str(err or "").strip()
                gt_note = str(gt.reason or "").strip()
                notes: List[str] = []
                if build_detail:
                    notes.append(f"build_detail:{build_detail}")
                if gt_note:
                    notes.append(f"gt_note:{gt_note}")
                audit_note = "|".join(notes)
                if audit_note:
                    samples_with_audit_note += 1
                _append_event(
                    events_path,
                    {
                        "event": "sample_failed",
                        "sample_id": sample_id,
                        "image_loaded": False,
                        "generation_ok": False,
                        "answer_schema_valid": False,
                        "gt_anomaly_present": gt_anomaly_present,
                        "pred_anomaly_present": None,
                        "gt_top_anomaly": gt_top_anomaly,
                        "pred_top_anomaly": None,
                        "anomaly_present_match": None,
                        "top_anomaly_match": None,
                        "ap_comparable": False,
                        "top_anomaly_comparable": False,
                        "comparison_ready": False,
                        "failure_reason": reason,
                        "comparability_note": "",
                        # auxiliary note channel; does not change primary failure_reason
                        "audit_note": audit_note,
                    },
                )
                continue

            image_loaded = False
            generation_ok = False
            answer_schema_valid = False
            pred_anomaly_present = None
            pred_top_anomaly = None
            anomaly_present_match = None
            top_anomaly_match = None
            fail_reason = ""
            comparability_note = ""
            audit_note = ""

            def _set_fail(reason: str) -> None:
                nonlocal fail_reason
                # failure_reason: single primary cause, first hit wins, immutable afterwards.
                if not fail_reason:
                    fail_reason = str(reason or "").strip()

            try:
                if ex.image_path is None or not ex.image_path.exists():
                    raise FileNotFoundError(f"image_missing:{ex.image_path}")
                img = Image.open(ex.image_path).convert("RGB")
                image_loaded = True
                prompt_text = _build_chat_prompt(processor, ex.prompt_text)
                inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
                inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                with torch.no_grad():
                    gen_ids = model.generate(**inputs, generation_config=gen_cfg)
                generation_ok = True
                samples_generation_ok += 1

                input_ids = inputs.get("input_ids")
                decode_ids = gen_ids
                if input_ids is not None and hasattr(input_ids, "shape") and hasattr(gen_ids, "shape"):
                    decode_ids = gen_ids[:, input_ids.shape[1] :]
                if hasattr(processor, "batch_decode"):
                    decoded = processor.batch_decode(decode_ids, skip_special_tokens=True)
                else:
                    tok = getattr(processor, "tokenizer", None)
                    if tok is not None and hasattr(tok, "batch_decode"):
                        decoded = tok.batch_decode(decode_ids, skip_special_tokens=True)
                    else:
                        decoded = [str(decode_ids)]
                text_out = str(decoded[0] if decoded else "").strip()

                answer_xml = PaperContract.extract_answer_xml(text_out)
                if answer_xml is None:
                    _set_fail("missing_answer_tags")
                else:
                    try:
                        ans_obj = json.loads(answer_xml)
                        if isinstance(ans_obj, dict):
                            ok, _ = PaperContract.validate_schema(ans_obj)
                            answer_schema_valid = bool(ok)
                            if answer_schema_valid:
                                samples_schema_valid += 1
                                pred_ap = ans_obj.get("anomaly_present")
                                pred_ta = ans_obj.get("top_anomaly")
                                pred_anomaly_present = pred_ap if isinstance(pred_ap, bool) else None
                                # Keep stripped raw string for explicit comparability policy.
                                pred_top_anomaly = str(pred_ta).strip() if isinstance(pred_ta, str) else None
                            else:
                                _set_fail("answer_schema_invalid")
                        else:
                            _set_fail("answer_json_parse_error")
                    except Exception:
                        _set_fail("answer_json_parse_error")
            except Exception as e:
                _set_fail(f"generation_failed:{type(e).__name__}:{e}")

            samples_evaluated += 1

            ap_comparable = isinstance(gt_anomaly_present, bool) and isinstance(pred_anomaly_present, bool)
            if ap_comparable:
                anomaly_present_comparable_count += 1
                anomaly_present_match = bool(gt_anomaly_present == pred_anomaly_present)
                if anomaly_present_match:
                    anomaly_present_match_count += 1

            gt_top_norm, gt_top_kind = _classify_top_anomaly_value(gt_top_anomaly)
            pred_top_norm, pred_top_kind = _classify_top_anomaly_value(pred_top_anomaly)
            top_comparable = gt_top_kind == "category" and pred_top_kind == "category"
            if top_comparable:
                top_anomaly_comparable_count += 1
                top_anomaly_match = bool(str(gt_top_norm).lower() == str(pred_top_norm).lower())
                if top_anomaly_match:
                    top_anomaly_match_count += 1

            comparison_ready = bool(ap_comparable or top_comparable)
            if not fail_reason and (not generation_ok or not answer_schema_valid):
                if not generation_ok:
                    _set_fail("generation_failed:UnknownError:unknown")
                else:
                    _set_fail("answer_schema_invalid")

            gt_note = str(gt.reason or "").strip()
            if gt_note:
                audit_note = f"gt_note:{gt_note}"

            # event type is fully determined by primary failure presence.
            if fail_reason:
                event_type = "sample_failed"
                samples_failed += 1
            else:
                event_type = "sample_ok"
                if comparison_ready:
                    samples_comparison_ready += 1
                if not comparison_ready:
                    # sample_ok + schema_valid + no comparable axes
                    samples_gt_not_comparable += 1
                    notes: List[str] = []
                    if not ap_comparable:
                        notes.append("gt_anomaly_present_not_comparable")
                    if not top_comparable:
                        notes.append("gt_top_anomaly_not_comparable")
                    comparability_note = "|".join(notes)

            if event_type == "sample_ok" and answer_schema_valid and (not comparison_ready):
                samples_schema_valid_but_not_comparable += 1
            if audit_note:
                samples_with_audit_note += 1

            _append_event(
                events_path,
                {
                    "event": event_type,
                    "sample_id": str(ex.sample_id),
                    "image_loaded": bool(image_loaded),
                    "generation_ok": bool(generation_ok),
                    "answer_schema_valid": bool(answer_schema_valid),
                    "gt_anomaly_present": gt_anomaly_present,
                    "pred_anomaly_present": pred_anomaly_present,
                    "gt_top_anomaly": gt_top_anomaly,
                    "pred_top_anomaly": pred_top_anomaly,
                    "anomaly_present_match": anomaly_present_match,
                    "top_anomaly_match": top_anomaly_match,
                    "ap_comparable": bool(ap_comparable),
                    "top_anomaly_comparable": bool(top_comparable),
                    "comparison_ready": bool(comparison_ready),
                    # field responsibilities:
                    # - failure_reason: primary failure only (empty for sample_ok)
                    # - comparability_note: comparability-only note (empty unless not comparable)
                    # - audit_note: auxiliary audit context only, never a primary failure cause
                    "failure_reason": fail_reason,
                    "comparability_note": comparability_note,
                    "audit_note": audit_note,
                },
            )

        replay_counts, replay_invariant_mismatches = _replay_eval_events(events_path)
        summary_counts: Dict[str, int] = {
            "samples_evaluated": int(samples_evaluated),
            "samples_failed": int(samples_failed),
            "samples_generation_ok": int(samples_generation_ok),
            "samples_schema_valid": int(samples_schema_valid),
            "samples_comparison_ready": int(samples_comparison_ready),
            "samples_gt_not_comparable": int(samples_gt_not_comparable),
            "samples_schema_valid_but_not_comparable": int(samples_schema_valid_but_not_comparable),
            "samples_with_audit_note": int(samples_with_audit_note),
            "anomaly_present_comparable_count": int(anomaly_present_comparable_count),
            "anomaly_present_match_count": int(anomaly_present_match_count),
            "top_anomaly_comparable_count": int(top_anomaly_comparable_count),
            "top_anomaly_match_count": int(top_anomaly_match_count),
        }
        audit_mismatches: List[str] = list(replay_invariant_mismatches)
        for key, summary_val in summary_counts.items():
            replay_val = int(replay_counts.get(key, 0))
            if replay_val != summary_val:
                audit_mismatches.append(f"count_mismatch:{key}:summary={summary_val},replay={replay_val}")
        audit_pass = len(audit_mismatches) == 0

        summary.update(
            {
                "status": "PASS" if audit_pass else "FAIL",
                "base_model": str(base_model),
                "adapter_dir": str(adapter_dir),
                "processor_dir": str(processor_dir) if processor_dir is not None else "",
                "val_jsonl": str(val_jsonl),
                "samples_requested": int(samples_requested),
                "samples_evaluated": int(replay_counts["samples_evaluated"]),
                "samples_generation_ok": int(replay_counts["samples_generation_ok"]),
                "samples_schema_valid": int(replay_counts["samples_schema_valid"]),
                "samples_failed": int(replay_counts["samples_failed"]),
                "samples_comparison_ready": int(replay_counts["samples_comparison_ready"]),
                "samples_gt_not_comparable": int(replay_counts["samples_gt_not_comparable"]),
                "samples_schema_valid_but_not_comparable": int(replay_counts["samples_schema_valid_but_not_comparable"]),
                "samples_with_audit_note": int(replay_counts["samples_with_audit_note"]),
                "anomaly_present_comparable_count": int(replay_counts["anomaly_present_comparable_count"]),
                "anomaly_present_match_count": int(replay_counts["anomaly_present_match_count"]),
                "top_anomaly_comparable_count": int(replay_counts["top_anomaly_comparable_count"]),
                "top_anomaly_match_count": int(replay_counts["top_anomaly_match_count"]),
                "replay_counts": {k: int(v) for k, v in replay_counts.items()},
                "audit_pass": bool(audit_pass),
                "audit_mismatches": list(audit_mismatches),
                "script_sha256": _sha256_file(Path(__file__).resolve()),
                "events_path": str(events_path),
            }
        )
    except Exception as e:
        summary["errors"].append(f"{type(e).__name__}: {e}")
        _append_event(events_path, {"event": "error", "error": f"{type(e).__name__}: {e}"})
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print(f"phase4_1_eval_summary={summary_path}")
    print(f"status={summary.get('status')}")
    return 0 if summary.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
