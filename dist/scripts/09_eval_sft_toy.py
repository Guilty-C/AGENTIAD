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


def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _render_message(m: Mapping[str, Any]) -> str:
    role = str(m.get("role") or "")
    name = str(m.get("name") or "")
    if role == "assistant" and "tool_call" in m:
        return f"ASSISTANT({name}) TOOL_CALL: {_json_dumps_stable(m.get('tool_call'))}\n"
    if role.lower() == "tool":
        s = _json_dumps_stable(m.get("content"))
        if len(s) > 512:
            h = _sha256_upper_text(s)
            s = s[:256] + "...<TRUNC>..." + s[-128:] + "|sha256=" + h
        return f"TOOL({name}): {s}\n"
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


def _try_parse_json_dict(s: Any) -> Optional[Dict[str, Any]]:
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        try:
            obj = json.loads(s)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _normalize_text(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, str):
        return s.replace("\r\n", "\n").replace("\r", "\n").strip()
    if isinstance(s, dict):
        return json.dumps(s, ensure_ascii=False, sort_keys=True, separators=(",", ":")).strip()
    return str(s).strip()


def _extract_yesno(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    t = text.lower()
    if re.search(r"\b(yes|no)\b", t) is None:
        return None
    m = re.search(r"\"anomaly\"\s*:\s*\"(yes|no)\"", t)
    if m is not None:
        return str(m.group(1))
    m = re.search(r"\banomaly\b[^\n\r]{0,64}\b(yes|no)\b", t)
    if m is not None:
        return str(m.group(1))
    m = re.search(r"\b(yes|no)\b", t)
    if m is not None:
        return str(m.group(1))
    return None


def _extract_confidence(text: str) -> Optional[float]:
    if not isinstance(text, str):
        return None
    t = text.lower()
    m = re.search(r"\"confidence\"\s*:\s*([-+]?\d+(?:\.\d+)?)", t)
    if m is None:
        m = re.search(r"\bconfidence\b[^\n\r]{0,64}?([-+]?\d+(?:\.\d+)?)", t)
    if m is None:
        return None
    try:
        v = float(m.group(1))
    except Exception:
        return None
    if v < 0.0 or v > 1.0:
        return None
    return float(v)


def _extract_bbox4(text: str) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(text, str):
        return None
    t = text
    m = re.search(
        r"\"bbox\"\s*:\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]",
        t,
    )
    if m is None:
        m = re.search(
            r"\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]",
            t,
        )
    if m is None:
        return None
    try:
        x1 = float(m.group(1))
        y1 = float(m.group(2))
        x2 = float(m.group(3))
        y2 = float(m.group(4))
    except Exception:
        return None
    if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
        return None
    if not (x1 < x2 and y1 < y2):
        return None
    return float(x1), float(y1), float(x2), float(y2)


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def _hash_dir_files(path: Path) -> str:
    parts: List[bytes] = []
    for p in sorted([x for x in path.rglob("*") if x.is_file()], key=lambda x: str(x).replace("\\", "/")):
        rel = str(p.relative_to(path)).replace("\\", "/")
        parts.append(rel.encode("utf-8") + b"\n")
        parts.append(p.read_bytes())
        parts.append(b"\n")
    return _sha256_upper_bytes(b"".join(parts))


def main() -> int:
    project_root = _bootstrap_src()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    train_jsonl_path = _resolve_path(project_root, args.train_jsonl)
    adapter_dir: Optional[Path] = None
    if args.adapter_dir:
        adapter_dir = _resolve_path(project_root, args.adapter_dir)
        if not adapter_dir.exists():
            print(f"Missing adapter_dir: {adapter_dir}", file=sys.stderr)
            return 2

    snapshot_path = (adapter_dir.parent / "train_snapshot.json") if adapter_dir is not None else None
    snapshot: Dict[str, Any] = {}
    if snapshot_path is not None and snapshot_path.exists():
        try:
            snapshot = json.loads(_read_text(snapshot_path))
        except Exception:
            snapshot = {}

    base_model = str(args.base_model or snapshot.get("base_model") or "sshleifer/tiny-gpt2")
    data_hash = _sha256_upper_bytes(train_jsonl_path.read_bytes())
    adapter_hash = _hash_dir_files(adapter_dir) if adapter_dir is not None else ""

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        print(
            "Missing dependencies for LoRA eval.\n"
            "Install (CPU):\n"
            "  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  python -m pip install --upgrade transformers peft accelerate\n",
            file=sys.stderr,
        )
        return 2

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    base_ref = base
    is_adapter_loaded = adapter_dir is not None
    if adapter_dir is not None:
        base_for_adapter = AutoModelForCausalLM.from_pretrained(base_model).to(device)
        model = PeftModel.from_pretrained(base_for_adapter, adapter_dir).to(device)
    else:
        model = base_ref
    model.eval()
    base_ref.eval()
    max_ctx = int(getattr(getattr(model, "config", None), "n_positions", 1024) or 1024)
    lora_param_abs_sum = 0.0
    adapter_logits_delta_mean_abs = 0.0
    if adapter_dir is not None:
        try:
            import torch

            s = torch.tensor(0.0, device=device)
            for n, p in model.named_parameters():
                if ("lora_" in n) or ("lora" in n):
                    try:
                        s = s + p.detach().abs().sum()
                    except Exception:
                        continue
            lora_param_abs_sum = float(s.detach().cpu().item())
        except Exception:
            lora_param_abs_sum = 0.0

    items = _read_jsonl(train_jsonl_path)
    if not items:
        print("Empty train_jsonl.", file=sys.stderr)
        return 1

    n_items = int(len(items))
    n_final_msgs = 0
    n_final_json_parse_ok = 0
    n_tool_pairs_in_data = 0
    for it in items:
        msgs_any = it.get("messages")
        if not isinstance(msgs_any, list):
            continue
        msgs = [m for m in msgs_any if isinstance(m, dict)]
        for m in msgs:
            if m.get("role") == "assistant" and m.get("name") == "final":
                n_final_msgs += 1
                final_obj = _try_parse_json_dict(m.get("content"))
                if final_obj is not None:
                    n_final_json_parse_ok += 1
                break
        for i in range(len(msgs) - 1):
            m0 = msgs[i]
            m1 = msgs[i + 1]
            if m0.get("role") == "assistant" and "tool_call" in m0 and m1.get("role") == "tool":
                n_tool_pairs_in_data += 1

    data_final_json_rate = float(n_final_json_parse_ok) / float(n_final_msgs) if n_final_msgs else 0.0

    max_samples = max(1, int(args.max_samples))
    subset = items[:max_samples]

    json_ok = 0
    json_total = 0
    prefix_json_ok = 0
    exact_match_ok = 0
    exact_match_total = 0
    contains_lbrace_n = 0
    contains_anomaly_key_n = 0
    contains_final_json_prefix_n = 0
    gen_empty_n = 0
    gen_chars_total = 0
    has_toolcall_generated = False
    toolcall_probe_name: Optional[str] = None
    has_tool_structure_in_data = False
    toolcall_generated_in_main_samples = 0
    toolcall_generated_example: Optional[str] = None
    gen_first_sample_head: Optional[str] = None
    prompt_first_sample_tail: Optional[str] = None
    prompt_first_sample_token_len: Optional[int] = None
    prompt_first_sample_token_len_trunc: Optional[int] = None

    def _avg_logprob_per_token(mdl: Any, prompt_text: str, cont_text: str) -> Tuple[float, int]:
        import torch
        import torch.nn.functional as F

        full_text = prompt_text + cont_text
        max_len = max(8, max_ctx - 1)
        enc_full = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        enc_prompt = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        input_ids = enc_full["input_ids"]
        prompt_len = int(enc_prompt["input_ids"].shape[1])
        cont_len = int(input_ids.shape[1]) - int(prompt_len)
        if cont_len <= 0:
            return 0.0, 0

        out_logits = mdl(input_ids=input_ids).logits
        logp = F.log_softmax(out_logits[:, :-1, :], dim=-1)
        tgt = input_ids[:, 1:]
        token_logp = logp.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)

        start = max(0, prompt_len - 1)
        end = min(int(start + cont_len), int(token_logp.shape[1]))
        if end <= start:
            return 0.0, 0
        s = token_logp[:, start:end].sum()
        n = int(end - start)
        return float((s / max(1, n)).detach().cpu().item()), n

    gold_logp_model_sum: Optional["torch.Tensor"] = None
    gold_logp_base_sum: Optional["torch.Tensor"] = None
    gold_tok_total_model = 0
    gold_tok_total_base = 0
    prompt_tok_sum = 0
    prompt_tok_count = 0
    prompt_truncated_n = 0
    expected_tok_sum = 0
    expected_tok_count = 0

    anomaly_any = 0
    anomaly_match = 0
    anomaly_total = 0
    confidence_any = 0
    confidence_abs_err_sum = 0.0
    confidence_total = 0
    bbox_any = 0
    bbox_iou_sum = 0.0
    bbox_total = 0

    for it in subset:
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

        prompt = _render_prefix(msgs, final_idx)
        prompt += "ASSISTANT(final): "

        expected_final_text = _normalize_text(msgs[final_idx].get("content"))
        expected_obj = _try_parse_json_dict(msgs[final_idx].get("content"))
        expected_anomaly: Optional[str] = None
        expected_confidence: Optional[float] = None
        expected_bbox: Optional[Tuple[float, float, float, float]] = None
        if isinstance(expected_obj, dict):
            a = expected_obj.get("anomaly")
            if isinstance(a, str) and a.strip().lower() in {"yes", "no"}:
                expected_anomaly = a.strip().lower()
            c = expected_obj.get("confidence")
            if isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0:
                expected_confidence = float(c)
            bb = expected_obj.get("bbox")
            if isinstance(bb, list) and len(bb) == 4:
                try:
                    x1, y1, x2, y2 = [float(x) for x in bb]
                    if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0 and x1 < x2 and y1 < y2:
                        expected_bbox = (x1, y1, x2, y2)
                except Exception:
                    expected_bbox = None

        if prompt_first_sample_tail is None:
            prompt_first_sample_tail = prompt[-200:].replace("\r", " ").replace("\n", " ")
            try:
                enc_dbg = tokenizer(prompt, return_tensors=None, truncation=False)
                prompt_first_sample_token_len = int(len(enc_dbg.get("input_ids") or []))
            except Exception:
                prompt_first_sample_token_len = None
        try:
            enc_dbg2 = tokenizer(prompt, return_tensors=None, truncation=False)
            ptok = int(len(enc_dbg2.get("input_ids") or []))
            prompt_tok_sum += int(ptok)
            prompt_tok_count += 1
        except Exception:
            pass
        if expected_final_text:
            try:
                enc_dbg3 = tokenizer(expected_final_text, return_tensors=None, truncation=False)
                etok = int(len(enc_dbg3.get("input_ids") or []))
                expected_tok_sum += int(etok)
                expected_tok_count += 1
            except Exception:
                pass

        max_prompt_len = max(8, max_ctx - int(args.max_new_tokens) - 1)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
        try:
            enc_dbg4 = tokenizer(prompt, return_tensors=None, truncation=False)
            if int(len(enc_dbg4.get("input_ids") or [])) > int(max_prompt_len):
                prompt_truncated_n += 1
        except Exception:
            pass
        if prompt_first_sample_token_len_trunc is None:
            try:
                prompt_first_sample_token_len_trunc = int(enc["input_ids"].shape[1])
            except Exception:
                prompt_first_sample_token_len_trunc = None
        if (adapter_dir is not None) and adapter_logits_delta_mean_abs == 0.0:
            try:
                import torch

                with torch.no_grad():
                    logits_base = base_ref(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask")).logits
                    logits_peft = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask")).logits
                    d = (logits_peft[:, -1, :] - logits_base[:, -1, :]).abs().mean()
                    adapter_logits_delta_mean_abs = float(d.detach().cpu().item())
            except Exception:
                adapter_logits_delta_mean_abs = 0.0
        gen = model.generate(
            **enc,
            do_sample=False,
            top_p=1.0,
            max_new_tokens=int(args.max_new_tokens),
            pad_token_id=int(tokenizer.eos_token_id),
        )
        out = tokenizer.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        out_norm = _normalize_text(out)
        if gen_first_sample_head is None:
            s = out_norm.replace("\r", " ").replace("\n", " ").strip()
            gen_first_sample_head = s[:200]
        gen_chars_total += int(len(out_norm))
        if not out_norm:
            gen_empty_n += 1

        gold_avg_lp_model, gold_n_tok_model = _avg_logprob_per_token(model, prompt, expected_final_text)
        if gold_n_tok_model > 0:
            if gold_logp_model_sum is None:
                import torch

                gold_logp_model_sum = torch.tensor(0.0, device=device)
            gold_logp_model_sum = gold_logp_model_sum + float(gold_avg_lp_model) * int(gold_n_tok_model)
            gold_tok_total_model += int(gold_n_tok_model)
        gold_avg_lp_base, gold_n_tok_base = _avg_logprob_per_token(base_ref, prompt, expected_final_text)
        if gold_n_tok_base > 0:
            if gold_logp_base_sum is None:
                import torch

                gold_logp_base_sum = torch.tensor(0.0, device=device)
            gold_logp_base_sum = gold_logp_base_sum + float(gold_avg_lp_base) * int(gold_n_tok_base)
            gold_tok_total_base += int(gold_n_tok_base)

        js = _extract_first_json(out)
        json_total += 1
        pred_obj: Optional[Dict[str, Any]] = None
        if js is not None:
            try:
                obj = json.loads(js)
                if isinstance(obj, dict):
                    json_ok += 1
                    pred_obj = obj
            except Exception:
                pass
        if "FINAL_JSON:" in out:
            contains_final_json_prefix_n += 1
            js_p = _extract_json_after_prefix(out, "FINAL_JSON:")
            if js_p is not None:
                try:
                    obj_p = json.loads(js_p)
                    if isinstance(obj_p, dict):
                        prefix_json_ok += 1
                except Exception:
                    pass
        if "{" in out:
            contains_lbrace_n += 1
        if ("\"anomaly\"" in out) or ("anomaly" in out and "\"anomaly\"" not in out):
            contains_anomaly_key_n += 1

        pred_anomaly: Optional[str] = None
        pred_confidence: Optional[float] = None
        pred_bbox: Optional[Tuple[float, float, float, float]] = None
        if isinstance(pred_obj, dict):
            a = pred_obj.get("anomaly")
            if isinstance(a, str) and a.strip().lower() in {"yes", "no"}:
                pred_anomaly = a.strip().lower()
            c = pred_obj.get("confidence")
            if isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0:
                pred_confidence = float(c)
            bb = pred_obj.get("bbox")
            if isinstance(bb, list) and len(bb) == 4:
                try:
                    x1, y1, x2, y2 = [float(x) for x in bb]
                    if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0 and x1 < x2 and y1 < y2:
                        pred_bbox = (x1, y1, x2, y2)
                except Exception:
                    pred_bbox = None
        if pred_anomaly is None:
            pred_anomaly = _extract_yesno(out_norm)
        if pred_confidence is None:
            pred_confidence = _extract_confidence(out_norm)
        if pred_bbox is None:
            pred_bbox = _extract_bbox4(out_norm)

        if expected_anomaly is not None:
            anomaly_total += 1
            if pred_anomaly is not None:
                anomaly_any += 1
                if pred_anomaly == expected_anomaly:
                    anomaly_match += 1
        if expected_confidence is not None:
            confidence_total += 1
            if pred_confidence is not None:
                confidence_any += 1
                confidence_abs_err_sum += abs(float(pred_confidence) - float(expected_confidence))
        if expected_bbox is not None:
            bbox_total += 1
            if pred_bbox is not None:
                bbox_any += 1
                bbox_iou_sum += float(_bbox_iou(pred_bbox, expected_bbox))

        if expected_final_text:
            exact_match_total += 1
            if out_norm == expected_final_text:
                exact_match_ok += 1
        if "TOOL_CALL:" in out:
            tail = out.split("TOOL_CALL:", 1)[1]
            js_tool = _extract_first_json(tail)
            if js_tool is not None:
                try:
                    obj_tool = json.loads(js_tool)
                    if isinstance(obj_tool, dict):
                        toolcall_generated_in_main_samples += 1
                        if toolcall_generated_example is None:
                            toolcall_generated_example = tail.strip()
                except Exception:
                    pass

        toolcall_idx = None
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant" and "tool_call" in m:
                toolcall_idx = i
                break
        if toolcall_idx is not None and toolcall_probe_name is None:
            toolcall_probe_name = str(msgs[toolcall_idx].get("name") or "")
        for i in range(len(msgs) - 1):
            m0 = msgs[i]
            m1 = msgs[i + 1]
            if m0.get("role") == "assistant" and "tool_call" in m0 and m1.get("role") == "tool":
                has_tool_structure_in_data = True
                break

    if toolcall_probe_name:
        prompt2 = f"ASSISTANT({toolcall_probe_name}) TOOL_CALL: "
        max_prompt_len2 = max(8, max_ctx - 96 - 1)
        enc2 = tokenizer(prompt2, return_tensors="pt", truncation=True, max_length=max_prompt_len2).to(device)
        gen2 = model.generate(
            **enc2,
            do_sample=False,
            top_p=1.0,
            max_new_tokens=96,
            pad_token_id=int(tokenizer.eos_token_id),
        )
        out2 = tokenizer.decode(gen2[0][enc2["input_ids"].shape[1] :], skip_special_tokens=True)
        js2 = _extract_first_json(out2)
        if js2 is not None:
            try:
                obj2 = json.loads(js2)
                if isinstance(obj2, dict):
                    has_toolcall_generated = True
            except Exception:
                has_toolcall_generated = False

    json_parse_rate = float(json_ok) / float(json_total) if json_total else 0.0
    prefix_json_rate = float(prefix_json_ok) / float(json_total) if json_total else 0.0
    exact_match_rate = float(exact_match_ok) / float(exact_match_total) if exact_match_total else 0.0
    contains_lbrace_rate = float(contains_lbrace_n) / float(json_total) if json_total else 0.0
    contains_anomaly_key_rate = float(contains_anomaly_key_n) / float(json_total) if json_total else 0.0
    contains_final_json_prefix_rate = float(contains_final_json_prefix_n) / float(json_total) if json_total else 0.0
    gen_mean_chars = float(gen_chars_total) / float(json_total) if json_total else 0.0
    gold_final_avg_logprob_per_token = (
        float((gold_logp_model_sum / float(max(1, gold_tok_total_model))).detach().cpu().item())
        if (gold_logp_model_sum is not None)
        else 0.0
    )
    gold_final_avg_logprob_per_token_base = (
        float((gold_logp_base_sum / float(max(1, gold_tok_total_base))).detach().cpu().item()) if (gold_logp_base_sum is not None) else 0.0
    )
    gold_final_avg_logprob_per_token_delta = float(gold_final_avg_logprob_per_token - gold_final_avg_logprob_per_token_base)

    prompt_tok_mean = float(prompt_tok_sum) / float(prompt_tok_count) if prompt_tok_count else 0.0
    expected_final_tok_mean = float(expected_tok_sum) / float(expected_tok_count) if expected_tok_count else 0.0
    prompt_truncated_rate = float(prompt_truncated_n) / float(prompt_tok_count) if prompt_tok_count else 0.0
    anomaly_extract_rate = float(anomaly_any) / float(anomaly_total) if anomaly_total else 0.0
    anomaly_match_rate = float(anomaly_match) / float(anomaly_total) if anomaly_total else 0.0
    confidence_extract_rate = float(confidence_any) / float(confidence_total) if confidence_total else 0.0
    confidence_mae = float(confidence_abs_err_sum) / float(confidence_any) if confidence_any else 0.0
    bbox_extract_rate = float(bbox_any) / float(bbox_total) if bbox_total else 0.0
    bbox_iou_mean = float(bbox_iou_sum) / float(bbox_any) if bbox_any else 0.0

    print(f"seed={int(args.seed)}")
    print(f"base_model={base_model}")
    print(f"train_jsonl={str(train_jsonl_path)}")
    print(f"adapter_dir={(str(adapter_dir) if adapter_dir is not None else '(empty)')}")
    print(f"is_adapter_loaded={bool(is_adapter_loaded)}")
    print(f"data_hash={data_hash}")
    if adapter_dir is not None:
        print(f"adapter_hash={adapter_hash}")
    print(f"n_items={n_items}")
    print(f"n_final_msgs={int(n_final_msgs)}")
    print(f"n_final_json_parse_ok={int(n_final_json_parse_ok)}")
    print(f"n_tool_pairs_in_data={int(n_tool_pairs_in_data)}")
    print(f"data_final_json_rate={data_final_json_rate:.4f}")
    print(f"json_total={json_total}")
    print(f"json_ok={json_ok}")
    print(f"gen_empty_n={int(gen_empty_n)}")
    print(f"gen_mean_chars={gen_mean_chars:.2f}")
    print(f"gold_final_avg_logprob_per_token={gold_final_avg_logprob_per_token:.12f}")
    print(f"gold_final_avg_logprob_per_token_base={gold_final_avg_logprob_per_token_base:.12f}")
    print(f"gold_final_avg_logprob_per_token_delta={gold_final_avg_logprob_per_token_delta:.12f}")
    if adapter_dir is not None:
        print(f"lora_param_abs_sum={lora_param_abs_sum:.6f}")
    if prompt_first_sample_tail is not None:
        print(f"prompt_first_sample_tail={prompt_first_sample_tail}")
    if prompt_first_sample_token_len is not None:
        print(f"prompt_first_sample_token_len={int(prompt_first_sample_token_len)}")
    if prompt_first_sample_token_len_trunc is not None:
        print(f"prompt_first_sample_token_len_trunc={int(prompt_first_sample_token_len_trunc)}")
    if gen_first_sample_head is not None:
        print(f"gen_first_sample_head={gen_first_sample_head}")
    print(f"has_tool_structure_in_data={bool(has_tool_structure_in_data)}")
    if snapshot:
        cfg_hash = snapshot.get("config_hash")
        if cfg_hash:
            print(f"config_hash={cfg_hash}")

    if toolcall_generated_in_main_samples > 0 and toolcall_generated_example:
        ex = toolcall_generated_example.replace("\r", " ").replace("\n", " ").strip()
        if len(ex) > 200:
            ex = ex[:200]
        print(f"toolcall_generated_in_main_samples_example={ex}")

    print(f"prompt_tok_mean={prompt_tok_mean:.2f}")
    print(f"prompt_truncated_rate={prompt_truncated_rate:.4f}")
    print(f"expected_final_tok_mean={expected_final_tok_mean:.2f}")
    print(f"anomaly_extract_rate={anomaly_extract_rate:.4f}")
    print(f"anomaly_match_rate={anomaly_match_rate:.4f}")
    print(f"confidence_extract_rate={confidence_extract_rate:.4f}")
    print(f"confidence_mae={confidence_mae:.4f}")
    print(f"bbox_extract_rate={bbox_extract_rate:.4f}")
    print(f"bbox_iou_mean={bbox_iou_mean:.4f}")
    print(f"exact_match_rate={exact_match_rate:.4f}")
    print(f"prefix_json_rate={prefix_json_rate:.4f}")
    print(f"contains_lbrace_rate={contains_lbrace_rate:.4f}")
    print(f"contains_anomaly_key_rate={contains_anomaly_key_rate:.4f}")
    print(f"contains_final_json_prefix_rate={contains_final_json_prefix_rate:.4f}")

    print(f"toolcall_generated_in_main_samples_hits={int(toolcall_generated_in_main_samples)}")
    print(f"has_toolcall_generated_in_main_samples={bool(toolcall_generated_in_main_samples > 0)}")
    print(f"has_toolcall_generated={bool(has_toolcall_generated)}")
    print(f"json_parse_rate={json_parse_rate:.4f}")
    print(f"gold_final_avg_logprob_per_token_delta={gold_final_avg_logprob_per_token_delta:.12f}")
    print(f"adapter_logits_delta_mean_abs={adapter_logits_delta_mean_abs:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
