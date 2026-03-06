from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
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


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    data = yaml.safe_load(_read_text(path))
    if not isinstance(data, dict):
        raise ValueError("config must be YAML mapping")
    return data


def _merge_cfg(base: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in cur.items():
        if k == "base_config":
            continue
        out[k] = v
    return out


def _load_cfg(repo_root: Path, cfg_path: Path) -> Dict[str, Any]:
    cur = _load_yaml(cfg_path)
    base_raw = str(cur.get("base_config") or "").strip()
    if not base_raw:
        return cur
    base_path = (repo_root / base_raw).resolve() if not Path(base_raw).is_absolute() else Path(base_raw).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"base_config not found: {base_path}")
    base = _load_cfg(repo_root, base_path)
    return _merge_cfg(base, cur)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ln in _read_text(path).splitlines():
        s = ln.strip()
        if not s:
            continue
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError(f"jsonl line not object: {path}")
        items.append(obj)
    return items


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


def _resolve_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    p1 = (Path.cwd() / p).resolve()
    if p1.exists():
        return p1
    return (repo_root / p).resolve()


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(event)
    rec.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


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
    spec = importlib.util.spec_from_file_location("agentiad_l2_infer_mod", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _set_seed(seed: int) -> None:
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


def _bool_cfg(x: Any, default: bool) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _infer_device(device_raw: str) -> str:
    import torch

    d = str(device_raw or "auto").strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return d


def _resolve_torch_dtype(torch_mod: Any, spec: Any, use_cuda: bool) -> Any:
    if spec is None:
        return None
    if isinstance(spec, str):
        s = spec.strip().lower()
        if not s or s == "none":
            return None
        if s == "auto":
            return torch_mod.float16 if use_cuda else torch_mod.float32
        if s in {"float16", "fp16", "half"}:
            return torch_mod.float16
        if s in {"bfloat16", "bf16"}:
            return torch_mod.bfloat16
        if s in {"float32", "fp32"}:
            return torch_mod.float32
        return spec
    return spec


def _canonical_answer_content(content: str, paper_contract: Any) -> Optional[str]:
    try:
        answer_xml = paper_contract.extract_answer_xml(content)
        if answer_xml is None:
            return None
        obj = json.loads(answer_xml)
        if isinstance(obj, dict):
            ok, _ = paper_contract.validate_schema(obj)
            if ok:
                return f"{paper_contract.TAG_ANSWER_START}\n{_safe_json(obj)}\n{paper_contract.TAG_ANSWER_END}"
        return f"{paper_contract.TAG_ANSWER_START}\n{answer_xml.strip()}\n{paper_contract.TAG_ANSWER_END}"
    except Exception:
        return None


def _build_placeholder_image(path: Path, size: int) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        Image.new("RGB", (int(size), int(size)), color=(127, 127, 127)).save(path)
    return path


def _resolve_image_candidate(
    raw: str,
    sample_dir: Optional[Path],
    mmad_root: Optional[Path],
    repo_root: Path,
) -> Optional[Path]:
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
class VLMExample:
    sample_id: str
    prompt_text: str
    target_text: str
    image_path: Path


def _trajectory_to_example(
    item: Mapping[str, Any],
    paper_contract: Any,
    repo_root: Path,
    mmad_root: Optional[Path],
    placeholder_image: Path,
    require_tool_call: bool,
) -> Tuple[Optional[VLMExample], Optional[str], bool]:
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
    target_parts: List[str] = []
    has_tool_call = False
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
            else:
                s = str(content or "").strip()
                if s:
                    context_lines.append(f"USER[{name}]: {s}")
            continue
        if role == "tool":
            context_lines.append(f"TOOL[{name}]: {_safe_json(m.get('content'))}")
            continue
        if role != "assistant":
            continue
        tool_call = m.get("tool_call")
        if isinstance(tool_call, Mapping):
            has_tool_call = True
            target_parts.append(
                f"{paper_contract.TAG_TOOL_CALL_START}\n{_safe_json(dict(tool_call))}\n{paper_contract.TAG_TOOL_CALL_END}"
            )
        content = m.get("content")
        if str(name) == "final" and isinstance(content, str):
            ans = _canonical_answer_content(content, paper_contract)
            if ans:
                target_parts.append(ans)

    if not target_parts:
        return None, "missing_supervision_targets", False
    if require_tool_call and not has_tool_call:
        return None, "missing_tool_call_supervision", False
    if not any(str(x).startswith(paper_contract.TAG_ANSWER_START) for x in target_parts):
        return None, "missing_final_answer_supervision", False

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

    used_placeholder = False
    if resolved_image is None:
        resolved_image = placeholder_image
        used_placeholder = True

    prompt_text = (
        f"{paper_contract.SYSTEM_PROMPT}\n\n"
        "Trajectory context:\n"
        + ("\n".join(context_lines) if context_lines else "(empty)")
        + "\n\nContinue the assistant trajectory strictly according to the contract."
    )
    target_text = "\n".join(target_parts)
    return VLMExample(sample_id=sample_id, prompt_text=prompt_text, target_text=target_text, image_path=resolved_image), None, used_placeholder


class VLMDataCollator:
    def __init__(self, processor: Any, max_length: int):
        self.processor = processor
        self.max_length = int(max_length)
        self._chat_cache: Dict[Tuple[str, str, bool], str] = {}

    def _chat_text(self, prompt_text: str, target_text: str, add_generation_prompt: bool) -> str:
        key = (prompt_text, target_text, bool(add_generation_prompt))
        if key in self._chat_cache:
            return self._chat_cache[key]
        if hasattr(self.processor, "apply_chat_template") and getattr(self.processor, "chat_template", None):
            if add_generation_prompt:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": target_text}],
                    },
                ]
            txt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=bool(add_generation_prompt))
        else:
            base = f"<image>\n{prompt_text}\n"
            txt = base + ("Assistant:" if add_generation_prompt else f"Assistant:\n{target_text}")
        self._chat_cache[key] = txt
        return txt

    def __call__(self, batch: Sequence[VLMExample]) -> Dict[str, Any]:
        from PIL import Image

        prompts: List[str] = []
        fulls: List[str] = []
        images: List[Any] = []

        for ex in batch:
            img = Image.open(ex.image_path).convert("RGB")
            images.append(img)
            prompts.append(self._chat_text(ex.prompt_text, ex.target_text, add_generation_prompt=True))
            fulls.append(self._chat_text(ex.prompt_text, ex.target_text, add_generation_prompt=False))

        enc_full = self.processor(
            text=fulls,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc_prompt = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = enc_full["input_ids"]
        labels = input_ids.clone()
        labels[:] = -100

        full_attn = enc_full.get("attention_mask")
        prompt_attn = enc_prompt.get("attention_mask")
        if full_attn is None or prompt_attn is None:
            raise RuntimeError("processor output missing attention_mask; cannot build SFT labels")

        bs = int(input_ids.shape[0])
        for i in range(bs):
            prompt_len = int(prompt_attn[i].sum().item())
            full_len = int(full_attn[i].sum().item())
            start = max(0, min(prompt_len, full_len))
            if full_len > start:
                labels[i, start:full_len] = input_ids[i, start:full_len]

        supervision_token_count = labels.ne(-100).sum(dim=1)
        enc_full["labels"] = labels
        enc_full["supervision_token_count"] = supervision_token_count
        return enc_full


def _infer_target_modules(model: Any) -> List[str]:
    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "Wqkv",
        "wo",
        "wq",
        "wk",
        "wv",
        "c_attn",
        "c_proj",
    ]
    found: List[str] = []
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates and leaf not in found:
            found.append(leaf)
    return found if found else ["q_proj", "v_proj"]


def main() -> int:
    repo_root = _bootstrap_repo()
    from agentiad_repro.paper_contract import PaperContract

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_jsonl", "--train-jsonl", dest="train_jsonl", type=str, default=None)
    parser.add_argument("--val_jsonl", "--val-jsonl", dest="val_jsonl", type=str, default=None)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--base_model", "--base-model", dest="base_model", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--vlm-model-id", type=str, default=None)
    args = parser.parse_args()

    cfg_path = _resolve_path(repo_root, str(args.config))
    cfg = _load_cfg(repo_root, cfg_path)

    train_jsonl = _resolve_path(repo_root, str(args.train_jsonl or cfg.get("train_jsonl") or ""))
    val_jsonl_raw = str(args.val_jsonl or cfg.get("val_jsonl") or "").strip()
    val_jsonl = _resolve_path(repo_root, val_jsonl_raw) if val_jsonl_raw else None

    out_dir_raw = str(args.output_dir or cfg.get("output_dir") or "").strip()
    if not out_dir_raw:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir_raw = f"dist/outputs/phase4_1_vlm_lora_smoke_{ts}"
    output_dir = _resolve_path(repo_root, out_dir_raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "phase4_1_summary.json"
    summary: Dict[str, Any] = {
        "phase": "4.1",
        "mode": "vlm_lora_sft_smoke",
        "status": "FAIL",
        "config_path": str(cfg_path),
        "output_dir": str(output_dir),
        "errors": [],
    }
    event_log = output_dir / "logs" / "train_events.jsonl"
    _append_event(
        event_log,
        {
            "event": "start",
            "config_path": str(cfg_path),
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl) if val_jsonl is not None else "",
            "output_dir": str(output_dir),
        },
    )

    try:
        if not train_jsonl.exists():
            raise FileNotFoundError(f"train_jsonl not found: {train_jsonl}")
        if val_jsonl is not None and not val_jsonl.exists():
            raise FileNotFoundError(f"val_jsonl not found: {val_jsonl}")

        seed = int(cfg.get("seed", 0))
        _set_seed(seed)

        max_train_samples = int(args.max_train_samples if args.max_train_samples is not None else cfg.get("max_train_samples", 32))
        max_val_samples = int(args.max_val_samples if args.max_val_samples is not None else cfg.get("max_val_samples", 8))
        max_steps = int(args.max_steps if args.max_steps is not None else cfg.get("max_steps", 1))
        batch_size = int(cfg.get("batch_size", 1))
        grad_accum = int(cfg.get("grad_accum", 1))
        lr = float(cfg.get("learning_rate", 5e-5))
        max_length = int(cfg.get("max_length", 2048))
        require_tool_call = _bool_cfg(cfg.get("require_tool_call", True), True)
        placeholder_size = int(cfg.get("placeholder_image_size", 448))

        l2_mod = _load_l2_module(repo_root)

        vlm_model_id = str(
            args.base_model
            or args.vlm_model_id
            or cfg.get("base_model")
            or cfg.get("vlm_model_id")
            or cfg.get("model_id")
            or ""
        ).strip()
        if not vlm_model_id:
            raise ValueError("missing base_model/vlm_model_id in config and CLI")
        vlm_model_id = _normalize_model_ref(vlm_model_id)
        _append_event(event_log, {"event": "model_resolved", "base_model": vlm_model_id})

        offline_env = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        local_only = _bool_cfg(cfg.get("local_files_only"), offline_env)
        if local_only and hasattr(l2_mod, "_ensure_local_only_model_ready_or_raise"):
            l2_mod._ensure_local_only_model_ready_or_raise(vlm_model_id, local_only=True)

        mmad_root_raw = str(cfg.get("mmad_root") or os.environ.get("MMAD_ROOT") or "").strip()
        mmad_root = Path(mmad_root_raw).resolve() if mmad_root_raw else None

        placeholder_image = _build_placeholder_image(output_dir / "placeholder" / "missing_image.png", placeholder_size)

        train_items = _read_jsonl(train_jsonl)
        val_items = _read_jsonl(val_jsonl) if val_jsonl is not None else []
        _append_event(
            event_log,
            {
                "event": "dataset_loaded",
                "train_source_count": int(len(train_items)),
                "val_source_count": int(len(val_items)),
            },
        )

        train_examples: List[VLMExample] = []
        val_examples: List[VLMExample] = []
        reject_reasons: Dict[str, int] = {}
        train_placeholder_count = 0
        val_placeholder_count = 0

        for it in train_items:
            ex, err, used_ph = _trajectory_to_example(
                item=it,
                paper_contract=PaperContract,
                repo_root=repo_root,
                mmad_root=mmad_root,
                placeholder_image=placeholder_image,
                require_tool_call=require_tool_call,
            )
            if ex is None:
                key = str(err or "unknown")
                reject_reasons[key] = int(reject_reasons.get(key, 0)) + 1
                continue
            train_examples.append(ex)
            if used_ph:
                train_placeholder_count += 1
            if len(train_examples) >= max_train_samples:
                break

        for it in val_items:
            ex, err, used_ph = _trajectory_to_example(
                item=it,
                paper_contract=PaperContract,
                repo_root=repo_root,
                mmad_root=mmad_root,
                placeholder_image=placeholder_image,
                require_tool_call=require_tool_call,
            )
            if ex is None:
                continue
            val_examples.append(ex)
            if used_ph:
                val_placeholder_count += 1
            if len(val_examples) >= max_val_samples:
                break

        if not train_examples:
            raise RuntimeError("no valid train examples after trajectory conversion")
        _append_event(
            event_log,
            {
                "event": "dataset_converted",
                "train_used_count": int(len(train_examples)),
                "val_used_count": int(len(val_examples)),
                "train_placeholder_image_count": int(train_placeholder_count),
                "val_placeholder_image_count": int(val_placeholder_count),
                "rejected_reasons": {k: int(v) for k, v in sorted(reject_reasons.items(), key=lambda kv: kv[0])},
            },
        )

        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        try:
            from transformers import AutoModelForImageTextToText
        except Exception:
            AutoModelForImageTextToText = None

        processor: Any
        try:
            processor = AutoProcessor.from_pretrained(vlm_model_id, local_files_only=local_only)
        except Exception:
            processor = AutoTokenizer.from_pretrained(vlm_model_id, local_files_only=local_only)

        if not callable(getattr(processor, "__call__", None)):
            raise RuntimeError("processor/tokenizer is not callable")

        model_kwargs: Dict[str, Any] = {}
        device = _infer_device(str(cfg.get("device", "auto")))
        use_cuda = str(device).startswith("cuda")
        torch_dtype_eff = _resolve_torch_dtype(torch, cfg.get("torch_dtype", "auto"), use_cuda=use_cuda)
        if torch_dtype_eff is not None:
            model_kwargs["torch_dtype"] = torch_dtype_eff

        accelerate_ok = False
        device_map = cfg.get("device_map", "auto")
        if use_cuda and str(device_map).strip().lower() not in {"", "none"}:
            try:
                import accelerate  # noqa: F401

                accelerate_ok = True
                model_kwargs["device_map"] = device_map
            except Exception:
                accelerate_ok = False

        def _load_model(model_cls: Any) -> Any:
            try:
                return model_cls.from_pretrained(vlm_model_id, local_files_only=local_only, **model_kwargs)
            except ValueError as e:
                msg = str(e)
                if "requires `accelerate`" in msg and "device_map" in model_kwargs:
                    return model_cls.from_pretrained(vlm_model_id, local_files_only=local_only)
                raise

        model: Any
        if AutoModelForImageTextToText is not None:
            try:
                model = _load_model(AutoModelForImageTextToText)
            except Exception:
                model = _load_model(AutoModelForCausalLM)
        else:
            model = _load_model(AutoModelForCausalLM)

        if hasattr(l2_mod, "_ensure_decoder_only_left_padding"):
            l2_mod._ensure_decoder_only_left_padding(processor, model)

        if (not use_cuda) or (not accelerate_ok):
            model = model.to(device)

        target_modules_raw = cfg.get("target_modules")
        if isinstance(target_modules_raw, str):
            target_modules = [x.strip() for x in target_modules_raw.split(",") if x.strip()]
        elif isinstance(target_modules_raw, list):
            target_modules = [str(x).strip() for x in target_modules_raw if str(x).strip()]
        else:
            target_modules = []
        if not target_modules:
            target_modules = _infer_target_modules(model)

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.train()

        collator = VLMDataCollator(processor=processor, max_length=max_length)
        train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=False, collate_fn=collator)
        val_loader = DataLoader(val_examples, batch_size=batch_size, shuffle=False, collate_fn=collator) if val_examples else None

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("no trainable LoRA parameters")
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        infer_model_device = getattr(l2_mod, "_infer_model_device", None)
        model_device = infer_model_device(model) if callable(infer_model_device) else next(model.parameters()).device

        global_step = 0
        accum = 0
        train_loss_last = None
        train_batches_skipped_no_supervision = 0
        val_batches_seen = 0
        val_batches_with_supervision = 0
        val_batches_skipped_no_supervision = 0
        t0 = time.time()
        while global_step < max_steps:
            progressed = False
            for batch in train_loader:
                progressed = True
                sup_count_tensor = batch.pop("supervision_token_count", None)
                sup_tokens = 0
                if sup_count_tensor is not None:
                    try:
                        sup_tokens = int(sup_count_tensor.sum().item())
                    except Exception:
                        sup_tokens = 0
                if sup_tokens <= 0:
                    train_batches_skipped_no_supervision += 1
                    _append_event(
                        event_log,
                        {
                            "event": "train_batch_skipped_no_supervision",
                            "skip_index": int(train_batches_skipped_no_supervision),
                        },
                    )
                    continue
                batch = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss / float(max(1, grad_accum))
                if not bool(torch.isfinite(loss).item()):
                    train_batches_skipped_no_supervision += 1
                    _append_event(
                        event_log,
                        {"event": "train_batch_skipped_non_finite_loss", "loss": float(out.loss.detach().cpu().item())},
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                accum += 1
                train_loss_last = float(out.loss.detach().cpu().item())
                if accum % max(1, grad_accum) == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    print(f"[train] step={global_step} loss={train_loss_last:.6f}")
                    _append_event(
                        event_log,
                        {"event": "train_step", "step": int(global_step), "loss": float(train_loss_last)},
                    )
                if global_step >= max_steps:
                    break
            if not progressed:
                break

        val_loss = None
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    val_batches_seen += 1
                    sup_count_tensor = batch.pop("supervision_token_count", None)
                    sup_tokens = 0
                    if sup_count_tensor is not None:
                        try:
                            sup_tokens = int(sup_count_tensor.sum().item())
                        except Exception:
                            sup_tokens = 0
                    if sup_tokens <= 0:
                        val_batches_skipped_no_supervision += 1
                        _append_event(
                            event_log,
                            {
                                "event": "val_batch_skipped_no_supervision",
                                "batch_index": int(val_batches_seen),
                            },
                        )
                        continue
                    batch = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in batch.items()}
                    out = model(**batch)
                    if not bool(torch.isfinite(out.loss).item()):
                        _append_event(
                            event_log,
                            {
                                "event": "val_batch_skipped_non_finite_loss",
                                "batch_index": int(val_batches_seen),
                                "loss": float(out.loss.detach().cpu().item()),
                            },
                        )
                        continue
                    val_batches_with_supervision += 1
                    val_loss = float(out.loss.detach().cpu().item())
                    break
            model.train()

        adapter_dir = output_dir / "adapter"
        processor_dir = output_dir / "processor"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        processor_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(processor_dir)

        summary.update(
            {
                "status": "PASS",
                "vlm_model_id": vlm_model_id,
                "local_files_only": bool(local_only),
                "device": str(device),
                "seed": int(seed),
                "train_jsonl": str(train_jsonl),
                "val_jsonl": str(val_jsonl) if val_jsonl is not None else "",
                "train_source_count": int(len(train_items)),
                "val_source_count": int(len(val_items)),
                "train_used_count": int(len(train_examples)),
                "val_used_count": int(len(val_examples)),
                "train_placeholder_image_count": int(train_placeholder_count),
                "val_placeholder_image_count": int(val_placeholder_count),
                "rejected_reasons": {k: int(v) for k, v in sorted(reject_reasons.items(), key=lambda kv: kv[0])},
                "max_steps": int(max_steps),
                "steps_completed": int(global_step),
                "batch_size": int(batch_size),
                "grad_accum": int(grad_accum),
                "learning_rate": float(lr),
                "train_loss_last": train_loss_last,
                "val_loss_first_batch": val_loss,
                "val_batches_seen": int(val_batches_seen),
                "val_batches_with_supervision": int(val_batches_with_supervision),
                "val_batches_skipped_no_supervision": int(val_batches_skipped_no_supervision),
                "train_batches_skipped_no_supervision": int(train_batches_skipped_no_supervision),
                "target_modules": list(target_modules),
                "adapter_dir": str(adapter_dir),
                "processor_dir": str(processor_dir),
                "duration_sec": float(max(0.0, time.time() - t0)),
                "train_jsonl_sha256": _sha256_file(train_jsonl),
                "val_jsonl_sha256": _sha256_file(val_jsonl) if val_jsonl is not None else "",
                "script_sha256": _sha256_file(Path(__file__).resolve()),
                "first_train_sample_id": train_examples[0].sample_id if train_examples else "",
                "first_train_image_path": str(train_examples[0].image_path) if train_examples else "",
                "event_log_path": str(event_log),
            }
        )
    except Exception as e:
        summary["errors"].append(f"{type(e).__name__}: {e}")
        _append_event(event_log, {"event": "error", "error": f"{type(e).__name__}: {e}"})
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print(f"phase4_1_summary={summary_path}")
    print(f"status={summary.get('status')}")
    return 0 if summary.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
