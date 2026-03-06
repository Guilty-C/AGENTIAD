# [REMOTE EXECUTION DETECTED? CHECK GUIDELINES]
# This project enforces a strict "Zero-Pollution" remote execution protocol for shared lab servers.
# See REMOTE_EXECUTION_GUIDE.txt for the mandatory "Upload -> Tmp Run -> Cleanup" workflow.

import sys
import shutil
import subprocess
import os
import json
import hashlib
import zipfile
import random
import argparse
import time
import platform
from pathlib import Path
import pathlib
import re
import csv
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Mapping, Optional, Sequence, Set, Tuple
from contextlib import contextmanager

# --- Configuration & Paths ---

PROJECT_ROOT = Path(__file__).parent.resolve()
DIST_SCRIPTS = PROJECT_ROOT / "dist/scripts"
L2_SCRIPT = DIST_SCRIPTS / "06_run_agentiad_infer.py"
L3_SCRIPT = DIST_SCRIPTS / "08_build_sft_trajectories.py"
L4_SCRIPT = DIST_SCRIPTS / "09_train_lora_sft_toy.py"
L5_SCRIPT = DIST_SCRIPTS / "10_build_grpo_rollouts_toy.py"
L6_SCRIPT = DIST_SCRIPTS / "10_train_grpo_toy.py"
VERIFY_SCRIPT = Path(__file__)

# Constants
EXPECTED_SHA_VERIFY_ALL = "942512B15BD85860B777DA215F6C8DEA78FB4F73BD16237FD11E4570E8B47FBA"

# --- Infrastructure / Utils ---

# Idempotent sys.path injection
_src_path_str = str(PROJECT_ROOT / "src")
if _src_path_str not in sys.path:
    sys.path.insert(0, _src_path_str)

@dataclass
class StageContext:
    work_dir: Path
    seeds: List[int]
    max_samples: Optional[int]
    allow_flags: bool
    no_adapter: bool
    allow_full_dataset: bool = False
    phase1_baseline: bool = False
    phase2_full_infer: bool = False
    strict_j_mode: bool = False
    dataset_split: str = "test"
    env_overrides: Dict[str, str] = field(default_factory=dict)
    strict_contract: bool = False
    vlm_model_id: Optional[str] = None
    requested_vlm_model_id: Optional[str] = None
    requested_vlm_model_local_dir: Optional[str] = None
    vlm_model_source: str = "config"
    vlm_max_side: Optional[int] = None
    sdp_backend: Optional[str] = None
    vlm_retry_n: Optional[int] = None
    vlm_max_side_source: str = "default"
    sdp_backend_source: str = "default"
    vlm_retry_n_source: str = "default"
    scripts_dir: Path = field(default_factory=lambda: DIST_SCRIPTS)
    cached_cfg_max_samples: Optional[int] = None

    def __post_init__(self):
        # Config-aware initialization
        # Try to read max_samples from mr_config.yaml in work_dir
        mr_cfg = self.work_dir / "mr_config.yaml"
        if mr_cfg.exists():
            try:
                import yaml
                # Safe load
                with open(mr_cfg, "r", encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f)
                if isinstance(cfg_data, dict) and "max_samples" in cfg_data:
                    val = cfg_data["max_samples"]
                    if val is not None:
                        try:
                            self.cached_cfg_max_samples = int(val)
                        except:
                            pass
            except:
                pass

    def get_effective_max_samples(self) -> Optional[int]:
        # Priority: CLI -> Config -> Full Run (None) -> Safe Default (2)
        # Matches L3 logic: CLI > config > None
        
        # 1. CLI Override
        if self.max_samples is not None:
            return self.max_samples
            
        # 2. Config Override
        if self.cached_cfg_max_samples is not None:
            return self.cached_cfg_max_samples
            
        # 3. Full Run
        if self.allow_full_dataset:
            return None
            
        # 4. Safe Default
        return 2

    def get_script(self, name: str) -> Path:
        p = self.scripts_dir / name
        if not p.exists():
            candidates = [
                Path("dist/scripts") / name,
                Path("../dist/scripts") / name,
                Path("../../dist/scripts") / name,
                PROJECT_ROOT / "dist/scripts" / name
            ]
            for c in candidates:
                if c.exists(): return c.resolve()
            raise FileNotFoundError(f"Script not found: {name}")
        return p

@dataclass
class StageResult:
    success: bool = True
    gates: Dict[str, bool] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=lambda: {"allow_flags_used": False, "allow_flags_violation": False, "evidence_checks": [], "gates_na": {}})
    errors: List[str] = field(default_factory=list)
    measurements: Dict[str, float] = field(default_factory=lambda: {"evidence_check_sec": 0.0})
    remediations: List[str] = field(default_factory=list)

class Timer:
    def __init__(self, name, measurements_dict=None, key=None):
        self.name = name
        self.measurements = measurements_dict
        self.key = key
        self.duration = 0.0
    def __enter__(self):
        self.start = time.time()
        print(f"[Timer] Start {self.name}...", file=sys.stderr)
        return self
    def __exit__(self, *args):
        self.duration = time.time() - self.start
        print(f"[Timer] {self.name} took {self.duration:.2f}s", file=sys.stderr)
        if self.measurements is not None and self.key:
            self.measurements[self.key] = round(self.duration, 2)

class CmdResult:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

class CmdRunner:
    @staticmethod
    def run(cmd, env_overrides=None, cwd=None, timeout=None, stream_output=False) -> Optional[CmdResult]:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        if env_overrides:
            env.update(env_overrides)
        
        start_t = time.time()
        cmd_name = Path(cmd[1]).name if len(cmd) > 1 else cmd[0]
        
        if stream_output:
            print(f"[Cmd] Starting {cmd_name}...", file=sys.stderr)
            try:
                process = subprocess.Popen(
                    cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                    text=True, bufsize=1, encoding='utf-8', errors='replace'
                )
                output_lines = []
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None: break
                    if line:
                        sys.stderr.write(f"  [{cmd_name}] {line}")
                        output_lines.append(line)
                returncode = process.poll()
                dur = time.time() - start_t
                print(f"[Cmd] {cmd_name} finished in {dur:.2f}s (RC={returncode})", file=sys.stderr)
                full_out = "".join(output_lines).encode('utf-8')
                res = CmdResult(returncode, full_out, full_out)
                return res
            except Exception as e:
                print(f"[Cmd] Exception running {cmd_name}: {e}", file=sys.stderr)
                return None
        else:
            try:
                res_proc = subprocess.run(cmd, capture_output=True, env=env, cwd=cwd, timeout=timeout)
                dur = time.time() - start_t
                print(f"[Cmd] {cmd_name} took {dur:.2f}s (RC={res_proc.returncode})", file=sys.stderr)
                if res_proc.returncode != 0:
                     print(f"[Cmd] STDOUT: {res_proc.stdout.decode('utf-8', errors='replace')[:1000]} [TRUNCATED]", file=sys.stderr)
                     print(f"[Cmd] STDERR: {res_proc.stderr.decode('utf-8', errors='replace')[:1000]} [TRUNCATED]", file=sys.stderr)
                return CmdResult(res_proc.returncode, res_proc.stdout, res_proc.stderr)
            except subprocess.TimeoutExpired:
                print(f"[Cmd] {cmd[0]} TIMED OUT after {timeout}s", file=sys.stderr)
                return None

class _TeeStream:
    def __init__(self, terminal_stream, log_fp):
        self._terminal_stream = terminal_stream
        self._log_fp = log_fp

    def write(self, data):
        if data is None:
            return 0
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        self._terminal_stream.write(data)
        self._log_fp.write(data)
        self.flush()
        return len(data)

    def flush(self):
        self._terminal_stream.flush()
        self._log_fp.flush()

    def isatty(self):
        return bool(getattr(self._terminal_stream, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self._terminal_stream, "encoding", "utf-8")

    def fileno(self):
        return self._terminal_stream.fileno()

@contextmanager
def _tee_terminal_to_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_fp:
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(orig_stdout, log_fp)
        sys.stderr = _TeeStream(orig_stderr, log_fp)
        try:
            yield log_fp
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr

def _safe_subprocess_text(cmd: List[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode == 0:
            return (proc.stdout or "").strip() or (proc.stderr or "").strip()
    except Exception:
        pass
    return ""

def _build_terminal_log_path(output_dir: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return output_dir / "logs" / f"terminal_{ts}_{os.getpid()}.txt"

def _emit_terminal_log_header(log_path: Path, args: argparse.Namespace, output_dir: Path):
    cmdline = " ".join(str(x) for x in [sys.executable, Path(__file__).name, *sys.argv[1:]])
    git_commit = _safe_subprocess_text(["git", "rev-parse", "HEAD"])[:40]
    if not git_commit:
        git_commit = "UNKNOWN"
    hf_home = str(
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface")
    )
    env_summary = {
        "HF_HOME": hf_home,
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", ""),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        "MMAD_ROOT": os.environ.get("MMAD_ROOT", ""),
    }
    optional_env_keys = [
        "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "VLM_MODEL_ID",
        "VLM_MODEL_LOCAL_DIR", "HUGGINGFACE_HUB_CACHE",
    ]
    for k in optional_env_keys:
        v = os.environ.get(k, "")
        if v:
            env_summary[k] = v
    print(f"[Log] terminal_log_path={log_path}")
    print(f"[Log] start_time_local={time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Log] cwd={Path.cwd()}")
    print(f"[Log] output_dir={output_dir}")
    print(f"[Log] argv={cmdline}")
    print(f"[Log] mode={getattr(args, 'mode', 'unknown')}")
    print(f"[Log] python={sys.version.splitlines()[0]}")
    print(f"[Log] platform={platform.platform()}")
    print(f"[Log] git_commit={git_commit}")
    print(f"[Log] env_summary={json.dumps(env_summary, ensure_ascii=False, sort_keys=True)}")

def _cleanup_work_dir_preserve_logs(work_dir: Path):
    if not work_dir.exists():
        return
    for child in work_dir.iterdir():
        if child.name == "logs" and child.is_dir():
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)

def _is_offline_flag_1(v: Any) -> bool:
    return str(v or "").strip() == "1"

def _effective_env(env_overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})
    return env

def _resolve_hf_home_and_cache(env: Dict[str, str]) -> tuple[Path, Path]:
    hf_home_raw = str(env.get("HF_HOME", "") or env.get("HUGGINGFACE_HUB_CACHE", "")).strip()
    hf_home = Path(hf_home_raw).expanduser() if hf_home_raw else (Path.home() / ".cache" / "huggingface")
    hub_cache_raw = str(env.get("HUGGINGFACE_HUB_CACHE", "")).strip()
    if hub_cache_raw:
        hub_cache = Path(hub_cache_raw).expanduser()
    elif hf_home.name.lower() == "hub":
        hub_cache = hf_home
    else:
        hub_cache = hf_home / "hub"
    return hf_home.resolve(), hub_cache.resolve()

def _platform_copy_model_cmd() -> str:
    if os.name == "nt":
        return "Copy-Item -Recurse -Force <DOWNLOADED_MODEL_DIR> <LOCAL_VLM_DIR>"
    return "cp -a <DOWNLOADED_MODEL_DIR> <LOCAL_VLM_DIR>"

def _looks_like_repo_id(model_id: str) -> bool:
    s = str(model_id or "").strip()
    return bool(s) and not Path(s).expanduser().exists()

def _local_model_dir_ready(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    has_config = (model_dir / "config.json").exists()
    has_tokenizer = (
        (model_dir / "tokenizer.json").exists()
        or (model_dir / "tokenizer_config.json").exists()
        or (model_dir / "preprocessor_config.json").exists()
    )
    return bool(has_config and has_tokenizer)

def _mode_requires_infer_vlm_gate(mode_name: str) -> bool:
    mode = str(mode_name or "").strip()
    return mode in {"default", "contract_smoke", "phase1_baseline", "phase2_full_infer", "strict_j"}

def _offline_vlm_gate_required(mode_name: str, strict_contract: bool, env: Dict[str, str]) -> bool:
    if not strict_contract:
        return False
    if not _mode_requires_infer_vlm_gate(mode_name):
        return False
    # Hard gate in strict-contract if either core offline switch is on.
    return _is_offline_flag_1(env.get("HF_HUB_OFFLINE")) or _is_offline_flag_1(env.get("TRANSFORMERS_OFFLINE"))

def _check_offline_local_vlm_ready(
    *,
    mode_name: str,
    strict_contract: bool,
    requested_model_id: str,
    requested_local_dir: str,
    effective_model_id: str,
    env: Dict[str, str],
) -> Dict[str, Any]:
    check_required = _offline_vlm_gate_required(mode_name, strict_contract, env)
    if not check_required:
        return {"required": False, "ok": True, "reason": "not_required"}

    hf_home, hub_cache = _resolve_hf_home_and_cache(env)
    req_local = str(requested_local_dir or "").strip()
    req_model = str(requested_model_id or "").strip()
    eff_model = str(effective_model_id or req_model or "").strip()
    checked_notes: List[str] = []
    ready = False
    ready_source = ""

    if req_local:
        p = Path(req_local).expanduser()
        if p.exists() and p.is_dir() and _local_model_dir_ready(p):
            ready = True
            ready_source = f"vlm_model_local_dir:{p.resolve()}"
        else:
            checked_notes.append(f"vlm_model_local_dir_not_ready:{p}")

    if (not ready) and eff_model:
        p = Path(eff_model).expanduser()
        if p.exists() and p.is_dir():
            if _local_model_dir_ready(p):
                ready = True
                ready_source = f"vlm_model_id_local_path:{p.resolve()}"
            else:
                checked_notes.append(f"vlm_model_id_local_path_not_ready:{p}")
        else:
            checked_notes.append(f"vlm_model_id_not_local_dir:{eff_model}")
    elif not ready and not eff_model:
        checked_notes.append("vlm_model_id_not_set")

    hf_endpoint = str(env.get("HF_ENDPOINT", "")).strip()
    remediation_model = eff_model if _looks_like_repo_id(eff_model) else "Qwen/Qwen2.5-VL-3B-Instruct"
    remediation_a = (
        f"{_platform_copy_model_cmd()}; "
        "python verify_all.py --mode phase2_full --output-dir <OUTPUT_DIR> --seeds 0 --strict-contract "
        "--vlm-model-local-dir <LOCAL_VLM_DIR>"
    )
    remediation_b = (
        f"HF_HOME=\"{hf_home}\" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "
        f"huggingface-cli download \"{remediation_model}\" --local-dir <DOWNLOADED_MODEL_DIR>"
    )
    if ready:
        return {
            "required": True,
            "ok": True,
            "ready_source": ready_source,
            "checked_notes": checked_notes,
            "hf_home": str(hf_home),
            "hf_cache": str(hub_cache),
        }

    msg_lines = [
        "离线模式必须提供本地 VLM 目录（strict-contract + offline）。",
        f"current_vlm_model_id={eff_model or 'NOT_SET'}",
        f"current_vlm_model_local_dir={req_local or 'NOT_SET'}",
        f"HF_HOME={hf_home}",
        f"HF_CACHE={hub_cache}",
    ]
    if hf_endpoint:
        msg_lines.append(f"HF_ENDPOINT={hf_endpoint}")
    msg_lines.append(f"checked={'; '.join(checked_notes) if checked_notes else 'none'}")
    msg_lines.append("fix_a_local_dir=" + remediation_a)
    msg_lines.append("fix_b_preload_cache=" + remediation_b)
    return {
        "required": True,
        "ok": False,
        "msg": " | ".join(msg_lines),
        "checked_notes": checked_notes,
        "hf_home": str(hf_home),
        "hf_cache": str(hub_cache),
        "remediation_a": remediation_a,
        "remediation_b": remediation_b,
    }

def _check_offline_model_readiness_for_probe(
    ctx: StageContext,
) -> Dict[str, Any]:
    env = _effective_env(ctx.env_overrides)
    if ctx.phase2_full_infer:
        mode_name = "phase2_full_infer"
    elif ctx.phase1_baseline:
        mode_name = "phase1_baseline"
    elif ctx.strict_j_mode:
        mode_name = "strict_j"
    else:
        mode_name = "contract_smoke"
    requested_local_dir = str(ctx.requested_vlm_model_local_dir or "").strip()
    requested_model_id = str(ctx.requested_vlm_model_id or "").strip()
    effective_model_id = str(ctx.vlm_model_id or requested_model_id or "").strip()
    return _check_offline_local_vlm_ready(
        mode_name=mode_name,
        strict_contract=bool(ctx.strict_contract),
        requested_model_id=requested_model_id,
        requested_local_dir=requested_local_dir,
        effective_model_id=effective_model_id,
        env=env,
    )

class CmdBuilder:
    def __init__(self, script_path: Path):
        self.parts = [sys.executable, str(script_path)]
    def arg(self, key: str, value: Any = None):
        self.parts.append(key)
        if value is not None: self.parts.append(str(value))
        return self
    def flag(self, key: str, condition: bool = True):
        if condition: self.parts.append(key)
        return self
    def build(self) -> List[str]:
        return self.parts

# Helper methods for CmdBuilder
def with_config(b, p): return b.arg("--config", p)
def with_run_name(b, n): return b.arg("--run_name", n)
def with_evidence_dir(b, p): return b.arg("--evidence_dir", p)
def with_trace_dir(b, p): return b.arg("--trace_dir", p)
def with_out_jsonl(b, p): return b.arg("--out_jsonl", p)
CmdBuilder.with_config = with_config
CmdBuilder.with_run_name = with_run_name
CmdBuilder.with_evidence_dir = with_evidence_dir
CmdBuilder.with_trace_dir = with_trace_dir
CmdBuilder.with_out_jsonl = with_out_jsonl

def _require_cmd_ok(res: StageResult, cmd_res: Optional[CmdResult], label: str, stable_stage: str, extra_msg: str = "", record_cmd_failed: bool = True) -> bool:
    if cmd_res is None:
        res.success = False
        msg = f"{label} failed: CmdRunner returned None (timeout/exception) {extra_msg}"
        res.errors.append(msg)
        if record_cmd_failed:
             res.artifacts["evidence_checks"].append({
                 "stage": label, 
                 "stable_stage": stable_stage,
                 "label": label,
                 "code": "CMD_FAILED", 
                 "msg": msg
             })
        return False
    if cmd_res.returncode != 0:
        res.success = False
        # Extract tail of stdout/stderr for context
        out_sample = cmd_res.stdout.decode('utf-8', errors='replace')[-300:] if cmd_res.stdout else ""
        err_sample = cmd_res.stderr.decode('utf-8', errors='replace')[-300:] if cmd_res.stderr else ""
        msg = f"{label} failed: RC={cmd_res.returncode} {extra_msg} | OUT={out_sample} | ERR={err_sample}"
        res.errors.append(msg)
        if record_cmd_failed:
             res.artifacts["evidence_checks"].append({
                 "stage": label, 
                 "stable_stage": stable_stage,
                 "label": label,
                 "code": "CMD_FAILED", 
                 "msg": msg
             })
        return False
    return True

class J1:
    @staticmethod
    def _norm_flag(s: str) -> str:
        s = str(s).strip()
        s = s.replace("-", "_")
        return s

    SAFE_ALLOW_FLAGS = {_norm_flag("--allow_full_dataset"), _norm_flag("--allow-full-dataset")}

    @staticmethod
    def check(cmd_list: List[str]) -> bool:
        for arg in cmd_list:
            s = str(arg)
            norm_s = J1._norm_flag(s)
            # Check prefix on normalized string to handle --allow-foo and --allow_foo
            if norm_s.startswith("--allow_") and norm_s not in J1.SAFE_ALLOW_FLAGS:
                 return True
        return False

    @staticmethod
    def record_if_violation(cmd: List[str], res: StageResult, label: str = "cmd"):
        # Find flags that are violations
        # We report the ORIGINAL flag for clarity, but check against normalized whitelist
        violations = []
        for arg in cmd:
            s = str(arg)
            norm_s = J1._norm_flag(s)
            if norm_s.startswith("--allow_"):
                if norm_s not in J1.SAFE_ALLOW_FLAGS:
                    violations.append(s)
        
        if not violations: return False

        res.artifacts["allow_flags_used"] = True
        res.artifacts["allow_flags_violation"] = True
        
        # Enhanced debug info (Task C)
        all_allow_flags = [str(a) for a in cmd if J1._norm_flag(str(a)).startswith("--allow_")]
        normalized_violations = [J1._norm_flag(v) for v in violations]
        
        res.errors.append(f"J1 Violation: allow flag detected in {label}: {violations} (Normalized Violations: {normalized_violations}, Whitelist: {list(J1.SAFE_ALLOW_FLAGS)})")
        return True

class Evidence:
    @staticmethod
    def verify_zip(zip_path: Path, script_name: str, expected_trace_count: Optional[int] = None) -> tuple[str, str, float]:
        t = Timer(f"Verify Zip {zip_path.name}")
        with t:
            if not zip_path.exists(): return "MISSING_ZIP", "Missing zip", t.duration
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    names = zf.namelist()
                    if "INDEX.txt" not in names: return "MISSING_INDEX", "Missing INDEX.txt", t.duration
                    
                    found_script = False
                    for n in names:
                        if n.endswith(script_name): found_script = True; break
                    if not found_script:
                        if f"dist/scripts/{script_name}" not in names:
                            return "MISSING_SCRIPT", f"Missing script {script_name}", t.duration
                    
                    trace_files = [n for n in names if n.endswith("trace.json")]
                    
                    if expected_trace_count is None:
                        # Policy: If None, we don't enforce count, but we still check integrity if traces exist.
                        # We do NOT return NO_TRACES here if empty, as it's allowed.
                        pass
                    else:
                        # Policy: strict count match
                        if len(trace_files) != expected_trace_count:
                            return "TRACE_COUNT_MISMATCH", f"J3 Fail: Trace count {len(trace_files)} != Expected {expected_trace_count}", t.duration
                    
                    has_tool_call = False
                    has_fingerprint = False
                    for tf in trace_files:
                        try:
                            with zf.open(tf) as f:
                                data = json.load(f)
                                turns = data.get("turns", [])
                                for t_turn in turns:
                                    tc = t_turn.get("tool_call")
                                    if isinstance(tc, dict) and str(tc.get("name") or "").strip():
                                        has_tool_call = True
                                    tcs = t_turn.get("tool_calls")
                                    if isinstance(tcs, list):
                                        for c in tcs:
                                            if isinstance(c, dict) and str(c.get("name") or "").strip():
                                                has_tool_call = True
                                                break

                                    candidates = []
                                    if "tool_result" in t_turn:
                                        candidates.append(t_turn["tool_result"])
                                    if "tool_results" in t_turn:
                                        candidates.extend(t_turn["tool_results"])
                                    for tr in candidates:
                                        if isinstance(tr, dict) and any(k in tr for k in ["result_sha", "size", "path_hash"]):
                                            has_fingerprint = True
                        except:
                            pass
                        if has_tool_call and has_fingerprint:
                            break
                    if has_tool_call and not has_fingerprint:
                        return "MISSING_FINGERPRINT", f"Missing tool fingerprints (J2). Checked {len(trace_files)} traces.", t.duration
            except Exception as e: return "EXCEPTION", f"Zip Check Exception: {e}", t.duration
        return "OK", "OK", t.duration

    @staticmethod
    def get_toolcall_rate(zip_path: Path) -> float:
        if not zip_path.exists(): return 0.0
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                traces = [n for n in zf.namelist() if n.endswith("trace.json")]
                if not traces: return 0.0
                with_tool = 0
                for tn in traces:
                    try:
                        t_data = json.load(zf.open(tn))
                        turns = t_data.get("turns", [])
                        # Count tool calls by name (not just result)
                        has_tc = False
                        for t in turns:
                             # Check tool_call dict
                             tc = t.get("tool_call")
                             if isinstance(tc, dict) and str(tc.get("name") or "").strip():
                                 has_tc = True
                                 break
                             # Check tool_calls list
                             tcs = t.get("tool_calls")
                             if isinstance(tcs, list):
                                 for c in tcs:
                                     if isinstance(c, dict) and str(c.get("name") or "").strip():
                                         has_tc = True
                                         break
                             if has_tc: break
                        if has_tc: with_tool += 1
                    except: pass
                return with_tool / len(traces)
        except: return 0.0


def resolve_evidence_zip(ev_dir: Path):
    expected = ev_dir / "evidence_package.zip"
    if expected.exists():
        return expected, "", ["evidence_package.zip"]

    candidates = sorted([p for p in ev_dir.glob("*.zip") if p.is_file()], key=lambda p: p.name.lower())
    found = [p.name for p in candidates]
    if len(candidates) == 1:
        return candidates[0], f"fallback to {candidates[0].name}", found
    return None, "", found


def parse_l2_asset_audit(merged_output: bytes) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not merged_output:
        return out
    try:
        lines = merged_output.decode("utf-8", errors="replace").splitlines()
    except Exception:
        return out
    for raw in lines:
        line = raw.strip()
        if line.startswith("MMAD_ROOT_RESOLVED="):
            out["mmad_root_resolved"] = line.split("=", 1)[1].strip() or "NOT_SET"
        elif line.startswith("MMAD_ASSET_MODE="):
            out["mmad_asset_mode"] = line.split("=", 1)[1].strip() or "unknown"
    return out

def parse_l2_json_from_cmd(cmd_res: Optional[CmdResult]) -> Optional[Dict[str, Any]]:
    if not cmd_res:
        return None
    merged_output = cmd_res.stderr if cmd_res.stderr else cmd_res.stdout
    if not merged_output:
        return None
    try:
        for line in merged_output.decode("utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("L2_RESULT_JSON="):
                return json.loads(s[len("L2_RESULT_JSON="):])
    except Exception as e:
        print(f"Warning: Failed to parse L2_RESULT_JSON: {e}", file=sys.stderr)
    return None


def _read_first_model_id_from_csv(csv_path: Path) -> Optional[str]:
    try:
        if not csv_path.exists() or not csv_path.is_file():
            return None
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
            if not isinstance(first, dict):
                return None
            val = str(first.get("model_id", "") or "").strip()
            return val or None
    except Exception:
        return None


def _read_first_model_id_from_trace_zip(ev_zip: Path) -> Optional[str]:
    try:
        if not ev_zip.exists() or not ev_zip.is_file():
            return None
        with zipfile.ZipFile(ev_zip, "r") as zf:
            for name in zf.namelist():
                if not name.endswith("trace.json"):
                    continue
                try:
                    obj = json.load(zf.open(name))
                    if not isinstance(obj, dict):
                        continue
                    fp = obj.get("fingerprint", {})
                    if isinstance(fp, dict):
                        mid = str(fp.get("model_id", "") or "").strip()
                        if mid:
                            return mid
                    mid_top = str(obj.get("model_id", "") or "").strip()
                    if mid_top:
                        return mid_top
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _count_phase2_execution_errors_from_zip(ev_zip: Path, csv_name: str) -> Dict[str, int]:
    out = {
        "execution_error_count": 0,
        "unknown_count": 0,
        "strict_schema_invalid_count": 0,
        "rows_total": 0,
        "schema_error_examples": [],
    }
    schema_errors: List[str] = []

    def _record_schema_error(code: str) -> None:
        if len(schema_errors) < 100:
            schema_errors.append(str(code))

    try:
        if not ev_zip.exists() or not ev_zip.is_file():
            return out
        with zipfile.ZipFile(ev_zip, "r") as zf:
            if csv_name not in zf.namelist():
                return out
            content = zf.read(csv_name).decode("utf-8", errors="replace")
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                out["rows_total"] += 1
                raw_output = str((row or {}).get("raw_output", "") or "").strip()
                if not raw_output:
                    out["strict_schema_invalid_count"] += 1
                    _record_schema_error("raw_output_empty")
                    continue
                try:
                    raw_obj = json.loads(raw_output)
                except Exception:
                    out["strict_schema_invalid_count"] += 1
                    _record_schema_error("invalid_json_parse")
                    continue
                final_obj = raw_obj.get("final") if isinstance(raw_obj, dict) else None
                if not isinstance(final_obj, dict):
                    out["strict_schema_invalid_count"] += 1
                    _record_schema_error("final_not_dict")
                    continue
                anomaly = str(final_obj.get("anomaly", "") or "").strip().lower()
                defect_type = str(final_obj.get("defect_type", "") or "").strip().lower()
                required_keys_ok = all(k in final_obj for k in ["anomaly", "bbox", "defect_type", "confidence"])
                anomaly_ok = anomaly in {"yes", "no", "unknown"}
                bbox_val = final_obj.get("bbox")
                bbox_ok = False
                if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4:
                    try:
                        x1 = float(bbox_val[0])
                        y1 = float(bbox_val[1])
                        x2 = float(bbox_val[2])
                        y2 = float(bbox_val[3])
                        in_range = all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2])
                        bbox_ok = in_range and (x2 > x1) and (y2 > y1)
                    except Exception:
                        bbox_ok = False
                defect_type_ok = isinstance(final_obj.get("defect_type"), str) and bool(str(final_obj.get("defect_type")).strip())
                confidence_ok = final_obj.get("confidence", "__MISSING__") is None
                if not (required_keys_ok and anomaly_ok and bbox_ok and defect_type_ok and confidence_ok):
                    out["strict_schema_invalid_count"] += 1
                    if not required_keys_ok:
                        _record_schema_error("missing_required_keys")
                    if not anomaly_ok:
                        _record_schema_error("invalid_anomaly")
                    if not bbox_ok:
                        _record_schema_error("invalid_bbox")
                    if not defect_type_ok:
                        _record_schema_error("invalid_defect_type")
                    if not confidence_ok:
                        _record_schema_error("invalid_confidence")
                if anomaly == "unknown":
                    out["unknown_count"] += 1
                if defect_type == "execution_error":
                    out["execution_error_count"] += 1
    except Exception:
        return out
    out["schema_error_examples"] = schema_errors[:20]
    return out


def _phase2_sanity_check_evidence_zip(
    zip_path: str,
    seed: int,
    expected_n: Optional[int],
    require_toolcall_rate: bool = True,
    require_trace_policy: str = "auto",  # "auto"|"must_exist"|"must_not_exist"
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": True,
        "zip_path": str(zip_path),
        "seed": int(seed),
        "expected_n": expected_n,
        "missing_member": [],
        "csv_member": None,
        "summary_member": None,
        "summary": {},
        "csv_fields": [],
        "row_count": 0,
        "row_count_expected": expected_n,
        "sample_id_empty_count": 0,
        "raw_output_parse_invalid": 0,
        "final_not_dict": 0,
        "final_missing_keys": {"anomaly": 0, "defect_type": 0, "bbox": 0, "confidence": 0, "reason": 0},
        "invalid_anomaly": 0,
        "toolcall_rate_avg": None,
        "trace_policy": {
            "require": str(require_trace_policy),
            "trace_emitted_summary": None,
            "trace_count": 0,
            "ok": True,
            "detail": "",
        },
        "examples": {
            "raw_output_parse_invalid": [],
            "final_not_dict": [],
            "sample_id_empty": [],
            "missing_key_anomaly": [],
            "missing_key_defect_type": [],
            "missing_key_bbox": [],
            "missing_key_confidence": [],
            "missing_key_reason": [],
            "invalid_anomaly": [],
        },
        "errors": [],
    }

    def _add_example(bucket: str, sid: str) -> None:
        ex = out["examples"].setdefault(bucket, [])
        if len(ex) < 10:
            ex.append(str(sid))

    zp = Path(zip_path)
    if not zp.exists() or not zp.is_file():
        out["ok"] = False
        out["errors"].append(f"zip_missing:{zp}")
        return out

    try:
        with zipfile.ZipFile(zp, "r") as zf:
            members = zf.namelist()
            csv_member = f"tables/agentiad_infer_mr_s{int(seed)}.csv"
            if csv_member not in members:
                cands = [m for m in members if m.startswith("tables/") and m.endswith(".csv")]
                out["missing_member"].append({"required": csv_member, "candidates": cands[:20]})
                out["ok"] = False
            else:
                out["csv_member"] = csv_member

            expected_summary = f"logs/agentiad_infer_summary_mr_s{int(seed)}.json"
            summary_member = expected_summary if expected_summary in members else None
            if summary_member is None:
                log_cands = [m for m in members if m.startswith("logs/") and "agentiad_infer_summary" in m and m.endswith(".json")]
                seed_tag = f"_s{int(seed)}"
                seeded = [m for m in log_cands if seed_tag in m]
                summary_member = seeded[0] if seeded else (log_cands[0] if log_cands else None)
                if summary_member is None:
                    out["missing_member"].append({"required": expected_summary, "candidates": log_cands[:20]})
                    out["ok"] = False
            out["summary_member"] = summary_member

            summary_obj: Dict[str, Any] = {}
            if summary_member is not None:
                try:
                    summary_obj = json.loads(zf.read(summary_member).decode("utf-8", errors="replace"))
                    if not isinstance(summary_obj, dict):
                        summary_obj = {}
                        out["ok"] = False
                        out["errors"].append("summary_not_dict")
                except Exception as e:
                    out["ok"] = False
                    out["errors"].append(f"summary_parse_error:{type(e).__name__}")
                    summary_obj = {}
            out["summary"] = {
                "n_attempted": summary_obj.get("n_attempted"),
                "n_success": summary_obj.get("n_success"),
                "n_skipped": summary_obj.get("n_skipped"),
                "out_csv": summary_obj.get("out_csv"),
                "trace_root": summary_obj.get("trace_root") or summary_obj.get("trace_dir"),
                "toolcall_rate_avg": summary_obj.get("toolcall_rate_avg"),
                "toolcall_rate": summary_obj.get("toolcall_rate"),
                "trace_emitted": summary_obj.get("trace_emitted"),
                "trace_count": summary_obj.get("trace_count"),
            }

            trace_members = [m for m in members if m.endswith("trace.json")]
            out["trace_policy"]["trace_count"] = int(len(trace_members))
            trace_emitted = summary_obj.get("trace_emitted")
            out["trace_policy"]["trace_emitted_summary"] = trace_emitted if isinstance(trace_emitted, bool) else None

            rows_total = 0
            tool_rows = 0
            if out["csv_member"] is not None:
                try:
                    content = zf.read(out["csv_member"]).decode("utf-8", errors="replace")
                    reader = csv.DictReader(content.splitlines())
                    out["csv_fields"] = list(reader.fieldnames or [])
                    if "sample_id" not in out["csv_fields"] or "raw_output" not in out["csv_fields"]:
                        out["ok"] = False
                        out["errors"].append("csv_missing_required_fields")
                    for row in reader:
                        rows_total += 1
                        sid = str((row or {}).get("sample_id", "") or "").strip()
                        if not sid:
                            out["sample_id_empty_count"] += 1
                            _add_example("sample_id_empty", f"row_{rows_total}")
                        try:
                            pz_called = int(str((row or {}).get("pz_called", "0") or "0"))
                        except Exception:
                            pz_called = 0
                        try:
                            cr_called = int(str((row or {}).get("cr_called", "0") or "0"))
                        except Exception:
                            cr_called = 0
                        if (pz_called > 0) or (cr_called > 0):
                            tool_rows += 1

                        raw_output = str((row or {}).get("raw_output", "") or "").strip()
                        raw_obj: Any = None
                        if not raw_output:
                            out["raw_output_parse_invalid"] += 1
                            _add_example("raw_output_parse_invalid", sid or f"row_{rows_total}")
                            continue
                        try:
                            raw_obj = json.loads(raw_output)
                        except Exception:
                            out["raw_output_parse_invalid"] += 1
                            _add_example("raw_output_parse_invalid", sid or f"row_{rows_total}")
                            continue
                        if not isinstance(raw_obj, dict):
                            out["raw_output_parse_invalid"] += 1
                            _add_example("raw_output_parse_invalid", sid or f"row_{rows_total}")
                            continue
                        final_obj = raw_obj.get("final")
                        if not isinstance(final_obj, dict):
                            out["final_not_dict"] += 1
                            _add_example("final_not_dict", sid or f"row_{rows_total}")
                            continue
                        for k in ["anomaly", "defect_type", "bbox", "confidence", "reason"]:
                            if k not in final_obj:
                                out["final_missing_keys"][k] += 1
                                _add_example(f"missing_key_{k}", sid or f"row_{rows_total}")
                        anomaly = str(final_obj.get("anomaly", "") or "").strip().lower()
                        if anomaly not in {"yes", "no", "unknown"}:
                            out["invalid_anomaly"] += 1
                            _add_example("invalid_anomaly", sid or f"row_{rows_total}")
                except Exception as e:
                    out["ok"] = False
                    out["errors"].append(f"csv_read_error:{type(e).__name__}")

            out["row_count"] = int(rows_total)
            if expected_n is not None and rows_total != int(expected_n):
                out["ok"] = False
                out["errors"].append("row_count_mismatch")
            if int(out["sample_id_empty_count"]) > 0:
                out["ok"] = False
                out["errors"].append("sample_id_empty")
            if int(out["raw_output_parse_invalid"]) > 0:
                out["ok"] = False
                out["errors"].append("raw_output_parse_invalid")
            if int(out["final_not_dict"]) > 0:
                out["ok"] = False
                out["errors"].append("final_not_dict")
            if any(int(v) > 0 for v in out["final_missing_keys"].values()):
                out["ok"] = False
                out["errors"].append("final_missing_keys")
            if int(out["invalid_anomaly"]) > 0:
                out["ok"] = False
                out["errors"].append("invalid_anomaly")

            tool_rate = summary_obj.get("toolcall_rate_avg", None)
            if tool_rate is None:
                tool_rate = summary_obj.get("toolcall_rate", None)
            if tool_rate is None and rows_total > 0:
                tool_rate = float(tool_rows) / float(rows_total)
            try:
                out["toolcall_rate_avg"] = float(tool_rate) if tool_rate is not None else None
            except Exception:
                out["toolcall_rate_avg"] = None
            if require_toolcall_rate and (out["toolcall_rate_avg"] is None or float(out["toolcall_rate_avg"]) <= 0.0):
                out["ok"] = False
                out["errors"].append("toolcall_rate_zero")

            policy = str(require_trace_policy or "auto").strip().lower()
            trace_ok = True
            trace_detail = ""
            if policy == "must_exist":
                trace_ok = int(len(trace_members)) > 0
                trace_detail = f"must_exist: found_trace_count={len(trace_members)}"
            elif policy == "must_not_exist":
                trace_ok = int(len(trace_members)) == 0
                trace_detail = f"must_not_exist: found_trace_count={len(trace_members)}"
            else:
                if isinstance(trace_emitted, bool):
                    if trace_emitted:
                        trace_ok = int(len(trace_members)) > 0
                    else:
                        trace_ok = int(len(trace_members)) == 0
                    trace_detail = (
                        f"auto: trace_emitted={trace_emitted}, found_trace_count={len(trace_members)}"
                    )
                else:
                    trace_ok = True
                    trace_detail = f"auto: summary trace_emitted missing, found_trace_count={len(trace_members)}"
            out["trace_policy"]["ok"] = bool(trace_ok)
            out["trace_policy"]["detail"] = trace_detail
            if not trace_ok:
                out["ok"] = False
                out["errors"].append("trace_policy_mismatch")
    except zipfile.BadZipFile:
        out["ok"] = False
        out["errors"].append("bad_zip")
        return out
    except Exception as e:
        out["ok"] = False
        out["errors"].append(f"sanity_exception:{type(e).__name__}")
        return out

    if out["missing_member"]:
        out["ok"] = False
        out["errors"].append("missing_member")
    return out


def _count_missing_answer_tags_in_terminal_log(terminal_log_path: Path) -> int:
    p = Path(terminal_log_path)
    if not p.exists() or not p.is_file():
        return 0
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0
    # Count both plain parser errors and schema_repair-prefixed variants.
    return int(
        len(
            re.findall(
                r"Missing <answer> tags|\[schema_repair\].*Missing <answer> tags",
                text,
            )
        )
    )


def _stats_evidence_zip(zip_path: Path, terminal_log_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "zip_path": str(zip_path),
        "main_jsonl_lines": 0,
        "trace_json_count": 0,
        "missing_answer_tags_count": 0,
        "missing_answer_rate": 0.0,
        "denominator": "trace_json_count",
        "error": "",
    }
    zp = Path(zip_path)
    if not zp.exists() or not zp.is_file():
        out["error"] = f"zip_missing:{zp}"
        return out

    missing_count = _count_missing_answer_tags_in_terminal_log(Path(terminal_log_path))
    out["missing_answer_tags_count"] = int(missing_count)
    try:
        with zipfile.ZipFile(zp, "r") as zf:
            members = zf.namelist()
            main_members = [m for m in members if (m == "main.jsonl" or m.endswith("/main.jsonl"))]
            main_lines = 0
            for member in main_members:
                with zf.open(member, "r") as fh:
                    main_lines += sum(1 for _ in fh)
            trace_count = len(
                [
                    m
                    for m in members
                    if (m == "trace.json" or m.endswith("/trace.json") or m.endswith(".trace.json"))
                ]
            )
            out["main_jsonl_lines"] = int(main_lines)
            out["trace_json_count"] = int(trace_count)
            denom = int(main_lines) if int(main_lines) > 0 else int(trace_count)
            out["denominator"] = "main_jsonl_lines" if int(main_lines) > 0 else "trace_json_count"
            out["missing_answer_rate"] = (float(missing_count) / float(denom)) if denom > 0 else 0.0
            return out
    except Exception as e:
        out["error"] = f"zip_stats_error:{type(e).__name__}:{e}"
        return out


def _emit_phase2_zip_stats_line(output_dir: Path, seeds: Sequence[int], terminal_log_path: Path) -> Dict[str, Any]:
    # Ensure tee buffer is visible before reading terminal log for counting.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    zip_paths: List[Path] = []
    for seed in seeds:
        p = Path(output_dir) / f"seed_{int(seed)}" / "ev_06" / "evidence_package.zip"
        if p.exists() and p.is_file():
            zip_paths.append(p)

    per_zip: List[Dict[str, Any]] = [_stats_evidence_zip(zp, terminal_log_path) for zp in zip_paths]
    main_total = int(sum(int(x.get("main_jsonl_lines", 0) or 0) for x in per_zip))
    trace_total = int(sum(int(x.get("trace_json_count", 0) or 0) for x in per_zip))
    missing_count = _count_missing_answer_tags_in_terminal_log(Path(terminal_log_path))
    denom = main_total if main_total > 0 else trace_total
    rate = (float(missing_count) / float(denom)) if denom > 0 else 0.0
    out = {
        "zip_count": int(len(zip_paths)),
        "main_jsonl_lines": int(main_total),
        "trace_json_count": int(trace_total),
        "missing_answer_tags_count": int(missing_count),
        "missing_answer_rate": float(rate),
        "denominator": "main_jsonl_lines" if main_total > 0 else "trace_json_count",
        "zip_paths": [str(p) for p in zip_paths],
        "per_zip": per_zip,
    }
    print("[zip_stats] " + json.dumps(out, ensure_ascii=False, sort_keys=True))
    return out


def _phase2_fail_fast_abort(res: "StageResult", reason: str) -> None:
    res.gates["J2"] = False
    res.success = False
    msg = f"Phase2Sanity fail-fast: {reason}"
    res.errors.append(msg)
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _phase2_report_and_abort(
    res: "StageResult",
    seed: int,
    ev_zip: Path,
    sanity: Dict[str, Any],
) -> None:
    print(f"[Phase2Sanity] FAIL seed={seed} zip={ev_zip}", file=sys.stderr)
    for mm in sanity.get("missing_member", []) or []:
        print(f"  - missing_member: {mm}", file=sys.stderr)
    print(f"  - csv_fields: {sanity.get('csv_fields', [])}", file=sys.stderr)
    print(
        f"  - row_count: got {sanity.get('row_count')} expected {sanity.get('row_count_expected')}",
        file=sys.stderr,
    )
    print(
        "  - raw_output_parse_invalid: "
        f"{sanity.get('raw_output_parse_invalid', 0)} "
        f"(examples: {sanity.get('examples', {}).get('raw_output_parse_invalid', [])})",
        file=sys.stderr,
    )
    print(
        "  - final_not_dict: "
        f"{sanity.get('final_not_dict', 0)} "
        f"(examples: {sanity.get('examples', {}).get('final_not_dict', [])})",
        file=sys.stderr,
    )
    print(f"  - final_missing_keys: {sanity.get('final_missing_keys', {})}", file=sys.stderr)
    print(f"  - invalid_anomaly: {sanity.get('invalid_anomaly', 0)}", file=sys.stderr)
    print(f"  - toolcall_rate_avg: {sanity.get('toolcall_rate_avg')}", file=sys.stderr)
    tp = sanity.get("trace_policy", {}) or {}
    print(
        f"  - trace_policy: require={tp.get('require')} ok={tp.get('ok')} detail={tp.get('detail')}",
        file=sys.stderr,
    )
    reason = ", ".join(sanity.get("errors", []) or ["unknown_sanity_failure"])
    _phase2_fail_fast_abort(res, reason)


def _phase2_missing_zip_fail_fast(
    res: "StageResult",
    output_dir: Path,
    ev_dir: Path,
    seed: int,
) -> None:
    entries: List[str] = []
    if ev_dir.exists() and ev_dir.is_dir():
        try:
            entries = [p.name for p in sorted(ev_dir.iterdir(), key=lambda x: x.name)[:30]]
        except Exception:
            entries = []
    print(
        "Expected evidence zip missing; likely output-dir template typo or braces expansion issue",
        file=sys.stderr,
    )
    print(f"  - seed: {seed}", file=sys.stderr)
    print(f"  - resolved output_dir: {output_dir}", file=sys.stderr)
    print(f"  - expected ev_06 dir: {ev_dir}", file=sys.stderr)
    print(f"  - ev_06 listing(top30): {entries}", file=sys.stderr)
    _phase2_fail_fast_abort(res, "expected_evidence_zip_missing")


def handle_remediation(errors: Sequence[str]) -> List[str]:
    """
    Checks errors and suggests remediation steps for schema issues.
    """
    remediation: List[str] = []
    for error in errors:
        if error == "final_not_dict":
            remediation.append("Schema error: `final` is not an object. Ensure every row writes `final` as a JSON dict.")
        elif error == "invalid_json_parse":
            remediation.append("Schema error: failed to parse `raw_output`. Ensure `raw_output` is valid JSON with a `final` object.")
        elif error == "missing_required_keys":
            remediation.append("Schema violation: `final` must include `anomaly`, `defect_type`, `bbox`, and `confidence`.")
        elif error == "invalid_anomaly":
            remediation.append("Schema violation: `final.anomaly` must be `yes` or `no`; default invalid values to `no`.")
        elif error == "invalid_defect_type":
            remediation.append("Schema violation: `final.defect_type` must be a non-empty string; default missing/invalid values to `none`.")
        elif error == "invalid_bbox":
            remediation.append("Schema violation: `final.bbox` must be a normalized 4-value box with x2>x1 and y2>y1.")
        elif error == "invalid_confidence":
            remediation.append("Schema violation: `final.confidence` must be null.")
        elif error == "raw_output_empty":
            remediation.append("Schema error: `raw_output` is empty. Ensure inference always writes a schema-valid fallback result.")
    # Preserve order while deduplicating.
    dedup: List[str] = []
    seen: Set[str] = set()
    for msg in remediation:
        if msg not in seen:
            seen.add(msg)
            dedup.append(msg)
    return dedup


def _is_placeholder_vlm_model_id(model_id: str) -> bool:
    s = str(model_id or "").strip().lower()
    if not s:
        return False
    leaf = s.rsplit("/", 1)[-1]
    if leaf in {"distilgpt2", "gpt2"}:
        return True
    if leaf.startswith("distil"):
        return True
    return False

class CSVHash:
    @staticmethod
    def compute(zip_path: Path, csv_name: str) -> Optional[str]:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                if csv_name in zf.namelist():
                    try:
                        import pandas as pd
                        import io
                        content = zf.read(csv_name).decode("utf-8")
                        df = pd.read_csv(io.StringIO(content))
                        if "sample_id" in df.columns: df = df.sort_values("sample_id")
                        return hashlib.sha256(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).hexdigest().upper()
                    except:
                        return hashlib.sha256(zf.read(csv_name)).hexdigest().upper()
        return None

    @staticmethod
    def generate_from_traces(trace_dir: Path) -> tuple[Optional[str], Optional[str]]:
        try:
            import pandas as pd
            rows = []
            t_files = sorted(list(trace_dir.rglob("trace.json")), key=lambda p: str(p))
            def _safe_str(x): return "" if x is None else str(x)
            
            for tf in t_files:
                try:
                    tr = json.loads(tf.read_text(encoding="utf-8"))
                    final_path = tf.parent / "final.json"
                    final_obj = {}
                    if final_path.exists(): final_obj = json.loads(final_path.read_text(encoding="utf-8"))
                    
                    turns = tr.get("turns", [])
                    raw_obj = {
                        "round0": next((t.get("raw_output", "") for t in turns if t["round"]==0), ""),
                        "round1": next((t.get("raw_output", "") for t in turns if t["round"]==1), ""),
                        "round2": next((t.get("raw_output", "") for t in turns if t["round"]==2), ""),
                        "cr_called": any(t["name"]=="cr" for t in turns),
                        "ref_sample_id": "",
                        "final": {
                            "anomaly": final_obj.get("anomaly", "unknown"),
                            "bbox": final_obj.get("bbox"),
                            "defect_type": _safe_str(final_obj.get("defect_type")),
                            "confidence": final_obj.get("confidence")
                        }
                    }
                    for t in turns:
                        if t["name"]=="cr" and "tool_result" in t:
                            raw_obj["ref_sample_id"] = t["tool_result"].get("ref_sample_id", "")
                            
                    raw_str = json.dumps(raw_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    fp = tr.get("fingerprint", {})
                    
                    row = {
                        "sample_id": tr.get("sample_id", ""),
                        "class_name": tr.get("class_name", ""),
                        "gt_label": tr.get("gt_label", ""),
                        "pred_label": final_obj.get("anomaly") if final_obj.get("anomaly") in {"yes", "no"} else "UNKNOWN",
                        "raw_output": raw_str,
                        "model_id": fp.get("model_id", ""),
                        "seed": fp.get("seed", 0),
                        "prompt_hash": fp.get("prompt_hash", ""),
                        "config_hash": fp.get("config_hash", ""),
                        "pz_called": 1 if any(t["name"]=="pz" for t in turns) else 0,
                        "cr_called": 1 if raw_obj["cr_called"] else 0,
                        "bbox_norm": json.dumps(final_obj.get("bbox"), ensure_ascii=False, separators=(",", ":")),
                        "ref_sample_id": raw_obj["ref_sample_id"]
                    }
                    rows.append(row)
                except Exception: pass
            
            if not rows: return None, "No rows generated"
            df = pd.DataFrame(rows, columns=[
                "sample_id", "class_name", "gt_label", "pred_label", "raw_output",
                "model_id", "seed", "prompt_hash", "config_hash",
                "pz_called", "cr_called", "bbox_norm", "ref_sample_id"
            ])
            df = df.sort_values("sample_id").reset_index(drop=True)
            return hashlib.sha256(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).hexdigest().upper(), None
        except Exception as e:
            return None, str(e)


def build_phase1_acceptance_payload(work_dir: Path, seeds: List[int], payload: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    per_seed_metrics_path: List[str] = []
    evidence_paths: List[str] = []
    n_total = 0

    artifacts = payload.get("artifacts") or {}
    dataset_binding_hash = artifacts.get("manifest_hash") or artifacts.get("ids_sha256")
    dataset_binding_hash_source = (
        "manifest_hash" if artifacts.get("manifest_hash")
        else ("ids_sha256" if artifacts.get("ids_sha256") else "")
    )
    if not dataset_binding_hash:
        errors.append("dataset_binding_hash unavailable (manifest_hash/ids_sha256 both missing)")

    for seed in seeds:
        s_dir = work_dir / f"seed_{seed}"
        metrics_json = s_dir / "baseline_metrics.json"
        per_seed_metrics_path.append(str(metrics_json))

        ev_zip, _, found = resolve_evidence_zip(s_dir / "ev_06")
        if ev_zip is None or not ev_zip.exists():
            evidence_paths.append("")
            errors.append(
                f"seed {seed}: missing evidence zip (expected evidence_package.zip in {s_dir / 'ev_06'}; found {found})"
            )
        else:
            evidence_paths.append(str(ev_zip))

        if not metrics_json.exists():
            errors.append(f"seed {seed}: missing baseline_metrics.json")
            continue

        try:
            m = json.loads(metrics_json.read_text(encoding="utf-8"))
            n_total += int(m.get("n_valid", 0))
        except Exception as e:
            errors.append(f"seed {seed}: failed to parse baseline_metrics.json ({e})")

    summary_path = work_dir / "aggregate" / "baseline_summary.json"
    if not summary_path.exists():
        errors.append("missing aggregate/baseline_summary.json")
    else:
        try:
            summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(summary_obj.get("n_total", -1)) < 0:
                errors.append("baseline_summary.json missing n_total")
        except Exception as e:
            errors.append(f"failed to parse baseline_summary.json ({e})")

    success = (
        len(seeds) == 3
        and len(errors) == 0
        and bool(payload.get("success", False))
    )

    return {
        "success": success,
        "errors": errors,
        "seeds": seeds,
        "dataset_binding_hash": dataset_binding_hash,
        "dataset_binding_hash_source": dataset_binding_hash_source,
        "per_seed_metrics_path": per_seed_metrics_path,
        "summary_path": str(summary_path),
        "n_total": n_total,
        "evidence_paths": evidence_paths,
        "audit_note": "Phase1 baseline excludes SFT/GRPO training stages; acceptance uses baseline inference outputs only.",
    }

def build_phase2_full_infer_acceptance_payload(work_dir: Path, seeds: List[int], payload: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    evidence_paths: List[str] = []
    gates = payload.get("gates", {}) or {}
    measurements = payload.get("measurements", {}) or {}

    for seed in seeds:
        s_dir = work_dir / f"seed_{seed}"
        ev_zip, _, found = resolve_evidence_zip(s_dir / "ev_06")
        if ev_zip is None or not ev_zip.exists():
            evidence_paths.append("")
            errors.append(
                f"seed {seed}: missing evidence zip (expected evidence_package.zip in {s_dir / 'ev_06'}; found {found})"
            )
        else:
            evidence_paths.append(str(ev_zip))

    if not bool(gates.get("J2", False)):
        errors.append("J2 must pass in phase2_full_infer")
    if not bool(gates.get("J3", False)):
        errors.append("J3 must pass in phase2_full_infer")
    tool_avg = float(measurements.get("toolcall_rate_avg", 0.0) or 0.0)
    if tool_avg <= 0:
        errors.append(f"J6/toolcall_rate_avg must be > 0 in phase2_full_infer, got {tool_avg}")

    success = (
        len(errors) == 0
        and bool(payload.get("success", False))
    )

    return {
        "success": success,
        "errors": errors,
        "seeds": seeds,
        "evidence_paths": evidence_paths,
        "toolcall_rate_avg": tool_avg,
        "audit_note": "Phase2 full infer acceptance requires infer-only evidence completeness and non-zero tool usage.",
    }

# --- Stages ---

def _expected_count_from_ids(ids_path: Path, eff_max: Optional[int]) -> Optional[int]:
    """
    Calculate expected trace count based on effective max samples and available IDs.
    Policy:
    - If eff_max is None (Full Run), we enforce STRICT count against all IDs found in ids.txt.
      Rationale: If the user explicitly requested a full run (via --allow-full-dataset),
      we must verify that every single ID in the input list produced a trace.
      This hardens J3 for full runs.
    - If eff_max is set, we return min(len(ids), eff_max).
    """
    if not ids_path.exists(): return None
    try:
        lines = [x for x in ids_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        n_ids = len(lines)
        if eff_max is None:
            # Full run -> Strict expectation: all IDs must have traces
            return n_ids
        return min(n_ids, eff_max)
    except:
        return None


def _resolve_expected_max_samples(ctx: StageContext) -> Optional[int]:
    eff_max = ctx.get_effective_max_samples()
    if eff_max is not None:
        return int(eff_max)
    env = _effective_env(ctx.env_overrides)
    dev_raw = str(env.get("DEV_MAX_SAMPLES", "") or "").strip()
    if not dev_raw:
        return None
    if not re.fullmatch(r"[1-9]\d*", dev_raw):
        return None
    return int(dev_raw)

def _generate_ids_from_dataset(requested_split: str) -> tuple[List[str], str]:
    from datasets import load_dataset
    ds = load_dataset("jiang-cc/MMAD")
    split = requested_split
    if split not in ds:
        if "test" in ds:
            split = "test"
        else:
            split = list(ds.keys())[0]
    d0 = ds[split]
    valid_ids: List[str] = []
    for i in range(len(d0)):
        row = d0[i]
        sid = row.get("sample_id") or row.get("id")
        if not sid:
            sid = f"{split}_{i}"
        valid_ids.append(str(sid).strip())
    return valid_ids, split

class Stage:
    def execute(self, ctx: StageContext, res: StageResult): raise NotImplementedError

class PreflightDeps(Stage):
    def execute(self, ctx: StageContext, res: StageResult):
        missing_deps = []
        try: subprocess.check_call([sys.executable, "-c", "import yaml"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: missing_deps.append("PyYAML")
        for mod, pkg in [("torch", "torch"), ("transformers", "transformers"), ("accelerate", "accelerate"), ("datasets", "datasets"), ("PIL", "pillow"), ("pyarrow", "pyarrow")]:
            try: subprocess.check_call([sys.executable, "-c", f"import {mod}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except: missing_deps.append(pkg)
        if missing_deps:
            res.success = False
            res.gates["J4_DEPENDENCIES"] = False
            res.errors.append(f"Missing L2 dependencies: {', '.join(missing_deps)}")
            res.remediations.append(f"conda install -y -c conda-forge torch transformers accelerate datasets pillow pyarrow")
        else:
            res.gates["J4_DEPENDENCIES"] = True

class ProbeIds(Stage):
    def execute(self, ctx: StageContext, res: StageResult):
        offline_check = _check_offline_model_readiness_for_probe(ctx)
        if offline_check.get("required"):
            res.artifacts["offline_model_check"] = {
                "required": True,
                "ok": bool(offline_check.get("ok", False)),
                "hf_home": offline_check.get("hf_home", ""),
                "hf_cache": offline_check.get("hf_cache", ""),
                "checked_notes": offline_check.get("checked_notes", []),
                "ready_source": offline_check.get("ready_source", ""),
            }
            if not offline_check.get("ok", False):
                res.success = False
                err_msg = str(offline_check.get("msg", "offline model readiness check failed"))
                res.errors.append(err_msg)
                res.gates["J2"] = False
                res.artifacts["evidence_checks"].append({
                    "stage": "Probe",
                    "stable_stage": "ProbeIds",
                    "label": "Probe",
                    "code": "OFFLINE_MODEL_NOT_READY",
                    "msg": err_msg,
                })
                for rk in ["remediation_a", "remediation_b"]:
                    rv = str(offline_check.get(rk, "")).strip()
                    if rv and rv not in res.remediations:
                        res.remediations.append(rv)
                return

        if ctx.phase1_baseline or ctx.phase2_full_infer:
            with Timer("Phase1 Dataset Binding"):
                try:
                    valid_ids, used_split = _generate_ids_from_dataset(ctx.dataset_split)
                    
                    if not valid_ids:
                        res.success = False
                        res.errors.append("No IDs found in dataset for Phase1")
                        return

                    (ctx.work_dir / "ids.txt").write_text("\n".join(valid_ids), encoding="utf-8")
                    res.artifacts["ids_sha256"] = hashlib.sha256("\n".join(valid_ids).encode()).hexdigest()
                    res.artifacts["phase1_full_mode"] = True
                    res.artifacts["audit_ids_total"] = len(valid_ids)
                    res.artifacts["audit_ids_source"] = "generated_dataset"
                    mode_name = "phase2_full_infer" if ctx.phase2_full_infer else "phase1_baseline"
                    res.artifacts["ids_source_note"] = f"{mode_name} uses generated full dataset ids (split={used_split})"
                    return
                except Exception as e:
                    res.success = False
                    mode_name = "Phase2" if ctx.phase2_full_infer else "Phase1"
                    res.errors.append(f"{mode_name} Dataset Load Failed: {e}")
                    return

        with Timer("Data Binding"):
            manifest_path = PROJECT_ROOT / "data/mmad/mmad_manifest.json"
            if not manifest_path.exists():
                manifest_path = Path("data/mmad/mmad_manifest.json").resolve()
            if not manifest_path.exists():
                 res.success = False
                 msg = "Manifest missing"
                 res.errors.append(msg)
                 res.artifacts["evidence_checks"].append({
                     "stage": "DataBinding", 
                     "stable_stage": "DataBinding",
                     "label": "DataBinding",
                     "code": "DATA_BINDING_FAILED", 
                     "msg": msg
                 })
                 return
            
            res.artifacts["manifest_hash"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            valid_ids = []
            if (ctx.work_dir / "ids.txt").exists() and (ctx.work_dir / "ids.txt").stat().st_size > 0:
                 valid_ids = [x.strip() for x in (ctx.work_dir / "ids.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
            ids_source = "ids.txt" if valid_ids else ""
            ids_source_note = "reused existing ids.txt from output_dir" if valid_ids else ""
            eff_max = ctx.get_effective_max_samples()
            if ctx.strict_j_mode and ctx.allow_full_dataset:
                # strict_j full run must use full dataset ids source, not a potentially truncated ids.txt.
                valid_ids = []
                ids_source = ""
                ids_source_note = "strict_j + allow_full_dataset: force generated full dataset ids source"
            elif ctx.strict_j_mode and eff_max is not None and eff_max > 32 and len(valid_ids) < eff_max:
                valid_ids = []
                ids_source = ""
                ids_source_note = f"strict_j + max_samples={eff_max}>32: existing ids.txt insufficient, switching ids source"
            
            if not valid_ids:
                if ctx.strict_j_mode and ctx.allow_full_dataset:
                    try:
                        valid_ids, used_split = _generate_ids_from_dataset(ctx.dataset_split)
                        ids_source = "generated_dataset"
                        ids_source_note = f"strict_j + allow_full_dataset: generated full ids from dataset (split={used_split})"
                    except Exception as e:
                        res.success = False
                        res.errors.append(f"Strict J full ids generation failed: {e}")
                        return
                elif ctx.strict_j_mode and eff_max is not None and eff_max > 32:
                    try:
                        valid_ids, used_split = _generate_ids_from_dataset(ctx.dataset_split)
                        if len(valid_ids) < eff_max:
                            res.success = False
                            res.errors.append(
                                f"Strict J ids source too small: required >= {eff_max}, got {len(valid_ids)} from dataset split {used_split}"
                            )
                            return
                        ids_source = "generated_dataset"
                        ids_source_note = f"strict_j + max_samples={eff_max}>32: generated dataset ids (split={used_split}) to avoid probe-32 cap"
                    except Exception as e:
                        res.success = False
                        res.errors.append(f"Strict J ids generation failed for max_samples={eff_max}: {e}")
                        return
                else:
                    probe_dir = ctx.work_dir / "probe"
                    probe_dir.mkdir(parents=True, exist_ok=True)
                    probe_cfg = ctx.work_dir / "probe_config.yaml"
                    probe_cfg.write_text("model_id: distilgpt2\nrun_name: probe\n", encoding="utf-8")
                    
                    cmd = CmdBuilder(ctx.get_script("06_run_agentiad_infer.py")).with_config(probe_cfg).with_run_name("probe").arg("--seed", "42").arg("--max_samples", "32").arg("--evidence_dir", probe_dir).arg("--split", ctx.dataset_split).build()
                    if J1.record_if_violation(cmd, res, "probe cmd"): return
                    probe_res = CmdRunner.run(cmd, ctx.env_overrides, stream_output=True)
                    if probe_res:
                        merged_probe = probe_res.stderr if probe_res.stderr else probe_res.stdout
                        l2_asset_probe = parse_l2_asset_audit(merged_probe or b"")
                        if "mmad_root_resolved" in l2_asset_probe:
                            res.artifacts["mmad_root_resolved"] = l2_asset_probe["mmad_root_resolved"]
                        if "mmad_asset_mode" in l2_asset_probe:
                            res.artifacts["mmad_asset_mode"] = l2_asset_probe["mmad_asset_mode"]
                        probe_l2 = parse_l2_json_from_cmd(probe_res)
                        if isinstance(probe_l2, dict):
                            if "error" in probe_l2:
                                res.artifacts["probe_l2_error"] = probe_l2.get("error")
                            if "dataset_splits_available" in probe_l2:
                                res.artifacts["probe_dataset_splits_available"] = probe_l2.get("dataset_splits_available")
                            if "remediation" in probe_l2:
                                res.artifacts["probe_remediation"] = probe_l2.get("remediation")
                    
                    if _require_cmd_ok(res, probe_res, "Probe", "ProbeIds"):
                        zip_path, _, _ = resolve_evidence_zip(probe_dir)
                        if zip_path is not None and zip_path.exists():
                            with zipfile.ZipFile(zip_path, "r") as zf:
                                for n in zf.namelist():
                                    if n.endswith("trace.json"):
                                        try:
                                            d = json.load(zf.open(n))
                                            if "sample_id" in d: valid_ids.append(d["sample_id"])
                                        except: pass
                        valid_ids = sorted(list(set(valid_ids)))[:32]
                        ids_source = "probe"
                        ids_source_note = "default probe ids source (max_samples=32)"
                    else:
                        # _require_cmd_ok already set res.success=False and errors
                        res.gates["J3"] = False
                        return

            if not valid_ids:
                 res.success = False
                 msg = "No valid IDs found"
                 res.errors.append(msg)
                 res.artifacts["evidence_checks"].append({
                     "stage": "DataBinding", 
                     "stable_stage": "DataBinding",
                     "label": "DataBinding",
                     "code": "DATA_BINDING_FAILED", 
                     "msg": msg
                 })
                 return
            (ctx.work_dir / "ids.txt").write_text("\n".join(valid_ids), encoding="utf-8")
            res.artifacts["ids_sha256"] = hashlib.sha256("\n".join(valid_ids).encode()).hexdigest()
            res.artifacts["audit_ids_total"] = len(valid_ids)
            res.artifacts["audit_ids_source"] = ids_source or "generated/ids.txt"
            res.artifacts["ids_source_note"] = ids_source_note or "ids source selected by data binding policy"

class AgentInfer06(Stage):
    def execute(self, ctx: StageContext, res: StageResult):
        mr_cfg = ctx.work_dir / "mr_config.yaml"
        if ctx.phase2_full_infer:
            cfg_obj: Dict[str, Any] = {}
            if mr_cfg.exists():
                try:
                    import yaml

                    loaded = yaml.safe_load(mr_cfg.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        cfg_obj = dict(loaded)
                except Exception:
                    cfg_obj = {}
            if ctx.vlm_model_id:
                cfg_obj["vlm_model_id"] = str(ctx.vlm_model_id)
                cfg_obj["model_id"] = str(ctx.vlm_model_id)
            elif "model_id" not in cfg_obj and "vlm_model_id" not in cfg_obj:
                cfg_obj["model_id"] = "distilgpt2"
            if ctx.vlm_max_side is not None:
                cfg_obj["vlm_max_side"] = int(ctx.vlm_max_side)
            if ctx.sdp_backend is not None:
                cfg_obj["sdp_backend"] = str(ctx.sdp_backend)
            if ctx.vlm_retry_n is not None:
                cfg_obj["vlm_retry_n"] = int(ctx.vlm_retry_n)
            cfg_obj["run_name"] = "minireal"
            try:
                import yaml

                mr_cfg.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")
            except Exception:
                mr_cfg.write_text("model_id: distilgpt2\nrun_name: minireal\n", encoding="utf-8")
        elif not mr_cfg.exists():
            mr_cfg.write_text("model_id: distilgpt2\nrun_name: minireal\n", encoding="utf-8")
        (ctx.work_dir / "L1_baseline.csv").write_text("idx,split,answer,pred,correct,method,triggered\n")
        
        if ctx.no_adapter: return 

        for seed in ctx.seeds:
            with Timer(f"Seed {seed} Pipeline"):
                s_dir = ctx.work_dir / f"seed_{seed}"
                s_dir.mkdir(parents=True, exist_ok=True)
                ev_06 = s_dir / "ev_06"
                
                cmd = CmdBuilder(ctx.get_script("06_run_agentiad_infer.py")).with_config(mr_cfg).with_run_name(f"mr_s{seed}").arg("--seed", seed).arg("--id_list", ctx.work_dir / "ids.txt").with_evidence_dir(ev_06)
                cmd.arg("--split", ctx.dataset_split)
                if ctx.phase2_full_infer and ctx.vlm_max_side is not None:
                    cmd.arg("--vlm-max-side", int(ctx.vlm_max_side))
                if ctx.phase2_full_infer and ctx.sdp_backend is not None:
                    cmd.arg("--sdp-backend", str(ctx.sdp_backend))
                if ctx.phase2_full_infer and ctx.vlm_retry_n is not None:
                    cmd.arg("--vlm-retry-n", int(ctx.vlm_retry_n))
                
                if ctx.phase1_baseline:
                    cmd.arg("--enable_tools", "false")
                
                if ctx.allow_full_dataset:
                    cmd.arg("--allow_full_dataset")
                else:
                    eff_max = ctx.get_effective_max_samples()
                    if eff_max is not None:
                         cmd.arg("--max_samples", eff_max)
                    else:
                         cmd.arg("--max_samples", "99999")
                
                if ctx.allow_flags: cmd.arg("--allow_code_execution")
                
                cmd_list = cmd.build()
                if J1.record_if_violation(cmd_list, res, f"06 cmd seed {seed}"): return
                cmd_res = CmdRunner.run(cmd_list, ctx.env_overrides, stream_output=True)
                merged_output = b""
                if cmd_res:
                    merged_output = cmd_res.stderr if cmd_res.stderr else cmd_res.stdout
                merged_text = (merged_output or b"").decode("utf-8", errors="replace")
                l2_asset_info = parse_l2_asset_audit(merged_output or b"")
                if "mmad_root_resolved" in l2_asset_info:
                    res.artifacts["mmad_root_resolved"] = l2_asset_info["mmad_root_resolved"]
                if "mmad_asset_mode" in l2_asset_info:
                    res.artifacts["mmad_asset_mode"] = l2_asset_info["mmad_asset_mode"]
                if l2_asset_info.get("mmad_asset_mode") == "hub_disabled_no_assets":
                    remediation = "export MMAD_ROOT=<local_mmad_root>; MMAD_ROOT must contain DS-MVTec/ and MVTec-AD/"
                    if remediation not in res.remediations:
                        res.remediations.append(remediation)
                if not _require_cmd_ok(res, cmd_res, f"S{seed}-06", "AgentInfer06"):
                    # CMD_FAILED already recorded by helper
                    return
                
                # Task B: Parse L2_RESULT_JSON for effective_n
                l2_effective_n = None
                seed_audit_mismatch_recorded = False
                l2_data = parse_l2_json_from_cmd(cmd_res)
                actual_model_id = ""
                if isinstance(l2_data, dict):
                    l2_dataset_split = str(l2_data.get("dataset_split", "") or "")
                    res.artifacts[f"seed_{seed}_l2_dataset_split"] = l2_dataset_split
                    actual_model_id = str(
                        l2_data.get("vlm_model_id", "") or l2_data.get("model_id", "")
                    ).strip()
                    if not actual_model_id:
                        out_csv_raw = str(l2_data.get("out_csv", "") or "").strip()
                        if out_csv_raw:
                            actual_model_id = _read_first_model_id_from_csv(Path(out_csv_raw)) or ""
                    if ctx.strict_j_mode and l2_dataset_split and l2_dataset_split != ctx.dataset_split:
                        seed_audit_mismatch_recorded = True
                        res.artifacts["evidence_checks"].append({
                            "stage": f"S{seed}-06",
                            "stable_stage": "AgentInfer06",
                            "label": f"S{seed}-06",
                            "code": "DATASET_SPLIT_MISMATCH",
                            "msg": f"Strict J fail: dataset_split mismatch (expected={ctx.dataset_split}, actual={l2_dataset_split})"
                        })
                        res.success = False
                        res.gates["J3"] = False
                        res.errors.append(
                            f"S{seed}-06 dataset_split mismatch under strict_j: expected {ctx.dataset_split}, got {l2_dataset_split}"
                        )
                    if "effective_n" in l2_data:
                        l2_effective_n = int(l2_data["effective_n"])
                        res.artifacts[f"seed_{seed}_l2_effective_n"] = l2_effective_n
                        # Also record skip info for audit
                        if "n_skipped" in l2_data:
                             res.artifacts[f"seed_{seed}_l2_skipped"] = l2_data["n_skipped"]
                        if "skip_reasons" in l2_data:
                             res.artifacts[f"seed_{seed}_l2_skip_reasons"] = l2_data["skip_reasons"]
                        if "skip_reason_examples" in l2_data:
                             res.artifacts[f"seed_{seed}_l2_skip_reason_examples"] = l2_data["skip_reason_examples"]
                        if "allow_full_dataset" in l2_data:
                             res.artifacts[f"seed_{seed}_l2_allow_full_dataset"] = l2_data["allow_full_dataset"]
                        if "max_samples_effective" in l2_data:
                             res.artifacts[f"seed_{seed}_l2_max_samples_effective"] = l2_data["max_samples_effective"]

                        image_env = l2_data.get("image_loader_env", {}) if isinstance(l2_data, dict) else {}
                        offline_all = (
                            str(image_env.get("HF_DATASETS_OFFLINE", "")) == "1"
                            and str(image_env.get("HF_HUB_OFFLINE", "")) == "1"
                            and str(image_env.get("TRANSFORMERS_OFFLINE", "")) == "1"
                        )
                        skip_examples = l2_data.get("skip_reason_examples", {}) if isinstance(l2_data, dict) else {}
                        if (
                            str(res.artifacts.get("mmad_asset_mode", "")) == "local_root"
                            and offline_all
                            and isinstance(skip_examples, dict)
                            and "hub_download_error" in skip_examples
                        ):
                            res.success = False
                            res.errors.append(
                                f"S{seed}-06 local_root offline violation: hub_download_error present in skip_reason_examples. "
                                "Remediation: export MMAD_ROOT=<local_mmad_root>; MMAD_ROOT must contain DS-MVTec/ and MVTec-AD/."
                            )
                            remediation = "export MMAD_ROOT=<local_mmad_root>; MMAD_ROOT must contain DS-MVTec/ and MVTec-AD/"
                            if remediation not in res.remediations:
                                res.remediations.append(remediation)

                        # Audit check: mismatch between requested and effective
                        if "n_requested_ids" in l2_data:
                            n_req = int(l2_data["n_requested_ids"])
                            n_skipped = int(l2_data.get("n_skipped", 0))
                            skip_reasons = l2_data.get("skip_reasons", {})
                            max_eff_raw = l2_data.get("max_samples_effective")

                            max_eff_int: Optional[int] = None
                            if isinstance(max_eff_raw, int):
                                max_eff_int = max_eff_raw
                            elif isinstance(max_eff_raw, str):
                                s_eff = max_eff_raw.strip()
                                if s_eff and s_eff.upper() != "NONE":
                                    try:
                                        max_eff_int = int(s_eff)
                                    except Exception:
                                        max_eff_int = None

                            expected_n = n_req if max_eff_int is None else min(n_req, max_eff_int)
                            has_skip_signal = n_skipped > 0 or (isinstance(skip_reasons, dict) and len(skip_reasons) > 0)
                            should_flag_mismatch = (
                                l2_effective_n != expected_n and (has_skip_signal or l2_effective_n < expected_n)
                            )

                            if should_flag_mismatch:
                                seed_audit_mismatch_recorded = True
                                res.artifacts["evidence_checks"].append({
                                    "stage": f"S{seed}-06",
                                    "stable_stage": "AgentInfer06",
                                    "label": f"S{seed}-06",
                                    "code": "EFFECTIVE_N_MISMATCH",
                                    "msg": (
                                        f"Expected {expected_n} != Effective {l2_effective_n} "
                                        f"(n_requested_ids={n_req}, max_samples_effective={max_eff_raw}, "
                                        f"Skipped {n_skipped}; Reasons={skip_reasons})"
                                    )
                                })
                                if ctx.phase1_baseline:
                                    res.success = False
                                    res.gates["J3"] = False
                                    res.errors.append(
                                        f"Phase1 dataset completeness failure: effective_n ({l2_effective_n}) != expected ({expected_n}); n_skipped={n_skipped}, skip_reasons={skip_reasons}. Remediation: export MMAD_ROOT=<local_mmad_root>; MMAD_ROOT must contain DS-MVTec/ and MVTec-AD/."
                                    )
                                    remediation = "export MMAD_ROOT=<local_mmad_root>; MMAD_ROOT must contain DS-MVTec/ and MVTec-AD/"
                                    if remediation not in res.remediations:
                                        res.remediations.append(remediation)

                if not actual_model_id and isinstance(l2_data, dict):
                    out_csv_raw = str(l2_data.get("out_csv", "") or "").strip()
                    if out_csv_raw:
                        actual_model_id = _read_first_model_id_from_csv(Path(out_csv_raw)) or ""
                if (not actual_model_id) or str(actual_model_id).strip().upper() == "UNKNOWN":
                    ev06_model_zip, _, _ = resolve_evidence_zip(ev_06)
                    if ev06_model_zip is not None and ev06_model_zip.exists():
                        actual_model_id = _read_first_model_id_from_trace_zip(ev06_model_zip) or actual_model_id
                if not actual_model_id:
                    actual_model_id = "UNKNOWN"

                res.artifacts[f"seed_{seed}_audit_vlm_model_id"] = actual_model_id
                cur_audit = str(res.artifacts.get("audit_vlm_model_id", "") or "").strip()
                if (not cur_audit) or cur_audit.upper() == "UNKNOWN":
                    res.artifacts["audit_vlm_model_id"] = actual_model_id
                    res.artifacts["audit_vlm_model_source"] = ctx.vlm_model_source

                if ctx.phase2_full_infer:
                    fallback_marker = "[OfflineFallback] Activating fallback to distilgpt2"
                    explicit_user_model = ctx.vlm_model_source in {"cli", "env"} and bool(ctx.vlm_model_id)
                    remediation = (
                        "Model fallback detected. Set --vlm-model-local-dir to a downloaded VLM directory "
                        "(e.g. /data2/lrrelevant/hf_offline/models/Qwen2.5-VL-3B-Instruct-AWQ) and rerun phase2_full_infer."
                    )
                    if fallback_marker in merged_text:
                        res.success = False
                        res.gates["J2"] = False
                        res.errors.append(
                            f"S{seed}-06 hard-fail: detected model load fallback to distilgpt2. {remediation}"
                        )
                        if remediation not in res.remediations:
                            res.remediations.append(remediation)
                        return
                    if _is_placeholder_vlm_model_id(actual_model_id) and not explicit_user_model:
                        res.success = False
                        res.gates["J2"] = False
                        res.errors.append(
                            f"S{seed}-06 hard-fail: placeholder model_id '{actual_model_id}' detected in phase2_full_infer. {remediation}"
                        )
                        if remediation not in res.remediations:
                            res.remediations.append(remediation)
                        return

                eff_max = _resolve_expected_max_samples(ctx)
                subset_size = _expected_count_from_ids(ctx.work_dir / "ids.txt", eff_max)
                if ctx.phase2_full_infer:
                    res.artifacts[f"seed_{seed}_audit_expected_max_samples"] = "NONE" if eff_max is None else int(eff_max)
                
                # If Phase 1 and we have effective_n, use it for strict J3
                if ctx.phase1_baseline and l2_effective_n is not None:
                     subset_size = l2_effective_n

                ev06_zip, ev06_note, ev06_found = resolve_evidence_zip(ev_06)
                if ev06_zip is None:
                    if ctx.phase2_full_infer:
                        _phase2_missing_zip_fail_fast(res, ctx.work_dir, ev_06, int(seed))
                    code = "MISSING_ZIP"
                    msg = f"Missing zip: expected evidence_package.zip in {ev_06}; found {ev06_found}"
                    dur = 0.0
                else:
                    if ctx.phase2_full_infer:
                        sanity = _phase2_sanity_check_evidence_zip(
                            zip_path=str(ev06_zip),
                            seed=int(seed),
                            expected_n=subset_size,
                            require_toolcall_rate=True,
                            require_trace_policy="auto",
                        )
                        res.artifacts[f"seed_{seed}_phase2_sanity"] = sanity
                        if not bool(sanity.get("ok")):
                            _phase2_report_and_abort(res, int(seed), ev06_zip, sanity)
                    code, msg, dur = Evidence.verify_zip(ev06_zip, "06_run_agentiad_infer.py", expected_trace_count=subset_size)
                    if ev06_note:
                        msg = f"{msg} ({ev06_note})"
                if not (seed_audit_mismatch_recorded and code == "OK"):
                    res.artifacts["evidence_checks"].append({
                        "stage": f"S{seed}-06", 
                        "stable_stage": "AgentInfer06",
                        "label": f"S{seed}-06",
                        "code": code, 
                        "msg": msg
                    })
                res.measurements["evidence_check_sec"] = res.measurements.get("evidence_check_sec", 0.0) + dur
                if code != "OK": res.success = False; res.errors.append(f"S{seed}-06 Evidence: {msg}")
                if ctx.phase2_full_infer and ev06_zip is not None and ev06_zip.exists():
                    csv_name = f"tables/agentiad_infer_mr_s{seed}.csv"
                    phase2_counts = _count_phase2_execution_errors_from_zip(ev06_zip, csv_name)
                    exec_cnt = int(phase2_counts.get("execution_error_count", 0))
                    unknown_cnt = int(phase2_counts.get("unknown_count", 0))
                    schema_bad_cnt = int(phase2_counts.get("strict_schema_invalid_count", 0))
                    res.artifacts[f"seed_{seed}_execution_error_count"] = exec_cnt
                    res.artifacts[f"seed_{seed}_unknown_count"] = unknown_cnt
                    res.artifacts[f"seed_{seed}_strict_schema_invalid_count"] = schema_bad_cnt
                    res.artifacts["execution_error_count"] = int(res.artifacts.get("execution_error_count", 0)) + exec_cnt
                    res.artifacts["unknown_count"] = int(res.artifacts.get("unknown_count", 0)) + unknown_cnt
                    res.artifacts["strict_schema_invalid_count"] = int(res.artifacts.get("strict_schema_invalid_count", 0)) + schema_bad_cnt
                    print(
                        f"[Phase2Audit] seed={seed} execution_error_count={exec_cnt} unknown_count={unknown_cnt} strict_schema_invalid_count={schema_bad_cnt}",
                        file=sys.stderr,
                    )
                    if exec_cnt > 0 or unknown_cnt > 0 or schema_bad_cnt > 0:
                        if exec_cnt == 0 and (unknown_cnt > 0 or schema_bad_cnt > 0):
                            schema_errors = phase2_counts.get("schema_error_examples", []) if isinstance(phase2_counts, dict) else []
                            schema_msgs = handle_remediation(schema_errors if isinstance(schema_errors, list) else [])
                            remediation = (
                                "Phase2 hard gate failed. execution_error_count=0 but unknown/schema_invalid detected; "
                                "focus remediation on schema repair for `final` and strict JSON normalization."
                            )
                            if schema_msgs:
                                remediation = remediation + " " + " ".join(schema_msgs)
                        else:
                            remediation = (
                                "Phase2 hard gate failed. Reduce VLM memory pressure via --vlm-max-side (e.g. 640), "
                                "and verify GPU memory/driver stability."
                            )
                        res.success = False
                        res.gates["J2"] = False
                        res.errors.append(
                            f"S{seed}-06 hard-fail: execution_error_count={exec_cnt}, unknown_count={unknown_cnt}, strict_schema_invalid_count={schema_bad_cnt}. {remediation}"
                        )
                        if remediation not in res.remediations:
                            res.remediations.append(remediation)
                        return

class BuildTraj08(Stage):
    def execute(self, ctx: StageContext, res: StageResult):
        if ctx.no_adapter or ctx.phase1_baseline: return
        mr_cfg = ctx.work_dir / "mr_config.yaml"
        for seed in ctx.seeds:
            if not res.success: break
            s_dir = ctx.work_dir / f"seed_{seed}"
            ev_06 = s_dir / "ev_06"
            ev_08 = s_dir / "ev_08"
            l3_jsonl = s_dir / "l3.jsonl"
            
            # Prereq Check
            zip_path, zip_note, zip_found = resolve_evidence_zip(ev_06)
            if zip_path is None:
                res.success = False
                res.errors.append(f"S{seed}-08 prereq missing: expected evidence_package.zip in ev_06; found {zip_found}")
                return

            tmp_unzip = s_dir / "tmp_unzip_06"
            if tmp_unzip.exists(): shutil.rmtree(tmp_unzip, ignore_errors=True)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf: zf.extractall(tmp_unzip)
            except Exception as e:
                res.success = False
                msg = f"S{seed}-08 unzip failed: {e}"
                res.errors.append(msg)
                res.artifacts["evidence_checks"].append({
                    "stage": f"S{seed}-08", 
                    "stable_stage": "BuildTraj08",
                    "label": f"S{seed}-08",
                    "code": "UNZIP_FAILED", 
                    "msg": msg
                })
                if tmp_unzip.exists(): shutil.rmtree(tmp_unzip, ignore_errors=True)
                return
            
            trace_dir = tmp_unzip / "traces" if (tmp_unzip / "traces").exists() else tmp_unzip
            
            builder = CmdBuilder(ctx.get_script("08_build_sft_trajectories.py")).with_config(mr_cfg).with_run_name(f"mr_s{seed}").with_trace_dir(trace_dir).with_out_jsonl(l3_jsonl).with_evidence_dir(ev_08)
            eff_max = ctx.get_effective_max_samples()
            if eff_max is not None:
                 builder.arg("--max_samples", eff_max)
            else:
                 builder.arg("--allow_full_dataset")
            
            cmd = builder.build()
            if J1.record_if_violation(cmd, res, "08 cmd"): return
            cmd_res = CmdRunner.run(cmd, ctx.env_overrides, stream_output=True)
            if not _require_cmd_ok(res, cmd_res, f"S{seed}-08", "BuildTraj08"):
                # CMD_FAILED already recorded by helper
                return

            # Evidence Check
            eff_max = ctx.get_effective_max_samples()
            subset_size = _expected_count_from_ids(ctx.work_dir / "ids.txt", eff_max)
            
            ev08_zip, ev08_note, ev08_found = resolve_evidence_zip(ev_08)
            if ev08_zip is None:
                code = "MISSING_ZIP"
                msg = f"Missing zip: expected evidence_package.zip in {ev_08}; found {ev08_found}"
                dur = 0.0
            else:
                code, msg, dur = Evidence.verify_zip(ev08_zip, "08_build_sft_trajectories.py", expected_trace_count=subset_size)
                if ev08_note:
                    msg = f"{msg} ({ev08_note})"
            res.artifacts["evidence_checks"].append({
                "stage": f"S{seed}-08", 
                "stable_stage": "BuildTraj08",
                "label": f"S{seed}-08",
                "code": code, 
                "msg": msg
            })
            res.measurements["evidence_check_sec"] = res.measurements.get("evidence_check_sec", 0.0) + dur
            if code != "OK": res.success = False; res.errors.append(f"S{seed}-08 Evidence: {msg}")

            # Cleanup
            if tmp_unzip.exists(): shutil.rmtree(tmp_unzip, ignore_errors=True)

            l3_lines = 0
            if l3_jsonl.exists():
                with open(l3_jsonl, "r", encoding="utf-8") as f: l3_lines = sum(1 for _ in f)
            
            # Use effective max for line check
            eff_max = ctx.get_effective_max_samples()
            subset_size = _expected_count_from_ids(ctx.work_dir / "ids.txt", eff_max)
            
            # If subset_size is None (full run), we skip the exact line count check (or check against total IDs if we trusted them)
            # Policy: if full run, we can't strictly enforce count without knowing total available traces.
            # But usually full run = all IDs.
            # Let's check against IDs if subset_size is None, assuming full run means "all in ids.txt"
            # NOTE: With updated _expected_count_from_ids, subset_size should NOT be None even for full run.
            # It returns n_ids if eff_max is None. So this fallback block is likely redundant but kept for safety.
            if subset_size is None:
                 # Re-calculate from IDs for full run check
                 try:
                     lines = [x for x in (ctx.work_dir / "ids.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
                     subset_size = len(lines)
                 except:
                     subset_size = -1 # Skip check
            
            if subset_size > 0 and l3_lines != subset_size:
                 res.success = False; res.errors.append(f"J3 Fail: L3 lines {l3_lines} != Expected {subset_size}")

class SFTTrain09(Stage):
    def execute(self, ctx: StageContext, res: StageResult):
        if ctx.no_adapter or ctx.phase1_baseline: return
        for seed in ctx.seeds:
            if not res.success: break
            s_dir = ctx.work_dir / f"seed_{seed}"
            l3_jsonl = s_dir / "l3.jsonl"
            ev_09 = s_dir / "ev_09"
            l4_out = s_dir / "l4_out"
            
            cmd = CmdBuilder(ctx.get_script("09_train_lora_sft_toy.py")).arg("--train_jsonl", l3_jsonl).arg("--output_dir", l4_out).with_evidence_dir(ev_09).arg("--base_model", "distilgpt2").arg("--max_steps", "50").arg("--batch_size", "1").arg("--seed", seed).arg("--lr", "1e-3").arg("--grad_accum", "1").arg("--lora_alpha", "128").build()
            if J1.record_if_violation(cmd, res, "09 cmd"): return
            cmd_res = CmdRunner.run(cmd, ctx.env_overrides, stream_output=True)
            if not _require_cmd_ok(res, cmd_res, f"S{seed}-09", "SFTTrain09"):
                # CMD_FAILED already recorded by helper
                return
            
            ev09_zip, ev09_note, ev09_found = resolve_evidence_zip(ev_09)
            if ev09_zip is None:
                code = "MISSING_ZIP"
                msg = f"Missing zip: expected evidence_package.zip in {ev_09}; found {ev09_found}"
                dur = 0.0
            else:
                # Evidence Check (Training, no traces usually)
                code, msg, dur = Evidence.verify_zip(ev09_zip, "09_train_lora_sft_toy.py", expected_trace_count=None)
                if ev09_note:
                    msg = f"{msg} ({ev09_note})"
            res.artifacts["evidence_checks"].append({
                "stage": f"S{seed}-09", 
                "stable_stage": "SFTTrain09",
                "label": f"S{seed}-09",
                "code": code, 
                "msg": msg
            })
            res.measurements["evidence_check_sec"] = res.measurements.get("evidence_check_sec", 0.0) + dur
            if code != "OK": res.success = False; res.errors.append(f"S{seed}-09 Evidence: {msg}")

class SFTInfer(Stage):
    def execute(self, ctx: StageContext, res: StageResult):
        if ctx.no_adapter or not res.success or ctx.phase1_baseline: return
        # Using first seed's adapter for SFT inference
        first_seed = ctx.seeds[0] if ctx.seeds else 0
        seed0_adapter = ctx.work_dir / f"seed_{first_seed}/l4_out/adapter"
        ev_sft = ctx.work_dir / "ev_sft"
        mr_cfg = ctx.work_dir / "mr_config.yaml"
        
        cmd = CmdBuilder(ctx.get_script("06_run_agentiad_infer.py")).with_config(mr_cfg).with_run_name("mr_sft").arg("--seed", "0").arg("--id_list", ctx.work_dir / "ids.txt").with_evidence_dir(ev_sft).arg("--split", ctx.dataset_split).arg("--adapter_path", seed0_adapter)
        
        eff_max = ctx.get_effective_max_samples()
        if eff_max is not None:
             cmd.arg("--max_samples", eff_max)
        else:
             cmd.arg("--max_samples", "99999")
        
        cmd_list = cmd.build()
        if J1.record_if_violation(cmd_list, res, "SFT cmd"): return
        cmd_res = CmdRunner.run(cmd_list, ctx.env_overrides, stream_output=True)
        sft_audit_mismatch_recorded = False
        l2_data = parse_l2_json_from_cmd(cmd_res)
        if isinstance(l2_data, dict):
            l2_dataset_split = str(l2_data.get("dataset_split", "") or "")
            res.artifacts["sft_l2_dataset_split"] = l2_dataset_split
            if "effective_n" in l2_data:
                res.artifacts["sft_l2_effective_n"] = int(l2_data["effective_n"])
            if ctx.strict_j_mode and l2_dataset_split and l2_dataset_split != ctx.dataset_split:
                sft_audit_mismatch_recorded = True
                res.artifacts["evidence_checks"].append({
                    "stage": "SFTInfer",
                    "stable_stage": "SFTInfer",
                    "label": "SFTInfer",
                    "code": "DATASET_SPLIT_MISMATCH",
                    "msg": f"Strict J fail: dataset_split mismatch (expected={ctx.dataset_split}, actual={l2_dataset_split})"
                })
                res.success = False
                res.gates["J3"] = False
                res.errors.append(
                    f"SFTInfer dataset_split mismatch under strict_j: expected {ctx.dataset_split}, got {l2_dataset_split}"
                )
        if not _require_cmd_ok(res, cmd_res, "SFTInfer", "SFTInfer"):
            # CMD_FAILED already recorded by helper
            return

        # Evidence Check
        eff_max = ctx.get_effective_max_samples()
        subset_size = _expected_count_from_ids(ctx.work_dir / "ids.txt", eff_max)
        
        evsft_zip, evsft_note, evsft_found = resolve_evidence_zip(ev_sft)
        if evsft_zip is None:
            code = "MISSING_ZIP"
            msg = f"Missing zip: expected evidence_package.zip in {ev_sft}; found {evsft_found}"
            dur = 0.0
        else:
            code, msg, dur = Evidence.verify_zip(evsft_zip, "06_run_agentiad_infer.py", expected_trace_count=subset_size)
            if evsft_note:
                msg = f"{msg} ({evsft_note})"
        if not (sft_audit_mismatch_recorded and code == "OK"):
            res.artifacts["evidence_checks"].append({
                "stage": "SFTInfer", 
                "stable_stage": "SFTInfer",
                "label": "SFTInfer",
                "code": code, 
                "msg": msg
            })
        res.measurements["evidence_check_sec"] = res.measurements.get("evidence_check_sec", 0.0) + dur
        if code != "OK": res.success = False; res.errors.append(f"SFT Evidence: {msg}")

class GRPOBuild(Stage):
    def execute(self, ctx: StageContext, res: StageResult):
        if ctx.no_adapter or not res.success or ctx.phase1_baseline: return
        grpo_cfg = ctx.work_dir / "mr_grpo_config.yaml"

        import yaml

        first_seed = ctx.seeds[0] if ctx.seeds else 0
        l3_source = ctx.work_dir / f"seed_{first_seed}/l3.jsonl"
        rollouts_jsonl = ctx.work_dir / "rollouts.jsonl"
        ev_10_build = ctx.work_dir / "ev_10_build"

        l3_lines = 0
        if l3_source.exists():
            with open(l3_source, "r", encoding="utf-8") as f: l3_lines = sum(1 for _ in f)
        rollouts_target = l3_lines * 2 if l3_lines > 0 else 10

        sft_adapter = ctx.work_dir / f"seed_{first_seed}/l4_out/adapter"
        effective_model = str(ctx.vlm_model_id or "").strip()
        mr_cfg = ctx.work_dir / "mr_config.yaml"
        if (not effective_model) and mr_cfg.exists():
            try:
                loaded = yaml.safe_load(mr_cfg.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    effective_model = str(loaded.get("vlm_model_id") or loaded.get("model_id") or "").strip()
            except Exception:
                pass
        if not effective_model:
            res.success = False
            res.errors.append("GRPOBuild missing effective VLM model id (ctx.vlm_model_id / mr_config model_id).")
            return

        cfg_obj: Dict[str, Any] = {}
        base_cfg = PROJECT_ROOT / "dist" / "configs" / "grpo_toy.yaml"
        if base_cfg.exists():
            try:
                loaded = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cfg_obj = dict(loaded)
            except Exception:
                cfg_obj = {}
        cfg_obj["phase4_mode"] = "real_vlm"
        cfg_obj["rollout_mode"] = "real_vlm"
        cfg_obj["base_model_id"] = str(effective_model)
        cfg_obj["rollout_samples"] = int(rollouts_target)
        cfg_obj["rollouts_per_prompt"] = int(cfg_obj.get("rollouts_per_prompt") or 2)
        cfg_obj["local_files_only"] = bool(str(os.environ.get("TRANSFORMERS_OFFLINE", "0")).strip() == "1")
        if sft_adapter.exists():
            cfg_obj["adapter_init"] = str(sft_adapter)
        grpo_cfg.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

        cmd = CmdBuilder(ctx.get_script("10_build_grpo_rollouts_toy.py")).with_config(grpo_cfg).arg("--train_jsonl", l3_source).arg("--output_jsonl", rollouts_jsonl).with_evidence_dir(ev_10_build).arg("--seed", "42").arg("--max_samples", rollouts_target)
        sft_adapter = ctx.work_dir / f"seed_{first_seed}/l4_out/adapter"
        if sft_adapter.exists(): cmd.arg("--adapter_init", sft_adapter)

        cmd_list = cmd.build()
        if J1.record_if_violation(cmd_list, res, "GRPO build cmd"): return
        cmd_res = CmdRunner.run(cmd_list, ctx.env_overrides, stream_output=True)
        if not _require_cmd_ok(res, cmd_res, "GRPOBuild", "GRPOBuild"):
             # CMD_FAILED already recorded by helper
             return

        if not rollouts_jsonl.exists():
            res.success = False
            res.errors.append(f"GRPOBuild missing rollouts output: {rollouts_jsonl}")
            return
        try:
            with open(rollouts_jsonl, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                res.success = False
                res.errors.append("GRPOBuild produced empty rollouts.jsonl")
                return
            first = json.loads(lines[0])
            if str(first.get("schema_version") or "") != "grpo_rollout_v1":
                res.success = False
                res.errors.append("GRPOBuild first rollout schema_version != grpo_rollout_v1")
                return
            mode_vals = Counter()
            for ln in lines[: min(256, len(lines))]:
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                mode_vals[str(obj.get("rollout_mode") or "unset")] += 1
            if int(mode_vals.get("real_vlm", 0)) <= 0:
                res.success = False
                res.errors.append("GRPOBuild rollout_mode does not include real_vlm in inspected records.")
                return
            res.artifacts["rollouts_count"] = int(len(lines))
            res.artifacts["rollouts_mode_counts_head"] = {k: int(v) for k, v in sorted(mode_vals.items(), key=lambda kv: kv[0])}
        except Exception as e:
            res.success = False
            res.errors.append(f"GRPOBuild rollout inspection failed: {type(e).__name__}: {e}")
            return

        res.artifacts["rollouts_sha256"] = hashlib.sha256(rollouts_jsonl.read_bytes()).hexdigest()
        res.artifacts["grpo_phase4_mode"] = "real_vlm"

        # Evidence Check
        # rollouts produces traces but count might vary or be large
        ev10b_zip, ev10b_note, ev10b_found = resolve_evidence_zip(ev_10_build)
        if ev10b_zip is None:
            code = "MISSING_ZIP"
            msg = f"Missing zip: expected evidence_package.zip in {ev_10_build}; found {ev10b_found}"
            dur = 0.0
        else:
            code, msg, dur = Evidence.verify_zip(ev10b_zip, "10_build_grpo_rollouts_toy.py", expected_trace_count=None)
            if ev10b_note:
                msg = f"{msg} ({ev10b_note})"
        res.artifacts["evidence_checks"].append({
            "stage": "GRPOBuild", 
            "stable_stage": "GRPOBuild",
            "label": "GRPOBuild",
            "code": code, 
            "msg": msg
        })
        res.measurements["evidence_check_sec"] = res.measurements.get("evidence_check_sec", 0.0) + dur
        if code != "OK": res.success = False; res.errors.append(f"GRPOBuild Evidence: {msg}")

class GRPOTrain(Stage):
    def execute(self, ctx: StageContext, res: StageResult):
        if ctx.no_adapter or not res.success or ctx.phase1_baseline: return
        grpo_cfg = ctx.work_dir / "mr_grpo_config.yaml"
        rollouts_jsonl = ctx.work_dir / "rollouts.jsonl"
        l6_out = ctx.work_dir / "l6_out"
        ev_10_train = ctx.work_dir / "ev_10_train"
        
        first_seed = ctx.seeds[0] if ctx.seeds else 0
        sft_adapter = ctx.work_dir / f"seed_{first_seed}/l4_out/adapter"
        
        cmd = CmdBuilder(ctx.get_script("10_train_grpo_toy.py")).with_config(grpo_cfg).arg("--train_jsonl", rollouts_jsonl).arg("--output_dir", l6_out).with_evidence_dir(ev_10_train).arg("--seed", "42").arg("--max_steps", "10").arg("--lr", "1e-2")
        if sft_adapter.exists(): cmd.arg("--adapter_init", sft_adapter)
        
        cmd_list = cmd.build()
        if J1.record_if_violation(cmd_list, res, "GRPO train cmd"): return
        cmd_res = CmdRunner.run(cmd_list, ctx.env_overrides, stream_output=True)
        if not _require_cmd_ok(res, cmd_res, "GRPOTrain", "GRPOTrain"):
            # CMD_FAILED already recorded by helper
            return

        required_outputs = {
            "train_snapshot": l6_out / "train_snapshot.json",
            "grpo_metrics_csv": l6_out / "grpo_metrics.csv",
            "reward_curve_csv": l6_out / "reward_curve.csv",
            "policy_delta_csv": l6_out / "policy_delta.csv",
            "reward_audit_json": l6_out / "reward_audit.json",
            "reward_events_jsonl": l6_out / "reward_events.jsonl",
        }
        missing = [k for k, p in required_outputs.items() if (not p.exists())]
        if missing:
            res.success = False
            res.errors.append(f"GRPOTrain missing required artifacts: {','.join(missing)}")
            return
        for k, p in required_outputs.items():
            if p.exists() and p.is_file():
                res.artifacts[f"{k}_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
                res.artifacts[f"{k}_bytes"] = int(p.stat().st_size)

        snap_path = l6_out / "train_snapshot.json"
        if snap_path.exists():
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
            if "timestamp" in snap: del snap["timestamp"]
            res.artifacts["snapshot_hash"] = hashlib.sha256(json.dumps(snap, sort_keys=True).encode()).hexdigest()
            res.artifacts["reward_audit"] = snap.get("reward_audit_check", "FAIL")
            if "lora_param_abs_delta" in snap: res.artifacts["lora_delta"] = snap["lora_param_abs_delta"]
        reward_audit_path = l6_out / "reward_audit.json"
        if reward_audit_path.exists():
            try:
                ra = json.loads(reward_audit_path.read_text(encoding="utf-8"))
                res.artifacts["reward_audit_check_file"] = str(ra.get("reward_audit_check") or "")
                if not bool(ra.get("real_phase4", False)):
                    res.success = False
                    res.errors.append("GRPOTrain reward_audit.json indicates real_phase4=false.")
                    return
            except Exception:
                res.artifacts["reward_audit_check_file"] = "PARSE_ERROR"

        # Evidence Check
        ev10t_zip, ev10t_note, ev10t_found = resolve_evidence_zip(ev_10_train)
        if ev10t_zip is None:
            code = "MISSING_ZIP"
            msg = f"Missing zip: expected evidence_package.zip in {ev_10_train}; found {ev10t_found}"
            dur = 0.0
        else:
            code, msg, dur = Evidence.verify_zip(ev10t_zip, "10_train_grpo_toy.py", expected_trace_count=None)
            if ev10t_note:
                msg = f"{msg} ({ev10t_note})"
        res.artifacts["evidence_checks"].append({
            "stage": "GRPOTrain", 
            "stable_stage": "GRPOTrain",
            "label": "GRPOTrain",
            "code": code, 
            "msg": msg
        })
        res.measurements["evidence_check_sec"] = res.measurements.get("evidence_check_sec", 0.0) + dur
        if code != "OK": res.success = False; res.errors.append(f"GRPOTrain Evidence: {msg}")

class GRPOInfer(Stage):
    def execute(self, ctx: StageContext, res: StageResult):
        if ctx.no_adapter or not res.success or ctx.phase1_baseline: return
        l6_out = ctx.work_dir / "l6_out"
        ev_grpo = ctx.work_dir / "ev_grpo"
        mr_cfg = ctx.work_dir / "mr_config.yaml"
        
        grpo_adapter = l6_out / "adapter"
        if not grpo_adapter.exists():
             if (l6_out / "adapter_model.bin").exists() or (l6_out / "adapter_model.safetensors").exists():
                 grpo_adapter = l6_out
        
        if grpo_adapter.exists():
            cmd = CmdBuilder(ctx.get_script("06_run_agentiad_infer.py")).with_config(mr_cfg).with_run_name("mr_grpo_infer").arg("--seed", "0").arg("--id_list", ctx.work_dir / "ids.txt").with_evidence_dir(ev_grpo).arg("--split", ctx.dataset_split).arg("--adapter_path", grpo_adapter)
            
            eff_max = ctx.get_effective_max_samples()
            if eff_max is not None:
                 cmd.arg("--max_samples", eff_max)
            else:
                 cmd.arg("--max_samples", "99999")
            
            cmd_list = cmd.build()
            if J1.record_if_violation(cmd_list, res, "GRPO Infer cmd"): return
            cmd_res = CmdRunner.run(cmd_list, ctx.env_overrides, stream_output=True)
            grpo_audit_mismatch_recorded = False
            l2_data = parse_l2_json_from_cmd(cmd_res)
            if isinstance(l2_data, dict):
                l2_dataset_split = str(l2_data.get("dataset_split", "") or "")
                res.artifacts["grpo_l2_dataset_split"] = l2_dataset_split
                if "effective_n" in l2_data:
                    res.artifacts["grpo_l2_effective_n"] = int(l2_data["effective_n"])
                if ctx.strict_j_mode and l2_dataset_split and l2_dataset_split != ctx.dataset_split:
                    grpo_audit_mismatch_recorded = True
                    res.artifacts["evidence_checks"].append({
                        "stage": "GRPOInfer",
                        "stable_stage": "GRPOInfer",
                        "label": "GRPOInfer",
                        "code": "DATASET_SPLIT_MISMATCH",
                        "msg": f"Strict J fail: dataset_split mismatch (expected={ctx.dataset_split}, actual={l2_dataset_split})"
                    })
                    res.success = False
                    res.gates["J3"] = False
                    res.errors.append(
                        f"GRPOInfer dataset_split mismatch under strict_j: expected {ctx.dataset_split}, got {l2_dataset_split}"
                    )
            if not _require_cmd_ok(res, cmd_res, "GRPOInfer", "GRPOInfer"):
                # CMD_FAILED already recorded by helper
                return

            # Evidence Check
            eff_max = ctx.get_effective_max_samples()
            subset_size = _expected_count_from_ids(ctx.work_dir / "ids.txt", eff_max)
            
            evgrpo_zip, evgrpo_note, evgrpo_found = resolve_evidence_zip(ev_grpo)
            if evgrpo_zip is None:
                code = "MISSING_ZIP"
                msg = f"Missing zip: expected evidence_package.zip in {ev_grpo}; found {evgrpo_found}"
                dur = 0.0
            else:
                code, msg, dur = Evidence.verify_zip(evgrpo_zip, "06_run_agentiad_infer.py", expected_trace_count=subset_size)
                if evgrpo_note:
                    msg = f"{msg} ({evgrpo_note})"
            if not (grpo_audit_mismatch_recorded and code == "OK"):
                res.artifacts["evidence_checks"].append({
                    "stage": "GRPOInfer", 
                    "stable_stage": "GRPOInfer",
                    "label": "GRPOInfer",
                    "code": code, 
                    "msg": msg
                })
            res.measurements["evidence_check_sec"] = res.measurements.get("evidence_check_sec", 0.0) + dur
            if code != "OK": res.success = False; res.errors.append(f"GRPOInfer Evidence: {msg}")

class Phase1Metrics(Stage):
    @staticmethod
    def _normalize_yesno(x: Any) -> str:
        if x is None:
            return ""
        s = str(x).strip().lower()
        if s in {"yes", "y", "true", "1", "anomaly", "abnormal"}:
            return "yes"
        if s in {"no", "n", "false", "0", "normal"}:
            return "no"
        return ""

    @classmethod
    def _resolve_correct_series(cls, df):
        import pandas as pd

        if "correct" in df.columns:
            return pd.to_numeric(df["correct"], errors="coerce").fillna(0.0)
        if "gt_label" in df.columns and "pred_label" in df.columns:
            gt = df["gt_label"].map(cls._normalize_yesno)
            pred = df["pred_label"].map(cls._normalize_yesno)
            return (gt == pred).astype(float)
        return None

    def execute(self, ctx: StageContext, res: StageResult):
        if not ctx.phase1_baseline or not res.success: return
        
        import pandas as pd
        import numpy as np
        
        all_dfs = []
        per_seed_metrics = []
        agg_dir = ctx.work_dir / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)
        
        for seed in ctx.seeds:
            s_dir = ctx.work_dir / f"seed_{seed}"
            # Ensure per-seed output directory exists
            s_dir.mkdir(parents=True, exist_ok=True)

            # SSOT: Read from evidence zip
            zip_path, _, found = resolve_evidence_zip(s_dir / "ev_06")
            if zip_path is None or not zip_path.exists():
                res.errors.append(
                    f"Evidence zip missing for seed {seed}: expected evidence_package.zip in {s_dir / 'ev_06'}; found {found}"
                )
                continue
            
            csv_name = f"tables/agentiad_infer_mr_s{seed}.csv"
            
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    if csv_name not in zf.namelist():
                         res.errors.append(f"Missing {csv_name} inside evidence_package.zip for seed {seed}")
                         continue
                    
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f)

                correct_series = self._resolve_correct_series(df)
                if correct_series is None:
                    res.errors.append(
                        f"Phase1Metrics seed {seed} missing accuracy fields: need 'correct' or ('gt_label' and 'pred_label')"
                    )
                    continue
                if "correct" not in df.columns:
                    df["correct"] = correct_series
                seed_acc = float(correct_series.mean()) if len(correct_series) > 0 else 0.0
                seed_n = int(len(correct_series))
                
                df["seed"] = seed
                all_dfs.append(df)
                
                # Copy to stable layout
                target_csv = s_dir / "baseline_metrics.csv"
                target_csv.write_text(df.to_csv(index=False), encoding="utf-8")
                per_seed_json = s_dir / "baseline_metrics.json"
                seed_payload = {
                    "seed": int(seed),
                    "accuracy": seed_acc,
                    "n_valid": seed_n,
                    "source_csv": target_csv.name,
                }
                per_seed_json.write_text(
                    json.dumps(seed_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                per_seed_metrics.append(seed_payload)
                
                # Per-class
                if "class_name" in df.columns and "correct" in df.columns:
                     per_class = df.groupby("class_name").agg(
                         acc=("correct", "mean"),
                         count=("correct", "count")
                     ).reset_index()
                     per_class.to_csv(s_dir / "baseline_per_class.csv", index=False)
                else:
                     # Minimal fallback
                     (s_dir / "baseline_per_class.csv").write_text("class_name,acc,count\n", encoding="utf-8")

            except Exception as e:
                res.errors.append(f"Metrics Aggregation Seed {seed} failed: {e}")

        if not all_dfs:
            res.errors.append("No metrics CSVs found for aggregation")
            return

        # Aggregation
        seed_accs = []
        for df in all_dfs:
             correct_series = self._resolve_correct_series(df)
             if correct_series is None:
                 res.errors.append("Phase1Metrics aggregation missing accuracy fields in one seed dataframe")
                 continue
             acc = correct_series.mean()
             seed_accs.append(acc)

        mean_acc = 0.0
        std_acc = 0.0
        if seed_accs:
            mean_acc = float(np.mean(seed_accs))
            std_acc = float(np.std(seed_accs))
            
        agg_df = pd.DataFrame([{
            "metric": "accuracy",
            "mean": mean_acc,
            "std": std_acc,
            "seeds": len(seed_accs)
        }])
        agg_csv = agg_dir / "metrics_mean_std.csv"
        agg_df.to_csv(agg_csv, index=False)
        n_total = int(sum(int(m.get("n_valid", 0)) for m in per_seed_metrics))
        summary_json = agg_dir / "baseline_summary.json"
        summary_payload = {
            "mode": "phase1_baseline",
            "baseline_scope": "inference_only_no_sft_no_grpo_training",
            "seeds": [int(s) for s in ctx.seeds],
            "per_seed": sorted(per_seed_metrics, key=lambda x: x["seed"]),
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "n_total": n_total,
            "audit_note": "Toy workload may produce identical table_hashes.sft and table_hashes.grpo when both inference paths resolve to equivalent predictions on sampled inputs; Phase1 baseline does not include training stages.",
        }
        summary_json.write_text(
            json.dumps(summary_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        
        res.artifacts["phase1_metrics"] = {
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "n_total": n_total,
            "summary_path": str(summary_json),
            "per_seed_metrics_path": [str(ctx.work_dir / f"seed_{s}" / "baseline_metrics.json") for s in ctx.seeds],
            "audit_note": "Phase1 baseline excludes SFT/GRPO training stages."
        }
        
        # Evidence Zip for Aggregate
        self._package_aggregate(agg_dir, [agg_csv])

    def _package_aggregate(self, agg_dir: Path, files: List[Path]):
        try:
            zip_path = agg_dir / "evidence_package.zip"
            index_path = agg_dir / "INDEX.txt"
            idx_lines = []
            
            for fp in files:
                if not fp.exists(): continue
                sha = hashlib.sha256(fp.read_bytes()).hexdigest().upper()
                size = fp.stat().st_size
                idx_lines.append(f"{fp.name} {size} {sha}")
            
            index_path.write_text("\n".join(idx_lines) + "\n", encoding="utf-8")
            
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in files:
                    if fp.exists(): zf.write(fp, arcname=fp.name)
                zf.write(index_path, arcname="INDEX.txt")
                
            if zip_path.exists():
                sha_zip = hashlib.sha256(zip_path.read_bytes()).hexdigest().upper()
                with open(index_path, "a", encoding="utf-8") as f:
                    f.write(f"file=evidence_package.zip sha256={sha_zip} (content_hash)\n")
        except Exception as e:
            print(f"Aggregate packaging failed: {e}", file=sys.stderr)

class GateEvaluator:
    @staticmethod
    def evaluate(ctx: StageContext, res: StageResult):
        def _set_gate(name: str, value: bool):
            # Fail-sticky: once a gate is set False by any stage, evaluator must not override it to True.
            res.gates[name] = bool(res.gates.get(name, True) and value)

        # Table Hashes (Task A Requirement: distinguish baseline/sft/grpo)
        first_seed = ctx.seeds[0] if ctx.seeds else 0
        
        # Baseline/Agent: Usually same in this repro, derived from 06 run of first seed
        agent_csv, _, _ = resolve_evidence_zip(ctx.work_dir / f"seed_{first_seed}/ev_06")
        agent_hash = CSVHash.compute(agent_csv, f"tables/agentiad_infer_mr_s{first_seed}.csv") if agent_csv else None
        
        # SFT: Derived from SFT Infer run
        sft_csv, _, _ = resolve_evidence_zip(ctx.work_dir / "ev_sft")
        sft_hash = CSVHash.compute(sft_csv, "tables/agentiad_infer_mr_sft.csv") if sft_csv else None
        
        # GRPO: Derived from GRPO Infer run
        grpo_csv, _, _ = resolve_evidence_zip(ctx.work_dir / "ev_grpo")
        grpo_hash = CSVHash.compute(grpo_csv, "tables/agentiad_infer_mr_grpo_infer.csv") if grpo_csv else None
        
        res.artifacts["table_hashes"] = {
            "agent": agent_hash,
            "sft": sft_hash,
            "grpo": grpo_hash
        }
        res.artifacts["gates_na"]["baseline"] = "removed_in_repro_no_true_baseline"
        res.artifacts["table_hashes_note"] = "baseline and agent derived from same 06 run in this repro"
        
        # Tool Rates (J6) - Iterate all seeds
        tool_rates = []
        for s in ctx.seeds:
            ev_pkg, _, _ = resolve_evidence_zip(ctx.work_dir / f"seed_{s}/ev_06")
            if ev_pkg is not None and ev_pkg.exists():
                tool_rates.append(Evidence.get_toolcall_rate(ev_pkg))
        
        if tool_rates:
            import statistics
            res.measurements["toolcall_rate_min"] = min(tool_rates)
            res.measurements["toolcall_rate_max"] = max(tool_rates)
            res.measurements["toolcall_rate_avg"] = statistics.mean(tool_rates)
            # J6 Metric
            res.artifacts["has_toolcall_rate_excl_synth"] = res.measurements["toolcall_rate_avg"]
            res.artifacts["toolcall_rate_def"] = "fraction_of_traces_with_tool_call_name"
            
            # Gate J6: tool usage > 0 (soft check here, strict check in scorer)
            # Policy: If avg is 0, we do not fail J6 in workload mode, but mark it NA or Note.
            avg_rate = res.measurements["toolcall_rate_avg"]
            if ctx.phase1_baseline:
                 # STRICT: Must be 0
                 if avg_rate == 0:
                      _set_gate("J6", True)
                 else:
                      _set_gate("J6", False)
                      res.errors.append(f"J6 Fail: Phase1 Baseline must have 0 tool usage, got {avg_rate}")
            elif ctx.phase2_full_infer:
                if avg_rate > 0:
                    _set_gate("J6", True)
                else:
                    _set_gate("J6", False)
                    res.errors.append(f"J6 Fail: phase2_full_infer requires toolcall_rate_avg > 0, got {avg_rate}")
            else:
                if avg_rate > 0:
                    _set_gate("J6", True)
                else:
                    _set_gate("J6", True)
                    res.artifacts["gates_na"]["J6"] = "workload_mode_not_enforced_use_strict_j"
        else:
            # No tool rates available (e.g. no seeds or failed)
            if ctx.phase2_full_infer:
                _set_gate("J6", False)
                res.errors.append("J6 Fail: phase2_full_infer has no tool rate data")
            else:
                _set_gate("J6", True)
                res.artifacts["gates_na"]["J6"] = "no_data_available"

        # J2: Evidence Integrity / Auditability
        # Only integrity-breaking codes should fail J2. Audit notes (e.g. EFFECTIVE_N_MISMATCH)
        # must not force J2 failure.
        fail_j2_codes = {
            "MISSING_ZIP",
            "MISSING_INDEX",
            "MISSING_SCRIPT",
            "EXCEPTION",
            "CMD_FAILED",
            "UNZIP_FAILED",
            "MISSING_FINGERPRINT",
            "RMTREE_FAILED",
            "REF_DIR_MISSING",
            "REF_ZIP_MISSING",
            "SENTINEL_UNZIP_FAILED",
            "REF_L3_MISSING",
        }
        evidence_ok = True
        if res.artifacts["evidence_checks"]:
            evidence_ok = not any(
                check.get("code") in fail_j2_codes
                for check in res.artifacts["evidence_checks"]
            )
        else:
            # If no checks ran but success is True (maybe skipped?), J2 might be N/A or False.
            # If success is True and we ran stages, we should have checks.
            # If Probe failed early, success is False.
            if res.success and ctx.seeds:
                # At least one seed run -> should have checks.
                evidence_ok = False

        _set_gate("J2", evidence_ok)

        # J3: Coverage (Checked via TRACE_COUNT_MISMATCH code in verify_zip)
        coverage_ok = True
        no_split_mismatch = True
        if res.artifacts["evidence_checks"]:
             coverage_ok = all(check["code"] != "TRACE_COUNT_MISMATCH" for check in res.artifacts["evidence_checks"])
             if ctx.phase1_baseline:
                 coverage_ok = coverage_ok and all(
                     check.get("code") != "EFFECTIVE_N_MISMATCH" for check in res.artifacts["evidence_checks"]
                 )
             if ctx.strict_j_mode:
                 no_split_mismatch = all(
                     check.get("code") != "DATASET_SPLIT_MISMATCH"
                     for check in res.artifacts["evidence_checks"]
                 )
        res.gates["J3"] = res.gates.get("J3", True) and coverage_ok and no_split_mismatch

        # J1: Flags (Checked per stage)
        # J1 Gate Pass = NO VIOLATION.
        # artifacts["allow_flags_used"] is for audit, not for gate failure if whitelisted.
        _set_gate("J1", not res.artifacts.get("allow_flags_violation", False))
        
        # J9: Determinism (N/A for single run mode, unless Sentinel)
        # Explicitly marking N/A for workload mode -> True with artifacts note
        _set_gate("J9", True)
        res.artifacts["gates_na"]["J9"] = "N/A in normal workload mode"
        
        return res

class SentinelPipeline(Stage):
    def __init__(self, ref_dir_str: str):
        self.ref_dir_str = ref_dir_str

    def execute(self, ctx: StageContext, res: StageResult):
        print(f"[Sentinel] Running in Sentinel Determinism Mode (ref={self.ref_dir_str})", file=sys.stderr)
        ref_dir = Path(self.ref_dir_str).resolve()
        if not ref_dir.exists():
            res.success = False
            msg = "Sentinel ref dir missing"
            res.errors.append(msg)
            res.artifacts["evidence_checks"].append({
                "stage": "SentinelPrep", 
                "stable_stage": "SentinelPrep",
                "label": "SentinelPrep",
                "code": "REF_DIR_MISSING", 
                "msg": msg
            })
            return
            
        try:
            if ctx.work_dir.exists(): shutil.rmtree(ctx.work_dir, ignore_errors=True)
            ctx.work_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            res.success = False
            msg = f"Sentinel cleanup failed: {e}"
            res.errors.append(msg)
            res.artifacts["evidence_checks"].append({
                "stage": "SentinelPrep", 
                "stable_stage": "SentinelPrep",
                "label": "SentinelPrep",
                "code": "RMTREE_FAILED", 
                "msg": msg
            })
            return
        
        first_seed = ctx.seeds[0] if ctx.seeds else 0
        ref_ev_06_zip = ref_dir / f"seed_{first_seed}/ev_06/evidence_package.zip"
        if not ref_ev_06_zip.exists():
            res.success = False
            msg = "Ref Run1 missing 06 zip"
            res.errors.append(msg)
            res.artifacts["evidence_checks"].append({
                "stage": "SentinelPrep", 
                "stable_stage": "SentinelPrep",
                "label": "SentinelPrep",
                "code": "REF_ZIP_MISSING", 
                "msg": msg
            })
            return
            
        unzip_dir = ctx.work_dir / "unzip_sentinel"
        try:
            try:
                with zipfile.ZipFile(ref_ev_06_zip, "r") as zf: zf.extractall(unzip_dir)
            except Exception as e:
                res.success = False
                msg = f"Sentinel unzip failed: {e}"
                res.errors.append(msg)
                res.artifacts["evidence_checks"].append({
                    "stage": "SentinelBuildTraj08", 
                    "stable_stage": "SentinelBuildTraj08",
                    "label": "SentinelBuildTraj08",
                    "code": "SENTINEL_UNZIP_FAILED", 
                    "msg": msg
                })
                return
            
            trace_dir = unzip_dir / "traces" if (unzip_dir / "traces").exists() else unzip_dir
            
            mr_cfg = ctx.work_dir / "mr_config.yaml"
            mr_cfg.write_text("model_id: distilgpt2\nrun_name: minireal\n", encoding="utf-8")
            
            ev_08_sentinel = ctx.work_dir / "ev_08_sentinel"
            l3_sentinel = ctx.work_dir / "l3_sentinel.jsonl"
            
            builder = CmdBuilder(ctx.get_script("08_build_sft_trajectories.py")).with_config(mr_cfg).with_run_name(f"mr_s{first_seed}").with_trace_dir(trace_dir).with_out_jsonl(l3_sentinel).with_evidence_dir(ev_08_sentinel)
            
            eff_max = ctx.get_effective_max_samples()
            if eff_max is not None:
                 builder.arg("--max_samples", eff_max)
            else:
                 builder.arg("--allow_full_dataset")
            
            cmd = builder.build()
            cmd_res = CmdRunner.run(cmd, ctx.env_overrides, stream_output=True)
            if not _require_cmd_ok(res, cmd_res, "SentinelBuildTraj08", "SentinelBuildTraj08"):
                 # CMD_FAILED already recorded by helper
                 return
                 
            # Sentinel Evidence Check
            code, msg, dur = Evidence.verify_zip(ev_08_sentinel / "evidence_package.zip", "08_build_sft_trajectories.py", expected_trace_count=None)
            res.artifacts["evidence_checks"].append({
                "stage": "SentinelBuildTraj08", 
                "stable_stage": "SentinelBuildTraj08",
                "label": "SentinelBuildTraj08",
                "code": code, 
                "msg": msg
            })
            if code != "OK":
                 res.success = False
                 res.errors.append(f"Sentinel evidence check failed: {msg}")
                 res.gates["J9"] = False
                 return
                 
            ref_l3 = ref_dir / f"seed_{first_seed}/l3.jsonl"
            if not ref_l3.exists():
                res.success = False
                msg = "Ref Run1 missing l3.jsonl"
                res.errors.append(msg)
                res.artifacts["evidence_checks"].append({
                    "stage": "SentinelPrep", 
                    "stable_stage": "SentinelPrep",
                    "label": "SentinelPrep",
                    "code": "REF_L3_MISSING", 
                    "msg": msg
                })
                return
            
            hash_ref = self.get_stable_l3_hash(ref_l3)
            hash_new = self.get_stable_l3_hash(l3_sentinel)
            
            if hash_ref != hash_new:
                res.success = False; res.errors.append(f"Determinism Failed: l3.jsonl mismatch (Ref={hash_ref[:8]} vs New={hash_new[:8]})")
                res.gates["J9"] = False
                return
            
            res.gates["J9"] = True # Passed Determinism Check
            print(f"[Sentinel] 08 Determinism OK (Hash={hash_new[:8]})", file=sys.stderr)

            # Check Adapter Hash
            ref_adapter = ref_dir / f"seed_{first_seed}/l4_out/adapter/adapter_model.bin"
            if not ref_adapter.exists(): ref_adapter = ref_dir / f"seed_{first_seed}/l4_out/adapter/adapter_model.safetensors"
            if ref_adapter.exists():
                 res.artifacts["adapter_hash"] = hashlib.sha256(ref_adapter.read_bytes()).hexdigest()
            
            # Check Table Hash
            gen_hash, err = CSVHash.generate_from_traces(trace_dir)
            res.artifacts["table_hashes"] = {"agent": gen_hash}
            agent_hash = CSVHash.compute(ref_ev_06_zip, f"tables/agentiad_infer_mr_s{first_seed}.csv")
            
            if gen_hash and agent_hash and gen_hash != agent_hash:
                 res.success = False; res.errors.append(f"Table Determinism Failed: Ref={agent_hash[:8]} vs Gen={gen_hash[:8]}")
                 res.gates["J9"] = False
        finally:
            if unzip_dir.exists(): shutil.rmtree(unzip_dir, ignore_errors=True)

    @staticmethod
    def get_stable_l3_hash(path):
        lines = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    def clean(o):
                        if isinstance(o, dict):
                            for k in ["paths", "env_hash", "trajectory_fingerprint_hash", "timestamp", "start_timestamp", "end_timestamp"]:
                                if k in o: del o[k]
                            if "images" in o and isinstance(o["images"], list):
                                for img in o["images"]:
                                    if isinstance(img, dict) and "path" in img: del img["path"]
                            for k, v in o.items(): clean(v)
                        elif isinstance(o, list):
                            for v in o: clean(v)
                    clean(obj)
                    lines.append(json.dumps(obj, sort_keys=True))
        except Exception as e: return f"ERROR: {e}"
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest().upper()


def get_file_sha256(p):
    """
    Compute SHA256 of a file.
    
    Self-hash policy:
    The EXPECTED_SHA_VERIFY_ALL line is stripped to allow deterministic self-verification.
    This does NOT affect hashing of any other file.
    """
    if p.name == "verify_all.py":
        with open(p, "rb") as f:
            lines = f.readlines()
        content = b"".join(l for l in lines if not l.strip().startswith(b"EXPECTED_SHA_VERIFY_ALL ="))
        return hashlib.sha256(content).hexdigest().upper()
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()

# Legacy run_cmd wrapper for compatibility
def run_cmd(cmd, env_overrides=None, cwd=None, timeout=None, stream_output=False):
    return CmdRunner.run(cmd, env_overrides, cwd, timeout, stream_output)

def verify_evidence_zip_optimized(zip_path, script_name, expected_trace_count=None):
    return Evidence.verify_zip(zip_path, script_name, expected_trace_count)

def _parse_seeds_value(raw_seeds: Any) -> List[int]:
    parts: List[str] = []
    if raw_seeds is None:
        return [42]
    if isinstance(raw_seeds, str):
        parts = [raw_seeds]
    elif isinstance(raw_seeds, (list, tuple)):
        parts = [str(x) for x in raw_seeds]
    else:
        parts = [str(raw_seeds)]

    tokens: List[str] = []
    for part in parts:
        for seg in str(part).split(","):
            t = seg.strip()
            if t:
                tokens.append(t)
    if not tokens:
        raise ValueError("empty seeds; use --seeds 0 1 2 or --seeds 0,1,2")

    parsed: List[int] = []
    bad: List[str] = []
    for tok in tokens:
        if re.fullmatch(r"[+-]?\d+", tok):
            parsed.append(int(tok))
        else:
            bad.append(tok)
    if bad:
        raise ValueError(
            "invalid --seeds value(s): "
            + ", ".join(bad)
            + ". Expected integers; examples: --seeds 0 1 2 or --seeds 0,1,2"
        )
    return parsed


def _hash_file_upper(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest().upper()
    except Exception:
        return ""


def _write_jsonl_stable(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(dict(it), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl_dicts(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                out.append(obj)
    except Exception:
        return []
    return out


def _count_trace_files_in_zip(zip_path: Path) -> int:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return int(sum(1 for n in zf.namelist() if n.endswith("trace.json")))
    except Exception:
        return 0


def _discover_phase2_seed_zips(input_root: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for d in sorted(input_root.glob("seed_*"), key=lambda p: p.name):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"seed_(\d+)", d.name)
        if not m:
            continue
        seed = int(m.group(1))
        ev06 = d / "ev_06"
        zip_path, _, _ = resolve_evidence_zip(ev06)
        if zip_path is not None and zip_path.exists():
            out.append((seed, zip_path.resolve()))
    out.sort(key=lambda x: int(x[0]))
    return out


def _read_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _read_sample_ids_from_csv(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "error": "",
        "rows_total": 0,
        "sample_ids_total": 0,
        "sample_ids_unique": 0,
        "duplicate_count": 0,
        "missing_sample_id_count": 0,
        "ids_set": set(),
        "id_field": "",
    }
    if not path.exists():
        out["error"] = "missing_csv"
        return out
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = [str(x) for x in (reader.fieldnames or [])]
            id_field = "sample_id" if "sample_id" in fields else ("id" if "id" in fields else "")
            if not id_field:
                out["error"] = "missing_sample_id_field"
                return out
            out["id_field"] = id_field
            seen: Set[str] = set()
            dup = 0
            missing = 0
            total = 0
            for row in reader:
                if not isinstance(row, dict):
                    continue
                total += 1
                sid = str(row.get(id_field) or "").strip()
                if not sid:
                    missing += 1
                    continue
                if sid in seen:
                    dup += 1
                else:
                    seen.add(sid)
            out["rows_total"] = int(total)
            out["sample_ids_total"] = int(total - missing)
            out["sample_ids_unique"] = int(len(seen))
            out["duplicate_count"] = int(dup)
            out["missing_sample_id_count"] = int(missing)
            out["ids_set"] = seen
            out["ok"] = True
            return out
    except Exception as e:
        out["error"] = f"csv_read_error:{type(e).__name__}"
        return out


def _read_sample_ids_from_main_jsonl(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "error": "",
        "lines_total": 0,
        "sample_ids_total": 0,
        "sample_ids_unique": 0,
        "duplicate_count": 0,
        "missing_sample_id_count": 0,
        "parse_error_count": 0,
        "ids_set": set(),
    }
    if not path.exists():
        out["error"] = "missing_main_jsonl"
        return out
    try:
        seen: Set[str] = set()
        dup = 0
        missing = 0
        parse_err = 0
        total = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
            except Exception:
                parse_err += 1
                continue
            if not isinstance(obj, dict):
                parse_err += 1
                continue
            sid = str(obj.get("sample_id") or "").strip()
            if not sid:
                missing += 1
                continue
            if sid in seen:
                dup += 1
            else:
                seen.add(sid)
        out["lines_total"] = int(total)
        out["sample_ids_total"] = int(total - missing - parse_err)
        out["sample_ids_unique"] = int(len(seen))
        out["duplicate_count"] = int(dup)
        out["missing_sample_id_count"] = int(missing)
        out["parse_error_count"] = int(parse_err)
        out["ids_set"] = seen
        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = f"main_jsonl_read_error:{type(e).__name__}"
        return out


def _resolve_phase2_raw_shard_roots(input_root: Path) -> Optional[Tuple[Path, Path, Path]]:
    cands: List[Path] = [
        (input_root / "outputs"),
        input_root,
        (input_root.parent / "outputs"),
        (PROJECT_ROOT / "outputs"),
    ]
    uniq: List[Path] = []
    seen: Set[str] = set()
    for p in cands:
        try:
            rp = p.resolve()
        except Exception:
            continue
        k = str(rp)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(rp)

    for out_root in uniq:
        traces_root = out_root / "traces"
        logs_root = out_root / "logs"
        tables_root = out_root / "tables"
        if not traces_root.is_dir() or not logs_root.is_dir() or not tables_root.is_dir():
            continue
        has_shard = False
        for child in sorted(traces_root.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            if not re.fullmatch(r"mb4_s\d+_g\d+", child.name):
                continue
            if (child / "main.jsonl").exists():
                has_shard = True
                break
        if has_shard:
            return traces_root.resolve(), logs_root.resolve(), tables_root.resolve()
    return None


def _discover_phase2_raw_seed_shards(input_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "traces_root": "",
        "logs_root": "",
        "tables_root": "",
        "seeds": {},
    }
    roots = _resolve_phase2_raw_shard_roots(input_root)
    if roots is None:
        return out
    traces_root, logs_root, tables_root = roots
    seeds: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for run_dir in sorted(traces_root.iterdir(), key=lambda p: p.name):
        if not run_dir.is_dir():
            continue
        m = re.fullmatch(r"mb4_s(\d+)_g(\d+)", run_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        gpu = int(m.group(2))
        run_name = run_dir.name
        seeds.setdefault(seed, {})[gpu] = {
            "run_name": run_name,
            "seed": int(seed),
            "gpu": int(gpu),
            "trace_root": run_dir.resolve(),
            "main_jsonl": (run_dir / "main.jsonl").resolve(),
            "summary_json": (logs_root / f"agentiad_infer_summary_{run_name}.json").resolve(),
            "table_csv": (tables_root / f"agentiad_infer_{run_name}.csv").resolve(),
        }
    out["traces_root"] = str(traces_root)
    out["logs_root"] = str(logs_root)
    out["tables_root"] = str(tables_root)
    out["seeds"] = seeds
    return out


def _pick_trace_root_from_unzip(tmp_unzip: Path, seed: int) -> Optional[Path]:
    cands: List[Path] = []

    def _has_direct_sample_dirs(root: Path) -> bool:
        try:
            for child in sorted(root.iterdir(), key=lambda p: p.name):
                if child.is_dir() and (child / "trace.json").exists():
                    return True
        except Exception:
            return False
        return False

    traces_dir = tmp_unzip / "traces"
    if traces_dir.exists() and traces_dir.is_dir():
        if _has_direct_sample_dirs(traces_dir):
            cands.append(traces_dir)
        for child in sorted(traces_dir.iterdir(), key=lambda p: p.name):
            if child.is_dir() and _has_direct_sample_dirs(child):
                cands.append(child)

    if _has_direct_sample_dirs(tmp_unzip):
        cands.append(tmp_unzip)

    if not cands:
        for tr in sorted(tmp_unzip.rglob("trace.json"), key=lambda p: str(p)):
            sample_dir = tr.parent
            parent = sample_dir.parent
            if _has_direct_sample_dirs(parent):
                cands.append(parent)

    uniq: List[Path] = []
    seen: Set[str] = set()
    for p in cands:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(p.resolve())
    if not uniq:
        return None

    seed_tag = f"s{int(seed)}"
    uniq.sort(key=lambda p: (0 if seed_tag in p.name else 1, str(p)))
    return uniq[0]


def _parse_l3_build_metrics(cmd_res: Optional[CmdResult]) -> Dict[str, int]:
    out = {
        "N_total_candidates": 0,
        "N_written": 0,
        "skipped_trace": 0,
        "skipped_final": 0,
    }
    if not cmd_res:
        return out
    merged = cmd_res.stderr if cmd_res.stderr else cmd_res.stdout
    if not merged:
        return out
    text = merged.decode("utf-8", errors="replace")
    for key in list(out.keys()):
        m = re.search(rf"\b{re.escape(key)}=(\d+)\b", text)
        if m:
            out[key] = int(m.group(1))
    return out


def _load_paper_contract_cls_for_phase31():
    try:
        from agentiad_repro.paper_contract import PaperContract  # type: ignore
        return PaperContract
    except Exception:
        return None


def _validate_phase31_item(
    item: Dict[str, Any],
    *,
    paper_contract: Any,
    strict_contract: bool,
) -> Tuple[bool, str, bool]:
    if not isinstance(item, dict):
        return False, "item_not_dict", False

    if str(item.get("schema_version") or "") != "sft_trajectory_v1":
        return False, "schema_version_invalid", False

    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        return False, "messages_missing", False

    final_obj = item.get("final")
    if not isinstance(final_obj, dict):
        return False, "final_not_dict", False

    schema_ok = False
    if paper_contract is not None:
        try:
            schema_ok, _ = paper_contract.validate_schema(final_obj)
        except Exception:
            schema_ok = False
    else:
        schema_ok = all(k in final_obj for k in ("anomaly_present", "top_anomaly", "visual_descriptions"))
    if not schema_ok:
        return False, "final_schema_invalid", False

    last_msg = messages[-1] if messages else None
    if not isinstance(last_msg, dict):
        return False, "last_message_invalid", True
    if str(last_msg.get("role") or "") != "assistant" or str(last_msg.get("name") or "") != "final":
        return False, "last_message_not_final_assistant", True
    last_content = last_msg.get("content")
    if not isinstance(last_content, str):
        return False, "final_message_content_not_str", True

    if paper_contract is not None:
        try:
            answer_xml = paper_contract.extract_answer_xml(last_content)
            if answer_xml is None:
                return False, "final_message_missing_answer_xml", True
            ans_obj = json.loads(answer_xml)
            if not isinstance(ans_obj, dict):
                return False, "final_message_answer_not_dict", True
            ans_ok, _ = paper_contract.validate_schema(ans_obj)
            if not ans_ok:
                return False, "final_message_schema_invalid", True
        except Exception:
            return False, "final_message_parse_error", True

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role") or "") != "assistant":
            continue
        tc = msg.get("tool_call")
        if tc is None:
            continue
        if not isinstance(tc, dict):
            return False, "tool_call_not_dict", True
        tc_name = str(tc.get("name") or "").strip()
        if not tc_name:
            return False, "tool_call_name_missing", True
        if paper_contract is not None:
            ok_tc, _ = paper_contract.validate_tool_call(tc)
            if not ok_tc:
                return False, "tool_call_schema_invalid", True
        if strict_contract:
            if idx + 1 >= len(messages):
                return False, "tool_call_without_tool_result", True
            nxt = messages[idx + 1]
            if not isinstance(nxt, dict) or str(nxt.get("role") or "") != "tool":
                return False, "tool_call_not_followed_by_tool_role", True

    return True, "", True


def _stable_split_bucket(sample_id: str, traj_hash: str) -> str:
    key = f"{sample_id}|{traj_hash}"
    hv = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return "val" if (hv % 10 == 0) else "train"


def _tool_names_from_messages(messages: Sequence[Any]) -> List[str]:
    names: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if str(m.get("role") or "") != "assistant":
            continue
        tc = m.get("tool_call")
        if not isinstance(tc, dict):
            continue
        nm = str(tc.get("name") or "").strip()
        if nm:
            names.append(nm)
    return names


def _classify_pzcr(tool_names: Sequence[str]) -> str:
    s = set(str(x) for x in tool_names)
    has_pz = "crop_image_normalized" in s
    has_cr = "query_image" in s
    if has_pz and has_cr:
        return "PZ+CR"
    if has_pz and not has_cr:
        return "PZ-only"
    if (not has_pz) and has_cr:
        return "CR-only"
    return "NoTool"


def _audit_exported_sample(item: Dict[str, Any], paper_contract: Any) -> Dict[str, Any]:
    sample_id = str(item.get("sample_id") or "")
    msgs_any = item.get("messages")
    messages = msgs_any if isinstance(msgs_any, list) else []
    tool_names = _tool_names_from_messages(messages)
    pzcr_type = _classify_pzcr(tool_names)

    has_system_prompt = any(isinstance(m, dict) and str(m.get("role") or "") == "system" for m in messages)
    has_user_prompt = any(isinstance(m, dict) and str(m.get("role") or "") == "user" for m in messages)
    has_assistant_tool_call = any(
        isinstance(m, dict) and str(m.get("role") or "") == "assistant" and isinstance(m.get("tool_call"), dict)
        for m in messages
    )
    has_tool_result = any(isinstance(m, dict) and str(m.get("role") or "") == "tool" for m in messages)

    has_assistant_reasoning = False
    has_final_answer = False
    final_contract_ok = False
    final_has_explicit_cot = False
    last_tool_call_name = ""

    for m in messages:
        if not isinstance(m, dict):
            continue
        if str(m.get("role") or "") == "assistant":
            tc = m.get("tool_call")
            if isinstance(tc, dict):
                nm = str(tc.get("name") or "").strip()
                if nm:
                    last_tool_call_name = nm
            content = m.get("content")
            if isinstance(content, str):
                s = content.strip()
                if "<think>" in s or ("<answer>" in s and "</answer>" in s and "<think>" in s):
                    final_has_explicit_cot = True
                if str(m.get("name") or "") != "final":
                    if "<think>" in s or not (s.startswith("<answer>") and s.endswith("</answer>")):
                        has_assistant_reasoning = True

    if messages:
        last = messages[-1]
        if isinstance(last, dict) and str(last.get("role") or "") == "assistant" and str(last.get("name") or "") == "final":
            content = last.get("content")
            if isinstance(content, str):
                has_final_answer = True
                if paper_contract is not None:
                    try:
                        answer_xml = paper_contract.extract_answer_xml(content)
                        if answer_xml is not None:
                            ans_obj = json.loads(answer_xml)
                            if isinstance(ans_obj, dict):
                                ok, _ = paper_contract.validate_schema(ans_obj)
                                final_contract_ok = bool(ok)
                    except Exception:
                        final_contract_ok = False

    # CoT supervision is optional unless it is explicitly trace-derived.
    # Primary readiness target for current export is final <answer> + last tool call.
    loss_mask_ready = bool(has_final_answer and bool(last_tool_call_name))
    loss_mask_reasons: List[str] = []
    if not has_final_answer:
        loss_mask_reasons.append("missing_final_answer_message")
    if not last_tool_call_name:
        loss_mask_reasons.append("missing_tool_call_for_last_tool_supervision")

    return {
        "sample_id": sample_id,
        "tool_names": tool_names,
        "trajectory_type": pzcr_type,
        "has_system_prompt": has_system_prompt,
        "has_user_prompt": has_user_prompt,
        "has_assistant_reasoning": has_assistant_reasoning,
        "has_assistant_tool_call": has_assistant_tool_call,
        "has_tool_result": has_tool_result,
        "has_final_answer": has_final_answer,
        "final_answer_contract_ok": final_contract_ok,
        "final_has_explicit_cot": final_has_explicit_cot,
        "last_tool_call_name": last_tool_call_name,
        "loss_mask_ready_answer_plus_last_tool_call": loss_mask_ready,
        "loss_mask_ready_final_cot_plus_last_tool_call": bool(loss_mask_ready and final_has_explicit_cot),
        "loss_mask_not_ready_reasons": loss_mask_reasons,
    }


def _run_build_sft_traj_phase31(args: argparse.Namespace, work_dir: Path, env_overrides: Dict[str, str]) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {
        "allow_flags_used": False,
        "allow_flags_violation": False,
        "evidence_checks": [],
        "gates_na": {},
    }
    errors: List[str] = []
    remediations: List[str] = []
    measurements: Dict[str, Any] = {"evidence_check_sec": 0.0}
    gates: Dict[str, bool] = {"P3_1": True}

    input_dir_raw = str(getattr(args, "input_dir", "") or "").strip()
    if not input_dir_raw:
        return {
            "success": False,
            "gates": {"P3_1": False},
            "artifacts": artifacts,
            "errors": ["build_sft_traj requires --input-dir pointing to a Phase2 output root."],
            "measurements": measurements,
            "remediations": ["Use --input-dir <phase2_output_root> and --output-dir <phase3_output_root>."],
        }

    input_root = Path(input_dir_raw).resolve()
    if not input_root.exists() or not input_root.is_dir():
        return {
            "success": False,
            "gates": {"P3_1": False},
            "artifacts": artifacts,
            "errors": [f"input_dir not found or not a directory: {input_root}"],
            "measurements": measurements,
            "remediations": [
                "Pass a valid Phase2 root containing seed_*/ev_06/evidence_package.zip",
                "or raw shard layout outputs/traces|logs|tables.",
            ],
        }

    if input_root == work_dir.resolve():
        return {
            "success": False,
            "gates": {"P3_1": False},
            "artifacts": artifacts,
            "errors": ["output_dir must differ from input_dir to avoid destructive cleanup."],
            "measurements": measurements,
            "remediations": ["Use distinct directories for --input-dir and --output-dir."],
        }

    if args.max_samples is not None and int(args.max_samples) <= 0:
        return {
            "success": False,
            "gates": {"P3_1": False},
            "artifacts": artifacts,
            "errors": [f"invalid --max-samples={args.max_samples}; must be > 0"],
            "measurements": measurements,
            "remediations": ["Use --max-samples N where N>=1, or omit it."],
        }

    try:
        _cleanup_work_dir_preserve_logs(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {
            "success": False,
            "gates": {"P3_1": False},
            "artifacts": artifacts,
            "errors": [f"Workload cleanup failed for output_dir: {e}"],
            "measurements": measurements,
            "remediations": ["Choose a writable output_dir under dist/outputs or outputs."],
        }

    discovered_zip = _discover_phase2_seed_zips(input_root)
    discovered_raw = _discover_phase2_raw_seed_shards(input_root)
    raw_seeds_map: Dict[int, Dict[int, Dict[str, Any]]] = dict(discovered_raw.get("seeds") or {})
    phase2_input_layout = ""
    if raw_seeds_map:
        phase2_input_layout = "raw_shards_mb4_s{seed}_g{gpu}"
    elif discovered_zip:
        phase2_input_layout = "seed_ev06_zip"
    else:
        return {
            "success": False,
            "gates": {"P3_1": False},
            "artifacts": artifacts,
            "errors": [f"No Phase2 inputs discovered under {input_root}."],
            "measurements": measurements,
            "remediations": [
                "Expected either seed_<N>/ev_06/evidence_package.zip",
                "or outputs/traces/mb4_s{seed}_g{gpu}/main.jsonl + matching outputs/logs and outputs/tables files.",
            ],
        }

    discovered_seed_ids: List[int] = []
    if phase2_input_layout == "seed_ev06_zip":
        discovered_seed_ids = sorted(int(s) for s, _ in discovered_zip)
    else:
        discovered_seed_ids = sorted(int(s) for s in raw_seeds_map.keys())

    seeds_provided = bool(getattr(args, "seeds_provided", False))
    seed_filter: Set[int] = set(int(x) for x in (args.seeds or [])) if seeds_provided else set()
    if seeds_provided:
        selected_seed_ids = [int(s) for s in discovered_seed_ids if int(s) in seed_filter]
    else:
        selected_seed_ids = list(discovered_seed_ids)
    selected_seed_ids.sort()
    artifacts["phase2_input_dir"] = str(input_root)
    artifacts["phase2_input_layout"] = phase2_input_layout
    artifacts["phase2_seeds_discovered"] = [int(s) for s in discovered_seed_ids]
    artifacts["phase2_seeds_selected"] = [int(s) for s in selected_seed_ids]
    artifacts["phase2_seed_filter_applied"] = bool(seeds_provided)
    if phase2_input_layout == "raw_shards_mb4_s{seed}_g{gpu}":
        artifacts["phase2_raw_shard_roots"] = {
            "traces_root": str(discovered_raw.get("traces_root") or ""),
            "logs_root": str(discovered_raw.get("logs_root") or ""),
            "tables_root": str(discovered_raw.get("tables_root") or ""),
        }
    if not selected_seed_ids:
        return {
            "success": False,
            "gates": {"P3_1": False},
            "artifacts": artifacts,
            "errors": [f"No selected seeds found in input_dir for filter={sorted(seed_filter)}."],
            "measurements": measurements,
            "remediations": [f"Available seeds: {discovered_seed_ids}"],
        }

    cfg_path = input_root / "mr_config.yaml"
    cfg_source = "phase2_input_root"
    if not cfg_path.exists():
        cfg_path = PROJECT_ROOT / "configs" / "agent_pzcr.yaml"
        cfg_source = "repo_default_config"
    if not cfg_path.exists():
        return {
            "success": False,
            "gates": {"P3_1": False},
            "artifacts": artifacts,
            "errors": [f"Unable to resolve config file for L3 builder under {input_root} or {PROJECT_ROOT / 'configs' / 'agent_pzcr.yaml'}"],
            "measurements": measurements,
            "remediations": ["Ensure mr_config.yaml exists under the Phase2 output root."],
        }
    artifacts["phase3_config_path"] = str(cfg_path.resolve())
    artifacts["phase3_config_source"] = cfg_source

    paper_contract = _load_paper_contract_cls_for_phase31()
    strict_contract = bool(getattr(args, "strict_contract", False))
    if paper_contract is None:
        errors.append("PaperContract import unavailable; fallback validation used.")

    source_input_hashes: Dict[str, str] = {}
    per_seed_rows: List[Dict[str, Any]] = []
    reject_reasons: Counter = Counter()
    validation_examples: Dict[str, List[str]] = {}
    all_valid_items: List[Dict[str, Any]] = []
    schema_pass_count = 0
    source_samples_discovered = 0
    source_samples_selected_cap = 0
    budget_left: Optional[int] = int(args.max_samples) if args.max_samples is not None else None
    zip_by_seed: Dict[int, Path] = {int(s): p for s, p in discovered_zip}
    tmp_root: Optional[Path] = None
    if phase2_input_layout == "seed_ev06_zip":
        tmp_root = work_dir / "_tmp_phase3_unzip"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)

    if phase2_input_layout == "seed_ev06_zip":
        for seed in selected_seed_ids:
            ev06_zip = zip_by_seed.get(int(seed))
            if ev06_zip is None:
                continue
            trace_count = _count_trace_files_in_zip(ev06_zip)
            source_samples_discovered += int(trace_count)
            source_input_hashes[f"seed_{int(seed)}_ev06_zip"] = _hash_file_upper(ev06_zip)

            if budget_left is not None and budget_left <= 0:
                reject_reasons["max_samples_cap"] += int(trace_count)
                per_seed_rows.append({
                    "seed": int(seed),
                    "input_mode": "seed_ev06_zip",
                    "source_trace_count": int(trace_count),
                    "selected_for_conversion": 0,
                    "l3_written": 0,
                    "validated_ok": 0,
                    "validated_rejected": 0,
                    "note": "max_samples_cap",
                })
                continue

            selected_for_seed = int(trace_count) if budget_left is None else min(int(trace_count), int(budget_left))
            source_samples_selected_cap += selected_for_seed

            code, msg, dur = Evidence.verify_zip(ev06_zip, "06_run_agentiad_infer.py", expected_trace_count=None)
            measurements["evidence_check_sec"] = float(measurements.get("evidence_check_sec", 0.0)) + float(dur)
            artifacts["evidence_checks"].append({
                "stage": f"S{seed}-06-src",
                "stable_stage": "Phase31SourceZip",
                "label": f"S{seed}-06-src",
                "code": code,
                "msg": msg,
            })
            if code != "OK":
                reject_reasons["source_zip_verify_failed"] += selected_for_seed
                if strict_contract:
                    errors.append(f"S{seed} source zip verify failed: {msg}")
                per_seed_rows.append({
                    "seed": int(seed),
                    "input_mode": "seed_ev06_zip",
                    "source_trace_count": int(trace_count),
                    "selected_for_conversion": selected_for_seed,
                    "l3_written": 0,
                    "validated_ok": 0,
                    "validated_rejected": selected_for_seed,
                    "note": f"source_zip_verify_failed:{code}",
                })
                if budget_left is not None:
                    budget_left = max(0, int(budget_left) - selected_for_seed)
                continue

            if tmp_root is None:
                reject_reasons["source_unzip_failed"] += selected_for_seed
                errors.append(f"S{seed} unzip workspace missing.")
                continue

            tmp_seed = tmp_root / f"seed_{int(seed)}"
            if tmp_seed.exists():
                shutil.rmtree(tmp_seed, ignore_errors=True)
            tmp_seed.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(ev06_zip, "r") as zf:
                    zf.extractall(tmp_seed)
            except Exception as e:
                reject_reasons["source_unzip_failed"] += selected_for_seed
                if strict_contract:
                    errors.append(f"S{seed} unzip failed: {e}")
                per_seed_rows.append({
                    "seed": int(seed),
                    "input_mode": "seed_ev06_zip",
                    "source_trace_count": int(trace_count),
                    "selected_for_conversion": selected_for_seed,
                    "l3_written": 0,
                    "validated_ok": 0,
                    "validated_rejected": selected_for_seed,
                    "note": "source_unzip_failed",
                })
                if budget_left is not None:
                    budget_left = max(0, int(budget_left) - selected_for_seed)
                continue

            trace_root = _pick_trace_root_from_unzip(tmp_seed, int(seed))
            if trace_root is None:
                reject_reasons["trace_root_not_found"] += selected_for_seed
                if strict_contract:
                    errors.append(f"S{seed} trace root not found after unzip.")
                per_seed_rows.append({
                    "seed": int(seed),
                    "input_mode": "seed_ev06_zip",
                    "source_trace_count": int(trace_count),
                    "selected_for_conversion": selected_for_seed,
                    "l3_written": 0,
                    "validated_ok": 0,
                    "validated_rejected": selected_for_seed,
                    "note": "trace_root_not_found",
                })
                if budget_left is not None:
                    budget_left = max(0, int(budget_left) - selected_for_seed)
                continue

            seed_out = work_dir / f"seed_{int(seed)}"
            seed_out.mkdir(parents=True, exist_ok=True)
            l3_raw_jsonl = seed_out / "l3_raw.jsonl"
            ev08 = seed_out / "ev_08"
            run_name = trace_root.name
            trace_dir_for_l3 = trace_root.parent if trace_root.parent != trace_root else trace_root
            builder = (
                CmdBuilder(L3_SCRIPT)
                .with_config(cfg_path)
                .with_run_name(run_name)
                .with_trace_dir(trace_dir_for_l3)
                .with_out_jsonl(l3_raw_jsonl)
                .with_evidence_dir(ev08)
                .flag("--allow_skip", True)
            )
            if budget_left is not None:
                builder.arg("--max_samples", int(budget_left))
            cmd = builder.build()
            cmd_res = CmdRunner.run(cmd, env_overrides, stream_output=True)
            if not _require_cmd_ok(
                StageResult(success=True, artifacts={"evidence_checks": []}),
                cmd_res,
                f"S{seed}-08",
                "BuildTraj08",
                record_cmd_failed=False,
            ):
                reject_reasons["l3_builder_failed"] += selected_for_seed
                errors.append(f"S{seed}-08 build command failed")
                per_seed_rows.append({
                    "seed": int(seed),
                    "input_mode": "seed_ev06_zip",
                    "source_trace_count": int(trace_count),
                    "selected_for_conversion": selected_for_seed,
                    "l3_written": 0,
                    "validated_ok": 0,
                    "validated_rejected": selected_for_seed,
                    "note": "l3_builder_failed",
                })
                if budget_left is not None:
                    budget_left = max(0, int(budget_left) - selected_for_seed)
                continue

            l3_metrics = _parse_l3_build_metrics(cmd_res)
            reject_reasons["l3_skipped_trace"] += int(l3_metrics.get("skipped_trace", 0))
            reject_reasons["l3_skipped_final"] += int(l3_metrics.get("skipped_final", 0))

            ev08_zip, ev08_note, ev08_found = resolve_evidence_zip(ev08)
            if ev08_zip is None:
                artifacts["evidence_checks"].append({
                    "stage": f"S{seed}-08",
                    "stable_stage": "BuildTraj08",
                    "label": f"S{seed}-08",
                    "code": "MISSING_ZIP",
                    "msg": f"Missing zip: expected evidence_package.zip in {ev08}; found {ev08_found}",
                })
                if strict_contract:
                    errors.append(f"S{seed}-08 evidence zip missing.")
            else:
                code08, msg08, dur08 = Evidence.verify_zip(ev08_zip, "08_build_sft_trajectories.py", expected_trace_count=None)
                measurements["evidence_check_sec"] = float(measurements.get("evidence_check_sec", 0.0)) + float(dur08)
                if ev08_note:
                    msg08 = f"{msg08} ({ev08_note})"
                artifacts["evidence_checks"].append({
                    "stage": f"S{seed}-08",
                    "stable_stage": "BuildTraj08",
                    "label": f"S{seed}-08",
                    "code": code08,
                    "msg": msg08,
                })
                if strict_contract and code08 != "OK":
                    errors.append(f"S{seed}-08 evidence verify failed: {msg08}")

            raw_items = _read_jsonl_dicts(l3_raw_jsonl)
            if budget_left is not None and len(raw_items) > int(budget_left):
                extra = len(raw_items) - int(budget_left)
                reject_reasons["max_samples_cap"] += int(extra)
                raw_items = raw_items[: int(budget_left)]
            if budget_left is not None:
                budget_left = max(0, int(budget_left) - len(raw_items))

            raw_items.sort(key=lambda it: str(it.get("sample_id") or ""))
            valid_this_seed = 0
            rejected_this_seed = 0
            for it in raw_items:
                ok, reason, schema_ok = _validate_phase31_item(
                    it,
                    paper_contract=paper_contract,
                    strict_contract=strict_contract,
                )
                if schema_ok:
                    schema_pass_count += 1
                if not ok:
                    rejected_this_seed += 1
                    reject_reasons[reason] += 1
                    sid = str(it.get("sample_id") or "")
                    arr = validation_examples.setdefault(reason, [])
                    if sid and len(arr) < 10:
                        arr.append(f"seed={int(seed)} sample_id={sid}")
                    continue
                row = dict(it)
                row["_phase2_seed"] = int(seed)
                all_valid_items.append(row)
                valid_this_seed += 1

            per_seed_rows.append({
                "seed": int(seed),
                "input_mode": "seed_ev06_zip",
                "source_trace_count": int(trace_count),
                "selected_for_conversion": selected_for_seed,
                "l3_written": int(len(raw_items)),
                "validated_ok": int(valid_this_seed),
                "validated_rejected": int(rejected_this_seed),
                "l3_skipped_trace": int(l3_metrics.get("skipped_trace", 0)),
                "l3_skipped_final": int(l3_metrics.get("skipped_final", 0)),
                "trace_root": str(trace_root),
            })
    else:
        traces_root_for_l3 = Path(str(discovered_raw.get("traces_root") or ""))
        required_gpus = [0, 1, 2, 3]
        for seed in selected_seed_ids:
            shard_map = dict(raw_seeds_map.get(int(seed)) or {})
            missing_gpus = [g for g in required_gpus if g not in shard_map]
            seed_fail_reasons: List[str] = []
            seed_fail_notes: List[str] = []
            seed_union_ids: Set[str] = set()
            seed_expected_total = 0
            requested_vals: List[int] = []
            shard_checks: List[Dict[str, Any]] = []

            def _add_seed_fail(reason: str, note: str) -> None:
                if reason not in seed_fail_reasons:
                    seed_fail_reasons.append(reason)
                if note and len(seed_fail_notes) < 20:
                    seed_fail_notes.append(note)

            if not traces_root_for_l3.exists():
                _add_seed_fail("raw_trace_root_missing", f"missing trace root: {traces_root_for_l3}")
            if missing_gpus:
                _add_seed_fail("raw_missing_gpu_shard", f"missing_gpus={missing_gpus}")

            for gpu in required_gpus:
                shard = shard_map.get(gpu)
                if not isinstance(shard, dict):
                    continue
                run_name = str(shard.get("run_name") or f"mb4_s{int(seed)}_g{int(gpu)}")
                main_jsonl = Path(str(shard.get("main_jsonl") or ""))
                summary_json = Path(str(shard.get("summary_json") or ""))
                table_csv = Path(str(shard.get("table_csv") or ""))

                source_input_hashes[f"seed_{int(seed)}_g{int(gpu)}_main_jsonl"] = _hash_file_upper(main_jsonl)
                source_input_hashes[f"seed_{int(seed)}_g{int(gpu)}_summary_json"] = _hash_file_upper(summary_json)
                source_input_hashes[f"seed_{int(seed)}_g{int(gpu)}_table_csv"] = _hash_file_upper(table_csv)

                missing_files: List[str] = []
                if not main_jsonl.exists():
                    missing_files.append("main.jsonl")
                if not summary_json.exists():
                    missing_files.append("summary.json")
                if not table_csv.exists():
                    missing_files.append("table.csv")
                if missing_files:
                    _add_seed_fail("raw_missing_shard_file", f"{run_name} missing={missing_files}")
                    shard_checks.append({
                        "gpu": int(gpu),
                        "run_name": run_name,
                        "status": "missing_files",
                        "missing": missing_files,
                    })
                    continue

                summary_obj = _read_json_dict(summary_json)
                if summary_obj is None:
                    _add_seed_fail("raw_summary_parse_error", f"{run_name} summary parse failed")
                    summary_obj = {}
                n_requested = int(summary_obj.get("n_requested_ids", 0) or 0)
                n_success = int(summary_obj.get("n_success", 0) or 0)
                if n_requested > 0:
                    requested_vals.append(int(n_requested))

                csv_stats = _read_sample_ids_from_csv(table_csv)
                main_stats = _read_sample_ids_from_main_jsonl(main_jsonl)
                if not bool(csv_stats.get("ok")):
                    _add_seed_fail("raw_csv_read_error", f"{run_name} csv_error={csv_stats.get('error')}")
                if not bool(main_stats.get("ok")):
                    _add_seed_fail("raw_main_jsonl_read_error", f"{run_name} main_error={main_stats.get('error')}")

                csv_dup = int(csv_stats.get("duplicate_count", 0) or 0)
                main_dup = int(main_stats.get("duplicate_count", 0) or 0)
                if csv_dup > 0 or main_dup > 0:
                    _add_seed_fail("raw_duplicate_within_shard", f"{run_name} csv_dup={csv_dup} main_dup={main_dup}")

                csv_unique = int(csv_stats.get("sample_ids_unique", 0) or 0)
                main_unique = int(main_stats.get("sample_ids_unique", 0) or 0)
                if csv_unique > 0 and main_unique > 0 and csv_unique != main_unique:
                    _add_seed_fail("raw_csv_main_count_mismatch", f"{run_name} csv_unique={csv_unique} main_unique={main_unique}")
                if n_success > 0 and csv_unique > 0 and n_success != csv_unique:
                    _add_seed_fail("raw_n_success_mismatch_csv", f"{run_name} n_success={n_success} csv_unique={csv_unique}")

                shard_ids = set(csv_stats.get("ids_set") or set())
                if not shard_ids:
                    shard_ids = set(main_stats.get("ids_set") or set())
                overlap = seed_union_ids.intersection(shard_ids)
                if overlap:
                    _add_seed_fail("raw_duplicate_across_shards", f"{run_name} overlap_count={len(overlap)}")
                seed_union_ids.update(shard_ids)

                shard_checks.append({
                    "gpu": int(gpu),
                    "run_name": run_name,
                    "status": "ok",
                    "n_requested_ids": int(n_requested),
                    "n_success": int(n_success),
                    "csv_unique_sample_ids": int(csv_unique),
                    "main_unique_sample_ids": int(main_unique),
                    "csv_duplicates": int(csv_dup),
                    "main_duplicates": int(main_dup),
                    "main_jsonl": str(main_jsonl),
                    "summary_json": str(summary_json),
                    "table_csv": str(table_csv),
                })

            req_set = sorted(set(int(x) for x in requested_vals)) if requested_vals else []
            if req_set and len(req_set) != 1:
                _add_seed_fail("raw_requested_ids_inconsistent", f"requested_ids={req_set}")

            # Seed-level expected total must be defined at seed scope, not per-shard.
            # Use deduped union across g0..g3 as the canonical expected_total.
            seed_source_count = int(len(seed_union_ids))
            seed_expected_total = int(seed_source_count)
            source_samples_discovered += int(seed_source_count)

            if budget_left is not None and budget_left <= 0:
                reject_reasons["max_samples_cap"] += int(seed_source_count)
                per_seed_rows.append({
                    "seed": int(seed),
                    "input_mode": "raw_shards_mb4_s{seed}_g{gpu}",
                    "source_trace_count": int(seed_source_count),
                    "selected_for_conversion": 0,
                    "l3_written": 0,
                    "validated_ok": 0,
                    "validated_rejected": 0,
                    "shards_checked": shard_checks,
                    "note": "max_samples_cap",
                })
                continue

            selected_for_seed = int(seed_source_count) if budget_left is None else min(int(seed_source_count), int(budget_left))
            source_samples_selected_cap += selected_for_seed

            if seed_fail_reasons:
                primary_reason = str(seed_fail_reasons[0])
                if selected_for_seed > 0:
                    reject_reasons[primary_reason] += int(selected_for_seed)
                else:
                    reject_reasons[primary_reason] += 1
                if strict_contract:
                    errors.append(f"S{seed} raw shard validation failed: {seed_fail_reasons}")
                per_seed_rows.append({
                    "seed": int(seed),
                    "input_mode": "raw_shards_mb4_s{seed}_g{gpu}",
                    "source_trace_count": int(seed_source_count),
                    "selected_for_conversion": int(selected_for_seed),
                    "l3_written": 0,
                    "validated_ok": 0,
                    "validated_rejected": int(selected_for_seed),
                    "missing_gpus": missing_gpus,
                    "expected_total": int(seed_expected_total),
                    "union_count": int(seed_source_count),
                    "seed_fail_reasons": seed_fail_reasons,
                    "seed_fail_notes": seed_fail_notes,
                    "shards_checked": shard_checks,
                    "note": primary_reason,
                })
                if budget_left is not None:
                    budget_left = max(0, int(budget_left) - selected_for_seed)
                continue

            seed_out = work_dir / f"seed_{int(seed)}"
            seed_out.mkdir(parents=True, exist_ok=True)
            seed_l3_written = 0
            seed_valid_ok = 0
            seed_valid_rejected = 0
            seed_skipped_trace = 0
            seed_skipped_final = 0
            l3_runs_used: List[str] = []

            for gpu in required_gpus:
                if budget_left is not None and budget_left <= 0:
                    break
                shard = shard_map.get(gpu)
                if not isinstance(shard, dict):
                    continue
                run_name = str(shard.get("run_name") or f"mb4_s{int(seed)}_g{int(gpu)}")
                l3_runs_used.append(run_name)

                l3_raw_jsonl = seed_out / f"l3_raw_g{int(gpu)}.jsonl"
                ev08 = seed_out / f"ev_08_g{int(gpu)}"
                builder = (
                    CmdBuilder(L3_SCRIPT)
                    .with_config(cfg_path)
                    .with_run_name(run_name)
                    .with_trace_dir(traces_root_for_l3)
                    .with_out_jsonl(l3_raw_jsonl)
                    .with_evidence_dir(ev08)
                    .flag("--allow_skip", True)
                )
                if budget_left is not None:
                    builder.arg("--max_samples", int(budget_left))
                cmd = builder.build()
                cmd_res = CmdRunner.run(cmd, env_overrides, stream_output=True)
                if not _require_cmd_ok(
                    StageResult(success=True, artifacts={"evidence_checks": []}),
                    cmd_res,
                    f"S{seed}-08-g{gpu}",
                    "BuildTraj08",
                    record_cmd_failed=False,
                ):
                    remain = max(0, int(selected_for_seed) - int(seed_l3_written))
                    reject_reasons["l3_builder_failed"] += int(remain if remain > 0 else 1)
                    errors.append(f"S{seed}-08-g{gpu} build command failed")
                    continue

                l3_metrics = _parse_l3_build_metrics(cmd_res)
                seed_skipped_trace += int(l3_metrics.get("skipped_trace", 0))
                seed_skipped_final += int(l3_metrics.get("skipped_final", 0))
                reject_reasons["l3_skipped_trace"] += int(l3_metrics.get("skipped_trace", 0))
                reject_reasons["l3_skipped_final"] += int(l3_metrics.get("skipped_final", 0))

                ev08_zip, ev08_note, ev08_found = resolve_evidence_zip(ev08)
                if ev08_zip is None:
                    artifacts["evidence_checks"].append({
                        "stage": f"S{seed}-08-g{gpu}",
                        "stable_stage": "BuildTraj08",
                        "label": f"S{seed}-08-g{gpu}",
                        "code": "MISSING_ZIP",
                        "msg": f"Missing zip: expected evidence_package.zip in {ev08}; found {ev08_found}",
                    })
                    if strict_contract:
                        errors.append(f"S{seed}-08-g{gpu} evidence zip missing.")
                else:
                    code08, msg08, dur08 = Evidence.verify_zip(ev08_zip, "08_build_sft_trajectories.py", expected_trace_count=None)
                    measurements["evidence_check_sec"] = float(measurements.get("evidence_check_sec", 0.0)) + float(dur08)
                    if ev08_note:
                        msg08 = f"{msg08} ({ev08_note})"
                    artifacts["evidence_checks"].append({
                        "stage": f"S{seed}-08-g{gpu}",
                        "stable_stage": "BuildTraj08",
                        "label": f"S{seed}-08-g{gpu}",
                        "code": code08,
                        "msg": msg08,
                    })
                    if strict_contract and code08 != "OK":
                        errors.append(f"S{seed}-08-g{gpu} evidence verify failed: {msg08}")

                raw_items = _read_jsonl_dicts(l3_raw_jsonl)
                if budget_left is not None and len(raw_items) > int(budget_left):
                    extra = len(raw_items) - int(budget_left)
                    reject_reasons["max_samples_cap"] += int(extra)
                    raw_items = raw_items[: int(budget_left)]
                if budget_left is not None:
                    budget_left = max(0, int(budget_left) - len(raw_items))

                raw_items.sort(key=lambda it: str(it.get("sample_id") or ""))
                for it in raw_items:
                    ok, reason, schema_ok = _validate_phase31_item(
                        it,
                        paper_contract=paper_contract,
                        strict_contract=strict_contract,
                    )
                    if schema_ok:
                        schema_pass_count += 1
                    if not ok:
                        seed_valid_rejected += 1
                        reject_reasons[reason] += 1
                        sid = str(it.get("sample_id") or "")
                        arr = validation_examples.setdefault(reason, [])
                        if sid and len(arr) < 10:
                            arr.append(f"seed={int(seed)} sample_id={sid}")
                        continue
                    row = dict(it)
                    row["_phase2_seed"] = int(seed)
                    all_valid_items.append(row)
                    seed_valid_ok += 1
                seed_l3_written += int(len(raw_items))

            per_seed_rows.append({
                "seed": int(seed),
                "input_mode": "raw_shards_mb4_s{seed}_g{gpu}",
                "source_trace_count": int(seed_source_count),
                "selected_for_conversion": int(selected_for_seed),
                "l3_written": int(seed_l3_written),
                "validated_ok": int(seed_valid_ok),
                "validated_rejected": int(seed_valid_rejected),
                "l3_skipped_trace": int(seed_skipped_trace),
                "l3_skipped_final": int(seed_skipped_final),
                "missing_gpus": missing_gpus,
                "expected_total": int(seed_expected_total),
                "expected_total_source": "seed_union_sample_ids",
                "union_count": int(seed_source_count),
                "per_shard_n_requested_ids_values": req_set,
                "l3_runs_used": l3_runs_used,
                "shards_checked": shard_checks,
            })

    if tmp_root is not None and tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    all_valid_items.sort(
        key=lambda it: (
            int(it.get("_phase2_seed", 0)),
            str(it.get("sample_id") or ""),
            str(it.get("trajectory_fingerprint_hash") or ""),
        )
    )

    dedup_items: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[int, str]] = set()
    for it in all_valid_items:
        seed = int(it.get("_phase2_seed", 0))
        sid = str(it.get("sample_id") or "")
        k = (seed, sid)
        if k in seen_keys:
            reject_reasons["duplicate_seed_sample"] += 1
            continue
        seen_keys.add(k)
        dedup_items.append(it)
    all_valid_items = dedup_items

    tool_counter: Counter = Counter()
    samples_with_tool = 0
    train_items: List[Dict[str, Any]] = []
    val_items: List[Dict[str, Any]] = []
    traj_hashes: List[str] = []
    for it in all_valid_items:
        msgs_any = it.get("messages")
        msgs = msgs_any if isinstance(msgs_any, list) else []
        cur_tools = 0
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if str(m.get("role") or "") != "assistant":
                continue
            tc = m.get("tool_call")
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name") or "").strip() or "unknown"
            tool_counter[name] += 1
            cur_tools += 1
        if cur_tools > 0:
            samples_with_tool += 1

        cleaned = dict(it)
        cleaned.pop("_phase2_seed", None)
        sample_id = str(cleaned.get("sample_id") or "")
        traj_hash = str(cleaned.get("trajectory_fingerprint_hash") or "")
        if traj_hash:
            traj_hashes.append(traj_hash)
        bucket = _stable_split_bucket(sample_id, traj_hash)
        if bucket == "val":
            val_items.append(cleaned)
        else:
            train_items.append(cleaned)

    all_items_clean = [*train_items, *val_items]
    all_jsonl = work_dir / "trajectories_sft.jsonl"
    train_jsonl = work_dir / "train.jsonl"
    val_jsonl = work_dir / "val.jsonl"
    _write_jsonl_stable(all_jsonl, all_items_clean)
    _write_jsonl_stable(train_jsonl, train_items)
    _write_jsonl_stable(val_jsonl, val_items)

    inspected_samples = list(all_items_clean[:3])
    content_audits = [_audit_exported_sample(it, paper_contract) for it in inspected_samples]
    presence_totals = {
        "has_system_prompt": int(sum(1 for x in content_audits if x.get("has_system_prompt"))),
        "has_user_prompt": int(sum(1 for x in content_audits if x.get("has_user_prompt"))),
        "has_assistant_reasoning": int(sum(1 for x in content_audits if x.get("has_assistant_reasoning"))),
        "has_assistant_tool_call": int(sum(1 for x in content_audits if x.get("has_assistant_tool_call"))),
        "has_tool_result": int(sum(1 for x in content_audits if x.get("has_tool_result"))),
        "has_final_answer": int(sum(1 for x in content_audits if x.get("has_final_answer"))),
        "final_answer_contract_ok": int(sum(1 for x in content_audits if x.get("final_answer_contract_ok"))),
    }
    pzcr_counter = Counter(str(x.get("trajectory_type") or "UNKNOWN") for x in content_audits)
    loss_mask_ready_count = int(sum(1 for x in content_audits if x.get("loss_mask_ready_answer_plus_last_tool_call")))
    loss_mask_ready_with_cot_count = int(sum(1 for x in content_audits if x.get("loss_mask_ready_final_cot_plus_last_tool_call")))
    loss_mask_reason_counter: Counter = Counter()
    for x in content_audits:
        for r in x.get("loss_mask_not_ready_reasons", []) or []:
            loss_mask_reason_counter[str(r)] += 1
    split_definition_audit = {
        "policy": "val if sha256(sample_id|trajectory_fingerprint_hash)%10==0 else train",
        "implementation_source": "verify_all.py::_stable_split_bucket",
        "matches_existing_project_or_frozen_split_definition": "UNPROVEN",
        "note": "No explicit frozen Phase3 split spec was found in repository files during this implementation.",
    }

    cfg_sha = _hash_file_upper(cfg_path)
    source_hash_agg = hashlib.sha256(
        json.dumps(source_input_hashes, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest().upper()
    traj_hash_agg = hashlib.sha256("\n".join(sorted(traj_hashes)).encode("utf-8")).hexdigest().upper()
    hash_summary = {
        "config_file_sha256": cfg_sha,
        "source_phase2_input_sha256": dict(sorted(source_input_hashes.items(), key=lambda kv: kv[0])),
        "source_phase2_input_sha256_aggregate": source_hash_agg,
        "trajectory_fingerprint_hash_aggregate": traj_hash_agg,
        "trajectories_sft_jsonl_sha256": _hash_file_upper(all_jsonl),
        "train_jsonl_sha256": _hash_file_upper(train_jsonl),
        "val_jsonl_sha256": _hash_file_upper(val_jsonl),
    }

    sampling_truncation_total = int(reject_reasons.get("max_samples_cap", 0) or 0)
    data_reject_reasons: Dict[str, int] = {
        str(k): int(v)
        for k, v in reject_reasons.items()
        if str(k) != "max_samples_cap"
    }
    data_rejected_total = int(sum(int(v) for v in data_reject_reasons.values()))
    not_converted = int(source_samples_selected_cap - len(all_valid_items))
    if data_rejected_total < not_converted:
        data_reject_reasons["unaccounted_not_converted"] = int(
            data_reject_reasons.get("unaccounted_not_converted", 0) + (not_converted - data_rejected_total)
        )
        data_rejected_total = int(sum(int(v) for v in data_reject_reasons.values()))
    rejected_total_including_sampling = int(data_rejected_total + sampling_truncation_total)

    validation_report = {
        "source_samples_discovered": int(source_samples_discovered),
        "source_samples_selected_for_conversion": int(source_samples_selected_cap),
        "converted_samples": int(len(all_valid_items)),
        "rejected_or_skipped_total": int(data_rejected_total),
        "rejected_or_skipped_reasons": {k: int(v) for k, v in sorted(data_reject_reasons.items(), key=lambda kv: kv[0])},
        "sampling_truncation_total": int(sampling_truncation_total),
        "sampling_truncation_reasons": {"max_samples_cap": int(sampling_truncation_total)} if sampling_truncation_total > 0 else {},
        "rejected_or_skipped_total_including_sampling_truncation": int(rejected_total_including_sampling),
        "schema_pass_count": int(schema_pass_count),
        "validation_examples": {k: v for k, v in sorted(validation_examples.items(), key=lambda kv: kv[0])},
        "strict_contract": bool(strict_contract),
        "content_audit_inspected_samples": content_audits,
        "split_definition_audit": split_definition_audit,
        "loss_mask_readiness_audit": {
            "target_policy": "answer_plus_last_tool_call; explicit_final_cot_optional_and_trace_derived_only",
            "ready_count": int(loss_mask_ready_count),
            "inspected_sample_count": int(len(content_audits)),
            "ready_with_explicit_final_cot_count": int(loss_mask_ready_with_cot_count),
            "not_ready_reason_counts": {k: int(v) for k, v in sorted(loss_mask_reason_counter.items(), key=lambda kv: kv[0])},
        },
    }
    validation_path = work_dir / "validation_results.json"
    validation_path.write_text(json.dumps(validation_report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    summary = {
        "phase": "3.1",
        "mode": "build_sft_traj",
        "status": "PASS",
        "strict_contract": bool(strict_contract),
        "input_dir": str(input_root),
        "phase2_input_layout": phase2_input_layout,
        "output_dir": str(work_dir),
        "seeds_selected": [int(s) for s in selected_seed_ids],
        "source_samples_discovered": int(source_samples_discovered),
        "source_samples_selected_for_conversion": int(source_samples_selected_cap),
        "converted_samples": int(len(all_valid_items)),
        "rejected_or_skipped_total": int(data_rejected_total),
        "rejected_or_skipped_reasons": {k: int(v) for k, v in sorted(data_reject_reasons.items(), key=lambda kv: kv[0])},
        "sampling_truncation_total": int(sampling_truncation_total),
        "sampling_truncation_reasons": {"max_samples_cap": int(sampling_truncation_total)} if sampling_truncation_total > 0 else {},
        "rejected_or_skipped_total_including_sampling_truncation": int(rejected_total_including_sampling),
        "schema_pass_count": int(schema_pass_count),
        "per_tool_usage": {
            "total_tool_calls": int(sum(tool_counter.values())),
            "samples_with_tool_calls": int(samples_with_tool),
            "per_tool": {k: int(v) for k, v in sorted(tool_counter.items(), key=lambda kv: kv[0])},
        },
        "split_counts": {
            "train": int(len(train_items)),
            "val": int(len(val_items)),
        },
        "content_audit": {
            "inspected_sample_count": int(len(content_audits)),
            "inspected_samples": content_audits,
            "presence_totals": presence_totals,
        },
        "pz_cr_audit": {
            "inspected_sample_count": int(len(content_audits)),
            "counts": {k: int(v) for k, v in sorted(pzcr_counter.items(), key=lambda kv: kv[0])},
        },
        "split_definition_audit": split_definition_audit,
        "loss_mask_readiness_audit": {
            "target_policy": "answer_plus_last_tool_call; explicit_final_cot_optional_and_trace_derived_only",
            "inspected_sample_count": int(len(content_audits)),
            "ready_count": int(loss_mask_ready_count),
            "ready_all_inspected": bool(loss_mask_ready_count == len(content_audits) if content_audits else False),
            "ready_with_explicit_final_cot_count": int(loss_mask_ready_with_cot_count),
            "not_ready_reason_counts": {k: int(v) for k, v in sorted(loss_mask_reason_counter.items(), key=lambda kv: kv[0])},
        },
        "hash_summary": hash_summary,
        "artifacts": {
            "trajectories_sft_jsonl": str(all_jsonl),
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "validation_results_json": str(validation_path),
        },
        "per_seed": per_seed_rows,
    }

    if len(all_valid_items) == 0:
        summary["status"] = "FAIL"
        errors.append("No valid SFT trajectories were produced.")
        remediations.append("Check Phase2 source artifacts and L3 conversion logs per seed under output_dir/seed_*/.")

    if strict_contract and data_rejected_total > 0:
        summary["status"] = "FAIL"
        errors.append(
            f"strict-contract enabled: data_rejected_or_skipped_total={data_rejected_total} must be 0."
        )
        remediations.append("Fix invalid/partial traces in Phase2 outputs or rerun Phase2 for affected seeds.")

    summary_path = work_dir / "phase3_1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    artifacts["phase3_summary_path"] = str(summary_path)
    artifacts["phase3_validation_path"] = str(validation_path)
    artifacts["phase3_train_jsonl"] = str(train_jsonl)
    artifacts["phase3_val_jsonl"] = str(val_jsonl)
    artifacts["phase3_all_jsonl"] = str(all_jsonl)
    artifacts["phase3_hash_summary"] = hash_summary
    artifacts["phase3_per_seed"] = per_seed_rows
    artifacts["phase3_content_audit_inspected_count"] = int(len(content_audits))
    artifacts["phase3_pzcr_counts_inspected"] = {k: int(v) for k, v in sorted(pzcr_counter.items(), key=lambda kv: kv[0])}
    measurements["source_samples_discovered"] = int(source_samples_discovered)
    measurements["source_samples_selected_for_conversion"] = int(source_samples_selected_cap)
    measurements["converted_samples"] = int(len(all_valid_items))
    measurements["rejected_or_skipped_total"] = int(data_rejected_total)
    measurements["sampling_truncation_total"] = int(sampling_truncation_total)
    measurements["schema_pass_count"] = int(schema_pass_count)

    if summary["status"] != "PASS":
        gates["P3_1"] = False
    gates["P3_1_SCHEMA"] = bool(schema_pass_count == len(all_valid_items))

    print(
        f"[Phase3.1] source={int(source_samples_discovered)} selected={int(source_samples_selected_cap)} "
        f"converted={int(len(all_valid_items))} rejected={int(data_rejected_total)} truncated={int(sampling_truncation_total)} "
        f"schema_pass={int(schema_pass_count)} train={int(len(train_items))} val={int(len(val_items))}",
        file=sys.stderr,
    )
    print(f"[Phase3.1] {'PASS' if summary['status'] == 'PASS' else 'FAIL'} summary={summary_path}", file=sys.stderr)

    return {
        "success": bool(summary["status"] == "PASS"),
        "gates": gates,
        "artifacts": artifacts,
        "errors": errors,
        "measurements": measurements,
        "remediations": remediations,
    }


def run_workload(args):
    # Phase 1 Baseline Logic
    is_build_sft = args.mode == "build_sft_traj"
    is_phase1 = args.mode == "phase1_baseline"
    is_phase1_acceptance_only = args.mode == "phase1_acceptance_only"
    is_phase2_full = args.mode == "phase2_full_infer"
    phase2_seeds_policy = ""
    phase2_vlm_max_side = int(args.vlm_max_side)
    phase2_sdp_backend = str(args.sdp_backend or "auto").strip().lower()
    phase2_vlm_retry_n = int(args.vlm_retry_n)
    phase2_vlm_max_side_source = "default"
    phase2_sdp_backend_source = "default"
    phase2_vlm_retry_n_source = "default"
    if is_phase1:
        args.allow_full_dataset = True
        args.seeds = [0, 1, 2]
        if args.output_dir is None: args.output_dir = "dist/outputs/phase1_baseline"
    if is_phase1_acceptance_only:
        args.seeds = [0, 1, 2]
        if args.output_dir is None: args.output_dir = "dist/outputs/phase1_baseline"
    if is_phase2_full:
        if args.max_samples is None:
            args.allow_full_dataset = True
        else:
            args.allow_full_dataset = False
        if args.seeds is None or args.seeds == [42]:
            args.seeds = [0, 1, 2]
            phase2_seeds_policy = "default_phase2_align_phase1"
        else:
            phase2_seeds_policy = "cli_override"
        if args.output_dir is None:
            args.output_dir = "dist/outputs/phase2_full_infer"
        if str(args.dataset_split).strip().lower() != "train":
            return {
                "success": False,
                "gates": {"J2": False, "J3": False},
                "artifacts": {
                    "allow_flags_used": False,
                    "allow_flags_violation": False,
                    "evidence_checks": [{
                        "stage": "DataBinding",
                        "stable_stage": "DataBinding",
                        "label": "DataBinding",
                        "code": "DATASET_SPLIT_INVALID",
                        "msg": "phase2_full_infer requires MMAD single split: train",
                    }],
                    "gates_na": {},
                    "audit_dataset_split": str(args.dataset_split),
                    "ids_source_note": "phase2_full_infer rejected non-train dataset_split",
                },
                "errors": [f"phase2_full_infer invalid --dataset-split={args.dataset_split}; MMAD supports single split 'train'"],
                "measurements": {"evidence_check_sec": 0.0},
                "remediations": ["Use --dataset-split train for MMAD single-split dataset."],
            }

    phase2_vlm_model_id: Optional[str] = None
    phase2_requested_vlm_model_id: Optional[str] = None
    phase2_requested_vlm_local_dir: Optional[str] = None
    phase2_vlm_model_source = "config"
    if is_phase2_full:
        argv = [str(x) for x in sys.argv]
        has_vlm_max_side_cli = "--vlm-max-side" in argv
        has_sdp_backend_cli = "--sdp-backend" in argv
        has_vlm_retry_n_cli = "--vlm-retry-n" in argv
        if has_vlm_max_side_cli:
            phase2_vlm_max_side_source = "cli"
        if has_sdp_backend_cli:
            phase2_sdp_backend_source = "cli"
        if has_vlm_retry_n_cli:
            phase2_vlm_retry_n_source = "cli"
        mr_cfg_existing = Path(args.output_dir).resolve() / "mr_config.yaml"
        if mr_cfg_existing.exists() and (not has_vlm_max_side_cli or not has_sdp_backend_cli or not has_vlm_retry_n_cli):
            try:
                import yaml

                loaded_cfg = yaml.safe_load(mr_cfg_existing.read_text(encoding="utf-8"))
                if isinstance(loaded_cfg, dict):
                    if (not has_vlm_max_side_cli) and ("vlm_max_side" in loaded_cfg):
                        phase2_vlm_max_side = int(loaded_cfg.get("vlm_max_side"))
                        phase2_vlm_max_side_source = "config"
                    if (not has_sdp_backend_cli) and ("sdp_backend" in loaded_cfg):
                        phase2_sdp_backend = str(loaded_cfg.get("sdp_backend") or "auto").strip().lower()
                        phase2_sdp_backend_source = "config"
                    if (not has_vlm_retry_n_cli):
                        if "vlm_retry_n" in loaded_cfg:
                            phase2_vlm_retry_n = int(loaded_cfg.get("vlm_retry_n"))
                            phase2_vlm_retry_n_source = "config"
                        elif "max_retries" in loaded_cfg:
                            phase2_vlm_retry_n = int(loaded_cfg.get("max_retries"))
                            phase2_vlm_retry_n_source = "config"
            except Exception:
                pass
        if phase2_sdp_backend not in {"auto", "math", "flash", "mem_efficient"}:
            phase2_sdp_backend = "auto"
        cli_local = str(args.vlm_model_local_dir or "").strip()
        cli_id = str(args.vlm_model_id or "").strip()
        env_local = str(os.environ.get("VLM_MODEL_LOCAL_DIR", "") or "").strip()
        env_id = str(os.environ.get("VLM_MODEL_ID", "") or "").strip()
        if cli_id:
            phase2_requested_vlm_model_id = cli_id
        elif env_id:
            phase2_requested_vlm_model_id = env_id
        if cli_local:
            phase2_requested_vlm_local_dir = cli_local
        elif env_local:
            phase2_requested_vlm_local_dir = env_local
        if cli_local:
            phase2_vlm_model_id = cli_local
            phase2_vlm_model_source = "cli"
        elif cli_id:
            phase2_vlm_model_id = cli_id
            phase2_vlm_model_source = "cli"
        elif env_local:
            phase2_vlm_model_id = env_local
            phase2_vlm_model_source = "env"
        elif env_id:
            phase2_vlm_model_id = env_id
            phase2_vlm_model_source = "env"

    gate_env = _effective_env()
    gate_requested_model_id = str(args.vlm_model_id or gate_env.get("VLM_MODEL_ID", "")).strip()
    gate_requested_local_dir = str(args.vlm_model_local_dir or gate_env.get("VLM_MODEL_LOCAL_DIR", "")).strip()
    gate_effective_model_id = str(phase2_vlm_model_id or gate_requested_model_id).strip()
    offline_gate = _check_offline_local_vlm_ready(
        mode_name=str(args.mode),
        strict_contract=bool(getattr(args, "strict_contract", False)),
        requested_model_id=gate_requested_model_id,
        requested_local_dir=gate_requested_local_dir,
        effective_model_id=gate_effective_model_id,
        env=gate_env,
    )
    if offline_gate.get("required") and not offline_gate.get("ok", False):
        err_msg = str(offline_gate.get("msg", "Offline mode requires a local VLM directory (strict-contract + offline)."))
        remediations = []
        for key in ("remediation_a", "remediation_b"):
            value = str(offline_gate.get(key, "")).strip()
            if value:
                remediations.append(value)
        return {
            "success": False,
            "gates": {"J2": False, "J3": False},
            "artifacts": {
                "allow_flags_used": False,
                "allow_flags_violation": False,
                "evidence_checks": [{
                    "stage": "Probe",
                    "stable_stage": "ProbeIds",
                    "label": "Probe",
                    "code": "OFFLINE_VLM_LOCAL_REQUIRED",
                    "msg": err_msg,
                }],
                "gates_na": {},
                "offline_model_check": {
                    "required": True,
                    "ok": False,
                    "hf_home": offline_gate.get("hf_home", ""),
                    "hf_cache": offline_gate.get("hf_cache", ""),
                    "checked_notes": offline_gate.get("checked_notes", []),
                    "ready_source": "",
                },
            },
            "errors": [err_msg],
            "measurements": {"evidence_check_sec": 0.0},
            "remediations": remediations,
        }
    
    work_dir = Path(args.output_dir).resolve()

    if is_phase1_acceptance_only:
        payload = {
            "success": True,
            "gates": {},
            "artifacts": {
                "allow_flags_used": False,
                "allow_flags_violation": False,
                "evidence_checks": [],
                "gates_na": {},
            },
            "errors": [],
            "measurements": {"evidence_check_sec": 0.0},
            "remediations": [],
        }
        ids_path = work_dir / "ids.txt"
        if ids_path.exists() and ids_path.stat().st_size > 0:
            ids_lines = [x.strip() for x in ids_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            if ids_lines:
                payload["artifacts"]["ids_sha256"] = hashlib.sha256("\n".join(ids_lines).encode()).hexdigest()
                payload["artifacts"]["audit_ids_total"] = len(ids_lines)
                payload["artifacts"]["audit_ids_source"] = "ids.txt"
            else:
                payload["errors"].append("ids.txt is empty under output_dir; ids_sha256 unavailable")
        else:
            payload["errors"].append("ids.txt missing under output_dir; ids_sha256 unavailable")
        acceptance = build_phase1_acceptance_payload(work_dir, args.seeds, payload)
        payload["acceptance_a"] = acceptance
        if not acceptance.get("success", False):
            payload["success"] = False
            for err in acceptance.get("errors", []):
                if err not in payload["errors"]:
                    payload["errors"].append(err)
        return payload

    # Phase 3.1: Build SFT trajectories from existing Phase2 outputs
    if is_build_sft:
        # Ensure src import path availability for subprocess calls
        env_overrides = {}
        src_path = PROJECT_ROOT / "src"
        current_pp = os.environ.get("PYTHONPATH", "")
        if str(src_path) not in current_pp and src_path.exists():
            env_overrides["PYTHONPATH"] = f"{src_path}{os.pathsep}{current_pp}" if current_pp else str(src_path)
        return _run_build_sft_traj_phase31(args, work_dir, env_overrides)
    
    # 0. Early Returns for strict_j FAIL-A / FAIL-B semantics (Stable J9)
    if args.allow_flags:
        # FAIL-A: J1 Violation check must happen immediately
        # Current logic enforces FAIL-A strictness for strict J9.
        # However, for Phase 1, we might want to check if allow_flags is explicitly enabling UNSAFE things.
        # But Phase 1 baseline should never need --allow-flags.
        # So we keep the strict return, but add clarity.
        return {
            "success": False,
            "gates": {"J1": False},
            "artifacts": {
                "allow_flags_used": True, 
                "allow_flags_violation": True,
                "evidence_checks": [],
                "gates_na": {}
            },
            "errors": ["J1 Violation: allow flag requested"],
            "measurements": {"evidence_check_sec": 0.0}
        }
    
    if args.no_adapter:
        # FAIL-B: Adapter check must happen immediately, no probe/no run
        check_adapter = work_dir / "MISSING_ADAPTER"
        config_path = check_adapter / "adapter_config.json"
        # Since we haven't run anything, this will fail as expected
        return {
            "success": False,
            "gates": {"J2": False},
            "artifacts": {
                "allow_flags_used": False, 
                "allow_flags_violation": False,
                "evidence_checks": [],
                "gates_na": {}
            },
            "errors": [f"Adapter Check Failed: Missing {config_path}"],
            "measurements": {"evidence_check_sec": 0.0}
        }

    # Ensure src is in PYTHONPATH for subprocesses
    env_overrides = {}
    src_path = PROJECT_ROOT / "src"
    current_pp = os.environ.get("PYTHONPATH", "")
    if str(src_path) not in current_pp:
        if src_path.exists():
            env_overrides["PYTHONPATH"] = f"{src_path}{os.pathsep}{current_pp}" if current_pp else str(src_path)

    # Context Setup
    ctx = StageContext(
        work_dir=work_dir,
        seeds=args.seeds,
        max_samples=args.max_samples,
        allow_flags=args.allow_flags,
        no_adapter=args.no_adapter,
        allow_full_dataset=args.allow_full_dataset,
        phase1_baseline=is_phase1,
        phase2_full_infer=is_phase2_full,
        strict_j_mode=(args.mode == "strict_j"),
        strict_contract=bool(getattr(args, "strict_contract", False)),
        dataset_split=args.dataset_split,
        env_overrides=env_overrides,
        vlm_model_id=phase2_vlm_model_id,
        requested_vlm_model_id=phase2_requested_vlm_model_id,
        requested_vlm_model_local_dir=phase2_requested_vlm_local_dir,
        vlm_model_source=phase2_vlm_model_source,
        vlm_max_side=phase2_vlm_max_side,
        sdp_backend=phase2_sdp_backend,
        vlm_retry_n=phase2_vlm_retry_n,
        vlm_max_side_source=phase2_vlm_max_side_source,
        sdp_backend_source=phase2_sdp_backend_source,
        vlm_retry_n_source=phase2_vlm_retry_n_source,
    )
    
    res = StageResult()
    res.artifacts["audit_allow_full_dataset"] = bool(ctx.allow_full_dataset)
    res.artifacts["audit_dataset_split"] = str(ctx.dataset_split)
    if is_phase2_full:
        _phase2_audit_eff = _resolve_expected_max_samples(ctx)
        res.artifacts["audit_effective_max_samples"] = "NONE" if _phase2_audit_eff is None else int(_phase2_audit_eff)
    else:
        res.artifacts["audit_effective_max_samples"] = "NONE" if ctx.allow_full_dataset else int(ctx.get_effective_max_samples())
    res.artifacts["audit_ids_total"] = 0
    res.artifacts["audit_ids_source"] = "NOT_SET"
    res.artifacts["ids_source_note"] = "NOT_SET"
    if is_phase2_full:
        res.artifacts["ids_source_note"] = "phase2_full_infer forces full coverage"
        res.artifacts["audit_vlm_model_source"] = phase2_vlm_model_source
        res.artifacts["audit_seeds_policy"] = phase2_seeds_policy or "cli_override"
        res.artifacts["execution_error_count"] = 0
        res.artifacts["unknown_count"] = 0
        res.artifacts["strict_schema_invalid_count"] = 0
        res.artifacts["audit_vlm_max_side"] = int(phase2_vlm_max_side)
        res.artifacts["audit_sdp_backend"] = str(phase2_sdp_backend)
        res.artifacts["audit_vlm_retry_n"] = int(phase2_vlm_retry_n)
        res.artifacts["audit_vlm_max_side_source"] = phase2_vlm_max_side_source
        res.artifacts["audit_sdp_backend_source"] = phase2_sdp_backend_source
        res.artifacts["audit_vlm_retry_n_source"] = phase2_vlm_retry_n_source
        for seed in args.seeds:
            res.artifacts[f"seed_{seed}_audit_vlm_max_side"] = int(phase2_vlm_max_side)
            res.artifacts[f"seed_{seed}_audit_sdp_backend"] = str(phase2_sdp_backend)
            res.artifacts[f"seed_{seed}_audit_vlm_retry_n"] = int(phase2_vlm_retry_n)
            res.artifacts[f"seed_{seed}_audit_vlm_max_side_source"] = phase2_vlm_max_side_source
            res.artifacts[f"seed_{seed}_audit_sdp_backend_source"] = phase2_sdp_backend_source
            res.artifacts[f"seed_{seed}_audit_vlm_retry_n_source"] = phase2_vlm_retry_n_source
        if phase2_vlm_model_id:
            res.artifacts["audit_requested_vlm_model_id"] = phase2_vlm_model_id
        else:
            res.artifacts["using_default_vlm_model_id"] = True

    # SENTINEL MODE
    if args.sentinel_ref:
        pipeline = [SentinelPipeline(args.sentinel_ref)]
        try:
            pipeline[0].execute(ctx, res)
        except FileNotFoundError as e:
            res.success = False
            res.errors.append(str(e))
            
        return {
            "success": res.success,
            "gates": res.gates,
            "artifacts": res.artifacts,
            "errors": res.errors,
            "measurements": res.measurements
        }

    # NORMAL WORKLOAD
    try:
        _cleanup_work_dir_preserve_logs(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {
            "success": False,
            "gates": {},
            "artifacts": {
                "allow_flags_used": False, 
                "allow_flags_violation": False,
                "evidence_checks": [{
                    "stage": "WorkloadPrep", 
                    "stable_stage": "WorkloadPrep",
                    "label": "WorkloadPrep",
                    "code": "RMTREE_FAILED", 
                    "msg": f"Workload cleanup failed: {e}"
                }],
                "gates_na": {}
            },
            "errors": [f"Workload cleanup failed: {e}"],
            "measurements": {"evidence_check_sec": 0.0}
        }
    
    # J0: Clean Room Verification (Execution Path)
    current_exec = Path(__file__).resolve()
    print(f"[Workload] Executing from: {current_exec}", file=sys.stderr)
    res.artifacts["execution_path"] = str(current_exec)

    # Pipeline Definition
    if is_phase2_full:
        pipeline = [
            PreflightDeps(),
            ProbeIds(),
            AgentInfer06(),
        ]
    else:
        pipeline = [
            PreflightDeps(),
            ProbeIds(),
            AgentInfer06(),
            BuildTraj08(),
            SFTTrain09(),
            SFTInfer(),
            GRPOBuild(),
            GRPOTrain(),
            GRPOInfer(),
            Phase1Metrics()
        ]
    
    # Run Pipeline
    for stage in pipeline:
        if not res.success: break
        try:
            stage.execute(ctx, res)
        except FileNotFoundError as e:
            res.success = False
            res.errors.append(str(e))
        
    # Evaluation
    GateEvaluator.evaluate(ctx, res)
    
    # Task B: Enforce success/exit_code consistency for Phase 1
    if ctx.phase1_baseline:
         # Must pass all gates to be considered successful
         gates_passed = all(res.gates.get(k, True) for k in ["J1","J2","J3","J4_DEPENDENCIES","J6"])
         
         # Also check for allow_flags_violation or any errors
         allow_violation = res.artifacts.get("allow_flags_violation", False)
         has_errors = len(res.errors) > 0
         
         if not gates_passed or allow_violation or has_errors:
             res.success = False
             if "Phase1 Gate Failure" not in res.errors:
                 res.errors.append("Phase1 Gate Failure: Strict compliance failed (gates/flags/errors)")

    payload = {
        "success": res.success,
        "gates": res.gates,
        "artifacts": res.artifacts,
        "errors": res.errors,
        "measurements": res.measurements,
        "remediations": res.remediations
    }
    payload["artifacts"].setdefault("mmad_root_resolved", "NOT_SET")
    payload["artifacts"].setdefault("mmad_asset_mode", "unknown")
    if ctx.phase1_baseline:
        payload["acceptance_a"] = build_phase1_acceptance_payload(ctx.work_dir, ctx.seeds, payload)
        if not payload["acceptance_a"].get("success", False):
            payload["success"] = False
            for err in payload["acceptance_a"].get("errors", []):
                if err not in payload["errors"]:
                    payload["errors"].append(err)
        payload["artifacts"]["phase1_baseline_spec"] = {
            "seed_policy_fixed": [0, 1, 2],
            "scope": "full_mmad_baseline_no_sft_no_grpo_training",
        }
    if ctx.phase2_full_infer:
        payload["acceptance_b"] = build_phase2_full_infer_acceptance_payload(ctx.work_dir, ctx.seeds, payload)
        if not payload["acceptance_b"].get("success", False):
            payload["success"] = False
            for err in payload["acceptance_b"].get("errors", []):
                if err not in payload["errors"]:
                    payload["errors"].append(err)
        payload["artifacts"]["phase2_full_infer_spec"] = {
            "scope": "infer_only_no_sft_no_grpo_training",
            "stages": ["PreflightDeps", "ProbeIds", "AgentInfer06"],
        }
    return payload

def build_arg_parser():
    parser = argparse.ArgumentParser(description="AgentIAD Reproduction Verification Orchestrator")
    parser.add_argument("--mode", type=str, default="default", help="Execution mode")
    parser.add_argument("--input-dir", type=str, default=None, dest="input_dir", help="Phase2 output root (required for --mode build_sft_traj)")
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples", help="Max samples per stage")
    parser.add_argument("--strict-contract", action="store_true", dest="strict_contract", help="Enforce strict contract")
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir", help="Output directory")
    parser.add_argument("--allow-flags", action="store_true", dest="allow_flags", help="Allow unsafe flags")
    parser.add_argument("--no-adapter", action="store_true", dest="no_adapter", help="Skip adapter checks")
    parser.add_argument("--allow-full-dataset", action="store_true", dest="allow_full_dataset", help="Allow full dataset runs")
    parser.add_argument("--dataset-split", type=str, default=None, dest="dataset_split", help="Dataset split for phase1")
    parser.add_argument("--vlm-model-id", type=str, default=None, dest="vlm_model_id", help="Explicit VLM model id (repo id or local path)")
    parser.add_argument("--vlm-model-local-dir", type=str, default=None, dest="vlm_model_local_dir", help="Explicit local VLM model directory; higher priority than --vlm-model-id")
    parser.add_argument("--vlm-max-side", type=int, default=768, dest="vlm_max_side", help="Override VLM max image side for phase2 infer")
    parser.add_argument("--sdp-backend", type=str, default="auto", dest="sdp_backend", help="Override SDP backend for phase2 infer (auto/math/flash/mem_efficient)")
    parser.add_argument("--vlm-retry-n", type=int, default=2, dest="vlm_retry_n", help="Override VLM retry count for phase2 infer")
    parser.add_argument("--sentinel-ref", type=str, default="", dest="sentinel_ref", help="Sentinel reference directory")
    parser.add_argument("--seeds", type=str, nargs="+", default=["42"], help="Random seeds, supports '--seeds 0 1 2' or '--seeds 0,1,2'")
    return parser

def main():
    # Remote Route-A reference command:
    # export MMAD_ROOT="/data2/lrrelevant/datasets/MMAD_ROOT"
    # CUDA_VISIBLE_DEVICES=0 python verify_all.py --mode strict_j --allow-full-dataset --dataset-split train --seeds 0 1 2
    parser = build_arg_parser()
    seeds_flag_provided = any(str(x).strip() == "--seeds" for x in sys.argv[1:])
    args = parser.parse_args()
    setattr(args, "seeds_provided", bool(seeds_flag_provided))
    if str(args.mode).strip() == "phase2_full":
        args.mode = "phase2_full_infer"
    if str(args.mode).strip() == "build_sft_traj" and not seeds_flag_provided:
        args.seeds = []
    else:
        try:
            args.seeds = _parse_seeds_value(args.seeds)
        except ValueError as e:
            print(f"[ArgsError] {e}", file=sys.stderr)
            sys.exit(2)
    if args.dataset_split is None:
        # MMAD is single-split in this workflow; keep historical default for other modes.
        args.dataset_split = "train" if str(args.mode) in {"build_sft_traj", "phase2_full_infer"} else "test"

    if args.output_dir is None and args.mode in {"phase1_baseline", "phase1_acceptance_only"}:
        args.output_dir = "dist/outputs/phase1_baseline"
    elif args.output_dir is None and args.mode == "phase2_full_infer":
        args.output_dir = "dist/outputs/phase2_full_infer"
    elif args.output_dir is None and args.mode == "build_sft_traj":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"dist/outputs/phase3_traj_{timestamp}"
    elif args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/workload_{timestamp}"

    output_dir = Path(args.output_dir).resolve()
    terminal_log_path = _build_terminal_log_path(output_dir)

    with _tee_terminal_to_log(terminal_log_path):
        _emit_terminal_log_header(terminal_log_path, args, output_dir)
        try:
            res = run_workload(args)
        except Exception as e:
            res = {
                "success": False,
                "errors": [f"Unhandled Exception: {str(e)}"]
            }

        if isinstance(res, dict):
            payload = res
        else:
            payload = {
                "success": getattr(res, "success", False),
                "gates": getattr(res, "gates", {}),
                "artifacts": getattr(res, "artifacts", {}),
                "errors": getattr(res, "errors", []),
                "measurements": getattr(res, "measurements", {}),
            }

        if not isinstance(payload.get("artifacts"), dict):
            payload["artifacts"] = {}
        payload["artifacts"]["terminal_log_path"] = str(terminal_log_path)
        if args.mode == "phase2_full_infer":
            try:
                zip_stats = _emit_phase2_zip_stats_line(output_dir, args.seeds, terminal_log_path)
                payload["artifacts"]["zip_stats"] = zip_stats
            except Exception as e:
                print(f"[zip_stats] " + json.dumps({"error": f"zip_stats_emit_failed:{type(e).__name__}:{e}"}, ensure_ascii=False, sort_keys=True))

        print("WORKLOAD_RESULT=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
        if args.mode in {"phase1_baseline", "phase1_acceptance_only", "phase2_full_infer"}:
            acceptance = payload.get("acceptance_a", {
                "success": False,
                "errors": ["acceptance payload missing"],
                "seeds": [0, 1, 2],
                "dataset_binding_hash": "",
                "dataset_binding_hash_source": "",
                "per_seed_metrics_path": [],
                "summary_path": "",
                "n_total": 0,
                "evidence_paths": [],
            })
            if args.mode == "phase2_full_infer":
                acceptance = payload.get("acceptance_b", {
                    "success": False,
                    "errors": ["acceptance payload missing"],
                    "seeds": payload.get("seeds", []),
                    "evidence_paths": [],
                    "toolcall_rate_avg": 0.0,
                })
            print("ACCEPTANCE_JSON=" + json.dumps(acceptance, ensure_ascii=False, sort_keys=True))
        print(f"terminal_log_path={terminal_log_path}")
        sys.exit(0 if payload.get("success", False) else 1)

if __name__ == "__main__":
    main()
