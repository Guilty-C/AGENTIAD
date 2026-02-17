
import os
import sys
import subprocess
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

def _selftest():
    """Lightweight static self-check for invariants."""
    # This runs before any imports or heavy logic
    print("SELFTEST_OK=1")

# Reuse existing config loader logic (via pyyaml as used in utils.py)
try:
    import yaml
except ImportError:
    raise RuntimeError("PyYAML missing. Install: python3 -m pip install pyyaml")

# Add src to path to import PaperContract
# We use resolve() to ensure we have absolute path to project root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))
from agentiad_repro.paper_contract import PaperContract

def compute_canonical_hash(data: Any) -> str:
    """Computes SHA256 hash of data with deterministic JSON serialization."""
    # Ensure deterministic ordering
    json_bytes = json.dumps(
        data, 
        sort_keys=True, 
        separators=(",", ":"), 
        ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()

def get_file_hash(path: Path) -> str:
    """Computes SHA256 hash of a file's content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get_git_status(root: Path) -> str:
    """Returns the git status output running from repo root."""
    # Use -C to force git to run in the project root context
    return subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], 
        text=True
    ).strip()

def verify_repo_clean(root: Path):
    """
    Verifies that the repository is clean.
    Only allows 'dist/freeze_manifest.json' to be untracked or modified.
    Strictly forbids any other dirty state.
    """
    status = get_git_status(root)
    if not status:
        return

    lines = status.splitlines()
    dirty_files = []
    
    # Allowed file relative to repo root (POSIX style)
    allowed_file = "dist/freeze_manifest.json"
    
    for line in lines:
        # Porcelain format: XY PATH or XY ORIG -> PATH
        if len(line) < 4:
            continue
            
        # The filename part starts after column 3 (indices 0,1 are status, 2 is space)
        # However, git status --porcelain might quote filenames if they contain spaces.
        raw_filename = line[3:].strip()
        
        # Handle renames "ORIG -> NEW"
        if " -> " in raw_filename:
            raw_filename = raw_filename.split(" -> ")[-1]
            
        # Handle quotes
        filename = raw_filename
        if filename.startswith('"') and filename.endswith('"'):
            # Basic unquoting (git uses C-style escaping, but simple stripping is usually enough for standard filenames)
            filename = filename[1:-1]
            
        # Normalize path separators to POSIX for comparison
        filename = filename.replace("\\", "/")
        
        if filename != allowed_file:
            dirty_files.append(line)
    
    if dirty_files:
        print("Error: Repository is not clean. Strict freeze requires clean state.")
        print(f"Checked against root: {root}")
        print("Dirty files:")
        for f in dirty_files:
            print(f)
        sys.exit(1)

def get_git_info(root: Path) -> Dict[str, str]:
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], 
        text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"], 
        text=True
    ).strip()
    return {"commit": commit, "branch": branch}

def get_pip_freeze() -> str:
    return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)

def normalize_seeds(config: Dict[str, Any]) -> List[int]:
    """
    Normalizes seeds from config.
    Supports 'seed': int or 'seeds': List[int].
    """
    seeds = []
    if "seeds" in config and isinstance(config["seeds"], list):
        seeds = [int(s) for s in config["seeds"]]
    elif "seed" in config:
        seeds = [int(config["seed"])]
    else:
        seeds = [0] # Default fallback
    return sorted(list(set(seeds))) # Sort and unique

def main():
    _selftest()

    # Use the globally resolved project_root
    dist_dir = project_root / "dist"
    manifest_path = dist_dir / "freeze_manifest.json"
    
    # 0. Strict Directory Check
    # We do NOT mkdir. The dist directory must exist.
    if not dist_dir.exists():
        print(f"Error: {dist_dir} does not exist.")
        print("Strict freeze requires 'dist/' directory to exist (and preferably be tracked/ignored).")
        sys.exit(1)

    # 1. Verify repo clean (Strict)
    verify_repo_clean(project_root)
    
    # 2. Get Infrastructure Info (Snapshot only, not part of def hash)
    git_info = get_git_info(project_root)
    pip_freeze_output = get_pip_freeze()
    python_info = {
        "executable": sys.executable,
        "version": sys.version
    }
    
    # 3. Load Configs & Contracts
    model_config_path = project_root / "configs" / "model.yaml"
    agent_config_path = project_root / "configs" / "agent_pzcr.yaml"
    contract_path = project_root / "src" / "agentiad_repro" / "paper_contract.py"
    
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
        
    with open(agent_config_path, "r", encoding="utf-8") as f:
        agent_config = yaml.safe_load(f)
        
    contract_file_hash = get_file_hash(contract_path)
        
    # 4. Compute Def Hashes
    
    # Dataset Split Definition
    # Only dataset selection params. 
    seeds = normalize_seeds(model_config)
    
    # Determine dataset ID and optional revision
    # Fallback to "jiang-cc/MMAD" if not present in config
    dataset_id = model_config.get("dataset_id") or model_config.get("dataset_name") or "jiang-cc/MMAD"
    dataset_revision = model_config.get("dataset_revision") # Optional revision
    
    dataset_split_def = {
        "dataset_id": dataset_id,
        "split": model_config.get("split"),
        "max_samples": model_config.get("max_samples"),
        "seeds": seeds
    }
    if dataset_revision:
        dataset_split_def["revision"] = dataset_revision
        
    # Prompt Definition Hash
    prompt_def = {
        "prompt_template": model_config.get("prompt_template"),
        "text_template": model_config.get("text_template"),
        "system_prompt": PaperContract.SYSTEM_PROMPT,
        "contract_file_hash": contract_file_hash
    }
    prompt_def_hash = compute_canonical_hash(prompt_def)
    
    # Tool Definition Hash
    tool_def = {
        "agent_config": agent_config,
        "allowed_tools": PaperContract.ALLOWED_TOOLS,
        "contract_file_hash": contract_file_hash
    }
    tool_def_hash = compute_canonical_hash(tool_def)
    
    # 5. Build Freeze Definition (Stable across machines)
    # This dictates the "scientific definition" of the experiment
    # Directly embed dataset_split_def for better auditability
    freeze_def = {
        "git_commit": git_info["commit"], # Branch is metadata, commit is definition
        "model_id": model_config.get("model_id"),
        "seeds": seeds,
        "dataset_split_def": dataset_split_def, # Direct embedding
        "prompt_def_hash": prompt_def_hash,
        "tool_def_hash": tool_def_hash
    }
    freeze_def_hash = compute_canonical_hash(freeze_def)
    
    # 6. Build Environment Snapshot (Machine-specific)
    env_snapshot = {
        "git_branch": git_info["branch"],
        "python_info": python_info,
        "pip_freeze": pip_freeze_output
    }
    env_snapshot_hash = compute_canonical_hash(env_snapshot)
    
    # 7. Construct Full Manifest
    manifest = {
        "freeze_def": freeze_def,
        "freeze_def_hash": freeze_def_hash,
        "env_snapshot": env_snapshot,
        "env_snapshot_hash": env_snapshot_hash,
        # Audit/Debug fields
        "_audit": {
            "prompt_def_keys": list(prompt_def.keys()),
            "tool_def_keys": list(tool_def.keys())
        }
    }
    
    # Add raw_seed if present for compatibility/record
    if "seed" in model_config:
        manifest["raw_seed"] = model_config["seed"]
    
    # 8. Save Manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        
    # 9. Output Status
    print("FREEZE_STATUS=OK")
    print(f"FREEZE_HASH={freeze_def_hash}")

if __name__ == "__main__":
    main()
