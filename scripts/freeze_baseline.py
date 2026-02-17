
import os
import sys
import subprocess
import json
import hashlib
from pathlib import Path
from typing import Any, Dict

# Reuse existing config loader logic (via pyyaml as used in utils.py)
try:
    import yaml
except ImportError:
    raise RuntimeError("PyYAML missing. Install: python3 -m pip install pyyaml")

# Add src to path to import PaperContract
sys.path.append(str(Path(__file__).parent.parent / "src"))
from agentiad_repro.paper_contract import PaperContract

def get_git_status() -> str:
    """Returns the git status output."""
    return subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()

def verify_repo_clean():
    """Verifies that the repository is clean (ignoring this script and output artifact)."""
    status = get_git_status()
    if not status:
        return

    lines = status.splitlines()
    allowed_files = {"scripts/freeze_baseline.py", "dist/freeze_manifest.json", "dist/"}
    
    dirty_files = []
    for line in lines:
        # line format is "XY filename"
        parts = line.split()
        if len(parts) >= 2:
            filename = parts[-1]
            if filename not in allowed_files and not filename.startswith("dist/"):
                dirty_files.append(line)
    
    if dirty_files:
        print("Error: Repository is not clean. Please commit or stash changes.")
        print("Dirty files:")
        for f in dirty_files:
            print(f)
        sys.exit(1)

def get_git_info() -> Dict[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    return {"commit": commit, "branch": branch}

def compute_hash(data: Any) -> str:
    """Computes SHA256 hash of data with deterministic JSON serialization."""
    # Ensure deterministic ordering
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

def get_pip_freeze() -> str:
    return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)

def main():
    project_root = Path(__file__).parent.parent
    dist_dir = project_root / "dist"
    
    # 1. Verify repo clean
    verify_repo_clean()
    
    # 2. Get Git Info
    git_info = get_git_info()
    
    # 3. Load Configs
    model_config_path = project_root / "configs" / "model.yaml"
    agent_config_path = project_root / "configs" / "agent_pzcr.yaml"
    
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
        
    with open(agent_config_path, "r", encoding="utf-8") as f:
        agent_config = yaml.safe_load(f)
        
    # 4. Compute Hashes
    # Dataset split hash
    dataset_split = model_config.get("split")
    dataset_split_hash = compute_hash(dataset_split)
    
    # Prompt template hash (Config template + System Prompt)
    prompt_data = {
        "template": model_config.get("prompt_template"),
        "text_template": model_config.get("text_template"),
        "system_prompt": PaperContract.SYSTEM_PROMPT
    }
    prompt_template_hash = compute_hash(prompt_data)
    
    # Tool enable config hash (Agent config + Allowed Tools contract)
    tool_data = {
        "agent_config": agent_config,
        "allowed_tools": PaperContract.ALLOWED_TOOLS
    }
    tool_enable_config_hash = compute_hash(tool_data)
    
    # 5. Capture Info
    seeds = model_config.get("seed")
    model_id = model_config.get("model_id")
    
    # 6. Dump pip freeze
    pip_freeze_output = get_pip_freeze()
    
    # 7. Construct Manifest
    manifest = {
        "git_info": git_info,
        "dataset_split_hash": dataset_split_hash,
        "prompt_template_hash": prompt_template_hash,
        "tool_enable_config_hash": tool_enable_config_hash,
        "seeds": seeds,
        "model_id": model_id,
        "pip_freeze": pip_freeze_output
    }
    
    # 8. Save Manifest
    if not dist_dir.exists():
        dist_dir.mkdir()
        
    manifest_path = dist_dir / "freeze_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
        
    # 9. Output Status
    manifest_hash = compute_hash(manifest)
    print("FREEZE_STATUS=OK")
    print(f"FREEZE_HASH={manifest_hash}")

if __name__ == "__main__":
    main()
