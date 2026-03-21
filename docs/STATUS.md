# AgentIAD MMAD Repro Status

Last update: 2026-01-20

## Checklist Audit Summary (Master Checklist v1.0)

| Level | Status | Evidence / Gaps |
| --- | --- | --- |
| Global (Project structure & reproducibility norms) | PARTIAL | `scripts/05_eval_vlm.py` + `configs/model.yaml` exist; no outputs tables/traces/ckpts yet; status log maintained here. |
| Level 1 (VLM baseline) | BROKEN | `scripts/05_eval_vlm.py` and `configs/model.yaml` exist; smoke test blocked by HF Hub `ProxyError` when fetching `jiang-cc/MMAD`. |
| Level 2 (AgentIAD PZ/CR inference) | MISSING | Missing `src/agentiad_repro/tools/pz.py`, `src/agentiad_repro/tools/cr.py`, `scripts/06_run_agentiad_infer.py`; no traces. |
| Level 3 (Trajectories) | MISSING | Missing `scripts/07_build_trajectories.py`, `scripts/08_validate_trajectories.py`, `data/trajectories/trajectories_sft.jsonl`. |
| Level 4 (SFT LoRA) | MISSING | Missing `scripts/09_train_sft_lora.py`, `configs/sft.yaml`, `outputs/ckpts/sft_lora/`. |
| Level 5 (GRPO) | MISSING | Missing `scripts/10_train_grpo_openrlhf.*`, `configs/grpo.yaml`, reward modules. |

## Latest Verification Evidence

### LOOP-1 Commands
- `python -m compileall -q scripts/00_env_check.py`
- `python scripts/00_env_check.py`

### LOOP-2 Commands
- `python -m compileall -q scripts/05_eval_vlm.py`
- `python scripts/05_eval_vlm.py --config configs/model.yaml`
- `python - <<'PY'\nimport hashlib, pathlib, yaml, subprocess\ncfg_path = pathlib.Path('configs/model.yaml')\ntext = cfg_path.read_text(encoding='utf-8')\nconfig_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()\ncfg = yaml.safe_load(text)\nprompt = cfg.get('prompt_template', '')\nprompt_hash = hashlib.sha256(str(prompt).encode('utf-8')).hexdigest()\nmodel_id = cfg.get('model_id')\nseed = cfg.get('seed')\ntry:\n    git_commit = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()\nexcept Exception:\n    git_commit = None\nprint(f\"git_commit={git_commit}\")\nprint(f\"seed={seed}\")\nprint(f\"prompt_hash={prompt_hash}\")\nprint(f\"config_hash={config_hash}\")\nprint(f\"model_id={model_id}\")\nPY`

### Key Outputs
- `outputs/logs/env.json`

### Output Structure Check
- `outputs/logs/env.json` created by `00_env_check.py` (environment metadata log).
- L1 baseline CSV not produced due to dataset fetch failure (`ProxyError` when loading `jiang-cc/MMAD`).

### LOOP-2 Error (stderr excerpt)
- `ConnectionError: Couldn't reach 'jiang-cc/MMAD' on the Hub (ProxyError)`

## Current Blockers
- Hugging Face Hub access blocked by `ProxyError` when fetching `jiang-cc/MMAD` in `scripts/05_eval_vlm.py`. Need dataset access or mirrored local dataset to proceed with L1.
