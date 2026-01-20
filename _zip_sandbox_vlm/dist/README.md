# agentiad_repro

一个从零搭建的可复现小工程骨架：通过脚本完成环境检查、真实 MMAD 下载、真实 sanity、真实 baseline 评测并落盘产物。

## 目录结构

- `src/`：Python 包源码
- `scripts/`：可直接运行的脚本（00~04）
- `configs/`：路径配置
- `outputs/`：运行产物（logs/tables/traces）
- `data/`：数据与缓存（mmad/trajectories/cache）

## 从零复现（PowerShell）

按顺序执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python .\scripts\00_env_check.py
python .\scripts\01_get_mmad_real.py
python .\scripts\02_mmad_sanity_real.py
python .\scripts\04_eval_baseline_real.py
```

若要显式设置 HF 缓存目录，需先创建目录再 Resolve-Path（否则 Resolve-Path 会失败）：

```powershell
mkdir -Force .\data\cache | Out-Null
$env:HF_HOME = (Resolve-Path .\data\cache).Path
$env:HF_DATASETS_CACHE = (Resolve-Path .\data\cache).Path
```

## 验收产物

- `outputs/logs/env.json`
- `data/mmad/mmad_manifest.json`
- `outputs/logs/mmad_download_meta.json`
- `outputs/logs/mmad_sanity_real.json`
- `outputs/traces/mmad_samples/*.png`（至少 5 张）
- `outputs/tables/baseline_real_*.csv`

## Level 1 baseline（VLM yes/no）

该 baseline 用生成式 VLM 做异常检测：对每张图输出严格 JSON：`{"anomaly":"yes"}` 或 `{"anomaly":"no"}`。

依赖（不写入 requirements.txt，按需安装）：

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install --upgrade transformers
```

运行（`vlm_model_id` 必填，不能为空）：

```powershell
python .\dist\scripts\05_eval_vlm.py --config .\dist\configs\model.yaml
```

产物：

- `outputs/logs/baseline_vlm_summary.json`
- `outputs/tables/baseline_vlm_*.csv`

CLIP MCQ baseline（多选题）请使用 `dist/scripts/04_eval_baseline_real.py`。

## Level 2 agent（PZ/CR + CLIP）

该 agent 在 baseline margin 很小（不确定）时触发 PZ/CR：先用 query/template 差分找 ROI，再对 ROI 复算 CLIP MCQ 并与 baseline 融合。

运行：

```powershell
python .\scripts\06_eval_agent_pzcr.py --config .\configs\agent_pzcr.yaml
```

可选：预热（只拉取/解码 query/template 图片）：

```powershell
python .\scripts\06_eval_agent_pzcr.py --config .\configs\agent_pzcr.yaml --warmup --warmup_n 200
```

产物：

- `outputs/tables/agent_pzcr_*.csv`
- `outputs/traces/agent_pzcr_*.jsonl`
- `outputs/logs/agent_pzcr_summary.json`
可选依赖：如需更快的 PZ box（连通域），可安装 `opencv-python`（不写入 requirements.txt）。

## Dummy 数据（仅 smoke test）

仓库仍保留 `scripts/01_get_mmad.py`、`scripts/02_mmad_sanity.py`、`scripts/04_eval_baseline.py`，它们只生成/评测 dummy 数据用于快速冒烟测试，不代表真实 MMAD。

## 打包说明

默认 `outputs/` 与 `data/` 不纳入 git；验收时需要打包：

- `outputs/logs/`
- `outputs/tables/`
- `outputs/traces/mmad_samples/`
