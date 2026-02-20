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
python -m pip install --upgrade torch torchvision torchaudio --index-url `https://download.pytorch.org/whl/cpu`
python -m pip install --upgrade transformers accelerate
```

Windows CUDA 12.1 示例（按需安装）：

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url `https://download.pytorch.org/whl/cu121`
python -m pip install --upgrade transformers accelerate
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

## Level 2 VLM Agent（global→PZ→(uncertain)CR→final）

该链路用于固化 VLM Agent 推理与可审计 trace：每个样本都会落盘 `crop.png/trace.json/final.json`，当触发 CR 时会额外落盘 `ref.png`。每样本 trace 目录为：`outputs/traces/<run_name>/<sample_id>/`。

cold-start CPU 冒烟（不下载模型/数据，走 dry_run，但仍会落 crop/ref/trace/final）：

```powershell
python scripts/06_run_agentiad_infer.py --config configs/model.yaml --max_samples 5 --run_name DRYTEST --dry_run
```

真实 VLM 冒烟（会下载模型与 MMAD 数据；需在 `configs/model.yaml` 中设置 `vlm_model_id`（或 `model_id`）为可运行的 VLM）：

```powershell
python scripts/06_run_agentiad_infer.py --config configs/model.yaml --max_samples 5 --run_name L2_SMOKE_REAL --seed 0
```

GPU 实跑示例（沿用 Level 1 的 VLM 安装策略，不把 torch 写入 requirements.txt）：

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url `https://download.pytorch.org/whl/cu121`
python -m pip install --upgrade transformers accelerate datasets pillow

python scripts/06_run_agentiad_infer.py --config configs/model.yaml --max_samples 50 --run_name L2RUN
```

### Progress & Tool Switch (stderr-only)

参数：

- `--progress true|false|1|0|yes|no|on|off`，优先级：CLI > env(`AGENTIAD_PROGRESS`) > `cfg.progress` > default False
- `--enable_tools true|false|1|0|yes|no|on|off`，优先级：CLI > `cfg.enable_tools` > default True

明确口径：进度条/告警都在 stderr，stdout 只用于 `key=...` 审计。

审计注意事项：

- 不要用 `2>&1`，否则会把 stderr（进度条/告警）混入 stdout，污染 stdout 审计口径
- PowerShell 正确审计示例（屏蔽 stderr；只看 stdout 的 `key=...` 行）：

```powershell
python dist/scripts/06_run_agentiad_infer.py --config dist/configs/model.yaml --max_samples 3 --run_name L2_AUDIT --seed 0 --dry_run --progress true 2> $null |
  Select-String -Pattern '^run_name=','^device=','^hf_device_map=','^gen_do_sample=','^config_hash=','^prompt_hash=','^csv_sha256=','^first_sample_id=','^first_trace_fingerprint_hash=' |
  ForEach-Object { $_.Line }
```

统计口径说明（务必区分 rate 与计数）：

- `toolcall_rate` 定义为：`(#samples with any valid tool_call.name) / N`，取值范围 0~1
- `tool_calls_total` 是 tool_call 的总次数计数（可大于 N）
- 不要把 `tool_calls_total / N` 叫做 rate（它是“每样本平均 tool_call 次数”，不是 0~1 的比例）

Metrics Glossary（可核验字段名）：

- toolcall_rate = (#samples with >=1 valid tool_call.name) / N（范围 0~1）
- tool_calls_total = total number of tool calls across all samples（整数）
- 不要把 tool_calls_total / N 称为 rate

运行自检（只写命令；不要求实际运行）：

```powershell
python -m compileall -q dist/scripts/06_run_agentiad_infer.py
```

证据式自检命令（只写命令；不贴输出）：

```powershell
Select-String -Path dist/README.md -Pattern 'toolcall_rate','tool_calls_total','tool_calls_total / N'
```

smoke（tools off/on + progress true；观察 stderr 是否有进度条）：

```powershell
python dist/scripts/06_run_agentiad_infer.py --config dist/configs/model.yaml --max_samples 3 --run_name SMOKE_TOOLS_OFF --seed 0 --dry_run --enable_tools false --progress true | Out-Null
python dist/scripts/06_run_agentiad_infer.py --config dist/configs/model.yaml --max_samples 3 --run_name SMOKE_TOOLS_ON  --seed 0 --dry_run --enable_tools true  --progress true | Out-Null
```

可复制 stats 口径 python -c 示例（与 Metrics Glossary 一致）：

```powershell
python -c "import json
from pathlib import Path

def stats(trace_dir):
    td = Path(trace_dir)
    sample_dirs = [p for p in td.iterdir() if p.is_dir()]
    N = len(sample_dirs)
    n_with_tool_call = 0
    tool_calls_total = 0
    crop_exists = 0
    ref_exists = 0
    for sd in sample_dirs:
        if (sd/'crop.png').exists():
            crop_exists += 1
        if (sd/'ref.png').exists():
            ref_exists += 1
        tp = sd/'trace.json'
        has_tool = False
        if tp.exists():
            try:
                tr = json.loads(tp.read_text(encoding='utf-8'))
            except Exception:
                tr = None
            turns = tr.get('turns') if isinstance(tr, dict) else None
            if isinstance(turns, list):
                for t in turns:
                    if not isinstance(t, dict):
                        continue
                    tc = t.get('tool_call')
                    if isinstance(tc, dict) and str(tc.get('name') or '').strip():
                        tool_calls_total += 1
                        has_tool = True
        if has_tool:
            n_with_tool_call += 1
    toolcall_rate = (n_with_tool_call / N) if N else 0.0
    return {
        'N': N,
        'toolcall_rate': toolcall_rate,
        'tool_calls_total': tool_calls_total,
        'crop_exists_rate': (crop_exists / N) if N else 0.0,
        'ref_exists_rate': (ref_exists / N) if N else 0.0,
        'trace_dir': str(td),
    }

base = Path('dist/outputs/traces')
for name in ['SMOKE_TOOLS_OFF','SMOKE_TOOLS_ON']:
    print(name, json.dumps(stats(base/name), ensure_ascii=False, sort_keys=True))
"
```

## Level 3 SFT trajectories（tool-use JSONL）

该脚本会复用 Level 2 VLM Agent 推理链路，读取每样本 `trace.json/final.json` 并汇总成可用于 SFT 的 JSONL：每条包含 `messages`（user/assistant/tool）以及 `trace_fingerprint_hash/trajectory_fingerprint_hash` 等复现指纹。

cold-start CPU 冒烟（不下载模型/数据，走 dry_run）：

```powershell
python scripts/08_build_sft_trajectories.py --config configs/model.yaml --max_samples 5 --seed 0 --out_jsonl outputs/traces/trajectories_sft_toy.jsonl --dry_run
```

产物：

- `outputs/traces/trajectories_sft_*.jsonl`

## Dummy 数据（仅 smoke test）

仓库仍保留 `scripts/01_get_mmad.py`、`scripts/02_mmad_sanity.py`、`scripts/04_eval_baseline.py`，它们只生成/评测 dummy 数据用于快速冒烟测试，不代表真实 MMAD。

## 打包说明

默认 `outputs/` 与 `data/` 不纳入 git；验收时需要打包：

- `outputs/logs/`
- `outputs/tables/`
- `outputs/traces/mmad_samples/`

## Release / Packaging Reproducibility

该 release zip 由脚本生成：`python scripts/07_make_release_zip.py`。

为便于只拿 release zip 也能复核，release 内亦包含同名脚本：`dist/scripts/07_make_release_zip.py`（解压后在 `dist/` 目录内运行：`python scripts/07_make_release_zip.py`）。

该脚本会：

- 从 `dist/` 收集文件并写入 zip，排除以下项：
  - 路径段包含 `__pycache__`
  - 以 `.pyc` 结尾的文件
  - `dist/outputs/` 目录
  - `.venv/`、`.hf/`、`.git/` 目录
- 进行只读终验并输出以下检查项：
  - `ZIP_SHA256`
  - `ZIP_BACKSLASH_PATHS`
  - `ZIP_HAS_DIST_OUTPUTS`
  - `ZIP_HAS_PYC`
  - `ZIP_HAS_PYCACHE_DIR`
- 排除规则（写入 zip 时生效）：
  - 任何路径段包含 `__pycache__`
  - 任何以 `.pyc` 结尾的文件
  - `dist/outputs/`
  - `.venv/`、`.hf/`、`.git/`
- 严谨口径声明：
  - dist 内关键增强点已通过行号与 hash 自检确认存在；未对仓库源文件逐字一致性做证明

可直接运行生成并验证：

```powershell
python scripts/07_make_release_zip.py
```

示例终验输出关键行（节选）：

```
ZIP_SHA256 <SHA256>
ZIP_BACKSLASH_PATHS 0
ZIP_HAS_DIST_OUTPUTS False
ZIP_HAS_PYC False
ZIP_HAS_PYCACHE_DIR False
```

为提升 zip 的确定性，脚本会按归档路径排序写入文件；同一 dist 内容在相同脚本版本与环境下 zip hash 更稳定（跨机器仍可能因压缩实现差异略有不同）。

## Offline MMAD Asset Configuration (Recommended)

For offline Route A runs, configure local MMAD assets explicitly.

Option 1: set environment variable before running scripts:

```powershell
$env:MMAD_ROOT = "D:\\datasets\\mmad"
```

Option 2: set `paths.mmad_root` in `dist/configs/paths.yaml` (or `configs/paths.yaml`):

```yaml
paths:
  mmad_root: "D:/datasets/mmad"
```

`MMAD_ROOT` (resolved value) should contain both directories:
- `DS-MVTec/`
- `MVTec-AD/`

If assets are missing and hub/offline access is disabled, remediation is:
`export MMAD_ROOT=<local_mmad_root>; MMAD_ROOT must contain DS-MVTec/ and MVTec-AD/`
