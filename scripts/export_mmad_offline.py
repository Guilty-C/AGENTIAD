import os, json
from datasets import load_dataset

OUT_DIR = os.environ.get("MMAD_EXPORT_DIR", r"data\MMAD_REAL")
os.makedirs(OUT_DIR, exist_ok=True)

print("[export] loading dataset: jiang-cc/MMAD")
ds = load_dataset("jiang-cc/MMAD")
train = ds["train"]

save_path = os.path.join(OUT_DIR, "hf_dataset")
print("[export] saving to:", save_path)
train.save_to_disk(save_path)

manifest = {
    "source": "jiang-cc/MMAD",
    "split": "train",
    "num_rows": len(train),
    "features": list(train.features.keys()),
    "saved_to_disk": "hf_dataset",
}
with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("[export] done. rows=", len(train))
print("[export] wrote:", os.path.join(OUT_DIR, "manifest.json"))
