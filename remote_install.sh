
# 1. Setup persistent data directory
mkdir -p ~/data/mmad

# 2. Create temp workspace
TMP_DL="/tmp/mmad_dl_$(date +%s)"
mkdir -p "$TMP_DL"

# 3. Unzip uploaded scripts
unzip -q ~/mmad_download_tmp.zip -d "$TMP_DL"

cd "$TMP_DL"

# 4. Link data directory to persistent location
mkdir -p data
# remove if exists to avoid nesting
rm -rf data/mmad
ln -s ~/data/mmad data/mmad

# 5. Run download script
# Ensure python path allows importing from src
export PYTHONPATH="$TMP_DL/src:$PYTHONPATH"
python3 scripts/01_get_mmad.py

# 6. Verify result
echo "Checking downloaded file:"
ls -lh ~/data/mmad/mmad_dummy.jsonl

# 7. Cleanup
cd ~
rm -rf "$TMP_DL"
rm ~/mmad_download_tmp.zip
echo "Cleanup complete."
