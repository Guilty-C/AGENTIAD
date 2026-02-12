
import hashlib
from pathlib import Path

p = Path("verify_all.py")
with open(p, "rb") as f:
    lines = f.readlines()

content = b"".join(l for l in lines if not l.strip().startswith(b"EXPECTED_SHA_VERIFY_ALL ="))
sha = hashlib.sha256(content).hexdigest().upper()
print(f"Calculated SHA: {sha}")

new_lines = []
for l in lines:
    if l.strip().startswith(b"EXPECTED_SHA_VERIFY_ALL ="):
        new_lines.append(f'EXPECTED_SHA_VERIFY_ALL = "{sha}"\n'.encode("utf-8"))
    else:
        new_lines.append(l)

with open(p, "wb") as f:
    f.writelines(new_lines)
