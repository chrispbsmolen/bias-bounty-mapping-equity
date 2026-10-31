"""Mirror the public Source Cooperative challenge bucket into data/source_coop/.

Reads data/raw_docs/listing.tsv (an S3 ListObjectsV2 listing saved from
https://data.source.coop/humane-intelligence?list-type=2&prefix=bias-bounty-mapping-equity-challenge/).
Downloads with resumable curl into <name>.part and renames only when the byte count
matches the listing. Skips bulk zips (redundant copies) and refuses any key that looks
like a removed reference layer or target file. Usage: python3 analysis/00_download_data.py [seconds_budget]
"""
import os, sys, time, subprocess, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/"
OUT = os.path.join(ROOT, "data", "source_coop")
LIST = os.path.join(ROOT, "data", "raw_docs", "listing.tsv")
BANNED = re.compile(r"tiger-roads|microsoft|hifld|cbp|coverage-gap|usgs-structures", re.I)
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150
t0 = time.time()
rows = [l.rstrip("\n").split("\t") for l in open(LIST)][1:]
todo = []
for key, lm, etag, size in rows:
    size = int(size)
    if key.startswith("bulk-downloads/") and key.endswith(".zip"):
        continue
    if BANNED.search(key):
        print("REFUSED", key); continue
    dst = os.path.join(OUT, key)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        continue
    todo.append((size, key, dst))
todo.sort()  # small files first
print(f"{len(todo)} files left, {sum(s for s,_,_ in todo)/1e9:.2f} GB")
for size, key, dst in todo:
    left = budget - (time.time() - t0)
    if left < 5:
        break
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + ".part"
    subprocess.run(["curl", "-sS", "--fail", "-C", "-", "--max-time", str(int(left)),
                    "-o", part, BASE + key], check=False)
    got = os.path.getsize(part) if os.path.exists(part) else 0
    if got == size:
        os.rename(part, dst); print("done", key, size)
    elif got > size:
        print("OVERSIZE", key, got, size); break
    else:
        print("partial", key, got, "/", size); break
left = [k for s,k,d in todo if not (os.path.exists(d) and os.path.getsize(d)==s)]
print("remaining", len(left))
