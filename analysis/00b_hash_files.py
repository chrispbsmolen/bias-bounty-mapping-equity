"""Compute sha256 for every mirrored file (skipping files over 1 GB) into data/raw_docs/sha256.tsv. Resumable."""
import os, sys, time, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "source_coop"); H = os.path.join(ROOT, "data", "raw_docs", "sha256.tsv")
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150; t0 = time.time()
done = {}
if os.path.exists(H):
    for l in open(H): k, s, h = l.rstrip("\n").split("\t"); done[k] = h
rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "data", "raw_docs", "listing.tsv"))][1:]
with open(H, "a") as f:
    for key, lm, etag, size in rows:
        p = os.path.join(OUT, key)
        if key in done or not os.path.exists(p): continue
        if time.time() - t0 > budget: break
        if int(size) > 1e9: h = "skipped (over 1 GB)"
        else:
            m = hashlib.sha256()
            with open(p, "rb") as g:
                for b in iter(lambda: g.read(1 << 22), b""): m.update(b)
            h = m.hexdigest()
        f.write(f"{key}\t{size}\t{h}\n"); f.flush()
n = sum(1 for _ in open(H)); print("hashed", n)
