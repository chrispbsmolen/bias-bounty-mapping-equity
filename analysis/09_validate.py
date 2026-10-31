"""Step 09. Format check for every CSV in submissions/candidates against data/SampleSubmission.csv.
9,794 rows, same GEOIDs as text with leading zeros, same column order, no blanks, scores and gaps in [0, 1]."""
import os, glob, sys, pandas as pd, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = pd.read_csv(os.path.join(ROOT, "data", "SampleSubmission.csv"), dtype=str)
bad = 0
for f in sorted(glob.glob(os.path.join(ROOT, "submissions", "candidates", "*.csv"))):
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    errs = []
    if list(d.columns) != list(S.columns): errs.append("columns differ from sample")
    if len(d) != 9794: errs.append(f"{len(d)} rows")
    if not d.GEOID.str.fullmatch(r"\d{11}").all(): errs.append("GEOID not 11 digit text")
    if set(d.GEOID) != set(S.GEOID) or d.GEOID.duplicated().any(): errs.append("GEOID set differs or duplicates")
    if (d == "").any().any(): errs.append("blank cells")
    num = [c for c in S.columns if c.endswith("_gap") or c.startswith("poi_gap") or c == "coverage_gap_score"]
    x = d[num].astype(float)
    if x.isna().any().any() or (x < 0).any().any() or (x > 1).any().any(): errs.append("values outside [0, 1]")
    flags = [c for c in S.columns if "defined" in c]
    if not d[flags].isin(["True", "False"]).all().all(): errs.append("flags not True/False")
    if d.columns[1] != "coverage_gap_score": errs.append("score not column 2")
    bad += bool(errs)
    print(("FAIL " if errs else "ok   ") + os.path.basename(f), f"mean={x.coverage_gap_score.mean():.4f}", "; ".join(errs))
sys.exit(1 if bad else 0)
