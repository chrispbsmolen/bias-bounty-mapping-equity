"""Step 16. Refit the leaderboard blend after new probe scores land (made 27 Oct).
1. Copies every scored file named in submissions/probe_scores.csv into submissions/uploaded/ if not there yet.
2. Prints, for every scored probe, mean(v y), covariance and correlation with the target, and its best affine RMSE.
3. Forward selection. Starts from transport, facilities and nchs plus the intercept and adds the probe vector that
   lowers the in-sample public RMSE most, only while the gain is at least 0.0005 and at most 10 numbers are used.
4. Writes submissions/candidates/cand_LB_blend_<tag>.csv and cand_LB_blend_<tag>_shrink20.csv (20% toward the
   public mean), and results/model/blend_weights_<tag>.csv.
Usage: python3 analysis/16_refit.py <tag>        for example 2026-10-28"""
import os, sys, shutil, numpy as np, pandas as pd
from scipy.optimize import lsq_linear
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
m10 = importlib.import_module("10_fit_probes"); m8 = importlib.import_module("08_candidates")
ROOT = m10.ROOT; UP = os.path.join(ROOT, "submissions", "uploaded"); CAND = m10.CAND
MIN_GAIN, MAX_N = 0.0005, 10
START = ["probe_02_transport", "probe_03_facilities", "probe_17_strata_nchs", "const"]
EXCLUDE = {"probe_01_B0v2_structural", "probe_14_B0v1_structural"}  # composites of other probes

def main(tag):
    sc = pd.read_csv(os.path.join(ROOT, "submissions", "probe_scores.csv"))
    sc = sc[sc.file.str.endswith(".csv") & sc.public_rmse.notna()]
    for f in sc.file:
        if not os.path.exists(os.path.join(UP, f)) and os.path.exists(os.path.join(CAND, f)):
            shutil.copy2(os.path.join(CAND, f), os.path.join(UP, f)); print("copied", f, "to submissions/uploaded")
    V = m10.load_vectors(list(sc.file)); M0, yb = m10.M0, m10.PUBLIC_MEAN
    scores = {f[:-4]: r for f, r in zip(sc.file, sc.public_rmse)}
    b = {k: (M0 + (V[k] ** 2).mean() - r ** 2) / 2 for k, r in scores.items()}; b["const"] = yb
    rows = []
    for k in scores:
        v = V[k].values; cov = b[k] - yb * v.mean()
        rows.append({"file": k, "public_rmse": scores[k], "mean_vy": b[k], "cov": cov, "corr": cov / (v.std() * 0.089),
                     "best_affine_rmse": np.sqrt(max(0, M0 - yb ** 2 - cov ** 2 / v.var()))})
    print(pd.DataFrame(rows).set_index("file").round(5).to_string())
    def fit(names):
        X = V[names].values; G = X.T @ X / len(X); bb = np.array([b[n] for n in names])
        L = np.linalg.cholesky(G + 1e-9 * np.eye(len(names))); lo, hi = m10.bounds_for(names)
        w = lsq_linear(L.T, np.linalg.solve(L, bb), bounds=(lo, hi)).x
        return np.sqrt(max(0, M0 - 2 * w @ bb + w @ G @ w)), w
    pool = [k for k in scores if k.startswith("probe_") and k not in EXCLUDE and k not in START]
    cur = list(START); r, w = fit(cur); print("start", round(r, 5))
    while len(cur) < MAX_N and pool:
        trials = {k: fit(cur + [k])[0] for k in pool}
        best = min(trials, key=trials.get)
        if r - trials[best] < MIN_GAIN: break
        cur.append(best); pool.remove(best); r, w = fit(cur); print("add", best, round(r, 5))
    beta = pd.Series(w, index=cur); print(beta.round(4).to_string()); print("in-sample public RMSE", round(r, 5))
    pred = np.clip(V[cur].values @ w, 0, 1)
    o = pd.read_parquet(os.path.join(m10.MOD, "B0v2_summary.parquet")).set_index("GEOID").loc[V.index].reset_index()
    m8.write(f"cand_LB_blend_{tag}.csv", o, pred)
    m8.write(f"cand_LB_blend_{tag}_shrink20.csv", o, np.clip(yb + 0.8 * (pred - yb), 0, 1))
    beta.to_csv(os.path.join(m10.MOD, f"blend_weights_{tag}.csv"))
    print("written", f"cand_LB_blend_{tag}.csv", "and _shrink20, mean", round(pred.mean(), 4))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
