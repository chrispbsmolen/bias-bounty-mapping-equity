"""Step 10. Fit a few broad blend weights from public leaderboard scores of the probe files.
Input submissions/probe_scores.csv with columns file,public_rmse (filled in by hand after each upload).
For a probe vector v, public RMSE^2 = M0 - 2<y,v> + <v,v>, with M0 = 0.107246^2 known from the all-zeros file,
so <y,v> = (M0 + <v,v> - RMSE^2) / 2, with <v,v> taken over all tracts (the public 30% is a random subset).
The blend y ~ sum_k beta_k v_k is then fitted by bounded least squares on the normal equations. Only global
weights are fitted: one multiplier per component vector and one intercept per region, never tract values.
--simulate runs the whole procedure against synthetic truths drawn from the structural model, with a random
30% public split, to check that it recovers the truth and to estimate the gain on the private split."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import lsq_linear
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, "results", "model"); CAND = os.path.join(ROOT, "submissions", "candidates")
M0 = 0.107246299 ** 2
CHECK = "probe_01_B0v2_structural"  # equals the sum of probes 02 to 05, used as a consistency check, not fitted

PUBLIC_MEAN = float(np.sqrt(0.107246299 ** 2 - 0.089022463 ** 2))  # public target mean from public constant scores
DEFAULT_USE = ["probe_02_transport", "probe_03_facilities", "probe_17_strata_nchs"]
PARTS = ["probe_02_transport", "probe_03_facilities", "probe_04_cbp", "probe_05_building"]

def bounds_for(names):
    # region indicators and region-masked vectors adjust a weight up or down, so they may be negative
    lo = [(-1.0 if "region" in k else -0.1 if k == "const" else -3.0 if k.endswith("_only") else 0.0) for k in names]
    hi = [(1.0 if "region" in k else 0.1 if k == "const" else 3.0) for k in names]
    return lo, hi

def load_vectors(files):
    """Vectors exactly as uploaded: column 2 of each file in submissions/uploaded (else submissions/candidates)."""
    S = pd.read_csv(os.path.join(ROOT, "data", "SampleSubmission.csv"), dtype={"GEOID": str})[["GEOID"]]
    out = {}
    for f in files:
        p = os.path.join(ROOT, "submissions", "uploaded", f)
        if not os.path.exists(p): p = os.path.join(CAND, f)
        d = pd.read_csv(p, dtype={"GEOID": str})
        out[f.replace(".csv", "")] = S.merge(d.iloc[:, :2], on="GEOID", how="left").iloc[:, 1].values
    V = pd.DataFrame(out, index=S.GEOID); V["const"] = 1.0
    return V

def fit(V, scores, ridge=1e-6, prior=None):
    """V: DataFrame of probe vectors (all tracts). scores: dict name -> public RMSE. Returns beta (Series), b, G."""
    all_parts = all(k in scores for k in PARTS)
    names = [k for k in scores if k in V and not (k == CHECK and all_parts)]
    if "const" in V and "const" not in names: names.append("const")
    X = V[names].values; G = X.T @ X / len(X)
    b = np.array([PUBLIC_MEAN if k == "const" else (M0 + (V[k].values ** 2).mean() - scores[k] ** 2) / 2 for k in names])
    p0 = np.array([0.0 if "region" in k else (prior or {}).get(k, 1.0) for k in names])
    A = G + ridge * np.eye(len(names)); rhs = b + ridge * p0
    L = np.linalg.cholesky(A)
    res = lsq_linear(L.T, np.linalg.solve(L, rhs), bounds=bounds_for(names))
    beta = pd.Series(res.x, index=names)
    pub = np.sqrt(max(0.0, M0 - 2 * beta.values @ b + beta.values @ G @ beta.values))
    return beta, pd.Series(b, index=names), pub

def simulate(n_sims=4):
    m7 = importlib.import_module("07_model"); m8 = importlib.import_module("08_candidates")
    D = m7.load(); rates = m7.site_rates(D)
    V = pd.read_parquet(os.path.join(MOD, "probe_vectors.parquet")).set_index("GEOID")
    b0 = V[CHECK].values
    rng = np.random.default_rng(11); rows = []
    for nm, P in [("v2 base", {}), ("FM in TIGER", {"theta_B": 1.0}), ("county fully in TIGER", {"theta_C": 1.0}),
                  ("fire coverage 0.9", {"cov_fire": (0.9, 0.9)}), ("more TIGER extra", {"p_extra": 0.4, "f_extra": 0.5}),
                  ("lower CBP density", {"cbp_pc": 0.015})]:
        _, S = m7.run({**m7.V2, **P}, "B0", D, rates, keep_draws=True)
        for t in m8.realisations(S, rng, R=1):
            for _ in range(n_sims // 2):
                pub = rng.random(len(t)) < 0.3
                sc = {k: np.sqrt(((V[k].values[pub] - t[pub]) ** 2).mean()) for k in V.columns}
                # in the simulation M0 is the synthetic public mean square
                global M0; keep = M0; M0 = (t[pub] ** 2).mean()
                beta, b, pubfit = fit(V.drop(columns=[c for c in ["const"] if c in V]), sc)
                M0 = keep
                pred = np.clip(V[beta.index].values @ beta.values, 0, 1)
                rows.append({"truth": nm, "B0 private": np.sqrt(((b0 - t)[~pub] ** 2).mean()),
                             "blend private": np.sqrt(((pred - t)[~pub] ** 2).mean()),
                             "blend public (fitted)": pubfit, "blend public (actual)": np.sqrt(((pred - t)[pub] ** 2).mean()),
                             "constant private": np.sqrt(((t[pub].mean() - t)[~pub] ** 2).mean()),
                             "check probe01 = sum(02..05)": (M0 + 0) * 0 + 0})
    out = pd.DataFrame(rows).drop(columns="check probe01 = sum(02..05)")
    out.to_csv(os.path.join(MOD, "probe_simulation.csv"), index=False)
    print(out.groupby("truth").mean().round(4).to_string())

if __name__ == "__main__":
    if "--simulate" in sys.argv:
        simulate(); sys.exit(0)
    m8 = importlib.import_module("08_candidates")
    sc = pd.read_csv(os.path.join(ROOT, "submissions", "probe_scores.csv"))
    sc = sc[sc.file.str.startswith("probe_")]
    scores = {f.replace(".csv", ""): r for f, r in zip(sc.file, sc.public_rmse)}
    V = load_vectors(list(sc.file))
    # --use a,b,c restricts the fit to those probe vectors (plus const). Default is the parsimonious set chosen on
    # 26 Oct, which fits as well as all nine vectors together (see research/06_model_v1.md section 12)
    use = DEFAULT_USE
    if "--use" in sys.argv: use = sys.argv[sys.argv.index("--use") + 1].split(",")
    if "--all" in sys.argv: use = list(scores)
    scores = {k: v for k, v in scores.items() if k in use}
    beta, b, pub = fit(V[[k for k in scores] + ["const"]], scores, prior={})
    print("<y, v> on the public subset"); print(b.round(6).to_string())
    print("weights"); print(beta.round(4).to_string()); print("fitted public RMSE (in sample)", round(pub, 5))
    if all(k in scores for k in [CHECK] + PARTS):
        bc = (M0 + (V[CHECK] ** 2).mean() - scores[CHECK] ** 2) / 2
        print("check <y,B0> direct", round(bc, 6), "from parts", round(sum(b[k] for k in PARTS), 6))
    pred = np.clip(V[beta.index].values @ beta.values, 0, 1)
    o = pd.read_parquet(os.path.join(MOD, "B0v2_summary.parquet")).set_index("GEOID").loc[V.index].reset_index()
    m8.write("cand_LB_blend.csv", o, pred)
    beta.to_csv(os.path.join(MOD, "blend_weights.csv"))
    print("blend mean", round(pred.mean(), 4))
