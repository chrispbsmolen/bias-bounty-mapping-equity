"""Discovery step 09. Direct test of the online presence mechanism with ACS 2020-2024 tract tables.

Inputs  data/external/acs/acsdt5y2024-b28002.dat (internet subscription), acsdt5y2024-c16002.dat (limited English)
        results/discovery/stores.parquet via d06_models.prep()
Tract measures
  no_sub        share of households without an internet subscription, (E012 + E013) / E001 of B28002
  no_broadband  share of households without a broadband subscription of any type, 1 - E004 / E001 of B28002
  lim_eng       share of limited English speaking households, (E004 + E007 + E010 + E013) / E001 of C16002
Questions
  1. Within county, does the miss rate of independent stores rise with no_sub and lim_eng?
  2. Does adding them to the main model shrink the SVI minority Q4 effect?
Outputs results/discovery/acs_tracts.parquet, acs_effects.csv, acs_minority_mediation.csv, acs_binned.csv"""
import os, sys, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "discovery")
ACS = os.path.join(ROOT, "data", "external", "acs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d06_models import prep, BASE

def read_tracts(table):
    d = pd.read_csv(os.path.join(ACS, f"acsdt5y2024-{table}.dat"), sep="|", dtype={"GEO_ID": str})
    d = d[d.GEO_ID.str.startswith("1400000US")].copy()
    d["GEOID"] = d.GEO_ID.str[9:]
    return d

def acs_table():
    b = read_tracts("b28002"); c = read_tracts("c16002")
    t = pd.DataFrame({"GEOID": b.GEOID, "hh_b": b.B28002_E001,
                      "no_sub": (b.B28002_E012 + b.B28002_E013) / b.B28002_E001.where(b.B28002_E001 > 0),
                      "no_broadband": 1 - b.B28002_E004 / b.B28002_E001.where(b.B28002_E001 > 0)})
    u = pd.DataFrame({"GEOID": c.GEOID, "hh_c": c.C16002_E001,
                      "lim_eng": (c.C16002_E004 + c.C16002_E007 + c.C16002_E010 + c.C16002_E013) / c.C16002_E001.where(c.C16002_E001 > 0)})
    return t.merge(u, on="GEOID", how="outer")

def fit(s, rhs, y="miss_main"):
    m = smf.ols(f"{y} ~ {rhs} + C(county)", data=s).fit(cov_type="cluster", cov_kwds={"groups": s.county.astype("category").cat.codes})
    return m

def row(m, k, label, n):
    ci = m.conf_int().loc[k]
    return {"term": label, "coef_pp": m.params[k] * 100, "lo": ci[0] * 100, "hi": ci[1] * 100, "p": m.pvalues[k], "n": n}

def main():
    a = acs_table()
    s = prep()
    tr = s[["GEOID"]].drop_duplicates().merge(a, on="GEOID", how="left")
    tr.to_parquet(os.path.join(OUT, "acs_tracts.parquet"), index=False)
    print("store tracts", len(tr), "missing ACS", int(tr.no_sub.isna().sum()))
    s = s.merge(a, on="GEOID", how="left").dropna(subset=["no_sub", "lim_eng"])
    sd = {}
    for v in ["no_sub", "no_broadband", "lim_eng"]:
        sd[v] = s.drop_duplicates("GEOID")[v].std()
        s[v + "_z"] = (s[v] - s.drop_duplicates("GEOID")[v].mean()) / sd[v]
        s[v + "_q"] = pd.qcut(s[v].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    print("tract SD of no_sub, no_broadband, lim_eng", {k: round(v, 4) for k, v in sd.items()})
    print("correlation with SVI minority (tracts)", s.drop_duplicates("GEOID")[["svi_minority", "no_sub", "no_broadband", "lim_eng"]].corr().round(2).iloc[0].to_dict())
    rows = []
    ind = s[s.tier == "independent"]; nat = s[s.tier == "national chain"]
    for lab, d in [("all stores", s), ("independent", ind), ("national chain", nat)]:
        rhs = BASE + " + no_sub_z + lim_eng_z" if lab == "all stores" else BASE.replace("C(tier) + ", "") + " + no_sub_z + lim_eng_z"
        m = fit(d, rhs)
        rows.append(row(m, "no_sub_z", f"{lab}. No internet subscription, per SD", len(d)))
        rows.append(row(m, "lim_eng_z", f"{lab}. Limited English households, per SD", len(d)))
        m = fit(d, rhs.replace("no_sub_z", "no_broadband_z"))
        rows.append(row(m, "no_broadband_z", f"{lab}. No broadband, per SD (in place of no subscription)", len(d)))
        # each alone, without the SVI minority theme, to see the raw within-county association
        for v, nm in [("no_sub_z", "No internet subscription"), ("lim_eng_z", "Limited English households")]:
            r2 = (BASE.replace("C(tier) + ", "") if lab != "all stores" else BASE).replace(" + C(svi_minority_q)", "") + " + " + v
            m = fit(d, r2)
            rows.append(row(m, v, f"{lab}. {nm}, per SD, alone and without the minority theme", len(d)))
    # interaction with tier on all stores
    m = fit(s, BASE + " + C(tier):no_sub_z + C(tier):lim_eng_z + no_sub_z + lim_eng_z")
    for k in [i for i in m.params.index if ":" in i and ("no_sub" in i or "lim_eng" in i)]:
        rows.append(row(m, k, "interaction " + k, len(s)))
    E = pd.DataFrame(rows)
    E.round(4).to_csv(os.path.join(OUT, "acs_effects.csv"), index=False)
    print(E.round(2).to_string())
    # does it explain the minority effect. Same sample, minority Q4 coefficient with and without ACS terms
    med = []
    for lab, d, base in [("all stores", s, BASE), ("independent", ind, BASE.replace("C(tier) + ", ""))]:
        for spec, rhs in [("main model", base), ("plus no internet subscription", base + " + no_sub_z"),
                          ("plus limited English", base + " + lim_eng_z"), ("plus both", base + " + no_sub_z + lim_eng_z"),
                          ("plus both as quartiles", base + " + C(no_sub_q) + C(lim_eng_q)")]:
            m = fit(d, rhs)
            r = row(m, "C(svi_minority_q)[T.Q4]", f"{lab}. {spec}", len(d)); r["sample"] = lab; r["spec"] = spec
            med.append(r)
    M = pd.DataFrame(med)
    M.round(4).to_csv(os.path.join(OUT, "acs_minority_mediation.csv"), index=False)
    print(M[["term", "coef_pp", "lo", "hi", "p"]].round(2).to_string())
    # binned raw rates for the figure
    B = []
    for v in ["no_sub", "lim_eng"]:
        g = s.groupby(["tier", v + "_q"], observed=True).miss_main.agg(["sum", "size"]).reset_index()
        g["var"] = v; g = g.rename(columns={v + "_q": "q"})
        rng = s.groupby(v + "_q", observed=True)[v].agg(["min", "max"]).reset_index().rename(columns={v + "_q": "q"})
        B.append(g.merge(rng, on="q"))
    B = pd.concat(B); B["rate"] = B["sum"] / B["size"]
    B.round(4).to_csv(os.path.join(OUT, "acs_binned.csv"), index=False)
    print(B.round(3).to_string())

if __name__ == "__main__":
    main()
