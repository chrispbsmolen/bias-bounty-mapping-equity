"""Discovery step 06. Models of the miss rate (SNAP store not found in Overture places).

Unit is the store. Sample is storefront stores (delivery routes, farmers markets, commissaries and co-ops
dropped) in tracts with an SVI value. Main model is a linear probability model of miss (0 or 1) with
county fixed effects and standard errors clustered by county, so coefficients read as percentage points.
A logit with county fixed effects is the check. Every robustness row refits the main model with one
change. Outputs results/discovery/effects_main.csv, effects_logit.csv, effects_robust.csv,
effects_interaction.csv."""
import os, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "discovery")
FORMAT = {"Supermarket": "large grocery", "Super Store": "large grocery", "Large Grocery Store": "large grocery",
          "Medium Grocery Store": "small or medium grocery", "Small Grocery Store": "small or medium grocery",
          "Convenience Store": "convenience", "Combination Grocery/Other": "combination (dollar, pharmacy)"}

def prep():
    s = pd.read_parquet(os.path.join(OUT, "stores.parquet"))
    s = s[s.storefront & s.svi_overall.notna() & s.svi_minority.notna()].copy()
    s["format"] = s.store_type.map(FORMAT).fillna("specialty")
    s["tier"] = pd.Categorical(s.tier, ["national chain", "small chain", "independent"])
    s["format"] = pd.Categorical(s.format, ["large grocery", "combination (dollar, pharmacy)", "convenience",
                                            "small or medium grocery", "specialty"])
    s["ruca"] = pd.cut(s.ruca_primary, [0, 3.5, 6.5, 9.5, 10.5], labels=["metro", "micro", "small town", "rural"])
    s["logdens"] = np.log10(s.dens.clip(lower=1))
    for v in ["svi_overall", "svi_minority", "svi_socioeconomic"]:
        s[v + "_q"] = pd.cut(s[v], [-0.01, 0.25, 0.5, 0.75, 1.01], labels=["Q1", "Q2", "Q3", "Q4"])
    s["cvi_climate_q"] = pd.qcut(s.cvi_climate, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    s["reservation"] = (s.tribal_legal_pct.fillna(0) >= 0.5).astype(int)
    s["otsa"] = ((s.tribal_stat_pct.fillna(0) >= 0.5) & (s.reservation == 0)).astype(int)
    n_act = s.groupby("GEOID").snap_id.transform("size")
    s["churn"] = s.n_ended / (s.n_ended + n_act)
    s["new_store"] = (~s.established.astype(bool)).astype(int)
    for c in [c for c in s.columns if c.startswith("m_")]:
        s["miss" + c[1:]] = 1 - s[c]
    return s

BASE = ("C(tier) + C(format) + logdens + C(ruca) + C(svi_minority_q) + C(svi_socioeconomic_q)"
        " + reservation + otsa + new_store + churn")

def lpm(s, y="miss_main", rhs=BASE, fe="county"):
    f = f"{y} ~ {rhs}" + (f" + C({fe})" if fe else "")
    m = smf.ols(f, data=s).fit(cov_type="cluster", cov_kwds={"groups": s["county"].astype("category").cat.codes})
    t = pd.DataFrame({"coef": m.params, "lo": m.conf_int()[0], "hi": m.conf_int()[1], "p": m.pvalues})
    return t[~t.index.str.startswith("C(county") & ~t.index.str.startswith("C(region")], m

def main():
    s = prep()
    print("n", len(s), "miss", s.miss_main.mean().round(4), "counties", s.county.nunique())
    t, m = lpm(s)
    (t * [100, 100, 100, 1]).round(3).to_csv(os.path.join(OUT, "effects_main.csv"))
    print((t * [100, 100, 100, 1]).round(2).to_string())
    # same model without fixed effects, and with region effects only
    t0, _ = lpm(s, fe=None)
    t1, _ = lpm(s, fe="region")
    # logit with county fixed effects (counties where every store matched or every store missed are dropped)
    v = s.groupby("county").miss_main.transform("std") > 0
    sl = s[v]
    lg = smf.glm(f"miss_main ~ {BASE} + C(county)", data=sl, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": sl["county"].astype("category").cat.codes})
    L = pd.DataFrame({"OR": np.exp(lg.params), "lo": np.exp(lg.conf_int()[0]), "hi": np.exp(lg.conf_int()[1]), "p": lg.pvalues})
    L = L[~L.index.str.startswith("C(county")]
    L.round(3).to_csv(os.path.join(OUT, "effects_logit.csv"))
    print(L.round(2).to_string())
    # robustness rows. Key terms only
    keys = ["C(tier)[T.independent]", "C(tier)[T.small chain]", "C(svi_minority_q)[T.Q4]", "C(svi_socioeconomic_q)[T.Q4]",
            "reservation", "otsa", "logdens"]
    rows = []
    def add(name, tt, n):
        r = {"check": name, "n": n}
        for k in keys:
            r[k] = tt.coef.get(k, np.nan) * 100
            r[k + "_lo"] = tt.lo.get(k, np.nan) * 100
            r[k + "_hi"] = tt.hi.get(k, np.nan) * 100
        rows.append(r)
    add("main (250 m, name 85, food address)", t, len(s))
    add("no fixed effects", t0, len(s))
    add("region effects instead of county", t1, len(s))
    for y, nm in [("miss_r100", "radius 100 m"), ("miss_r150", "radius 150 m"), ("miss_r500", "radius 500 m"),
                  ("miss_t75", "name threshold 75"), ("miss_t95", "name threshold 95"),
                  ("miss_nameonly", "name only, no address rule"), ("miss_food", "matched place must be food retail"),
                  ("miss_addrany", "address match to any place counts"), ("miss_lenient", "lenient, 2 km and name 90"),
                  ("miss_food30", "main or any food place within 30 m")]:
        tt, _ = lpm(s, y=y)
        add(nm, tt, len(s))
    tt, _ = lpm(s[s.established.astype(bool)], rhs=BASE.replace(" + new_store", ""))
    add("stores authorized before 2024 only", tt, int(s.established.sum()))
    tt, _ = lpm(s[s.region != "south-central-tx"])
    add("without South-Central Texas", tt, int((s.region != "south-central-tx").sum()))
    tt, _ = lpm(s[s.ruca == "metro"], rhs=BASE.replace(" + C(ruca)", ""))
    add("metro tracts only", tt, int((s.ruca == "metro").sum()))
    tt, _ = lpm(s, rhs=BASE.replace("C(svi_minority_q) + C(svi_socioeconomic_q)", "C(svi_overall_q)"))
    r = {"check": "overall SVI instead of themes", "n": len(s)}
    for k in keys:
        kk = k.replace("svi_minority_q", "svi_overall_q") if "minority" in k else k
        r[k] = tt.coef.get(kk, np.nan) * 100; r[k + "_lo"] = tt.lo.get(kk, np.nan) * 100; r[k + "_hi"] = tt.hi.get(kk, np.nan) * 100
    rows.append(r)
    R = pd.DataFrame(rows)
    R.round(2).to_csv(os.path.join(OUT, "effects_robust.csv"), index=False)
    print(R[["check", "n"] + keys].round(1).to_string())
    # interaction. Does the minority gradient sit with independents
    ti, _ = lpm(s, rhs=BASE.replace("C(svi_minority_q)", "C(tier) * C(svi_minority_q)"))
    ti = ti[ti.index.str.contains("svi_minority")]
    (ti * [100, 100, 100, 1]).round(3).to_csv(os.path.join(OUT, "effects_interaction.csv"))
    print((ti * [100, 100, 100, 1]).round(2).to_string())
    # climate hazard, added to the main model one at a time. CVI climate is close to constant within a
    # county, so every hazard term is fitted with region effects, and the two continuous ones also with county effects
    s["usfs_WHP_z"] = (s.usfs_WHP_mean - s.usfs_WHP_mean.mean()) / s.usfs_WHP_mean.std()
    s["heat_z"] = (s.epht_heat_days_summer - s.epht_heat_days_summer.mean()) / s.epht_heat_days_summer.std()
    hz = []
    for v, k, lab in [("C(cvi_climate_q)", "C(cvi_climate_q)[T.Q4]", "CVI climate Q4 vs Q1"),
                      ("usfs_WHP_z", "usfs_WHP_z", "wildfire hazard, per SD"),
                      ("heat_z", "heat_z", "summer heat days, per SD")]:
        col = v.replace("C(", "").replace(")", "")
        ss = s.dropna(subset=[col])
        for fe in ["region", "county"]:
            if fe == "county" and v.startswith("C("):
                continue
            tt, _ = lpm(ss, rhs=BASE + " + " + v, fe=fe)
            hz.append({"term": lab, "effects": fe, "coef_pp": tt.coef[k] * 100, "lo": tt.lo[k] * 100,
                       "hi": tt.hi[k] * 100, "p": tt.p[k], "n": len(ss)})
    H = pd.DataFrame(hz)
    H.round(3).to_csv(os.path.join(OUT, "effects_hazard.csv"), index=False)
    print(H.round(2).to_string())

if __name__ == "__main__":
    main()
