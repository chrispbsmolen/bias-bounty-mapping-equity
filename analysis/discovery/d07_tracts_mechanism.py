"""Discovery step 07. Tract summaries, mechanism checks and the named places.

Outputs in results/discovery/
  tract_recall.csv          one row per scored tract with at least one SNAP storefront
  mechanism_source_mix.csv  upstream Overture source of matched stores by chain tier and SVI minority quartile
  mechanism_spanish.csv     recall of independents with and without Spanish words in the store name
  named_tracts.csv          tracts with the most missed stores, and tracts that look empty of food retail
  summary.json              headline numbers quoted in research/07_bias_discovery.md"""
import os, sys, json, re, duckdb, numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "discovery")
SC = os.path.join(ROOT, "data", "source_coop")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d06_models import prep, BASE
from d03_match import FOOD_TAX, FOOD_CAT, norm
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
SPANISH = {"tienda", "carniceria", "supermercado", "mercado", "panaderia", "abarrotes", "fruteria", "tortilleria",
           "raspas", "dulceria", "paleteria", "minisuper", "super", "el", "la", "los", "las", "y", "del", "de",
           "michoacana", "mexicana", "mexico", "latino", "latina", "hispana", "botanica", "pulga"}

def food_counts(con):
    tax = ", ".join(f"'{x}'" for x in FOOD_TAX)
    cat = ", ".join(f"'{x}'" for x in FOOD_CAT)
    parts = []
    for r in REGIONS:
        tr = os.path.join(SC, "strata", r, f"{r}-census-tracts.parquet")
        parts.append(con.sql(f"""select t.GEOID, count(p.id) n_overture_food from (select GEOID, geometry::GEOMETRY g from '{tr}') t
            left join (select id, ST_Point(lon, lat) pt from '{OUT}/pois_{r}.parquet'
                       where cat in ({cat}) or len(list_intersect(tax, [{tax}])) > 0) p on ST_Contains(t.g, p.pt)
            group by 1""").df())
    return pd.concat(parts)

def main():
    con = duckdb.connect(config={"memory_limit": "1500MB", "threads": 2})
    con.sql("LOAD spatial")
    s = prep()
    S = {}
    S["n_stores"] = int(len(s)); S["miss_main"] = float(s.miss_main.mean()); S["miss_lenient"] = float(s.miss_lenient.mean())
    S["miss_by_tier"] = s.groupby("tier", observed=True).miss_main.mean().round(4).to_dict()
    S["miss_lenient_by_tier"] = s.groupby("tier", observed=True).miss_lenient.mean().round(4).to_dict()
    S["n_by_tier"] = s.groupby("tier", observed=True).size().to_dict()
    S["miss_by_minq_tier"] = {f"{a}|{b}": round(v, 4) for (a, b), v in s.groupby(["tier", "svi_minority_q"], observed=True).miss_main.mean().items()}
    # source mix of matched stores
    m = s[s.m_main == 1]
    mix = pd.crosstab([m.tier, m.svi_minority_q], m.ov_dataset, normalize="index").round(4)
    mix.to_csv(os.path.join(OUT, "mechanism_source_mix.csv"))
    S["meta_share_matched_by_tier"] = (m.ov_dataset == "meta").groupby(m.tier, observed=True).mean().round(4).to_dict()
    S["atp_share_matched_by_tier"] = (m.ov_dataset == "AllThePlaces").groupby(m.tier, observed=True).mean().round(4).to_dict()
    S["web_share_matched_by_tier"] = (m.ov_n_web > 0).groupby(m.tier, observed=True).mean().round(4).to_dict()
    # Spanish names among independents
    s["spanish"] = [bool(set(norm(x).split()) & SPANISH) for x in s.snap_name]
    ind = s[s.tier == "independent"]
    sp = ind.groupby(["spanish", "svi_minority_q"], observed=True).agg(miss=("miss_main", "mean"), miss_len=("miss_lenient", "mean"), n=("miss_main", "size"))
    sp.round(4).to_csv(os.path.join(OUT, "mechanism_spanish.csv"))
    f = "miss_main ~ spanish + C(format) + logdens + C(ruca) + C(svi_minority_q) + C(svi_socioeconomic_q) + reservation + otsa + new_store + churn + C(county)"
    mm = smf.ols(f, data=ind).fit(cov_type="cluster", cov_kwds={"groups": ind.county.astype("category").cat.codes})
    S["spanish_name_effect_pp_indep"] = [round(mm.params["spanish[T.True]"] * 100, 2)] + [round(x * 100, 2) for x in mm.conf_int().loc["spanish[T.True]"]]
    mm2 = smf.ols(f.replace("spanish + ", ""), data=ind[~ind.spanish]).fit(cov_type="cluster", cov_kwds={"groups": ind[~ind.spanish].county.astype("category").cat.codes})
    S["minQ4_effect_pp_indep_nonspanish"] = [round(mm2.params["C(svi_minority_q)[T.Q4]"] * 100, 2)] + [round(x * 100, 2) for x in mm2.conf_int().loc["C(svi_minority_q)[T.Q4]"]]
    S["spanish_share_indep_by_minq"] = ind.groupby("svi_minority_q", observed=True).spanish.mean().round(4).to_dict()
    # tract level
    t = pd.read_parquet(os.path.join(OUT, "tracts.parquet"))
    t = t.merge(food_counts(con), on="GEOID", how="left")
    g = s.groupby("GEOID").agg(n_snap=("snap_id", "size"), n_found=("m_main", "sum"), n_found_len=("m_lenient", "sum"),
                               n_indep=("tier", lambda x: (x == "independent").sum()))
    tr = t.merge(g, on="GEOID", how="inner")
    tr["recall"] = tr.n_found / tr.n_snap
    tr["n_missed"] = tr.n_snap - tr.n_found
    tr["overture_per_1k"] = tr.n_overture / (tr.pop_total.clip(lower=1) / 1000)
    tr.to_csv(os.path.join(OUT, "tract_recall.csv"), index=False)
    big = tr[tr.n_snap >= 3]
    S["n_tracts_with_snap"] = int(len(tr)); S["n_tracts_3plus"] = int(len(big))
    S["spearman_recall_vs_overture_per_1k"] = [round(x, 4) for x in spearmanr(big.recall, big.overture_per_1k)]
    S["spearman_recall_vs_svi_minority"] = [round(x, 4) for x in spearmanr(big.recall, big.svi_minority, nan_policy="omit")]
    # tracts that look empty in Overture but have SNAP stores
    empty = tr[(tr.n_found == 0)]
    S["tracts_all_snap_missed_main"] = int(len(empty)); S["pop_all_snap_missed_main"] = int(empty.pop_total.sum())
    S["stores_in_all_missed_tracts"] = int(empty.n_snap.sum())
    empty_len = tr[tr.n_found_len == 0]
    S["tracts_all_snap_missed_lenient"] = int(len(empty_len)); S["pop_all_snap_missed_lenient"] = int(empty_len.pop_total.sum())
    nofood = tr[tr.n_overture_food == 0]
    S["tracts_no_overture_food_but_snap"] = int(len(nofood)); S["pop_no_overture_food_but_snap"] = int(nofood.pop_total.sum())
    S["stores_in_no_overture_food_tracts"] = int(nofood.n_snap.sum())
    low = tr[(tr.n_snap >= 4) & (tr.recall <= 0.75)]
    S["tracts_4plus_recall_le_75"] = int(len(low)); S["pop_tracts_4plus_recall_le_75"] = int(low.pop_total.sum())
    S["missed_in_minQ4"] = int(s[s.svi_minority_q == "Q4"].miss_main.sum()); S["n_minQ4"] = int((s.svi_minority_q == "Q4").sum())
    S["missed_total"] = int(s.miss_main.sum())
    # named tracts
    cols = ["GEOID", "region", "county", "aiannh_name", "pop_total", "svi_minority", "svi_overall", "n_snap", "n_found",
            "n_found_len", "n_indep", "recall", "n_overture", "overture_per_1k", "n_overture_food"]
    a = tr.sort_values(["n_missed", "recall"], ascending=[False, True]).head(15)[cols + ["n_missed"]]
    a["list"] = "most missed stores"
    b = nofood.sort_values("n_snap", ascending=False).head(15)[cols + ["n_missed"]]
    b["list"] = "no food retail in Overture"
    pd.concat([a, b]).to_csv(os.path.join(OUT, "named_tracts.csv"), index=False)
    # named places
    for nm, mask in [("navajo_nation", s.aiannh_name.eq("Navajo Nation Reservation")),
                     ("tohono_oodham", s.aiannh_name.eq("Tohono O'odham Nation Reservation")),
                     ("starr_county_tx", s.county.eq("48427")), ("hidalgo_county_tx", s.county.eq("48215")),
                     ("cameron_county_tx", s.county.eq("48061")), ("reservations_all", s.reservation.eq(1))]:
        x = s[mask]
        S["place_" + nm] = {"n": int(len(x)), "miss_main": round(float(x.miss_main.mean()), 4),
                            "miss_lenient": round(float(x.miss_lenient.mean()), 4), "tracts": int(x.GEOID.nunique()),
                            "indep_share": round(float((x.tier == "independent").mean()), 4)}
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))
    print(mix.to_string()); print(sp.round(3).to_string())
    print(pd.read_csv(os.path.join(OUT, "named_tracts.csv")).round(3).to_string())

if __name__ == "__main__":
    main()
