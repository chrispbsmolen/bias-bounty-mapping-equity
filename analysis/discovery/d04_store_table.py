"""Discovery step 04. One row per active SNAP store with match flags, the matched Overture record and
tract covariates from the provided strata tables.

Main match rule. An Overture place within 250 m whose name similarity is at least 85 (any category),
or a food retail Overture place within 250 m whose street number and street name agree with the SNAP
address (this catches a convenience store listed only under its fuel brand, such as Shell or Valero).
An address match to a non-food place (a cash machine or a nail salon in the same plaza) does not count.
Variants are written as extra columns for the robustness checks.
Outputs results/discovery/stores.parquet and results/discovery/tracts.parquet."""
import os, glob, duckdb, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SC = os.path.join(ROOT, "data", "source_coop")
OUT = os.path.join(ROOT, "results", "discovery")
SNAP = os.path.join(ROOT, "data", "external", "snap", "snap_retailers_2005_2025.csv")
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
# name: (radius m, name threshold, address rule, require food category for every match)
# address rule "food" means an address match counts only for a food retail place, "any" for any place, "none" never
VARIANTS = {"m_main": (250, 85, "food", False), "m_r100": (100, 85, "food", False), "m_r150": (150, 85, "food", False),
            "m_r500": (500, 85, "food", False), "m_t75": (250, 75, "food", False), "m_t95": (250, 95, "food", False),
            "m_nameonly": (250, 85, "none", False), "m_food": (250, 85, "food", True),
            "m_addrany": (250, 85, "any", False), "m_lenient": (2000, 90, "food", False)}
STRATA = ["GEOID", "STATEFP", "COUNTYFP", "ALAND", "pop_total", "pct_urban", "ur_class", "svi_overall",
          "svi_socioeconomic", "svi_household", "svi_minority", "svi_housing_transport", "cvi_overall", "cvi_climate",
          "ruca_primary", "tribal_any", "tribal_pct", "tribal_legal_pct", "tribal_stat_pct", "aiannh_name", "usfs_WHP_mean", "usfs_RPS_mean",
          "epht_heat_days_summer", "hwd_heatindex_mean", "usdm_summer_dsci", "nchs_2013"]

def tract_table(con):
    parts = []
    for r in REGIONS:
        st = os.path.join(SC, "strata", r, f"{r}-strata-tract-table.parquet")
        d = con.sql(f"select {', '.join(STRATA)} from '{st}'").df()
        d["region"] = r
        parts.append(d)
    t = pd.concat(parts, ignore_index=True)
    t["county"] = t.STATEFP.astype(str) + t.COUNTYFP.astype(str)
    t["dens"] = t.pop_total / (t.ALAND / 1e6).clip(lower=0.01)
    # Overture place count per tract (all categories, the numerator of the score's establishment half)
    cnt = []
    for r in REGIONS:
        tr = os.path.join(SC, "strata", r, f"{r}-census-tracts.parquet")
        q = f"""select t.GEOID, count(p.id) n_overture from (select GEOID, geometry::GEOMETRY g from '{tr}') t
            left join (select id, ST_Point(lon, lat) pt from '{OUT}/pois_{r}.parquet') p on ST_Contains(t.g, p.pt)
            group by 1"""
        cnt.append(con.sql(q).df())
    t = t.merge(pd.concat(cnt), on="GEOID", how="left")
    # SNAP churn. Stores in the tract whose authorization ended in 2024 or 2025
    con.sql(f"""create or replace table ended as select try_cast(Latitude as double) lat, try_cast(Longitude as double) lon
        from read_csv('{SNAP}', all_varchar=true, header=true)
        where State in ('OK','WA','AZ','CA','TX','NM') and trim(coalesce("End Date",'')) <> ''
        and year(strptime("End Date", '%m/%d/%Y')) >= 2024""")
    ch = []
    for r in REGIONS:
        tr = os.path.join(SC, "strata", r, f"{r}-census-tracts.parquet")
        ch.append(con.sql(f"""select t.GEOID, count(*) n_ended from (select GEOID, geometry::GEOMETRY g from '{tr}') t
            join ended e on ST_Contains(t.g, ST_Point(e.lon, e.lat)) group by 1""").df())
    t = t.merge(pd.concat(ch), on="GEOID", how="left")
    t["n_ended"] = t.n_ended.fillna(0)
    return t

def main():
    con = duckdb.connect(config={"memory_limit": "1500MB", "threads": 2})
    con.sql("LOAD spatial")
    s = pd.read_parquet(os.path.join(OUT, "snap_active.parquet"))
    pr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "pairs_*.parquet")))], ignore_index=True)
    nf = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "nearest_food_*.parquet")))])
    for col, (R, T, use_addr, food) in VARIANTS.items():
        am = pr.addr_match & (pr.food_cat if use_addr == "food" else (use_addr == "any"))
        ok = (pr.dist_m <= R) & ((pr.name_sim >= T) | am)
        if food:
            ok &= pr.food_cat
        s[col] = s.snap_id.isin(pr.loc[ok, "snap_id"]).astype(int)
    ok = (pr.dist_m <= 250) & ((pr.name_sim >= 85) | (pr.addr_match & pr.food_cat))
    best = pr[ok].sort_values(["snap_id", "name_sim", "dist_m"], ascending=[True, False, True]).drop_duplicates("snap_id")
    best = best[["snap_id", "poi_id", "poi_name", "cat", "dataset", "confidence", "n_web", "n_phone", "n_social",
                 "brand_wd", "food_cat", "dist_m", "name_sim"]].add_prefix("ov_").rename(columns={"ov_snap_id": "snap_id"})
    s = s.merge(best, on="snap_id", how="left").merge(nf, on="snap_id", how="left")
    s["m_food30"] = ((s.m_main == 1) | (s.near_food_m <= 30)).astype(int)
    s["tier"] = np.where(s.chain_n_national >= 100, "national chain", np.where(s.chain_n_national <= 2, "independent", "small chain"))
    s["storefront"] = ~s.store_type.isin(["Delivery Route", "Farmers' Market", "Military Commissary", "Food Buying Co-op"])
    s["established"] = s.auth_date < pd.Timestamp("2024-01-01")
    t = tract_table(con)
    t.to_parquet(os.path.join(OUT, "tracts.parquet"), index=False)
    s = s.merge(t, on=["GEOID", "region"], how="left")
    s.to_parquet(os.path.join(OUT, "stores.parquet"), index=False)
    print(len(s), s[[c for c in VARIANTS]].mean().round(4).to_dict())
    print(s.isna().sum()[lambda x: x > 0].to_dict())

if __name__ == "__main__":
    main()
