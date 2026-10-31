"""Step 06. One feature row per scored tract (9,794), from steps 02 to 05 plus ACS and strata.
Output results/feat/tract_features.parquet. Grey-zone USFS building columns are kept with prefix gz_."""
import os, numpy as np, pandas as pd, duckdb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "data", "source_coop"); F = os.path.join(ROOT, "results", "feat"); G = os.path.join(ROOT, "results", "geo")
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
TOP4 = "class in ('motorway','trunk','primary','secondary')"
STRATA = ["pop_total", "ALAND", "pct_urban", "ur_class", "svi_overall", "svi_housing_transport", "cvi_overall", "ruca_primary",
          "rucc_2023", "nchs_2013", "tribal_pct", "mtbs_wildfire_burned_pct_land", "mtbs_wildfire_last_year",
          "nifc_wildfire_burned_pct_land", "nifc_wildfire_last_year", "usfs_PopCount_sum"]
GREY = ["usfs_BuildingCount_sum", "usfs_HUCount_sum", "usfs_BuildingCover_mean", "carbonplan_buildings"]

def sites(df, radius=200.0):
    """Count distinct sites: greedy clustering of points within `radius` metres, per tract."""
    if len(df) == 0: return 0
    x = df.lon.values * 111320 * np.cos(np.radians(df.lat.values)); y = df.lat.values * 110540
    used = np.zeros(len(df), bool); k = 0
    for i in range(len(df)):
        if used[i]: continue
        k += 1; used |= (x - x[i]) ** 2 + (y - y[i]) ** 2 <= radius ** 2
    return k

rows = []
for r in REGIONS:
    q = lambda s: duckdb.sql(s).df()
    t = q(f"select GEOID, area_m2, perimeter_m, edge_m from '{G}/{r}_tracts.parquet'")
    ri = q(f"""select GEOID,
        sum(len_m) filter (where {TOP4}) O_top4, sum(len_m) filter (where {TOP4} and link) O_link,
        sum(len_m) filter (where class='motorway' and not link) O_mw, sum(len_m) filter (where class='tertiary') L_tert,
        sum(len_m) filter (where fa and not link) A_in, sum(len_m) filter (where fa and not link and {TOP4}) A_top4,
        sum(len_m) filter (where fb and not fa and not link) B_in, sum(len_m) filter (where fb and not fa and not link and {TOP4}) B_top4,
        sum(len_m) filter (where fc and not fa and not fb and not link) C_in, sum(len_m) filter (where fc and not fa and not fb and not link and {TOP4}) C_top4,
        sum(len_m) filter (where not fa and not fb and not fc and not link and {TOP4}) X_top4
        from '{F}/roads_in_{r}.parquet' group by 1""")
    rb = q(f"""select GEOID, grp, band_len, band_in, n_runs from '{F}/roads_band_{r}.parquet'""")
    rbw = rb.pivot(index="GEOID", columns="grp", values=["band_len", "band_in"]); rbw.columns = [f"{a}_{b}" for a, b in rbw.columns]
    rbw = rbw.reset_index()
    r6 = q(f"""select GEOID, sum(band_len) band_len_A_b60, sum(band_in) band_in_A_b60 from '{F}/roads_band_b60_{r}.parquet' where grp='A' group by 1""")
    rbw = rbw.merge(r6, on="GEOID", how="outer")
    b = q(f"""select GEOID, sum(n) n_bld, sum(n) filter (where src='Microsoft ML Buildings') n_ms,
        sum(n) filter (where src='OpenStreetMap') n_osm, sum(n) filter (where src like 'Esri%') n_esri,
        sum(n) filter (where src like 'Google%') n_goog, sum(n) filter (where has_ms) n_has_ms,
        sum(a) a_bld, sum(a) filter (where src<>'Microsoft ML Buildings') a_nonms,
        sum(n) filter (where src<>'Microsoft ML Buildings' and sz in ('s4','s5')) n_nonms_big,
        sum(n) filter (where src<>'Microsoft ML Buildings' and sz='s5') n_nonms_huge,
        sum(n) filter (where src='Microsoft ML Buildings' and sz in ('s0','s1')) n_ms_small,
        sum(n) filter (where src<>'Microsoft ML Buildings' and sz in ('s0','s1')) n_nonms_small,
        sum(a) filter (where src<>'Microsoft ML Buildings' and sz in ('s4','s5')) a_nonms_big,
        sum(n) filter (where cls in ('apartments','commercial','retail','industrial','warehouse','office')) n_cls_big,
        sum(n) filter (where cls in ('static_caravan')) n_caravan, sum(n) filter (where cls in ('house','detached','residential','semidetached_house')) n_house,
        sum(n) filter (where cls in ('garage','garages','shed','carport')) n_outbld,
        sum(n) filter (where cls='fire_station') n_bfire, sum(n) filter (where cls in ('school','kindergarten')) n_bschool,
        sum(n) filter (where sub in ('commercial','industrial','service')) n_nonres
        from '{F}/bld_{r}.parquet' group by 1""")
    p = q(f"""select GEOID, sum(n) n_poi, sum(n50) n_poi50, sum(n) filter (where cat='') n_poi_nocat, sum(n_closed) n_poi_closed
        from '{F}/poi_{r}.parquet' group by 1""")
    fac = q(f"select * from '{F}/fac_{r}.parquet'")
    fr = []
    for gid, d in fac.groupby("GEOID"):
        e = {"GEOID": gid}
        for k in ["fire", "ems", "sch"]:
            cat = d[d[f"is_{k}"]]; ev = d[d[f"is_{k}"] | d[f"alt_{k}"] | d[f"nm_{k}"]]
            e[f"O_{k}"] = len(cat); e[f"O_{k}50"] = int((cat.conf >= 0.5).sum())
            e[f"ev_{k}"] = len(ev); e[f"site_{k}"] = sites(ev); e[f"sitecat_{k}"] = sites(cat)
            e[f"evx_{k}"] = int((~d[f"is_{k}"] & (d[f"alt_{k}"] | d[f"nm_{k}"])).sum())
        fr.append(e)
    fr = pd.DataFrame(fr)
    acs = q(f"select GEOID, housing_units hu from '{SC}/reference/{r}/{r}-census-acs-housing.parquet'")
    st = q(f"select GEOID, {', '.join(STRATA + GREY)} from '{SC}/strata/{r}/{r}-strata-tract-table.parquet'")
    st = st.rename(columns={c: "gz_" + c for c in GREY})
    d = t
    for x in [ri, rbw, b, p, fr, acs, st]:
        d = d.merge(x, on="GEOID", how="left")
    d.insert(1, "region", r)
    rows.append(d)
D = pd.concat(rows, ignore_index=True)
num = D.select_dtypes("number").columns.difference([c for c in D.columns if c.startswith("gz_") or c in STRATA])
D[num] = D[num].fillna(0)
D.to_parquet(os.path.join(F, "tract_features.parquet"), index=False)
print(D.shape); print(D.groupby("region")[["O_top4", "A_in", "n_bld", "n_ms", "n_poi", "O_fire", "site_fire", "O_ems", "O_sch", "site_sch", "hu", "pop_total"]].sum().round(0).to_string())
