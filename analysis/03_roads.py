"""Step 03. Road features per tract from the Overture roads layer only.
For every kept segment we flag route tiers from Overture's own routes field:
  A = Interstate, US highway, main state network (incl. Loop, Spur, Business, Truck, toll roads)
  B = state secondary systems (Texas FM, RM, Park roads; California county sign routes CR; WA secondary)
  C = county route networks (US:<state>:<County>)
  N = no A/B/C route but the primary name looks like a numbered highway
Outputs (results/feat/):
  roads_in_<region>.parquet   length inside each tract by class, link, tier flags
  roads_band_<region>.parquet for each tract and road group, the length of long runs (>= RUN m)
                              inside the 30 m boundary band, and the part of those runs inside the tract
Rows are processed in chunks of row groups so each call stays short; chunks are cached."""
import os, sys, duckdb, pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "data", "source_coop")
GEO = os.path.join(ROOT, "results", "geo"); OUT = os.path.join(ROOT, "results", "feat")
TMP = os.path.join(ROOT, "results", "tmp"); os.makedirs(TMP, exist_ok=True)
RUN = 200.0
BW = float(os.environ.get("BAND_M", "30"))  # band half width; 30 m is the main run, others are sensitivity runs
SUF = "" if BW == 30 else f"_b{int(BW)}"
ST = "(AZ|CA|NM|OK|TX|WA|ID|OR|NV|AR|KS|MO|LA|CO|UT)"
RE_A = rf"^US:I(:.*)?$|^US:US(:(Business|Alternate|Spur|Truck|Bypass|Temporary))?$|^US:{ST}(:(Loop|Spur|Business|Truck|Beltway|Toll|NTTA|Turnpike|Alternate|Bypass|Connector|Express|Harris:HCTRA))?$"
RE_EXCL = r"Historic|Old|Scenic|Named|Byway|Future|NHS|NHT|NFSR|NFRS|NSFR|NFSF|BIA|Trunk|THC|DOE|MX|CA:"
RE_B = rf"^US:TX:(FM|RM|Park|PR|RE|UR)$|^US:CA:CR$|^US:WA:secondary$"
RE_C = rf"^US:{ST}:[A-Z][A-Za-z_ .-]+$"
RE_NAME = r"(?i)^(interstate|i-\s?\d|i \d|us(-| )\d|u\.s\.|us highway|us hwy|state (highway|route|hwy)|state road|sh(-| )\d|sr(-| )\d|farm[- ]to[- ]market|fm(-| )\d|ranch[- ]to[- ]market|rm(-| )\d|ranch road \d|loop \d|spur \d|business \d|bus(\.)? \d|(ok|az|ca|wa|tx|nm)(-| )\d|highway \d|hwy\.? \d|state highway|tx \d|ca \d|county (road|route|highway) \d)"
GROUPS = {"A": "fa", "AB": "fa or fb", "ABC": "fa or fb or fc", "ABCN": "fa or fb or fc or fn", "top4": "top4 and not link"}

def con_():
    c = duckdb.connect(config={"memory_limit": "2GB", "threads": 4, "preserve_insertion_order": False,
                               "temp_directory": TMP})
    c.sql("LOAD spatial"); return c

def chunk(region, i, rgs):
    o_in = os.path.join(OUT, f"_roads_in_{region}_{i}.parquet"); o_bd = os.path.join(OUT, f"_roads_band{SUF}_{region}_{i}.parquet")
    if os.path.exists(o_bd) and os.path.getsize(o_bd) > 0:
        return False
    if SUF and not (os.path.exists(o_in) and os.path.getsize(o_in) > 0): raise SystemExit("run the 30 m pass first")
    src = os.path.join(SC, "reference", region, f"{region}-overture-roads.parquet")
    tbl = pq.ParquetFile(src).read_row_groups(rgs, columns=["id", "class", "subclass", "routes", "names", "geometry"])
    c = con_()
    c.register("raw", tbl)
    c.sql(f"""create table seg as select id, class, coalesce(subclass,'') = 'link' as link,
        class in ('motorway','trunk','primary','secondary') as top4,
        coalesce(len(list_filter(routes, r -> regexp_matches(coalesce(r.network,''), '{RE_A}') and not regexp_matches(coalesce(r.network,''), '{RE_EXCL}'))),0) > 0 as fa,
        coalesce(len(list_filter(routes, r -> regexp_matches(coalesce(r.network,''), '{RE_B}'))),0) > 0 as fb,
        coalesce(len(list_filter(routes, r -> regexp_matches(coalesce(r.network,''), '{RE_C}') and not regexp_matches(coalesce(r.network,''), '{RE_A}|{RE_B}|{RE_EXCL}'))),0) > 0 as fc,
        regexp_matches(coalesce(names.primary,''), '{RE_NAME}') as fn0,
        ST_Transform(ST_GeomFromWKB(geometry), 'EPSG:4326', 'EPSG:5070', always_xy := true) g
        from raw where class in ('motorway','trunk','primary','secondary','tertiary') or len(routes) > 0
           or regexp_matches(coalesce(names.primary,''), '{RE_NAME}')""")
    c.sql("alter table seg add column fn boolean; update seg set fn = fn0 and not (fa or fb or fc)")
    c.sql(f"create table t as select GEOID, ST_GeomFromWKB(geom_wkb) g, ST_Buffer(ST_Boundary(ST_GeomFromWKB(geom_wkb)), {BW}) b from '{os.path.join(GEO, region + '_tracts.parquet')}'")
    if not SUF: c.sql(f"""copy (select t.GEOID, s.class, s.link, s.fa, s.fb, s.fc, s.fn, count(*) n,
        sum(ST_Length(ST_Intersection(s.g, t.g))) len_m
        from seg s join t on ST_Intersects(s.g, t.g) group by all) to '{o_in}' (format parquet)""")
    c.sql("""create table bp as select t.GEOID, s.top4, s.link, s.fa, s.fb, s.fc, s.fn, ST_Intersection(s.g, t.b) p
        from (select * from seg where (fa or fb or fc or fn or top4) and not link) s join t on ST_Intersects(s.g, t.b)""")
    c.sql(f"copy (select GEOID, top4, link, fa, fb, fc, fn, ST_AsWKB(p) p from bp where not ST_IsEmpty(p)) to '{o_bd}' (format parquet)")
    return True

def band_agg(region):
    out = os.path.join(OUT, f"roads_band{SUF}_{region}.parquet")
    c = con_()
    c.sql(f"create table t as select GEOID, ST_GeomFromWKB(geom_wkb) g from '{os.path.join(GEO, region + '_tracts.parquet')}'")
    c.sql(f"create table bp as select GEOID, top4, link, fa, fb, fc, fn, ST_GeomFromWKB(p) p from '{os.path.join(OUT, f'_roads_band{SUF}_{region}_*.parquet')}'")
    parts = []
    for gname, cond in GROUPS.items():
        parts.append(f"""select GEOID, '{gname}' grp, unnest(ST_Dump(ST_LineMerge(ST_Union_Agg(p))), recursive := true) from bp where {cond} group by GEOID""")
    c.sql("create table runs as " + " union all ".join(parts))
    c.sql(f"""copy (select r.GEOID, r.grp, count(*) n_runs, sum(ST_Length(r.geom)) band_len,
        sum(ST_Length(ST_Intersection(r.geom, t.g))) band_in
        from runs r join t using (GEOID) where ST_Length(r.geom) >= {RUN} group by all) to '{out}' (format parquet)""")
    if not SUF: c.sql(f"copy (select * from '{os.path.join(OUT, f'_roads_in_{region}_*.parquet')}') to '{os.path.join(OUT, f'roads_in_{region}.parquet')}' (format parquet)")
    print(region, "band", c.sql(f"select grp, count(*), sum(band_len)/1e3, sum(band_in)/1e3 from '{out}' group by 1 order by 1").fetchall())

if __name__ == "__main__":
    import time
    region = sys.argv[1]; per = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 140
    t0 = time.time(); n = pq.ParquetFile(os.path.join(SC, "reference", region, f"{region}-overture-roads.parquet")).metadata.num_row_groups
    for i, s in enumerate(range(0, n, per)):
        if time.time() - t0 > budget: print("budget hit, rerun"); sys.exit(0)
        if chunk(region, i, list(range(s, min(n, s + per)))): print(region, "chunk", i, round(time.time() - t0, 1))
    if not os.path.exists(os.path.join(OUT, f"roads_band{SUF}_{region}.parquet")):
        band_agg(region)
    print("done", region, round(time.time() - t0, 1))
