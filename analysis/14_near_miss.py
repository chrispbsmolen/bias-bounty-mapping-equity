"""Step 14. Near-miss highways. For every tract, the distance from its polygon to the nearest tier-A Overture
segment (Interstate, US, main state routes, links excluded) when that distance is under 100 m, and the tier-A
length within 15 m outside the tract. A TIGER line a few metres from Overture's can clip a tract corner
that Overture misses, which makes the road component defined with a gap of 1.
Output results/feat/near_miss_<region>.parquet."""
import os, sys, duckdb, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib; r3 = importlib.import_module("03_roads")
ROOT = r3.ROOT; SC = r3.SC; GEO = r3.GEO; OUT = r3.OUT
def run(region):
    out = os.path.join(OUT, f"near_miss_{region}.parquet")
    if os.path.exists(out) and os.path.getsize(out) > 0: return
    c = r3.con_()
    src = os.path.join(SC, "reference", region, f"{region}-overture-roads.parquet")
    c.sql(f"""create table seg as select ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true) g
        from read_parquet('{src}') where coalesce(subclass,'') <> 'link'
        and coalesce(len(list_filter(routes, r -> regexp_matches(coalesce(r.network,''), '{r3.RE_A}') and not regexp_matches(coalesce(r.network,''), '{r3.RE_EXCL}'))),0) > 0""")
    c.sql(f"create table t as select GEOID, ST_GeomFromWKB(geom_wkb) g from '{os.path.join(GEO, region + '_tracts.parquet')}'")
    c.sql(f"""copy (select t.GEOID, min(ST_Distance(s.g, t.g)) dmin,
        sum(ST_Length(ST_Difference(ST_Intersection(s.g, ST_Buffer(t.g, 15)), t.g))) out15
        from t join seg s on ST_DWithin(s.g, t.g, 100) group by 1) to '{out}' (format parquet)""")
    print(region, c.sql(f"select count(*), sum(dmin<1), sum(dmin<15) from '{out}'").fetchall(), flush=True)
if __name__ == "__main__":
    for r in sys.argv[1:] or r3.__dict__.get("REGIONS", ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]):
        run(r)
