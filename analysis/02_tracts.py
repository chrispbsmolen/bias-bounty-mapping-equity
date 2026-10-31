"""Step 02. Tract geometry in EPSG:5070 for every scored tract, with a 30 m boundary band.
Writes results/geo/<region>_tracts.parquet (GEOID, area_m2, perimeter_m, edge_m, geom_wkb, band_wkb).
edge_m is the length of tract boundary that is also region boundary (no neighbour tract)."""
import os, sys, duckdb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "data", "source_coop")
OUT = os.path.join(ROOT, "results", "geo")
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
BAND = 30.0

def run(region):
    out = os.path.join(OUT, f"{region}_tracts.parquet")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    con = duckdb.connect(config={"memory_limit": "2GB", "threads": 2, "preserve_insertion_order": False})
    con.sql("LOAD spatial")
    tr = os.path.join(SC, "strata", region, f"{region}-census-tracts.parquet")
    ss = os.path.join(SC, "reference", region, f"{region}-sample-submission.csv")
    aoi = os.path.join(SC, "boundaries", f"{region}-aoi.geojson")
    con.sql(f"""create table t as select t.GEOID,
        ST_Transform(t.geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true) g
        from '{tr}' t join read_csv('{ss}', types={{'GEOID':'VARCHAR'}}) s using (GEOID)""")
    con.sql(f"""create table a0 as select ST_Transform(geom, 'EPSG:4326', 'EPSG:5070', always_xy := true) g from ST_Read('{aoi}')""")
    con.sql("""create table a as select unnest(ST_Dump(ST_Boundary(ST_Union_Agg(g))), recursive := true) from a0""")
    con.sql("""create table ab as select ST_Buffer(geom, 1.0) b from a""")
    con.sql(f"""copy (select GEOID, ST_Area(g) area_m2, ST_Length(ST_Boundary(g)) perimeter_m,
        coalesce((select sum(ST_Length(ST_Intersection(ST_Boundary(t.g), ab.b))) from ab where ST_Intersects(t.g, ab.b)), 0) edge_m,
        ST_AsWKB(g) geom_wkb, ST_AsWKB(ST_Buffer(ST_Boundary(g), {BAND})) band_wkb
        from t) to '{out}' (format parquet)""")
    print(region, con.sql(f"select count(*), sum(area_m2)/1e6, sum(edge_m>0) from '{out}'").fetchall())

if __name__ == "__main__":
    for r in (sys.argv[1:] or REGIONS):
        run(r)
