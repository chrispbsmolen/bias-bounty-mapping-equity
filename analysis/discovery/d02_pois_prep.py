"""Discovery step 02. Slim copy of the provided Overture places for matching.

Input  data/source_coop/reference/<region>/<region>-overture-pois.parquet (Overture 2026-08-19.0, provided)
Output results/discovery/pois_<region>.parquet with id, name, categories, brand, address, confidence,
       upstream source datasets, contact fields and lon, lat."""
import os, sys, duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SC = os.path.join(ROOT, "data", "source_coop")
OUT = os.path.join(ROOT, "results", "discovery")
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]

def run(r):
    out = os.path.join(OUT, f"pois_{r}.parquet")
    if os.path.exists(out):
        return
    con = duckdb.connect(config={"memory_limit": "1500MB", "threads": 2})
    con.sql("LOAD spatial")
    src = os.path.join(SC, "reference", r, f"{r}-overture-pois.parquet")
    con.sql(f"""copy (select id, names."primary" as poi_name, categories."primary" as cat, categories.alternate cat_alt,
        basic_category, taxonomy.hierarchy tax, brand.names."primary" as brand_name, brand.wikidata brand_wd,
        addresses[1].freeform addr, addresses[1].postcode postcode, confidence, operating_status,
        list_distinct(list_filter(list_transform(sources, x -> x.dataset), x -> x not in ('Overture', 'Overture-signals'))) datasets,
        coalesce(len(websites), 0) n_web, coalesce(len(phones), 0) n_phone, coalesce(len(socials), 0) n_social,
        ST_X(ST_Centroid(geometry::GEOMETRY)) lon, ST_Y(ST_Centroid(geometry::GEOMETRY)) lat
        from '{src}') to '{out}' (format parquet)""")
    print(r, con.sql(f"select count(*) from '{out}'").fetchone())

if __name__ == "__main__":
    for r in (sys.argv[1:] or REGIONS):
        run(r)
