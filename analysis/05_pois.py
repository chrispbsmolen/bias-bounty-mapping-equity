"""Step 05. Overture places per tract (point in polygon on the original lon/lat tract geometry).
Outputs results/feat/poi_<region>.parquet (per tract and primary category: count, count with confidence >= 0.5)
and results/feat/fac_<region>.parquet (every place that looks like a fire, EMS or school facility, by category
or by name, with its location, for site counting)."""
import os, sys, time, duckdb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "data", "source_coop"); OUT = os.path.join(ROOT, "results", "feat")
TMP = os.path.join(ROOT, "results", "tmp")
SCHOOL = "('elementary_school','middle_school','high_school','school','private_school','public_school')"
RE_FIRE = r"(?i)\b(fire|vfd|v\.f\.d|fire[- ]rescue|bomberos)\b"
RE_EMS = r"(?i)\b(ems|ambulance|emergency medical|paramedic|medic \d|rescue squad)\b"
RE_SCH = r"(?i)\b(elementary|middle school|high school|junior high|intermediate school|primary school|academy|isd|school)\b"

def run(region):
    o1 = os.path.join(OUT, f"poi_{region}.parquet"); o2 = os.path.join(OUT, f"fac_{region}.parquet")
    if os.path.exists(o2) and os.path.getsize(o2) > 0: return
    c = duckdb.connect(config={"memory_limit": "2GB", "threads": 4, "temp_directory": TMP}); c.sql("LOAD spatial")
    src = os.path.join(SC, "reference", region, f"{region}-overture-pois.parquet")
    c.sql(f"create table t as select GEOID, geometry g from '{os.path.join(SC, 'strata', region, region + '-census-tracts.parquet')}'")
    c.sql(f"""create table p as select t.GEOID, coalesce(categories.primary, '') cat, categories.alternate alt,
        coalesce(names.primary, '') nm, confidence conf, operating_status os, basic_category bc,
        ST_X(p.geometry) lon, ST_Y(p.geometry) lat
        from (select *, geometry from '{src}') p join t on ST_Contains(t.g, p.geometry)""")
    c.sql(f"""copy (select GEOID, cat, count(*) n, count(*) filter (where conf >= 0.5) n50,
        count(*) filter (where coalesce(os, 'open') <> 'open') n_closed from p group by all) to '{o1}' (format parquet)""")
    c.sql(f"""copy (select GEOID, cat, nm, conf, os, bc, lon, lat,
        cat = 'fire_department' is_fire, cat = 'ambulance_and_ems_services' is_ems, cat in {SCHOOL} is_sch,
        list_contains(alt, 'fire_department') alt_fire, list_contains(alt, 'ambulance_and_ems_services') alt_ems,
        len(list_filter(alt, x -> x in {SCHOOL})) > 0 alt_sch,
        regexp_matches(nm, '{RE_FIRE}') nm_fire, regexp_matches(nm, '{RE_EMS}') nm_ems, regexp_matches(nm, '{RE_SCH}') nm_sch
        from p where cat in ('fire_department','ambulance_and_ems_services') or cat in {SCHOOL}
          or list_contains(alt, 'fire_department') or list_contains(alt, 'ambulance_and_ems_services')
          or len(list_filter(alt, x -> x in {SCHOOL})) > 0
          or regexp_matches(nm, '{RE_FIRE}') or regexp_matches(nm, '{RE_EMS}') or regexp_matches(nm, '{RE_SCH}')) to '{o2}' (format parquet)""")
    print(region, c.sql(f"select sum(n) from '{o1}'").fetchall(), c.sql(f"select count(*), sum(is_fire::int), sum(is_ems::int), sum(is_sch::int) from '{o2}'").fetchall())

if __name__ == "__main__":
    for r in sys.argv[1:] or ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]:
        t0 = time.time(); run(r); print(round(time.time() - t0, 1), flush=True)
