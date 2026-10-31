"""Step 04. Overture building features per tract.
Each footprint is assigned to the tract containing the centre of its bounding box (the organisers used
centroids; the bbox centre is a close stand-in that needs no geometry decode).
Footprint size is approximated from the bbox in metres (bbox_m2). Source is the first entry of sources[].
Output results/feat/bld_<region>.parquet, one row per tract and (source, class, size band, has_ms)."""
import os, sys, time, duckdb, pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "data", "source_coop"); OUT = os.path.join(ROOT, "results", "feat")
TMP = os.path.join(ROOT, "results", "tmp")

def con_():
    c = duckdb.connect(config={"memory_limit": "2GB", "threads": 4, "preserve_insertion_order": False, "temp_directory": TMP})
    c.sql("LOAD spatial"); return c

def chunk(region, i, rgs, c):
    o = os.path.join(OUT, f"_bld_{region}_{i}.parquet")
    if os.path.exists(o) and os.path.getsize(o) > 0:
        return False
    src = os.path.join(SC, "reference", region, f"{region}-overture-buildings.parquet")
    tbl = pq.ParquetFile(src).read_row_groups(rgs, columns=["sources", "class", "subtype", "num_floors", "height", "bbox"])
    c.register("raw", tbl)
    c.sql(f"""copy (with b as (select
        coalesce(sources[1].dataset, 'none') src,
        coalesce(len(list_filter(sources, s -> s.dataset = 'Microsoft ML Buildings')), 0) > 0 has_ms,
        coalesce(class, '') cls, coalesce(subtype, '') sub,
        num_floors is not null or height is not null has_h,
        (bbox.xmax - bbox.xmin) * 111320 * cos(radians((bbox.ymin + bbox.ymax) / 2)) * (bbox.ymax - bbox.ymin) * 110540 bbox_m2,
        ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) pt from raw)
      select t.GEOID, src, has_ms, cls, sub, has_h,
        case when bbox_m2 < 40 then 's0' when bbox_m2 < 120 then 's1' when bbox_m2 < 400 then 's2'
             when bbox_m2 < 1500 then 's3' when bbox_m2 < 6000 then 's4' else 's5' end sz,
        count(*) n, sum(bbox_m2) a
      from b join t on ST_Contains(t.g, b.pt) group by all) to '{o}' (format parquet)""")
    c.unregister("raw")
    return True

if __name__ == "__main__":
    region = sys.argv[1]; per = int(sys.argv[2]) if len(sys.argv) > 2 else 4; budget = float(sys.argv[3]) if len(sys.argv) > 3 else 140
    t0 = time.time(); src = os.path.join(SC, "reference", region, f"{region}-overture-buildings.parquet")
    n = pq.ParquetFile(src).metadata.num_row_groups
    c = con_()
    c.sql(f"create table t as select GEOID, geometry g from '{os.path.join(SC, 'strata', region, region + '-census-tracts.parquet')}'")
    for i, s in enumerate(range(0, n, per)):
        if time.time() - t0 > budget: print("budget hit, rerun"); sys.exit(0)
        if chunk(region, i, list(range(s, min(n, s + per))), c): print(region, "chunk", i, round(time.time() - t0, 1), flush=True)
    out = os.path.join(OUT, f"bld_{region}.parquet")
    c.sql(f"copy (select GEOID, src, has_ms, cls, sub, has_h, sz, sum(n) n, sum(a) a from '{os.path.join(OUT, f'_bld_{region}_*.parquet')}' group by all) to '{out}' (format parquet)")
    print("done", region, c.sql(f"select sum(n) from '{out}'").fetchall(), round(time.time() - t0, 1))
