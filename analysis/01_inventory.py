"""Inventory every mirrored file: rows, columns and types, GeoParquet CRS. Writes results/inventory/files.tsv and columns.tsv."""
import os, json, csv, sys
import pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "source_coop"); OUT = os.path.join(ROOT, "results", "inventory")
csv.field_size_limit(sys.maxsize)
files, cols = [], []
for dp, _, fns in os.walk(SRC):
    for fn in sorted(fns):
        p = os.path.join(dp, fn); rel = os.path.relpath(p, SRC); size = os.path.getsize(p)
        if fn.endswith(".parquet"):
            f = pq.ParquetFile(p); md = f.metadata; sch = f.schema_arrow
            geo = sch.metadata.get(b"geo") if sch.metadata else None
            crs, gcols, gtypes = "", "", ""
            if geo:
                g = json.loads(geo); pc = g.get("primary_column", "")
                c = g["columns"].get(pc, {}); cr = c.get("crs", "absent (default OGC:CRS84)")
                if isinstance(cr, dict):
                    idd = cr.get("id", {}); crs = f'{idd.get("authority","")}:{idd.get("code","")}' if idd else cr.get("name", "projjson")
                else: crs = str(cr)
                gcols = pc; gtypes = ",".join(c.get("geometry_types", []))
            files.append([rel, size, md.num_rows, len(sch), md.num_row_groups, crs, gcols, gtypes])
            for fld in sch: cols.append([rel, fld.name, str(fld.type)[:200]])
        elif fn.endswith(".csv"):
            with open(p, newline="") as g:
                r = csv.reader(g); hdr = next(r); n = sum(1 for _ in r)
            files.append([rel, size, n, len(hdr), "", "", "", ""])
            for h in hdr: cols.append([rel, h, "csv"])
        else:
            files.append([rel, size, "", "", "", "", "", ""])
with open(os.path.join(OUT, "files.tsv"), "w") as f:
    f.write("path\tbytes\trows\tncols\trow_groups\tcrs\tgeom_col\tgeom_types\n")
    for r in sorted(files): f.write("\t".join(map(str, r)) + "\n")
with open(os.path.join(OUT, "columns.tsv"), "w") as f:
    f.write("path\tcolumn\ttype\n")
    for r in cols: f.write("\t".join(r) + "\n")
print(len(files), len(cols))
