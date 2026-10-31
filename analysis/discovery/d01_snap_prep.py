"""Discovery step 01. Clean the USDA SNAP retailer list and place each active store in a scored tract.

Input  data/external/snap/snap_retailers_2005_2025.csv (USDA FNS, current to 31 Dec 2025)
Output results/discovery/snap_active.parquet, one row per active SNAP store inside a scored tract,
       results/discovery/snap_chainkeys.csv, national counts of each normalized store name.
A store is active when its End Date is blank. Chain size is the number of active stores in the whole
country that share the same normalized name (digits, punctuation and legal suffixes removed)."""
import os, re, unicodedata, duckdb, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAP = os.path.join(ROOT, "data", "external", "snap", "snap_retailers_2005_2025.csv")
SC = os.path.join(ROOT, "data", "source_coop")
OUT = os.path.join(ROOT, "results", "discovery")
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
STOP = {"inc", "llc", "corp", "co", "ltd", "the", "store", "stores", "no", "num", "number", "dba", "and", "of"}

def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def collapse(toks):
    """Join runs of single letters, so H-E-B and H E B both become heb."""
    out, run = [], ""
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run += t
            continue
        if run:
            out.append(run); run = ""
        out.append(t)
    if run:
        out.append(run)
    return out

def chain_key(s):
    toks = [t for t in collapse(norm(s).split()) if not re.search(r"\d", t) and t not in STOP]
    return " ".join(toks)

def main():
    os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect()
    con.sql("LOAD spatial")
    s = con.sql(f"""select * from read_csv('{SNAP}', all_varchar=true, header=true)""").df()
    s.columns = [c.strip().lstrip("﻿") for c in s.columns]
    s["active"] = s["End Date"].fillna("").str.strip().eq("")
    s = s[s["active"]].copy()
    s = s.sort_values("Record ID").drop_duplicates("Record ID")
    s["ckey"] = s["Store Name"].map(chain_key)
    keys = s.groupby("ckey").size().rename("chain_n_national").reset_index()
    keys.sort_values("chain_n_national", ascending=False).to_csv(os.path.join(OUT, "snap_chainkeys.csv"), index=False)
    s = s.merge(keys, on="ckey", how="left")
    s = s[s["State"].isin(["OK", "WA", "AZ", "CA", "TX", "NM"])].copy()
    s["lat"] = pd.to_numeric(s["Latitude"], errors="coerce")
    s["lon"] = pd.to_numeric(s["Longitude"], errors="coerce")
    s = s.dropna(subset=["lat", "lon"])
    s["auth_date"] = pd.to_datetime(s["Authorization Date"], format="%m/%d/%Y", errors="coerce")
    con.register("s", s[["Record ID", "Store Name", "Store Type", "Street Number", "Street Name", "City", "State",
                         "Zip Code", "County", "lat", "lon", "auth_date", "ckey", "chain_n_national"]])
    parts = []
    for r in REGIONS:
        tr = os.path.join(SC, "strata", r, f"{r}-census-tracts.parquet")
        ss = os.path.join(SC, "reference", r, f"{r}-sample-submission.csv")
        q = f"""select s.*, t.GEOID, '{r}' region from s
            join (select t.GEOID, t.geometry::GEOMETRY g from '{tr}' t
                  join read_csv('{ss}', types={{'GEOID':'VARCHAR'}}) z using (GEOID)) t
            on ST_Contains(t.g, ST_Point(s.lon, s.lat))"""
        d = con.sql(q).df()
        print(r, len(d))
        parts.append(d)
    d = pd.concat(parts, ignore_index=True).drop_duplicates("Record ID")
    d = d.rename(columns={"Record ID": "snap_id", "Store Name": "snap_name", "Store Type": "store_type",
                          "Street Number": "st_num", "Street Name": "st_name", "City": "city", "State": "state",
                          "Zip Code": "zip", "County": "county_name"})
    d.to_parquet(os.path.join(OUT, "snap_active.parquet"), index=False)
    print("total", len(d))
    print(d.store_type.value_counts().to_string())

if __name__ == "__main__":
    main()
