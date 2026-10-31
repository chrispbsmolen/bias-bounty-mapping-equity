"""Discovery step 03. Find each active SNAP store among the Overture places.

Inputs  results/discovery/snap_active.parquet (step 01), results/discovery/pois_<region>.parquet (step 02)
Output  results/discovery/pairs_<region>.parquet, every SNAP to Overture pair within 2 km whose name
        similarity is at least 60 or whose street address agrees, plus every pair within 250 m.
        results/discovery/nearest_food_<region>.parquet, distance from each store to the nearest
        Overture place in a food retail category.
Distances are in metres in EPSG:5070 (always_xy). Name similarity is rapidfuzz token_set_ratio on
normalized names with store numbers removed (and plain ratio with spaces removed, whichever is higher), falling back to token_sort_ratio when the only shared
words are generic (market, food, mart and similar)."""
import os, re, sys, unicodedata
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from rapidfuzz import fuzz, process as rf_process

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "discovery")
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
NEAR, FAR = 250.0, 2000.0
STOP = {"inc", "llc", "corp", "co", "ltd", "the", "no", "num", "dba", "and", "of", "#"}
GENERIC = {"market", "mart", "food", "foods", "store", "stores", "grocery", "groceries", "shop", "gas", "mini",
           "stop", "express", "quick", "convenience", "center", "station", "travel", "fuel", "deli", "meat",
           "meats", "supermarket", "super", "tienda", "carniceria", "panaderia", "mercado", "abarrotes", "mart",
           "pharmacy", "drug", "drugs", "general", "country", "corner", "fresh", "family", "discount", "dollar",
           "plaza", "valley", "city", "town", "west", "east", "north", "south", "s", "y", "de", "la", "el", "los", "las"}
ABBR = {"street": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr", "highway": "hwy", "boulevard": "blvd",
        "lane": "ln", "court": "ct", "place": "pl", "parkway": "pkwy", "circle": "cir", "trail": "trl", "terrace": "ter",
        "north": "n", "south": "s", "east": "e", "west": "w", "freeway": "fwy", "expressway": "expy", "route": "rte",
        "us": "us", "state": "state", "farm": "fm", "ranch": "rm", "loop": "loop", "suite": "ste", "square": "sq"}
FOOD_TAX = {"food_and_beverage_store", "convenience_store", "fueling_station", "discount_store", "department_store",
            "superstore", "grocery_store"}
FOOD_CAT = {"pharmacy", "drugstore", "bakery", "butcher_shop", "farmers_market", "truck_stop", "warehouse_club",
            "wholesale_store", "meat_shop", "seafood_market", "donuts", "tortilleria", "candy_store"}

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

def name_clean(s):
    return " ".join(t for t in collapse(norm(s).split()) if not re.fullmatch(r"\d+[a-z]?", t) and t not in STOP)

def name_sim(a, b):
    if not a or not b:
        return 0.0
    nospace = fuzz.ratio(a.replace(" ", ""), b.replace(" ", ""))
    shared = set(a.split()) & set(b.split())
    if shared and shared <= GENERIC:
        return max(fuzz.token_sort_ratio(a, b), nospace)
    return max(fuzz.token_set_ratio(a, b), nospace)

def street_clean(s):
    return " ".join(ABBR.get(t, t) for t in norm(s).split())

def split_addr(freeform):
    f = norm(freeform)
    m = re.match(r"^(\d+[a-z]?)\s+(.*)$", f)
    return (m.group(1), street_clean(m.group(2))) if m else ("", street_clean(f))

def is_food(cat, tax):
    if cat in FOOD_CAT:
        return True
    if tax is not None and len(tax) and (set(tax) & FOOD_TAX):
        return True
    return False

def run(r, snap, tf):
    out = os.path.join(OUT, f"pairs_{r}.parquet")
    s = snap[snap.region == r].reset_index(drop=True)
    p = pd.read_parquet(os.path.join(OUT, f"pois_{r}.parquet"))
    p = p.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    px, py = tf.transform(p.lon.values, p.lat.values)
    sx, sy = tf.transform(s.lon.values, s.lat.values)
    p["food"] = [is_food(c, t) for c, t in zip(p.cat, p.tax)]
    p = p.drop(columns=["tax", "cat_alt"])
    p["nm"] = [name_clean(x) for x in p.poi_name]
    p["brand_nm"] = [name_clean(x) for x in p.brand_name]
    addr = [split_addr(x) for x in p.addr.fillna("")]
    p["a_num"] = [a[0] for a in addr]
    p["a_st"] = [a[1] for a in addr]
    tree = cKDTree(np.c_[px, py])
    s_nm = [name_clean(x) for x in s.snap_name]
    s_num = [norm(x).split(" ")[0] if norm(x) else "" for x in s.st_num.fillna("")]
    s_st = [street_clean(x) for x in s.st_name.fillna("")]
    nm_arr = p.nm.values.astype(object); br_arr = p.brand_nm.values.astype(object)
    num_arr = p.a_num.values.astype(object); st_arr = p.a_st.values.astype(object); food_arr = p.food.values
    rows = []
    near_food = np.full(len(s), np.inf)
    for i in range(len(s)):
        cand = tree.query_ball_point([sx[i], sy[i]], FAR)
        if not cand:
            continue
        cand = np.asarray(cand)
        d = np.hypot(px[cand] - sx[i], py[cand] - sy[i])
        fd = d[p.food.values[cand]]
        if len(fd):
            near_food[i] = fd.min()
        ts = np.maximum(rf_process.cdist([s_nm[i]], nm_arr[cand], scorer=fuzz.token_set_ratio)[0],
                        rf_process.cdist([s_nm[i]], br_arr[cand], scorer=fuzz.token_set_ratio)[0])
        numeq = (num_arr[cand] == s_num[i]) if s_num[i] else np.zeros(len(cand), bool)
        sel = (d <= NEAR) | (ts >= 60) | numeq
        for j, dj in zip(cand[sel], d[sel]):
            ns = max(name_sim(s_nm[i], nm_arr[j]), name_sim(s_nm[i], br_arr[j]))
            am = bool(s_num[i]) and s_num[i] == num_arr[j] and fuzz.token_set_ratio(s_st[i], st_arr[j]) >= 80
            if dj <= NEAR or ns >= 60 or am:
                rows.append((s.snap_id.iat[i], p.id.iat[j], float(dj), float(ns), am, bool(food_arr[j])))
    pr = pd.DataFrame(rows, columns=["snap_id", "poi_id", "dist_m", "name_sim", "addr_match", "food_cat"])
    keep = ["id", "poi_name", "cat", "datasets", "confidence", "n_web", "n_phone", "n_social", "brand_wd", "operating_status"]
    pr = pr.merge(p[keep].rename(columns={"id": "poi_id"}), on="poi_id", how="left")
    pr["dataset"] = [x[0] if x is not None and len(x) else None for x in pr.datasets]
    pr = pr.drop(columns=["datasets"])
    pr.to_parquet(out, index=False)
    pd.DataFrame({"snap_id": s.snap_id, "near_food_m": near_food}).to_parquet(
        os.path.join(OUT, f"nearest_food_{r}.parquet"), index=False)
    print(r, len(s), "stores", len(pr), "pairs")

if __name__ == "__main__":
    snap = pd.read_parquet(os.path.join(OUT, "snap_active.parquet"))
    tf = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    for r in (sys.argv[1:] or REGIONS):
        run(r, snap, tf)
