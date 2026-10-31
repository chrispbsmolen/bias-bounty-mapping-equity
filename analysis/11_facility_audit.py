"""Step 11. Facility audit with Overture data only.
Two near-independent Overture lists of the same facilities exist. OSM-sourced buildings tagged class
fire_station or school, and Meta-sourced places in the scored categories. Matching them by distance gives
capture-recapture estimates of how completely the scored place categories cover real facilities, split by
urban and rural tracts, plus how much of the name-only evidence is real.
Output results/model/facility_audit.csv and results/feat/osm_fac_sites.parquet."""
import os, warnings, numpy as np, pandas as pd, duckdb
warnings.filterwarnings("ignore", "This pattern")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "data", "source_coop"); F = os.path.join(ROOT, "results", "feat"); M = os.path.join(ROOT, "results", "model")
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
RAD = {"fire": 250.0, "sch": 300.0}
STRICT = {
    "fire": r"(?i)(fire (station|dept|department|district|rescue|company|protection district|hall)|volunteer fire|\bv\.?f\.?d\b|\bfd\b|fire & rescue|fire and rescue|fire-rescue|bomberos|engine company|fire co\b)",
    "sch": r"(?i)(elementary|middle school|high school|junior high|intermediate school|primary school|\bel\b|\bms\b|\bhs\b|charter school|k-?8|k-?12)"}
JUNK = {"fire": r"(?i)(restaurant|pizza|grill|bbq|barbecue|brew|cannabis|dispensary|fireplace|fireworks|equipment|extinguisher|sprinkler|alarm|safety|security|insurance|church|ministr|restoration|church)",
        "sch": r"(?i)(pre-?school|daycare|day care|driving|beauty|barber|cosmetology|dance|music|martial|karate|swim|cdl|truck|flight|tutor|language|yoga|massage|nursing|real estate|gun|dog|booster|pta|pto|alumni|foundation|bus |transportation|nutrition|athletic|stadium|field|gym)"}
OK_CATS = {"fire": {"fire_protection_service", "central_government_office", "public_service_and_government", "community_services_non_profits",
                    "", "ambulance_and_ems_services", "police_department", "organization", "fire_department"},
           "sch": {"education", "", "public_service_and_government", "community_services_non_profits", "religious_school", "campus_building",
                   "elementary_school", "middle_school", "high_school", "school", "private_school", "public_school", "charter_school"}}

def xy(lon, lat):
    return np.c_[lon * 111320 * np.cos(np.radians(lat)), lat * 110540]

def cluster(P, r):
    lab = -np.ones(len(P), int); k = 0
    for i in range(len(P)):
        if lab[i] >= 0: continue
        d = ((P - P[i]) ** 2).sum(1) <= r * r; lab[d & (lab < 0)] = k; k += 1
    return lab

def strict_mask(fac, k):
    return fac.cat.isin(OK_CATS[k]) & fac.nm.str.contains(STRICT[k]) & ~fac.nm.str.contains(JUNK[k])

if __name__ == "__main__":
    rows, osm_all = [], []
    for r in REGIONS:
        c = duckdb.connect(); c.sql("LOAD spatial")
        c.sql(f"create table t as select GEOID, ur_class, geometry g from '{SC}/strata/{r}/{r}-census-tracts.parquet'")
        b = c.sql(f"""select t.GEOID, t.ur_class, b.class, (bbox.xmin+bbox.xmax)/2 lon, (bbox.ymin+bbox.ymax)/2 lat
            from (select class, bbox from '{SC}/reference/{r}/{r}-overture-buildings.parquet' where class in ('fire_station','school')) b
            join t on ST_Contains(t.g, ST_Point((bbox.xmin+bbox.xmax)/2, (bbox.ymin+bbox.ymax)/2))""").df()
        fac = c.sql(f"select f.*, t.ur_class from '{F}/fac_{r}.parquet' f join t using (GEOID)").df()
        for k, cls in [("fire", "fire_station"), ("sch", "school")]:
            ob = b[b["class"] == cls].reset_index(drop=True)
            if len(ob) == 0: continue
            P1 = xy(ob.lon.values, ob.lat.values); l1 = cluster(P1, 150.0)
            s1 = pd.DataFrame(P1).groupby(l1).mean().values; u1 = ob.groupby(l1).ur_class.first().values
            osm_all.append(pd.DataFrame({"region": r, "type": k, "x": s1[:, 0], "y": s1[:, 1], "ur": u1, "GEOID": ob.groupby(l1).GEOID.first().values}))
            for lab, sel in [("category", fac[f"is_{k}"]), ("strict name only", ~fac[f"is_{k}"] & strict_mask(fac, k)),
                             ("loose name only", ~fac[f"is_{k}"] & (fac[f"nm_{k}"] | fac[f"alt_{k}"]) & ~strict_mask(fac, k))]:
                q = fac[sel].reset_index(drop=True)
                if len(q) == 0: continue
                P2 = xy(q.lon.values, q.lat.values); l2 = cluster(P2, 150.0)
                s2 = pd.DataFrame(P2).groupby(l2).mean().values; u2 = q.groupby(l2).ur_class.first().values
                d = np.sqrt(((s1[:, None, :] - s2[None, :, :]) ** 2).sum(2))
                m1 = d.min(1) <= RAD[k]; m2 = d.min(0) <= RAD[k]
                for ur in ["Urban", "Rural"]:
                    a = u1 == ur; z = u2 == ur
                    rows.append({"region": r, "type": k, "list2": lab, "ur": ur, "osm_sites": int(a.sum()), "overture_sites": int(z.sum()),
                                 "overture_places": int((q.ur_class == ur).sum()), "osm_matched": int(m1[a].sum()), "overture_matched": int(m2[z].sum())})
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(M, "facility_audit.csv"), index=False)
    pd.concat(osm_all).to_parquet(os.path.join(F, "osm_fac_sites.parquet"), index=False)
    g = A.groupby(["type", "list2", "ur"])[["osm_sites", "overture_sites", "overture_places", "osm_matched", "overture_matched"]].sum()
    g["share_osm_found"] = (g.osm_matched / g.osm_sites).round(3)
    g["share_overture_confirmed"] = (g.overture_matched / g.overture_sites).round(3)
    g["places_per_site"] = (g.overture_places / g.overture_sites).round(2)
    print(g.to_string())
