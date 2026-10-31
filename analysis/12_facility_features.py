"""Step 12. Stricter facility evidence per tract for model v2 (uses rules from step 11).
Junk names are dropped (restaurants named Fire, fireplace shops, preschools, driving schools, booster clubs),
name-only evidence must sit in a plausible category, fire departments are no longer read as EMS evidence,
and schools are counted as distinct names (a campus with an elementary and a middle school is two schools).
Output results/feat/fac_v2.parquet."""
import os, re, sys, warnings, numpy as np, pandas as pd, duckdb
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib; a11 = importlib.import_module("11_facility_audit")
F = a11.F
EMS_STRICT = r"(?i)(\bems\b|ambulance|emergency medical|paramedic|rescue squad|medic unit|medstar|acadian)"
EMS_CATS = {"health_and_medical", "medical_transportation", "public_service_and_government", "", "community_services_non_profits",
            "central_government_office", "emergency_service", "medical_service_organizations"}

def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\b(the|of|school|sch|campus|elem|elementary|el|middle|ms|high|hs|jr|sr|junior|senior|isd|cisd|academy)\b", " ", s)
    return " ".join(s.split())

def far_from(P, Q, r):
    if len(Q) == 0: return np.ones(len(P), bool)
    return np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(2)).min(1) > r

rows = []
fac = duckdb.sql(f"select * from '{F}/fac_*.parquet'").df()
fac["fire_strict"] = a11.strict_mask(fac, "fire") & ~fac.is_fire
fac["sch_junk"] = fac.nm.str.contains(a11.JUNK["sch"])
fac["sch_strict"] = a11.strict_mask(fac, "sch") & ~fac.is_sch
fac["ems_strict"] = ~fac.is_ems & fac.cat.isin(EMS_CATS) & fac.nm.str.contains(EMS_STRICT)
for gid, d in fac.groupby("GEOID"):
    P = a11.xy(d.lon.values, d.lat.values); e = {"GEOID": gid}
    for k, rad in [("fire", 250.0), ("ems", 250.0), ("sch", 300.0)]:
        cat = d[f"is_{k}"].values; st = d[f"{k}_strict"].values
        x = st & far_from(P, P[cat], rad)
        e[f"v2x_{k}"] = int(len(np.unique(a11.cluster(P[x], 150.0)))) if x.any() else 0
        e[f"v2cat_{k}"] = int(len(np.unique(a11.cluster(P[cat], 150.0)))) if cat.any() else 0
    good = d.is_sch & ~d.sch_junk
    e["v2cat_sch"] = len({norm(n) or f"_{i}" for i, n in enumerate(d.nm[good])})
    rows.append(e)
V = pd.DataFrame(rows)
V.to_parquet(os.path.join(F, "fac_v2.parquet"), index=False)
print(V.drop(columns="GEOID").sum().to_dict())
