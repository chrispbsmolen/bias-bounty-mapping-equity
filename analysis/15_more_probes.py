"""Step 15. Probe files for 27 Oct onward (made 26 Oct).
probe_18_strata_ruca      ruca_primary rescaled to 0 to 0.1 (missing or code 99 as 1, metropolitan core)
probe_19_strata_tribal    tribal_pct rescaled to 0 to 0.1
probe_20_cbp_by_housing   CBP contribution if establishments were spread by housing units instead of activity
probe_21_facilities_rural fire coverage 0.78 urban and 0.45 rural, facility contribution
All built on the v2 settings so they line up with the measured probes."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
m7 = importlib.import_module("07_model"); m8 = importlib.import_module("08_candidates")
D = m7.load(); rates = m7.site_rates(D)
v2 = m7.run(m7.V2, "B0", D, rates)
s = D.set_index("GEOID").loc[v2.GEOID]
def scale(x): x = x.astype(float); return 0.1 * (x - x.min()) / (x.max() - x.min())
m8.write("probe_18_strata_ruca.csv", v2, scale(s.ruca_primary.where(s.ruca_primary <= 10, 1.0).fillna(1.0)).values)
m8.write("probe_19_strata_tribal.csv", v2, scale(s.tribal_pct.fillna(0.0)).values)
r2 = dict(rates); r2["poi_pred"] = D.hu.values.astype(float)
c = m7.run({**m7.V2, "cbp_w": 0.0}, "B0", D, r2)
m8.write("probe_20_cbp_by_housing.csv", v2, c.c_C.values)
f = m7.run({**m7.V2, "cov_fire": (0.78, 0.45)}, "B0", D, rates)
m8.write("probe_21_facilities_rural.csv", v2, f.c_F.values)
for k, v in [("ruca", scale(s.ruca_primary.where(s.ruca_primary <= 10, 1.0).fillna(1.0)).values), ("tribal", scale(s.tribal_pct.fillna(0.0)).values), ("cbp_housing", c.c_C.values), ("fac_rural", f.c_F.values)]:
    print(k, round(float(np.mean(v)), 4), "corr with nchs", round(np.corrcoef(v, s.nchs_2013.fillna(1.0))[0, 1], 2), "corr with v2 cbp", round(np.corrcoef(v, v2.c_C)[0, 1], 2), "corr with v2 fac", round(np.corrcoef(v, v2.c_F)[0, 1], 2))

# 28 Oct. Building variant. The direct building probe entered the blend with weight 2.5, so the true building gap
# is larger than v2's. This version drops the assumed Overture surplus (m_b 0) and widens the spread to 0.08.
bw = m7.run({**m7.V2, "m_b": 0.0, "sig_b0": 0.08}, "B0", D, rates)
m8.write("probe_22_building_wide.csv", v2, bw.c_B.values)
print("building wide mean", round(bw.c_B.mean(), 4), "v2 building mean", round(v2.c_B.mean(), 4), "corr", round(np.corrcoef(bw.c_B, v2.c_B)[0, 1], 3))
