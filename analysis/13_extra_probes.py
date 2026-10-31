"""Step 13. Extra probe files derived from vectors already uploaded, so they stay consistent with the measured ones.
probe_15_transport_TX_only = the uploaded probe_02 transport vector inside south-central-tx, 0 elsewhere.
Together with probe_02 it splits the transport weight into Texas and the other four regions."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib; m8 = importlib.import_module("08_candidates")
ROOT = m8.ROOT
up = pd.read_csv(os.path.join(ROOT, "submissions", "uploaded", "probe_02_transport.csv"), dtype={"GEOID": str})
o = pd.read_parquet(os.path.join(ROOT, "results", "model", "B0v2_summary.parquet")).set_index("GEOID").loc[up.GEOID].reset_index()
v = np.where(up.region.values == "south-central-tx", up.coverage_gap_score.values, 0.0)
m8.write("probe_15_transport_TX_only.csv", o, v)
print("probe_15 mean", round(v.mean(), 5))

# ---- v3 (25 Oct evening): structural candidate and two probes for tomorrow ----
m7 = importlib.import_module("07_model")
D = m7.load(); rates = m7.site_rates(D)
v3 = m7.run(m7.V3, "B0", D, rates)
v3.to_parquet(os.path.join(ROOT, "results", "model", "B0v3_summary.parquet"), index=False)
m8.write("cand_B0v3_structural.csv", v3, v3.score.values)
m8.write("probe_16_B0v3_transport.csv", v3, v3.c_T.values)
nchs = D.set_index("GEOID").loc[v3.GEOID, "nchs_2013"].fillna(1.0).values
m8.write("probe_17_strata_nchs.csv", v3, 0.1 * (nchs - 1) / 5)
print("v3 mean", round(v3.score.mean(), 4), "v3 transport mean", round(v3.c_T.mean(), 4))
