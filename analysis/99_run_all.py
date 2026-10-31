"""Run the whole v1 pipeline in order (steps 02 to 09, with the facility steps 11 and 12 before the model). Cached outputs are reused, so a rerun is quick.
Usage: python3 analysis/99_run_all.py"""
import os, subprocess, sys
A = os.path.dirname(os.path.abspath(__file__))
R = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
def run(args, env=None):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable] + args, check=True, env={**os.environ, **(env or {})})
run([f"{A}/02_tracts.py"])
for band in ["30", "60"]:
    for r in R:
        run([f"{A}/03_roads.py", r, "4", "1e9"], {"BAND_M": band})
for r in R:
    run([f"{A}/04_buildings.py", r, "6", "1e9"])
run([f"{A}/05_pois.py"]); run([f"{A}/06_features.py"])
run([f"{A}/11_facility_audit.py"]); run([f"{A}/12_facility_features.py"]); run([f"{A}/14_near_miss.py"])
run([f"{A}/07_model.py", "B0v2"])
run([f"{A}/08_candidates.py"]); run([f"{A}/13_extra_probes.py"]); run([f"{A}/15_more_probes.py"]); run([f"{A}/09_validate.py"])
