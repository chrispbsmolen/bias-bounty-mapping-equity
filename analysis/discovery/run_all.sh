#!/usr/bin/env bash
# Rebuilds every Bias Discovery number and figure from data/source_coop, data/external/snap and data/external/acs.
# d09 runs before d08 because the last figure reads its output.
# Open-source tools only (Python 3.10, duckdb 1.5.6 with spatial, pandas, numpy, scipy, pyproj, rapidfuzz 3.14,
# statsmodels 0.15, matplotlib). No network access and no AI service needed. About 2 minutes on a laptop.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 analysis/discovery/d01_snap_prep.py
python3 analysis/discovery/d02_pois_prep.py
python3 analysis/discovery/d03_match.py
python3 analysis/discovery/d04_store_table.py
python3 analysis/discovery/d05_audit.py
python3 analysis/discovery/d06_models.py
python3 analysis/discovery/d07_tracts_mechanism.py
python3 analysis/discovery/d09_acs_mechanism.py
python3 analysis/discovery/d08_figures.py
