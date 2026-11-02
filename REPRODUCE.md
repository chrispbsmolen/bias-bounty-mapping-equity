# How to reproduce this entry

Everything runs on open-source Python with no AI service. Commands are run from the repository root.

## 1. Setup

Python 3.10 or later with duckdb (spatial extension), pandas, numpy, scipy, scikit-learn, pyarrow, pyproj, rapidfuzz, statsmodels and matplotlib.

```
pip install duckdb pandas numpy scipy scikit-learn pyarrow pyproj rapidfuzz statsmodels matplotlib
python3 analysis/00_download_data.py   # mirrors the public challenge bucket into data/
python3 analysis/00b_hash_files.py     # sha256 of every file, compare with data/MANIFEST.md
```

## 2. The two final submissions, byte for byte (seconds)

```
bash reproduce_finals.sh
```

It refits both blends from the probe files that were actually uploaded (`submissions/uploaded/`), their public scores (`submissions/probe_scores.csv`) and the frozen model summary (`results/model/`), and checks both outputs against the sha256 of the files submitted to Zindi. It needs only `data/SampleSubmission.csv` from the downloaded data. Expected output

```
MATCH  submissions/candidates/cand_LB_blend_2026-10-26.rebuilt.csv
MATCH  submissions/candidates/cand_LB_blend_2026-10-28.csv
```

The final files are also kept in `submissions/final/`.

## 3. The scored model from raw data (about 10 to 15 minutes)

```
mkdir -p results/geo results/feat results/model
python3 analysis/99_run_all.py
```

This rebuilds every component and probe vector from the provided data. The code kept changing while probes were being uploaded, so vectors rebuilt with the final code differ slightly from the uploaded ones (median 0.0007 to 0.002 per tract, at most about 0.07 for one CBP tract). The uploaded files in `submissions/uploaded/` are the record of what was scored, and step 2 uses them. Region and strata probes rebuild exactly.

On a small machine one Texas road query can take over two minutes.

## 4. The bias discovery (about 90 seconds after its data is in place)

```
mkdir -p data/external/snap data/external/acs
curl -L -o data/external/snap/snap-retailer-locator-data2005-2025.zip \
  https://www.fns.usda.gov/sites/default/files/resource-files/snap-retailer-locator-data2005-2025.zip
unzip -p data/external/snap/snap-retailer-locator-data2005-2025.zip \
  "Historical SNAP Retailer Locator Data 2005-2025.csv" > data/external/snap/snap_retailers_2005_2025.csv
curl -o data/external/acs/acsdt5y2024-b28002.dat \
  https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/acsdt5y2024-b28002.dat
curl -o data/external/acs/acsdt5y2024-c16002.dat \
  https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/acsdt5y2024-c16002.dat
bash analysis/discovery/run_all.sh
```

A clean rerun on 31 Oct 2026 on a different machine reproduced all 34 discovery tables and data files byte for byte. The write-up lists the sha256 of the external files.

## Data rules

The removed reference layers (Census TIGER/Line roads, Microsoft building footprints, HIFLD and USGS facility points, County Business Patterns) are not downloaded or used anywhere.
