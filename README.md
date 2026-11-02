# Bias Bounty Mapping Equity Challenge, csmolen

Code for Christopher Smolen's entry (Zindi username `csmolen`) in the Zindi Bias Bounty Mapping Equity Challenge, for the leaderboard and for the Best Bias Discovery and Best Documentation prizes. The methodology write-up explains every step and result.

## What is here

- `analysis/` holds the scored model (`00` to `16`, run with `99_run_all.py`) and the probe blend refit (`16_refit.py`).
- `analysis/discovery/` holds the bias discovery pipeline (`d01` to `d09`, run with `run_all.sh`).
- `data/MANIFEST.md` lists all 325 provided challenge files with URL, size and sha256. No data is stored in this repository.
- `submissions/probe_scores.csv` logs every leaderboard upload and its public score, the only inputs to the blend fit.
- `notes/AI_USE.md` records how Claude (Anthropic) was used. No AI service runs in any pipeline.
- `REPRODUCE.md` gives the steps from a fresh machine, and `reproduce_finals.sh` rebuilds both final submissions byte for byte in seconds.
- `submissions/uploaded/` holds every probe file that was uploaded and scored, and `submissions/final/` the two final submissions.

## Final submissions

- `XvNVpwKD`, `cand_LB_blend_2026-10-28`, public RMSE 0.066833
- `RziXa451`, `cand_LB_blend_2026-10-26`, public RMSE 0.068262

## Data rules

The removed reference layers (Census TIGER/Line roads, Microsoft building footprints, HIFLD and USGS facility points, County Business Patterns) are not downloaded or used anywhere. The scored model uses only the provided challenge data. The bias discovery also uses USDA SNAP retailer data and two ACS tables, all public domain, with URLs and hashes in the write-up.
