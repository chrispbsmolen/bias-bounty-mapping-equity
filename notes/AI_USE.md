# AI use record

Claude (Anthropic) is used for research, analysis, code drafting, debugging and documentation. The final solution must run on open-source tools with no AI service in the loop. Each entry below notes what Claude did.

- 24 Oct 2026. Folder setup and background research.
- 25 Oct 2026. Claude wrote and ran the v1 pipeline (`analysis/02` to `10`, `99_run_all.py`), the formula note, the model report and the candidate files. The pipeline runs on Python with DuckDB, pandas, numpy, scipy and scikit-learn, with no AI service.
- 25 Oct 2026. Claude wrote and ran the Bias Discovery pipeline (`analysis/discovery/d01` to `d08`, `run_all.sh`), did the 60-store hand audit recorded in `d05_audit.py`, made the figures and drafted `research/07_bias_discovery.md`. The pipeline runs on Python with DuckDB, pandas, numpy, scipy, pyproj, rapidfuzz, statsmodels and matplotlib, with no AI service.
- 25 Oct 2026, later. Claude wrote the facility audit and model v2 (`analysis/11`, `12`, changes to `07`, `08`, `10`, `99`) and updated the report and candidate README.
- 25 Oct 2026. Claude added the ACS internet and language test (`analysis/discovery/d09_acs_mechanism.py`, figure 6) and updated `research/07_bias_discovery.md`. No AI service in the pipeline.
- 25 Oct 2026. Claude drafted the write-up (`writeup/WRITEUP_draft.md`), rendered the PDF and reran the discovery pipeline to time it. Rendering uses pandoc and Chromium, no AI service.
