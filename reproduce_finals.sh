#!/usr/bin/env bash
# Rebuild the two final Zindi submissions from the uploaded probe files and check them byte for byte.
# Needs Python 3 with pandas, numpy, scipy and pyarrow, and data/SampleSubmission.csv from the challenge bucket
# (python3 analysis/00_download_data.py fetches it with the rest of the provided data).
# Runs in a few seconds. It restores submissions/probe_scores.csv when it ends.
set -euo pipefail
cd "$(dirname "$0")"
SCORES=submissions/probe_scores.csv
cp "$SCORES" "$SCORES.full"
trap 'mv "$SCORES.full" "$SCORES"' EXIT
mkdir -p submissions/candidates

# 26 Oct blend. The scores known on 26 Oct (zeros and probes 01 to 04, 06, 07, 16, 17), fitted by 10_fit_probes.py.
awk -F, 'NR==1 || $1 ~ /^(all_zeros_sample|probe_0[1-4]_|probe_0[67]_|probe_1[67]_)/' "$SCORES.full" > "$SCORES"
python3 analysis/10_fit_probes.py > /dev/null
cp submissions/candidates/cand_LB_blend.csv submissions/candidates/cand_LB_blend_2026-10-26.rebuilt.csv

# 28 Oct blend. Every score known on 28 Oct, refitted by 16_refit.py with forward selection.
grep -v '^cand_LB_blend_2026-10-28' "$SCORES.full" > "$SCORES"
python3 analysis/16_refit.py 2026-10-28 > /dev/null

check () {
  got=$(python3 -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1")
  if [ "$got" = "$2" ]; then echo "MATCH  $1"; else echo "DIFFER $1 ($got)"; exit 1; fi
}
check submissions/candidates/cand_LB_blend_2026-10-26.rebuilt.csv df7bc9766b5ba58fb4229a4e9fd7274f488e1d303f534a3124dfdfde4306856e
check submissions/candidates/cand_LB_blend_2026-10-28.csv 4580ac20125b7bfd3482f6040fcd2e04ea3fa7ef1cd2f72fb40c7675eda98d8b
