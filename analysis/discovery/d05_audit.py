"""Discovery step 05. Hand audit of 60 randomly drawn misses (seed 2026), with the verdicts recorded here.

The draw is reproduced by results/discovery/audit_sample_ids.csv. For each store we looked at every Overture
place within 300 m (top two by name similarity and the two nearest food retail places) and the nearest place
beyond 300 m with name similarity of at least 85. Verdicts
  A  absent. No Overture record of the store anywhere within 2 km under any plausible name
  B  present but more than 250 m from the SNAP point (location offset in one of the two sources)
  C  present nearby under a different name that the matcher scored below 85
  D  unclear (for example a chain store of the same name 500 m or more away that may be a different branch)
Output results/discovery/audit.csv."""
import os, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "discovery")
VERDICTS = {0: "A", 1: "B", 2: "A", 3: "A", 4: "D", 5: "A", 6: "A", 7: "B", 8: "A", 9: "A", 10: "B", 11: "A",
            12: "A", 13: "A", 14: "A", 15: "A", 16: "A", 17: "A", 18: "C", 19: "A", 20: "A", 21: "C", 22: "A",
            23: "B", 24: "A", 25: "A", 26: "C", 27: "B", 28: "D", 29: "A", 30: "A", 31: "A", 32: "D", 33: "A",
            34: "B", 35: "A", 36: "A", 37: "A", 38: "C", 39: "A", 40: "B", 41: "B", 42: "B", 43: "B", 44: "A",
            45: "A", 46: "A", 47: "A", 48: "D", 49: "B", 50: "B", 51: "B", 52: "A", 53: "A", 54: "A", 55: "B",
            56: "A", 57: "D", 58: "B", 59: "D"}

def main():
    s = pd.read_parquet(os.path.join(OUT, "stores.parquet"))
    f = os.path.join(OUT, "audit_sample_ids.csv")
    if not os.path.exists(f):
        s[s.storefront & (s.m_main == 0)].sample(60, random_state=2026)[["snap_id"]].to_csv(f, index=False)
    ids = pd.read_csv(f, dtype={"snap_id": str})
    a = ids.merge(s[["snap_id", "snap_name", "city", "tier", "store_type", "GEOID"]], on="snap_id", how="left")
    a["verdict"] = [VERDICTS[i] for i in range(len(a))]
    a.to_csv(os.path.join(OUT, "audit.csv"), index=False)
    print(a.verdict.value_counts().sort_index().to_string())
    print(pd.crosstab(a.tier, a.verdict).to_string())

if __name__ == "__main__":
    main()
