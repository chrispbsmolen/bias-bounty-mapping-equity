# The target formula, as far as it can be pinned

Written 25 Oct 2026. Sources are the bucket README (`data/raw_docs/README.md`, quoted as [README]), the challenge Info page as copied in `research/01_rules_and_intel.md` [INFO], the Zindi sample submission column layout [SAMPLE], and descriptions that participants posted before the reference layers were removed [BOARD]. We read those descriptions only. Nothing here uses a removed layer.

## Per component

Every component has the same capped form.

```
gap = 1 - min(1, overture / reference)        defined only when reference > 0
```

| Component | Overture side (we compute it exactly) | Reference side (removed, we model it) | Evidence |
|---|---|---|---|
| transport | length in EPSG:5070 of `motorway`, `trunk`, `primary`, `secondary` segments clipped to the tract, links included | length of TIGER `S1100` + `S1200` clipped to the tract | [README] "named-highway classes only", [INFO], links included per [BOARD] |
| building | count of Overture footprints whose centroid is in the tract | count of Microsoft US Building Footprints, same rule | [INFO] "Overture building counts per tract against Microsoft US Building Footprints", centroid rule per [BOARD] |
| fire, ems, schools | count of places with `categories.primary` equal to `fire_department`, `ambulance_and_ems_services`, or one of six school categories | USGS National Map (HIFLD) facility counts of that type | [README] quotes the capped formula for these three directly |
| cbp | count of all Overture places in the tract | County Business Patterns establishments allocated to the tract | [README] "all Overture places against CBP establishments", column `cbp_estab` per [BOARD] |

## Composition

```
fac      = mean of the defined ones among fire, ems, schools      (undefined if none defined)
poi      = mean of the defined ones among fac, cbp                (undefined if both undefined)
score    = mean of the defined ones among transport, building, poi
```

- "`coverage_gap_score` is the mean of the defined components only" and an undefined component "is then written as `0` with `_defined = false`" [README].
- "`poi_gap` is the mean of the two halves" and the facility half is "the mean over the defined types" [README].
- "Tracts with all three undefined are dropped" [README], so every scored tract has at least one defined component. Our model conditions on that.
- The sample submission carries exactly these seven gap and defined pairs (transport, building, poi, poi_fire, poi_ems, poi_schools, poi_cbp) [SAMPLE].

## Assumed, not quoted

1. The capped ratio for transport, building and cbp is assumed by analogy with the quoted facility formula. [BOARD] reconstructions say it held for all components.
2. That `poi` keeps only the defined half when the other half is undefined. This follows from "mean of the defined components" applied one level down.
3. Bounding box centre stands in for the footprint centroid (the two differ by less than 1e-4 RMSE per [BOARD]).
4. The reference roads are clipped to the tract polygon, so a TIGER line that lies exactly on a shared tract boundary counts in both tracts. [BOARD] reports 25% to 36% of TIGER highway length sits on tract boundaries. Our own data agrees, see below.
5. How CBP establishments were allocated to tracts is unknown. We assume they follow business activity, with a regional total near 0.025 establishments per resident (the national CBP rate).

## Published numbers we use as constraints (from organiser documentation)

- Share of tracts with at least one undefined component. Eastern Oklahoma 21%, Northern California 37%, South-Central Texas 28%, Maricopa 55% [INFO]. Building and POI are almost always defined, so this is the share of tracts with no TIGER highway.
- EMS "is close to 1:1 and rarely binds", "fire stations carry the component", hospitals are excluded [README].
- The Overture to TIGER length ratio "ranges from 0.71 to 1.59 across the four study regions" [INFO]. v1 could not reproduce it. In v2, counting 45% of county route length as TIGER highway brings eastern Oklahoma to 0.72, the low end of the range, while Texas stays at 1.68 (see section 9 of `research/06_model_v1.md`).

## Checks against our own data

- With TIGER approximated by Overture segments on Interstate, US and main state route networks, plus long Overture runs inside a 30 m band along the tract boundary (the double count), the predicted share of tracts with no highway is 22.4% (OK, published 21%), 54.4% (AZ, 55%), 38.2% (CA, 37%), 28.9% (TX, 28%). A 60 m band gives 21.6%, 54.1%, 37.6%, 28.3%.
- Adding Texas FM and RM roads to the proxy drops the Texas share to 18.5%, far from 28%, so most FM and RM roads are not S1200 in TIGER. Adding California county sign routes drops California to 28.8%, so those are not S1200 either.
