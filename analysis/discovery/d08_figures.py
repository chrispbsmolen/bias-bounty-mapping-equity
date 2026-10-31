"""Discovery step 08. Figures for the write-up, written to results/discovery/figures/."""
import os, sys, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "discovery")
FIG = os.path.join(OUT, "figures")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d06_models import prep
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
TIER_COL = {"national chain": BLUE, "small chain": AQUA, "independent": ORANGE}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def fig1(s):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(4)
    for tier in ["independent", "small chain", "national chain"]:
        g = s[s.tier == tier].groupby("svi_minority_q", observed=True).miss_main.agg(["sum", "size"])
        p = g["sum"] / g["size"]; lo, hi = wilson(g["sum"].values, g["size"].values)
        ax.fill_between(x, lo * 100, hi * 100, color=TIER_COL[tier], alpha=0.15, lw=0)
        ax.plot(x, p * 100, color=TIER_COL[tier], lw=2, marker="o", ms=7, mec=SURF, mew=2)
        ax.annotate(f"{tier}  {p.iloc[-1]*100:.0f}%", (3, p.iloc[-1] * 100), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
    ax.set_xticks(x, ["Q1\nfewest minority\nresidents", "Q2", "Q3", "Q4\nmost minority\nresidents"])
    ax.set_xlim(-0.2, 4.1); ax.set_ylim(0, 24)
    ax.set_ylabel("SNAP stores missing from Overture places (%)")
    ax.set_title("Chain stores are found everywhere. Independent stores vanish\nas the share of minority residents rises",
                 loc="left", fontsize=12, color=INK)
    ax.text(-0.12, -0.30, "Tract quartile of the SVI racial and ethnic minority theme. 28,615 active SNAP storefronts in the\n"
            "five challenge regions. Shaded bands are 95% Wilson intervals. A store counts as found when an Overture\n"
            "place within 250 m has a similar name (85 of 100) or is a food retailer at the same street address.",
            transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_miss_by_minority_and_chain.png"), dpi=200); plt.close(fig)

def fig2():
    t = pd.read_csv(os.path.join(OUT, "effects_main.csv"), index_col=0)
    h = pd.read_csv(os.path.join(OUT, "effects_hazard.csv"))
    rows = [("Independent store (vs national chain)", "C(tier)[T.independent]"), ("Small chain (vs national chain)", "C(tier)[T.small chain]"),
            ("Minority theme Q4 (vs Q1)", "C(svi_minority_q)[T.Q4]"), ("Socioeconomic theme Q4 (vs Q1)", "C(svi_socioeconomic_q)[T.Q4]"),
            ("Reservation or trust land", "reservation"), ("Oklahoma tribal statistical area", "otsa"),
            ("Density, each 10 times higher", "logdens"), ("Store authorized in 2024 or 2025", "new_store"),
            ("Convenience store (vs large grocery)", "C(format)[T.convenience]")]
    lab = [r[0] for r in rows]; c = [t.loc[r[1], "coef"] for r in rows]; lo = [t.loc[r[1], "lo"] for r in rows]; hi = [t.loc[r[1], "hi"] for r in rows]
    hz = h[(h.term == "CVI climate Q4 vs Q1") & (h.effects == "region")].iloc[0]
    lab.append("CVI climate Q4 (region effects only)"); c.append(hz.coef_pp); lo.append(hz.lo); hi.append(hz.hi)
    hz = h[(h.term == "summer heat days, per SD") & (h.effects == "county")].iloc[0]
    lab.append("Summer heat days, per SD"); c.append(hz.coef_pp); lo.append(hz.lo); hi.append(hz.hi)
    y = np.arange(len(lab))[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.axvline(0, color=INK2, lw=1)
    ax.hlines(y, lo, hi, color=BLUE, lw=2)
    ax.plot(c, y, "o", color=BLUE, ms=8, mec=SURF, mew=2)
    for yi, ci, l, hh in zip(y, c, lo, hi):
        ax.annotate(f"{ci:+.1f}  [{l:+.1f}, {hh:+.1f}]", (max(hh, 0), yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, lab); ax.set_xlim(-12, 66)
    ax.set_xlabel("Change in chance a SNAP store is missing (percentage points)")
    ax.set_title("What predicts a missing store, holding county fixed", loc="left", fontsize=12, color=INK)
    fig.text(0.02, 0.015, "Linear probability model with county fixed effects, 95% intervals clustered by county.\n"
            "Baseline miss rate 11.9%. All rows come from one model except the two hazard rows,\n"
            "which were each added to it separately.",
            fontsize=8, color=INK2, va="bottom")
    ax.set_title("What predicts a missing store, holding county fixed", loc="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(os.path.join(FIG, "fig2_effect_sizes.png"), dpi=200); plt.close(fig)

def fig3():
    R = pd.read_csv(os.path.join(OUT, "effects_robust.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 5.6), sharey=True)
    y = np.arange(len(R))[::-1]
    for ax, k, ttl, col in [(axs[0], "C(tier)[T.independent]", "Independent vs national chain", ORANGE),
                            (axs[1], "C(svi_minority_q)[T.Q4]", "Minority theme Q4 vs Q1", BLUE)]:
        ax.axvline(0, color=INK2, lw=1)
        ax.hlines(y, R[k + "_lo"], R[k + "_hi"], color=col, lw=2)
        ax.plot(R[k], y, "o", color=col, ms=7, mec=SURF, mew=2)
        ax.plot(R[k].iloc[0], y[0], "o", color=col, ms=10, mec=INK, mew=1.2)
        ax.set_title(ttl, loc="left", fontsize=11, color=INK); ax.set_xlabel("percentage points")
    labels = [c if not c.startswith("overall SVI") else "overall SVI instead of themes (*)" for c in R.check]
    axs[0].set_yticks(y, labels)
    fig.suptitle("Both gaps hold under almost every alternative match rule and sample", x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, 0.01, "Each row refits the main model with one change. The ringed dot is the main specification. The chain gap vanishes at a 100 m\n"
             "radius because big stores sit far from their geocoded point, and shrinks when any business at the same address counts as a match.\n"
             "(*) In this row the right panel shows overall SVI Q4, which is flat because poverty and minority status pull in opposite directions.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig(os.path.join(FIG, "fig3_robustness.png"), dpi=200); plt.close(fig)

def fig4(s):
    m = s[s.m_main == 1].copy()
    m["src"] = np.where(m.ov_dataset == "meta", "Meta (Facebook pages)", np.where(m.ov_dataset == "AllThePlaces", "AllThePlaces (chain store locators)", "other feeds"))
    order = ["national chain", "small chain", "independent"]
    tab = pd.crosstab(m.tier, m.src, normalize="index").loc[order] * 100
    web = (m.ov_n_web > 0).groupby(m.tier, observed=True).mean().loc[order] * 100
    cols = [("AllThePlaces (chain store locators)", AQUA), ("other feeds", GREY), ("Meta (Facebook pages)", BLUE)]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(3)[::-1]; left = np.zeros(3)
    for nm, col in cols:
        v = tab[nm].values
        ax.barh(y, v, left=left, color=col, height=0.55, edgecolor=SURF, linewidth=2, label=nm)
        for yi, l, vi in zip(y, left, v):
            if vi > 7:
                ax.text(l + vi / 2, yi, f"{vi:.0f}%", ha="center", va="center", fontsize=9, color="white" if col != GREY else INK)
        left += v
    for yi, w in zip(y, web.values):
        ax.text(101, yi, f"{w:.0f}% list a website", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, order); ax.set_xlim(0, 130); ax.set_xticks([0, 25, 50, 75, 100]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of SNAP stores found in Overture, by the feed that supplied the record (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=3, frameon=False, fontsize=8)
    ax.set_title("Independent stores reach the map mostly through Facebook pages", loc="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_source_mix.png"), dpi=200); plt.close(fig)

def fig5():
    S = json.load(open(os.path.join(OUT, "summary.json")))
    places = [("All five regions", None), ("Navajo Nation (AZ)", "place_navajo_nation"), ("Tohono O'odham Nation (AZ)", "place_tohono_oodham"),
              ("All reservation tracts", "place_reservations_all"), ("Starr County, TX", "place_starr_county_tx"),
              ("Hidalgo County, TX", "place_hidalgo_county_tx"), ("Cameron County, TX", "place_cameron_county_tx")]
    lab, a, b, n = [], [], [], []
    for l, k in places:
        if k is None:
            lab.append(l); a.append(S["miss_main"] * 100); b.append(S["miss_lenient"] * 100); n.append(S["n_stores"])
        else:
            lab.append(l); a.append(S[k]["miss_main"] * 100); b.append(S[k]["miss_lenient"] * 100); n.append(S[k]["n"])
    y = np.arange(len(lab))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.barh(y + 0.19, a, height=0.36, color=BLUE, label="main rule (within 250 m)")
    ax.barh(y - 0.19, b, height=0.36, color=ORANGE, label="lenient rule (same name within 2 km)")
    for yi, ai, bi, ni in zip(y, a, b, n):
        ax.text(ai + 0.8, yi + 0.19, f"{ai:.0f}%", va="center", fontsize=8, color=INK)
        ax.text(bi + 0.8, yi - 0.19, f"{bi:.0f}%   (n = {ni:,})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, lab); ax.set_xlim(0, 75); ax.grid(axis="y", visible=False)
    ax.set_xlabel("SNAP stores missing from Overture places (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Where the missing stores are", loc="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_named_places.png"), dpi=200); plt.close(fig)

def fig6():
    B = pd.read_csv(os.path.join(OUT, "acs_binned.csv"))
    E = pd.read_csv(os.path.join(OUT, "acs_effects.csv")).set_index("term")
    M = pd.read_csv(os.path.join(OUT, "acs_minority_mediation.csv")).set_index("term")
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    x = np.arange(4)
    for ax, v, ttl, key in [(axs[0], "no_sub", "Households without an internet subscription", "No internet subscription"),
                            (axs[1], "lim_eng", "Limited English speaking households", "Limited English households")]:
        for tier in ["independent", "small chain", "national chain"]:
            g = B[(B["var"] == v) & (B.tier == tier)].sort_values("q")
            lo, hi = wilson(g["sum"].values, g["size"].values)
            ax.fill_between(x, lo * 100, hi * 100, color=TIER_COL[tier], alpha=0.15, lw=0)
            ax.plot(x, g.rate * 100, color=TIER_COL[tier], lw=2, marker="o", ms=7, mec=SURF, mew=2)
            if v == "lim_eng":
                ax.annotate(tier, (3, g.rate.iloc[-1] * 100), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        g = B[(B["var"] == v) & (B.tier == "independent")].sort_values("q")
        ax.set_xticks(x, [f"{q}\n{a*100:.0f} to {b*100:.0f}%" for q, a, b in zip(g.q, g["min"], g["max"])])
        e = E.loc[f"independent. {key}, per SD"]
        ax.text(0.02, 0.97, f"Independents, within county, all controls\n{e.coef_pp:+.1f} points per SD [{e.lo:+.1f}, {e.hi:+.1f}]",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
        ax.set_title(ttl, loc="left", fontsize=11, color=INK); ax.set_xlim(-0.2, 4.0); ax.set_ylim(0, 26)
    axs[0].set_ylabel("SNAP stores missing from Overture places (%)")
    m0 = M.loc["independent. main model"]; m1 = M.loc["independent. plus both"]
    fig.suptitle("Independents in low-internet and limited-English tracts go missing more often in the raw data,\n"
                 "but these measures leave the minority effect almost unchanged", x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, 0.01, f"Tract quartiles of ACS 2020-2024 B28002 and C16002. Among independents, the SVI minority Q4 effect is {m0.coef_pp:+.1f} points "
             f"[{m0.lo:+.1f}, {m0.hi:+.1f}] without these\nmeasures and {m1.coef_pp:+.1f} [{m1.lo:+.1f}, {m1.hi:+.1f}] with both. Shaded bands are 95% Wilson intervals.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9)); fig.savefig(os.path.join(FIG, "fig6_internet_and_language.png"), dpi=200); plt.close(fig)

if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    s = prep()
    fig1(s); fig2(); fig3(); fig4(s); fig5(); fig6()
    print(sorted(os.listdir(FIG)))
