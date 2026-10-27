"""
capacity_planner.py - cellular capacity versus frequency-reuse planning tool (Mini Project 3.2).

All inputs live in CONFIG below. For each reuse factor N the tool computes
    total channels          = total spectrum / channel bandwidth
    channels per cell       = floor(total channels / N)        (a channel cannot be split; remainder unused)
    cells in the area       = floor(area / hexagon area),   hexagon area = (3*sqrt(3)/2) R^2
    system capacity         = channels per cell x cells          (simultaneous calls / channels in use)
    Q = D/R                 = sqrt(3N)
    first-tier SIR          = Q^n / i0   (n = path-loss exponent, i0 = number of first-tier interferers)
    Erlang capacity         = Erlang B traffic per cell at the target blocking probability, x cells
Cell splitting (smaller R) and sectoring (fewer interferers, channels split across sectors) are
evaluated as alternative capacity levers.

    python Mini_Project_3_2_Capacity_Planning/capacity_planner.py
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = []

# =========================================================================== CONFIGURATION
CONFIG = {
    "total_spectrum_kHz": 20_000,     # 20 MHz allocated to the operator (one link direction)
    "channel_bw_kHz": 200,            # GSM-like carrier
    "reuse_factors": [1, 3, 4, 7, 9, 12],
    "area_km2": 500.0,                # coverage area
    "R_km": 2.0,                      # cell radius (centre to corner)
    "pathloss_n": 4.0,                # path-loss exponent for the SIR estimate
    "sir_required_dB": 18.0,          # analogue-voice benchmark used in textbooks (digital needs less)
    "blocking": 0.02,                 # grade of service for the Erlang B figures
    "split_factors": [1, 2, 4],       # R, R/2, R/4
}


def say(s=""):
    print(s)
    LOG.append(s)


def hex_area(R):
    return 3 * math.sqrt(3) / 2 * R ** 2


def erlang_b(A, c):
    """Blocking probability for offered traffic A (Erl) on c channels (stable recursion)."""
    b = 1.0
    for k in range(1, c + 1):
        b = A * b / (k + A * b)
    return b


def erlang_capacity(c, gos):
    """Largest offered traffic (Erl) on c channels with blocking <= gos (bisection)."""
    if c <= 0:
        return 0.0
    lo, hi = 0.0, float(c) * 2 + 10
    for _ in range(80):
        mid = (lo + hi) / 2
        if erlang_b(mid, c) > gos:
            hi = mid
        else:
            lo = mid
    return lo


def sir_db(N, n=4.0, interferers=6):
    return 10 * math.log10(math.sqrt(3 * N) ** n / interferers)


def split_channels(ch, k):
    """Distribute ch channels over k sectors as evenly as possible."""
    return [ch // k + (1 if i < ch % k else 0) for i in range(k)]


def evaluate(N, R, sectors=1, cfg=CONFIG):
    total = cfg["total_spectrum_kHz"] // cfg["channel_bw_kHz"]
    ch_cell = total // N
    cells = int(cfg["area_km2"] // hex_area(R))
    interferers = {1: 6, 3: 2, 6: 1}[sectors]
    per_sector = split_channels(ch_cell, sectors)
    erl_cell = sum(erlang_capacity(c, cfg["blocking"]) for c in per_sector)
    return {
        "N": N, "R_km": R, "sectors": sectors, "total_channels": total, "channels_per_cell": ch_cell,
        "spare_channels": total - ch_cell * N, "cells": cells, "capacity_channels": ch_cell * cells,
        "capacity_per_km2": ch_cell * cells / cfg["area_km2"], "Q": math.sqrt(3 * N),
        "first_tier_interferers": interferers, "SIR_dB": sir_db(N, cfg["pathloss_n"], interferers),
        "erlang_per_cell": erl_cell, "erlang_total": erl_cell * cells,
    }


def main():
    cfg = CONFIG
    say("Configuration:")
    for k, v in cfg.items():
        say(f"  {k:20s} = {v}")
    A = hex_area(cfg["R_km"])
    say(f"  hexagon area = 2.598 R^2 = {A:.3f} km^2 -> {cfg['area_km2'] / A:.2f} cells fit, "
        f"{int(cfg['area_km2'] // A)} used")

    # ------------------------------------------------------------------ reuse-factor table
    say("=" * 100)
    rows = [evaluate(N, cfg["R_km"]) for N in cfg["reuse_factors"]]
    say(f"{'N':>3s}{'ch/cell':>9s}{'spare':>7s}{'cells':>7s}{'capacity':>10s}{'ch/km2':>8s}{'Q':>7s}"
        f"{'SIR dB':>8s}{'meets ' + str(int(cfg['sir_required_dB'])) + ' dB':>12s}{'Erl/cell':>10s}{'Erl total':>11s}")
    for r in rows:
        say(f"{r['N']:>3d}{r['channels_per_cell']:>9d}{r['spare_channels']:>7d}{r['cells']:>7d}"
            f"{r['capacity_channels']:>10d}{r['capacity_per_km2']:>8.2f}{r['Q']:>7.2f}{r['SIR_dB']:>8.1f}"
            f"{('yes' if r['SIR_dB'] >= cfg['sir_required_dB'] else 'no'):>12s}{r['erlang_per_cell']:>10.2f}"
            f"{r['erlang_total']:>11.1f}")

    # ------------------------------------------------------------------ cell splitting
    say("=" * 100)
    say("Cell splitting at N = 7 (Q and SIR unchanged because D and R shrink together)")
    split = []
    for f in cfg["split_factors"]:
        r = evaluate(7, cfg["R_km"] / f)
        split.append(r)
        say(f"  R = {r['R_km']:4.2f} km: cells {r['cells']:5d}, capacity {r['capacity_channels']:6d} channels "
            f"({r['capacity_channels'] / split[0]['capacity_channels']:5.2f}x), "
            f"{r['erlang_total']:7.1f} Erl ({r['erlang_total'] / split[0]['erlang_total']:5.2f}x), "
            f"SIR {r['SIR_dB']:.1f} dB")

    # ------------------------------------------------------------------ sectoring
    say("=" * 100)
    say("Sectoring (same sites, same spectrum): interferers 6 -> 2 (120 deg) -> 1 (60 deg)")
    sect = []
    for N, s in ((7, 1), (7, 3), (4, 3), (3, 6), (12, 1), (4, 1)):
        r = evaluate(N, cfg["R_km"], s)
        sect.append(r)
        per = split_channels(r["channels_per_cell"], s)
        say(f"  N = {N:2d}, {s} sector(s): channels per sector {per}, SIR {r['SIR_dB']:5.1f} dB "
            f"({'meets' if r['SIR_dB'] >= cfg['sir_required_dB'] else 'fails'} {cfg['sir_required_dB']:.0f} dB), "
            f"capacity {r['capacity_channels']} channels, {r['erlang_total']:.1f} Erl")

    # ------------------------------------------------------------------ save
    with open(os.path.join(HERE, "results_3_2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows + split[1:] + sect:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # ------------------------------------------------------------------ charts
    Ns = [r["N"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    x = np.arange(len(Ns))
    ax.bar(x, [r["capacity_channels"] for r in rows], color="#1f77b4", label="capacity (channels in area)")
    for i, r in enumerate(rows):
        ax.text(i, r["capacity_channels"] + 60, f"{r['capacity_channels']}\n({r['channels_per_cell']}/cell)",
                ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("system capacity (simultaneous channels)")
    ax.set_ylim(0, max(r["capacity_channels"] for r in rows) * 1.2)
    ax2 = ax.twinx()
    ax2.plot(x, [r["SIR_dB"] for r in rows], "o-", color="#d62728", label="first-tier SIR (omni)")
    ax2.axhline(cfg["sir_required_dB"], color="#d62728", ls=":", lw=1)
    ax2.text(len(Ns) - 0.6, cfg["sir_required_dB"] + 0.6, f"{cfg['sir_required_dB']:.0f} dB requirement",
             color="#d62728", fontsize=7, ha="right")
    ax2.set_ylabel("SIR (dB)", color="#d62728")
    ax.set_title("Capacity and interference versus reuse factor", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    ax = axes[1]
    Qs = [r["Q"] for r in rows]
    ax.plot(Qs, [r["capacity_channels"] for r in rows], "o-", color="#1f77b4")
    for r in rows:
        ax.annotate(f"N={r['N']}", (r["Q"], r["capacity_channels"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    qreq = (6 * 10 ** (cfg["sir_required_dB"] / 10)) ** (1 / cfg["pathloss_n"])
    ax.axvspan(0, qreq, color="#d62728", alpha=0.08)
    ax.axvline(qreq, color="#d62728", ls=":", lw=1)
    ax.text(qreq + 0.05, max(r["capacity_channels"] for r in rows) * 0.9,
            f"Q needed for {cfg['sir_required_dB']:.0f} dB\n(omni) = {qreq:.2f}", fontsize=8, color="#d62728")
    ax.set_xlabel("co-channel reuse ratio Q = D/R")
    ax.set_ylabel("system capacity (channels)")
    ax.set_title("Capacity versus Q: every gain in Q costs capacity", fontsize=10)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_capacity_vs_N_and_Q.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    levers = [("N=7 omni\n(baseline)", evaluate(7, 2.0, 1)), ("N=7\n3 sectors", evaluate(7, 2.0, 3)),
              ("N=4\n3 sectors", evaluate(4, 2.0, 3)), ("N=3\n6 sectors", evaluate(3, 2.0, 6)),
              ("N=7 omni\nR/2 split", evaluate(7, 1.0, 1)), ("N=4 3-sect.\n+ R/2 split", evaluate(4, 1.0, 3)),
              ("N=7 omni\nR/4 split", evaluate(7, 0.5, 1))]
    base = levers[0][1]["erlang_total"]
    vals = [lv[1]["erlang_total"] / base for lv in levers]
    cols = ["#7f7f7f", "#9467bd", "#9467bd", "#9467bd", "#2ca02c", "#ff7f0e", "#2ca02c"]
    ax.bar(range(len(levers)), vals, color=cols, edgecolor="black", lw=0.4)
    for i, (lab, r) in enumerate(levers):
        ok = r["SIR_dB"] >= cfg["sir_required_dB"]
        ax.text(i, vals[i] + 0.15, f"x{vals[i]:.2f}\nSIR {r['SIR_dB']:.1f} dB{'' if ok else ' (fails)'}",
                ha="center", fontsize=7, color="black" if ok else "#d62728")
    ax.set_xticks(range(len(levers)))
    ax.set_xticklabels([lv[0] for lv in levers], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Erlang capacity relative to baseline (log)")
    ax.set_ylim(0.3, 40)
    ax.set_title("Capacity levers at 2% blocking: sectoring (purple), splitting (green), both (orange)", fontsize=10)
    ax.grid(axis="y", alpha=0.35, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_capacity_levers.png"), dpi=150)
    plt.close(fig)
    say("=" * 100)
    say("Capacity levers relative to N = 7 omni (Erlang capacity at 2% blocking):")
    for lab, r in levers:
        say(f"  {lab.replace(chr(10), ' '):24s} x{r['erlang_total'] / base:6.2f}  "
            f"({r['erlang_total']:7.1f} Erl, {r['capacity_channels']} channels, SIR {r['SIR_dB']:.1f} dB, "
            f"{r['cells']} sites)")

    with open(os.path.join(HERE, "results_3_2.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
