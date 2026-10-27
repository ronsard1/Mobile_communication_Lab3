"""
Mini Project 3.1 - frequency-reuse planning and handover decision logic.

Part (a): hexagonal layouts for N = 1, 3, 4, 7 (and 9, 12), group labels, Q = D/R table with the
          reuse distance measured on the drawn layout as a check against sqrt(3N).
Part (b): synthetic serving/neighbour RSRP, naive / hysteresis / hysteresis+TTT triggers.
Lab tasks 1-6 follow.

    python Mini_Project_3_1_Reuse_Handover/mini_project_3_1.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import RegularPolygon, Polygon  # noqa: E402

from hexgrid import layout, measured_reuse_distance, valid_cluster_sizes, SHIFT, center, hex_cells  # noqa: E402
from handover import make_series, simulate  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


# ===================================================================== step 1
say(f"Step 1: NumPy {np.__version__}, Matplotlib {matplotlib.__version__}")
say(f"Valid cluster sizes N = i^2 + ij + j^2 up to 21: {valid_cluster_sizes(21)}")

# ===================================================================== steps 2-6 and lab task 1
CMAP = plt.get_cmap("tab20")


def draw_layout(ax, N, rings, R=1.0, sectors=False):
    cells = layout(N, rings, R)
    for (q, r, x, y, g) in cells:
        col = CMAP((g - 1) % 20) if N > 1 else "#9ecae1"
        ax.add_patch(RegularPolygon((x, y), 6, radius=R, orientation=0, facecolor=col,
                                    edgecolor="white" if not sectors else "black", lw=0.8,
                                    alpha=0.95 if g == 1 else 0.75))
        if g == 1:
            ax.add_patch(RegularPolygon((x, y), 6, radius=R, orientation=0, fill=False, edgecolor="black", lw=1.6))
        if not sectors:
            ax.text(x, y, str(g), ha="center", va="center", fontsize=6.5 if rings > 4 else 8,
                    fontweight="bold" if g == 1 else "normal")
    # mark the reuse distance D from the origin to the nearest co-channel cell
    i, j = SHIFT[N]
    x1, y1 = center(i, j, R)
    if N > 1:
        ax.annotate("", xy=(x1, y1), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color="black", lw=1.4))
        ax.text(x1 / 2, y1 / 2 + 0.35 * R, f"D = {math.hypot(x1, y1):.2f} R", fontsize=7.5, ha="center",
                bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=1))
    lim = R * 1.5 * rings + R
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    return cells


say("=" * 78)
say("Steps 2-6, lab task 1: layouts and reuse ratio")
say(f"{'N':>3s} {'(i,j)':>7s} {'Q = sqrt(3N)':>13s} {'D/R measured':>13s} {'groups seen':>12s} {'cells drawn':>12s}")
table = []
for N, rings in ((1, 4), (3, 4), (4, 4), (7, 5), (9, 5), (12, 6)):
    cells = layout(N, rings)
    Q = math.sqrt(3 * N)
    D = measured_reuse_distance(cells)
    groups = len({c[4] for c in cells})
    table.append((N, SHIFT[N], Q, D, groups, len(cells)))
    say(f"{N:>3d} {str(SHIFT[N]):>7s} {Q:>13.3f} {D:>13.3f} {groups:>12d} {len(cells):>12d}")
    assert abs(Q - D) < 1e-9, "layout reuse distance does not match sqrt(3N)"
say("  measured D/R equals sqrt(3N) for every N (asserted)")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, (N, rings) in zip(axes.ravel(), ((1, 4), (3, 4), (4, 4), (7, 5))):
    draw_layout(ax, N, rings)
    ax.set_title(f"N = {N}  (i, j) = {SHIFT[N]},  Q = D/R = {math.sqrt(3 * N):.2f}", fontsize=11)
fig.suptitle("Frequency-reuse layouts; bold outline = cells using group 1", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_reuse_layouts_1_3_4_7.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
for ax, (N, rings) in zip(axes, ((9, 5), (12, 6))):
    draw_layout(ax, N, rings)
    ax.set_title(f"N = {N}  (i, j) = {SHIFT[N]},  Q = {math.sqrt(3 * N):.2f}", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_reuse_layouts_9_12.png"), dpi=150)
plt.close(fig)

# SIR first-tier estimate used in the discussion (path-loss exponent 4, six interferers)
say("  first-tier SIR estimate S/I = Q^4 / 6 (n = 4, six co-channel interferers):")
for N, *_ in table:
    Q = math.sqrt(3 * N)
    say(f"    N = {N:2d}: Q = {Q:5.2f}, SIR = {10 * math.log10(Q ** 4 / 6):6.2f} dB, "
        f"channels per cell = total / {N}")

# ===================================================================== part (b) steps 7-14
say("=" * 78)
SEED = 9
S = make_series(seed=SEED)
THR = -100.0
configs = [("naive (0 dB, no TTT)", 0.0, 0), ("hysteresis 3 dB", 3.0, 0), ("hysteresis 3 dB + TTT 5", 3.0, 5)]
res = [simulate(S, h, t) for _, h, t in configs]
say(f"Steps 7-14: 300 samples, serving A falls, neighbour B rises, close stretch at samples 100-150 (seed {SEED})")
say(f"{'configuration':28s}{'handovers':>10s}{'ping-pongs':>11s}{'first HO':>9s}{'min serving':>12s}"
    f"{'samples < -100':>15s}{'sustained < -100':>17s}")
over = simulate(S, 8.0, 20)
for (name, h, t), r in list(zip(configs, res)) + [(("over-tuned 8 dB + TTT 20", 8.0, 20), over)]:
    say(f"{name:28s}{r.count:>10d}{r.ping_pongs():>11d}{r.events[0][0]:>9d}{r.min_serving():>12.1f}"
        f"{r.samples_below(THR):>15d}{('YES' if r.sustained_below(THR) else 'no'):>17s}")
say("  (sustained = 3 consecutive samples below -100 dBm, the too-late-handover criterion used below)")

col = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}


def stacked(series, results, titles, path, thr=THR, suptitle=None):
    t = np.arange(len(next(iter(series.values()))))
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.5 * len(results) + 0.6), sharex=True)
    for ax, r, ttl in zip(axes, results, titles):
        for c, v in series.items():
            ax.plot(t, v, lw=0.9, color=col[c], alpha=0.85, label=f"cell {c}")
        ax.plot(t, r.serving_rsrp, color="black", lw=1.4, label="serving cell RSRP")
        # band at the top showing which cell serves
        for c in series:
            m = r.serving == c
            ax.fill_between(t, -60, -62, where=m, color=col[c], step="mid")
        for (k, f, to) in r.events:
            ax.axvline(k, color="#d62728", lw=0.6, alpha=0.7)
        ax.axhline(thr, color="gray", ls=":", lw=1)
        ax.set_ylim(-112, -59)
        ax.set_ylabel("RSRP (dBm)")
        ax.set_title(f"{ttl}: {r.count} handovers, {r.ping_pongs()} ping-pongs, min serving {r.min_serving():.1f} dBm",
                     fontsize=9.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower left", ncol=4)
    axes[-1].set_xlabel("sample (red lines = handovers; colour band at top = serving cell; dotted = -100 dBm)")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


stacked(S, res, [c[0] for c in configs], os.path.join(FIG, "fig_handover_three_configs.png"))

# ===================================================================== lab task 2: three cells
say("=" * 78)
S3 = make_series(seed=SEED, extra=(-101.0, 0.06, "C"))
r3 = [simulate(S3, h, t) for _, h, t in configs]
say("Lab task 2: third neighbour C (close to A and B at the cell edge, weaker than B later); strongest candidate chosen")
for (name, h, t), r in zip(configs, r3):
    to_c = sum(1 for e in r.events if e[2] == "C")
    say(f"  {name:28s} handovers {r.count:3d} (into C: {to_c:2d}), ping-pongs {r.ping_pongs():2d}, "
        f"min serving {r.min_serving():6.1f} dBm")
stacked(S3, r3, [c[0] for c in configs], os.path.join(FIG, "fig_handover_three_cells.png"))

# ===================================================================== lab task 3: MRO sweep
say("=" * 78)
Hs = np.arange(0, 11, 1.0)
TTTs = np.arange(0, 21, 1)
TUNE_SEEDS = range(100, 200)              # 100 training realisations (evaluation in task 4 uses other seeds)
MAX_LATE = 0.02                           # accept at most 2% of runs with a sustained drop below -100 dBm
train = [make_series(seed=sd) for sd in TUNE_SEEDS]
cnt = np.zeros((len(Hs), len(TTTs)))
ppm = np.zeros((len(Hs), len(TTTs)))
late = np.zeros((len(Hs), len(TTTs)))
for a, h in enumerate(Hs):
    for b, tt in enumerate(TTTs):
        rs = [simulate(ser, h, int(tt)) for ser in train]
        cnt[a, b] = np.mean([r.count for r in rs])
        ppm[a, b] = np.mean([r.ping_pongs() for r in rs])
        late[a, b] = np.mean([r.sustained_below(THR) for r in rs])
ok = late <= MAX_LATE
best = None
for a in range(len(Hs)):
    for b in range(len(TTTs)):
        if ok[a, b]:
            key = (round(cnt[a, b], 3), ppm[a, b], Hs[a] + TTTs[b])
            if best is None or key < best[0]:
                best = (key, a, b)
_, ba, bb = best
H_BEST, TTT_BEST = float(Hs[ba]), int(TTTs[bb])
say(f"Lab task 3: sweep H 0-10 dB x TTT 0-20 samples ({len(Hs) * len(TTTs)} combinations) over "
    f"{len(TUNE_SEEDS)} random realisations")
say(f"  objective: minimise mean handover count subject to <= {100 * MAX_LATE:.0f}% of runs with a sustained "
    f"drop below {THR:.0f} dBm")
say(f"  acceptable combinations: {int(ok.sum())}; chosen H = {H_BEST:.0f} dB, TTT = {TTT_BEST} samples -> "
    f"mean {cnt[ba, bb]:.2f} handovers, {ppm[ba, bb]:.2f} ping-pongs, too-late in {100 * late[ba, bb]:.1f}% of runs")
for (h, tt) in ((0, 0), (3, 0), (3, 5), (8, 20)):
    a, b = int(h), int(tt)
    say(f"  for comparison H = {h} dB, TTT = {tt}: mean {cnt[a, b]:.2f} handovers, {ppm[a, b]:.2f} ping-pongs, "
        f"too-late {100 * late[a, b]:.1f}%")
rb = simulate(S, H_BEST, TTT_BEST)
say(f"  chosen setting on the seed-{SEED} series: {rb.count} handover(s), first at sample {rb.events[0][0]}, "
    f"min serving {rb.min_serving():.1f} dBm")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ext = (TTTs[0] - 0.5, TTTs[-1] + 0.5, Hs[0] - 0.5, Hs[-1] + 0.5)
im0 = axes[0].imshow(cnt, origin="lower", aspect="auto", cmap="viridis_r", extent=ext, vmax=6)
fig.colorbar(im0, ax=axes[0], label="mean handovers per run (clipped at 6)")
axes[0].set_title("Mean handover count (ping-pong where > 1)", fontsize=10)
im1 = axes[1].imshow(100 * late, origin="lower", aspect="auto", cmap="Reds", extent=ext, vmin=0, vmax=100)
fig.colorbar(im1, ax=axes[1], label="% runs with sustained serving < -100 dBm")
axes[1].contour(TTTs, Hs, late, levels=[MAX_LATE], colors="black", linewidths=1.5)
axes[1].set_title(f"Too-late risk (black line = {100 * MAX_LATE:.0f}% limit)", fontsize=10)
for ax in axes:
    ax.plot(TTT_BEST, H_BEST, marker="*", color="white", mec="black", ms=16)
    ax.plot(5, 3, marker="o", color="white", mec="black", ms=8)
    ax.set_xlabel("time-to-trigger (samples)")
    ax.set_ylabel("hysteresis (dB)")
fig.suptitle(f"MRO sweep over {len(TUNE_SEEDS)} realisations (star = chosen, circle = 3 dB / TTT 5)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_mro_sweep.png"), dpi=150)
plt.close(fig)


# ===================================================================== lab task 4: faster device (Monte Carlo)
def mc(seeds, **kw):
    out = {}
    for name, h, tt in (("naive", 0.0, 0), ("3 dB + TTT 5", 3.0, 5), (f"chosen {H_BEST:.0f} dB + TTT {TTT_BEST}", H_BEST, TTT_BEST)):
        c, pp, late, mn = [], [], 0, []
        for sd in seeds:
            r = simulate(make_series(seed=sd, **kw), h, tt)
            c.append(r.count)
            pp.append(r.ping_pongs())
            mn.append(r.min_serving())
            late += r.sustained_below(THR)
        out[name] = (np.mean(c), np.mean(pp), late / len(seeds), np.median(mn))
    return out


say("=" * 78)
say("Lab task 4: slow versus fast device, 200 random realisations each")
seeds = range(300, 500)            # evaluation realisations, disjoint from the tuning set
slow = mc(seeds)
fast = mc(seeds, slope_s=-0.48, slope_n=0.48, start_s=-80.0, start_n=-112.0, walk_sd=0.3, flat=(33, 50))
for label, d in (("pedestrian (base model)", slow), ("high speed (3x slope, 2x walk, 1/3 edge time)", fast)):
    say(f"  {label}")
    for name, (c, pp, late, mn) in d.items():
        say(f"    {name:22s} mean handovers {c:5.2f}, mean ping-pongs {pp:5.2f}, "
            f"runs with sustained drop < -100 dBm: {100 * late:5.1f}%, median min serving {mn:6.1f} dBm")
SF = make_series(seed=SEED, slope_s=-0.48, slope_n=0.48, walk_sd=0.3, flat=(33, 50))
stacked(SF, [simulate(SF, 3.0, 5), simulate(SF, H_BEST, TTT_BEST)],
        ["fast device, 3 dB + TTT 5", f"fast device, chosen {H_BEST:.0f} dB + TTT {TTT_BEST}"],
        os.path.join(FIG, "fig_handover_fast_device.png"))
fig, ax = plt.subplots(figsize=(8, 3.6))
names = list(slow)
x = np.arange(len(names))
ax.bar(x - 0.2, [100 * slow[n][2] for n in names], 0.4, label="pedestrian")
ax.bar(x + 0.2, [100 * fast[n][2] for n in names], 0.4, label="high speed")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("% of runs with sustained\nserving < -100 dBm")
ax.set_title("Too-late-handover risk: same parameters, faster device (200 runs each)", fontsize=10)
ax.grid(axis="y", alpha=0.35)
ax.legend(fontsize=8)
for i, n in enumerate(names):
    ax.text(i - 0.2, 100 * slow[n][2] + 1, f"{slow[n][0]:.1f} HO", ha="center", fontsize=7)
    ax.text(i + 0.2, 100 * fast[n][2] + 1, f"{fast[n][0]:.1f} HO", ha="center", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_speed_risk.png"), dpi=150)
plt.close(fig)

# ===================================================================== lab task 5: biased offset
say("=" * 78)
say("Lab task 5: cell individual offset favouring B (load balancing), hysteresis 3 dB + TTT 5")
off_rows = []
for off in (0.0, 2.0, 4.0, 6.0):
    r = simulate(S, 3.0, 5, offsets={"B": off})
    rn = simulate(S, 0.0, 0, offsets={"B": off})
    off_rows.append((off, r, rn))
    say(f"  offset B +{off:.0f} dB: first handover at sample {r.events[0][0]:3d}, handovers {r.count}, "
        f"ping-pongs {r.ping_pongs()}, min serving {r.min_serving():6.1f} dBm | naive with the same offset: "
        f"{rn.count} handovers, first at {rn.events[0][0]}")
stacked(S, [x[1] for x in off_rows[::2]], [f"3 dB + TTT 5, offset toward B +{x[0]:.0f} dB" for x in off_rows[::2]],
        os.path.join(FIG, "fig_handover_offset.png"))

# ===================================================================== lab task 6: sectoring
say("=" * 78)
say("Lab task 6: 120-degree sectoring, N = 7")


def interferers(N, sectors, n=4.0):
    """Count first-tier co-channel cells whose transmitting antenna points at the victim (downlink).
    Victim: mobile in the sector of the centre cell whose boresight is at 30 deg (sector 1, which
    covers azimuths -30 to 90 deg of a pointy-top hexagon).
    A co-channel sector uses the same boresight; it radiates over boresight +/- 180/sectors deg."""
    i, j = SHIFT[N]
    first_tier = []
    for k in range(6):                                     # rotate (i, j) by 60 deg steps in axial coords
        q, r = i, j
        for _ in range(k):
            q, r = -r, q + r
        first_tier.append(center(q, r))
    half = 180.0 / sectors
    count, contrib = 0, 0.0
    for (x, y) in first_tier:
        d = math.hypot(x, y)
        az = math.degrees(math.atan2(-y, -x))               # direction interferer -> victim cell
        diff = (az - 30.0 + 180) % 360 - 180
        if sectors == 1 or abs(diff) < half - 1e-9 or abs(abs(diff) - half) < 1e-9 and diff > 0:
            count += 1
            contrib += d ** (-n)
    return count, contrib


for N in (3, 4, 7, 12):
    Q = math.sqrt(3 * N)
    for s in (1, 3, 6):
        c, I = interferers(N, s)
        sir = 10 * math.log10(1.0 / I)                       # serving distance R = 1
        say(f"  N = {N:2d}, {s} sector(s) per site: {c} first-tier interferer(s), SIR = {sir:5.1f} dB")
fig, ax = plt.subplots(figsize=(7.2, 7.2))
cells = layout(7, 4)


def corner(x, y, ang, R=1.0):
    return (x + R * math.cos(math.radians(ang)), y + R * math.sin(math.radians(ang)))


SECT = ((330, 30, 90), (90, 150, 210), (210, 270, 330))    # three rhombi = three 120-degree sectors
for (q, r, x, y, g) in cells:
    for sidx, (a1, a2, a3) in enumerate(SECT):
        ax.add_patch(Polygon([(x, y), corner(x, y, a1), corner(x, y, a2), corner(x, y, a3)],
                             facecolor=CMAP((g - 1) % 20), alpha=[0.95, 0.55, 0.25][sidx], edgecolor="black", lw=0.3))
    ax.text(x + 0.35, y + 0.2, str(g), ha="center", va="center", fontsize=7, fontweight="bold")
i, j = SHIFT[7]
for k in range(6):
    q, r = i, j
    for _ in range(k):
        q, r = -r, q + r
    x, y = center(q, r)
    az = math.degrees(math.atan2(-y, -x))
    hit = abs((az - 30 + 180) % 360 - 180) <= 60 + 1e-9
    ax.annotate("", xy=(0.3, 0.3), xytext=(x + 0.3, y + 0.3),
                arrowprops=dict(arrowstyle="->", lw=1.8 if hit else 0.8, color="#d62728" if hit else "gray",
                                ls="-" if hit else "--"))
ax.add_patch(Polygon([(0, 0), corner(0, 0, 330), corner(0, 0, 30), corner(0, 0, 90)], fill=False,
                     edgecolor="#d62728", lw=2.5))
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("N = 7, three 120-degree sectors per site (darkest = sector 1, boresight 30 deg).\n"
             "Red arrows: co-channel sector-1 beams that point at the victim sector (2 of 6)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_sectoring_n7.png"), dpi=150)
plt.close(fig)

with open(os.path.join(HERE, "results_3_1.txt"), "w") as f:
    f.write("\n".join(LOG) + "\n")
print("results written; figures in", FIG)
