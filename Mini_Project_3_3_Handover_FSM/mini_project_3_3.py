"""
Mini Project 3.3 - run the connection/handover state machine through the scripted A -> B -> C drive
under three policies, write the full transition logs, count handovers, radio link failures and
ping-pongs, and draw the signal/serving-cell figure.

    python Mini_Project_3_3_Handover_FSM/mini_project_3_3.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fsm  # noqa: E402
from fsm import scenario, run, naive_policy, make_hyst_ttt_policy, TICK_MS, Q_OUT  # noqa: E402

FIG = os.path.join(HERE, "figures")
LOGS = os.path.join(HERE, "logs")
os.makedirs(FIG, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


x, rsrp = scenario(seed=3)
np.savetxt(os.path.join(HERE, "scenario_signals.csv"),
           np.column_stack([np.arange(len(x)) * TICK_MS, x, rsrp["A"], rsrp["B"], rsrp["C"]]),
           delimiter=",", header="t_ms,position_m,rsrp_A_dBm,rsrp_B_dBm,rsrp_C_dBm", comments="", fmt="%.3f")
say(f"Scenario: road past sites A (x=0), B (x=1000 m), C (x=2000 m), 60 m offset; 15 m/s (54 km/h); "
    f"{len(x)} ticks of {TICK_MS} ms = {len(x) * TICK_MS / 1000:.1f} s")
say(f"  RSRP = 15 dBm/RE - (128.1 + 37.6 log10 d_km) + shadowing (sigma 4 dB, decorrelation 50 m) "
    f"+ measurement noise (sigma 1.5 dB)")
say(f"  Q_OUT = {Q_OUT} dBm, T310 = {fsm.T310_TICKS * TICK_MS} ms, Q_RXLEVMIN = {fsm.Q_RXLEVMIN} dBm, "
    f"HO execution = {fsm.HO_EXEC_TICKS * TICK_MS} ms, preamble collision probability = {fsm.P_COLLISION}")
stack = np.vstack([rsrp[c] for c in "ABC"])
order = np.argsort(-stack, axis=0)                      # strongest cell first, per tick
for p, q in (("A", "B"), ("B", "C")):
    ip, iq = "ABC".index(p), "ABC".index(q)
    top2 = ((order[0] == ip) & (order[1] == iq)) | ((order[0] == iq) & (order[1] == ip))
    close = top2 & (np.abs(rsrp[p] - rsrp[q]) < 3.0)
    flips = np.where(top2[1:] & top2[:-1] & (np.sign(rsrp[p] - rsrp[q])[1:] != np.sign(rsrp[p] - rsrp[q])[:-1]))[0]
    idx = np.where(close)[0]
    say(f"  {p}/{q} are the two strongest cells and within 3 dB of each other for {len(idx) * TICK_MS / 1000:.1f} s "
        f"(x = {x[idx.min()]:.0f}-{x[idx.max()]:.0f} m); their order flips {len(flips)} times there: decision region")

policies = [naive_policy, make_hyst_ttt_policy(3, 320), make_hyst_ttt_policy(10, 5120)]
policies[2].label = "over-tuned: hysteresis 10 dB + TTT 5120 ms"
logs = []
say("=" * 90)
say(f"{'policy':44s}{'HO':>5s}{'HO fail':>8s}{'RLF':>5s}{'ping-pong':>10s}{'not connected':>15s}{'RACH':>6s}")
for pol in policies:
    L = run(x, rsrp, pol)
    logs.append(L)
    rach = sum(1 for e in L.events if e.dst == "RA_MSG1" and e.src == "IDLE")
    rlf_only = [e for e in L.events if e.src == "CONNECTED" and e.dst == "IDLE"]
    say(f"{pol.label:44s}{len(L.handovers()):>5d}{len(L.ho_failures()):>8d}{len(rlf_only):>5d}"
        f"{len(L.ping_pongs()):>10d}{L.outage_ms() / 1000:>13.2f} s{rach:>6d}")
    tag = {0: "naive", 1: "hyst3_ttt320", 2: "overtuned_hyst10_ttt5120"}[policies.index(pol)]
    with open(os.path.join(LOGS, f"transitions_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "from_state", "to_state", "cell", "trigger"])
        for e in L.events:
            w.writerow([e.t_ms, e.src, e.dst, e.cell, e.trigger])
say("  (HO fail = handover started but failed; RLF = radio link failure while CONNECTED; both return to IDLE."
    " 'not connected' includes attach, handover execution and re-establishment time)")

say("=" * 90)
for L in logs:
    say(f"Transition log excerpt - {L.policy} ({len(L.events)} transitions in total)")
    show = L.events[:5]
    if L.policy.startswith("naive"):
        pp = L.ping_pongs()
        t1 = pp[0][0]
        show += [e for e in L.events if t1 - 60 <= e.t_ms <= t1 + 400][:8]
    else:
        for k, e in enumerate(L.events[5:], start=5):
            if e.dst in ("HANDOVER", "IDLE") or e.src == "HANDOVER":
                show.append(e)
                if e.dst == "IDLE":
                    show += L.events[k + 1:k + 6]              # the re-establishment that follows
        show = show[:22]
    seen = set()
    for e in show:
        key = (e.t_ms, e.src, e.dst)
        if key in seen:
            continue
        seen.add(key)
        say(f"  t={e.t_ms / 1000:7.2f} s  {e.src:>9s} -> {e.dst:<9s} cell {e.cell}: {e.trigger}")
    pp = L.ping_pongs()
    if pp:
        say(f"  ping-pongs: {len(pp)}, first: {pp[0][2]} -> {pp[0][3]} at {pp[0][0] / 1000:.2f} s and back at "
            f"{pp[0][1] / 1000:.2f} s")

# ------------------------------------------------------------------ RACH contention check
say("=" * 90)
xs = np.full(40, 300.0)
static = {"A": np.full(40, -90.0), "B": np.full(40, -125.0), "C": np.full(40, -125.0)}
att_times, collisions = [], 0
example = None
for sd in range(1000):
    L = run(xs, static, naive_policy, seed=sd)
    col_ev = [e for e in L.events if "collision" in e.trigger]
    collisions += len(col_ev)
    if col_ev and example is None:
        example = L.events[:12]
    conn = next((e.t_ms for e in L.events if e.dst == "CONNECTED"), None)
    if conn is not None:
        att_times.append(conn)
att = np.array(att_times)
say(f"RACH check: 1000 attach attempts on a static -90 dBm cell, preamble collision probability {fsm.P_COLLISION}")
say(f"  connected in {len(att)}/1000; collisions observed {collisions}; attach time: 80 ms (no collision) in "
    f"{np.mean(att == 80) * 100:.1f}% of attaches, mean {att.mean():.1f} ms, max {att.max()} ms")
say("  example with a collision:")
for e in example:
    say(f"    t={e.t_ms:4d} ms {e.src:>9s} -> {e.dst:<9s} {e.trigger}")

# ------------------------------------------------------------------ figure
t = np.arange(len(x)) * TICK_MS / 1000
col = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
fig, axes = plt.subplots(len(logs), 1, figsize=(11, 2.7 * len(logs) + 0.5), sharex=True)
for ax, L in zip(axes, logs):
    for c in "ABC":
        ax.plot(t, rsrp[c], color=col[c], lw=0.6, alpha=0.7, label=f"cell {c}")
    serv = np.array([np.nan if s is None else rsrp[s][i] for i, s in enumerate(L.serving)])
    ax.plot(t, serv, color="black", lw=1.0, label="serving cell")
    for c in "ABC":
        m = np.array([s == c for s in L.serving])
        ax.fill_between(t, -62, -66, where=m, color=col[c], step="mid")
    m = np.array([s not in ("CONNECTED", "HANDOVER") for s in L.state])
    ax.fill_between(t, -62, -66, where=m, color="#d62728", step="mid", label="not connected")
    for e in L.events:
        if e.src == "HANDOVER" and e.dst == "CONNECTED":
            ax.axvline(e.t_ms / 1000, color="gray", lw=0.4, alpha=0.6)
        if e.dst == "IDLE" and e.src in ("CONNECTED", "HANDOVER"):
            ax.axvline(e.t_ms / 1000, color="#d62728", lw=1.6)
            ax.text(e.t_ms / 1000 + 0.6, -72, "HOF" if e.src == "HANDOVER" else "RLF", color="#d62728", fontsize=8,
                    fontweight="bold")
    ax.axhline(Q_OUT, color="#d62728", ls=":", lw=1)
    ax.set_ylim(-125, -60)
    ax.set_ylabel("RSRP (dBm)")
    rlf = sum(1 for e in L.events if e.src == "CONNECTED" and e.dst == "IDLE")
    ax.set_title(f"{L.policy}: {len(L.handovers())} handovers, {len(L.ping_pongs())} ping-pongs, "
                 f"{rlf} RLF, {len(L.ho_failures())} HO failure(s)", fontsize=9.5)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, ncol=5, loc="lower center")
axes[-1].set_xlabel("time (s)   [band at top: serving cell, red = not connected; dotted line = Q_OUT -110 dBm]")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_fsm_three_policies.png"), dpi=150)
plt.close(fig)

with open(os.path.join(HERE, "results_3_3.txt"), "w") as f:
    f.write("\n".join(OUT) + "\n")
print("logs in", LOGS, "figure in", FIG)
