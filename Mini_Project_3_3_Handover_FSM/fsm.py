"""
fsm.py - device connection lifecycle as an explicit finite-state machine (Mini Project 3.3).

States
  IDLE              no RRC connection; camps on the strongest cell when it is above Q_RXLEVMIN
  RA_MSG1           Random Access step 1: preamble sent on the PRACH
  RA_MSG2           step 2: Random Access Response received, carrying the timing advance (TA)
  RA_MSG3           step 3: RRC connection request sent on the granted uplink resource
  RA_MSG4           step 4: contention resolution; success -> CONNECTED, collision -> back off, retry MSG1
  CONNECTED         active link to a serving cell; measurements feed the handover POLICY
  HANDOVER          handover in progress (HO command from source, contention-free access at target)

Transitions (trigger)
  IDLE -> RA_MSG1                 a cell is above Q_RXLEVMIN (data to send / attach)
  RA_MSG1 -> RA_MSG2              RAR received within the window (cell above Q_RXLEVMIN)
  RA_MSG1 -> RA_MSG1              no RAR (cell too weak), preamble power ramped, retry
  RA_MSG2 -> RA_MSG3              uplink grant + TA applied
  RA_MSG3 -> RA_MSG4              connection request transmitted
  RA_MSG4 -> CONNECTED            contention resolved in our favour
  RA_MSG4 -> RA_MSG1              preamble collision (another device chose the same preamble)
  RA_* -> IDLE                    preamble attempts exhausted (RACH failure)
  CONNECTED -> HANDOVER           POLICY returns a target cell (naive or hysteresis + TTT)
  HANDOVER -> CONNECTED           access at the target completed (serving cell := target)
  HANDOVER -> IDLE                HO failure: HO command not decodable (source < Q_OUT) or target too weak
  CONNECTED -> IDLE               radio link failure: serving < Q_OUT for T310 consecutive ticks
After any failure the device stays IDLE for REEST_TICKS (cell search) before a new random access.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TICK_MS = 20                  # one FSM step = one measurement report = 20 ms
Q_OUT = -110.0                # dBm: link considered out of sync below this
Q_RXLEVMIN = -115.0           # dBm: minimum level for cell selection / random access
T310_TICKS = 25               # 500 ms below Q_OUT -> radio link failure
HO_EXEC_TICKS = 2             # 40 ms handover execution (HO command + access at target)
P_COLLISION = 0.10            # probability that contention resolution fails (preamble collision)
MAX_PREAMBLES = 10
PINGPONG_MS = 1000            # handover back to the previous cell within 1 s = ping-pong
REEST_TICKS = 50              # 1 s cell search / re-establishment after a failure before random access


# ------------------------------------------------------------------ handover policies (same signature)
def naive_policy(serving, rsrp_now, state):
    """Hand over the moment any neighbour is stronger than the serving cell."""
    best = max((c for c in rsrp_now if c != serving), key=lambda c: rsrp_now[c])
    return best if rsrp_now[best] > rsrp_now[serving] else None


def make_hyst_ttt_policy(hyst_db, ttt_ms):
    """Neighbour must beat serving by more than hyst_db for ttt_ms continuously (A3 event + TTT)."""
    need = max(1, int(round(ttt_ms / TICK_MS)))

    def policy(serving, rsrp_now, state):
        best = max((c for c in rsrp_now if c != serving), key=lambda c: rsrp_now[c])
        if rsrp_now[best] > rsrp_now[serving] + hyst_db:
            state["streak"] = state.get("streak", 0) + 1 if state.get("cand") == best else 1
            state["cand"] = best
            if state["streak"] >= need:
                state["streak"], state["cand"] = 0, None
                return best
        else:
            state["streak"], state["cand"] = 0, None              # condition broken: timer resets
        return None

    policy.label = f"hysteresis {hyst_db:g} dB + TTT {ttt_ms} ms"
    return policy


naive_policy.label = "naive (no margin, no TTT)"


# ------------------------------------------------------------------ scenario
def scenario(seed=3, speed_mps=15.0, x0=50.0, x1=1950.0, sigma_sh=4.0, decorr_m=50.0, meas_sd=1.5):
    """Device drives along a straight road past sites A (x=0), B (x=1000 m), C (x=2000 m), each 60 m
    off the road. RSRP = 15 dBm per RE - (128.1 + 37.6 log10 d_km) + correlated shadowing + measurement noise."""
    rng = np.random.default_rng(seed)
    n = int((x1 - x0) / speed_mps / (TICK_MS / 1000)) + 1
    x = x0 + speed_mps * (TICK_MS / 1000) * np.arange(n)
    sites = {"A": 0.0, "B": 1000.0, "C": 2000.0}
    step = speed_mps * TICK_MS / 1000
    rho = math.exp(-step / decorr_m)
    out = {}
    for c, xs in sites.items():
        d_km = np.sqrt((x - xs) ** 2 + 60.0 ** 2) / 1000
        sh = np.empty(n)
        sh[0] = rng.normal(0, sigma_sh)
        for k in range(1, n):                                        # Gudmundson AR(1) shadowing
            sh[k] = rho * sh[k - 1] + math.sqrt(1 - rho ** 2) * rng.normal(0, sigma_sh)
        out[c] = 15.0 - (128.1 + 37.6 * np.log10(d_km)) + sh + rng.normal(0, meas_sd, n)
    return x, out


# ------------------------------------------------------------------ the machine
@dataclass
class Event:
    t_ms: int
    src: str
    dst: str
    cell: str
    trigger: str


@dataclass
class RunLog:
    policy: str
    events: list = field(default_factory=list)
    serving: list = field(default_factory=list)    # serving cell per tick (None when not connected)
    state: list = field(default_factory=list)

    def handovers(self):
        return [e for e in self.events if e.src == "HANDOVER" and e.dst == "CONNECTED"]

    def ho_failures(self):
        return [e for e in self.events if e.src == "HANDOVER" and e.dst == "IDLE"]

    def rlfs(self):
        return [e for e in self.events if e.dst == "IDLE" and e.src in ("CONNECTED", "HANDOVER")]

    def ping_pongs(self):
        hos = []                                        # (t, from_cell, to_cell)
        prev = None
        for e in self.events:
            if e.src == "CONNECTED" and e.dst == "HANDOVER":
                prev = e.trigger.split("serving ")[1].split(" ")[0]
            if e.src == "HANDOVER" and e.dst == "CONNECTED":
                hos.append((e.t_ms, prev, e.cell))
        pp = []
        for (t1, f1, to1), (t2, f2, to2) in zip(hos, hos[1:]):
            if to2 == f1 and f2 == to1 and t2 - t1 <= PINGPONG_MS:
                pp.append((t1, t2, f1, to1))
        return pp

    def outage_ms(self):
        return TICK_MS * sum(1 for s in self.state if s != "CONNECTED")


def run(x, rsrp: dict, policy, seed=11) -> RunLog:
    rng = np.random.default_rng(seed)
    log = RunLog(policy.label)
    cells = list(rsrp)
    n = len(x)
    st = "IDLE"
    serving = None
    target = None
    pstate: dict = {}
    below = 0
    attempts = 0
    backoff = 0
    ho_left = 0
    idle_hold = 0
    sites = {"A": 0.0, "B": 1000.0, "C": 2000.0}

    def go(t, new, cell, why):
        nonlocal st
        log.events.append(Event(t * TICK_MS, st, new, cell or "-", why))
        st = new

    for t in range(n):
        now = {c: float(rsrp[c][t]) for c in cells}
        if st == "IDLE":
            best = max(now, key=now.get)
            if idle_hold > 0:
                idle_hold -= 1                                        # still searching after a failure
            elif now[best] >= Q_RXLEVMIN:
                serving, attempts, backoff = best, 0, 0
                go(t, "RA_MSG1", best, f"camped on strongest cell {best} ({now[best]:.1f} dBm), send preamble")
        elif st == "RA_MSG1":
            if backoff > 0:
                backoff -= 1
            elif now[serving] >= Q_RXLEVMIN:
                d = math.hypot(x[t] - sites[serving], 60.0)
                ta = round(2 * d / 3e8 / (16 / 30.72e6))            # TA in units of 16 Ts = 0.52 us
                go(t, "RA_MSG2", serving, f"RAR received, timing advance TA = {ta} ({2 * d / 3e8 * 1e6:.2f} us)")
            else:
                attempts += 1
                if attempts >= MAX_PREAMBLES:
                    go(t, "IDLE", serving, "RACH failure: preamble attempts exhausted")
                    serving = None
        elif st == "RA_MSG2":
            go(t, "RA_MSG3", serving, "uplink grant used: RRC connection request sent")
        elif st == "RA_MSG3":
            go(t, "RA_MSG4", serving, "awaiting contention resolution")
        elif st == "RA_MSG4":
            if rng.random() < P_COLLISION:
                attempts += 1
                backoff = int(rng.integers(1, 4))
                if attempts >= MAX_PREAMBLES:
                    go(t, "IDLE", serving, "RACH failure: attempts exhausted")
                    serving = None
                else:
                    go(t, "RA_MSG1", serving, f"preamble collision, back off {backoff} tick(s), retry")
            else:
                below, pstate = 0, {}
                go(t, "CONNECTED", serving, "contention resolved: RRC connected")
        elif st == "CONNECTED":
            below = below + 1 if now[serving] < Q_OUT else 0
            if below >= T310_TICKS:
                go(t, "IDLE", serving, f"RADIO LINK FAILURE: serving {serving} below {Q_OUT:.0f} dBm "
                                       f"for {T310_TICKS * TICK_MS} ms (too-late handover)")
                serving, idle_hold = None, REEST_TICKS
            else:
                tgt = policy(serving, now, pstate)
                if tgt is not None:
                    target, ho_left = tgt, HO_EXEC_TICKS
                    go(t, "HANDOVER", tgt, f"policy: serving {serving} {now[serving]:.1f} dBm, "
                                           f"target {tgt} {now[tgt]:.1f} dBm")
        elif st == "HANDOVER":
            if now[serving] < Q_OUT and ho_left == HO_EXEC_TICKS:
                go(t, "IDLE", target, f"HANDOVER FAILURE: HO command lost, source {serving} "
                                      f"{now[serving]:.1f} dBm < {Q_OUT:.0f} dBm")
                serving, idle_hold = None, REEST_TICKS
            elif now[target] < Q_RXLEVMIN:
                go(t, "IDLE", target, f"HANDOVER FAILURE: target {target} too weak ({now[target]:.1f} dBm)")
                serving, idle_hold = None, REEST_TICKS
            else:
                ho_left -= 1
                if ho_left <= 0:
                    serving, below, pstate = target, 0, {}
                    go(t, "CONNECTED", target, "handover complete (contention-free access at target)")
        log.serving.append(serving if st in ("CONNECTED", "HANDOVER") else None)
        log.state.append(st)
    return log
