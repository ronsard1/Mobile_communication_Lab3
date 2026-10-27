"""
handover.py - synthetic RSRP series and handover decision logic (Mini Project 3.1 part (b)).

All triggers share one engine, `simulate(series, hyst_db, ttt, offsets)`:
  * series   : dict cell -> RSRP array (dBm), one sample per measurement period
  * hyst_db  : hysteresis margin; a neighbour is a candidate only if it beats the serving cell by more
  * ttt      : time-to-trigger in samples; the candidate condition must hold that many consecutive
               samples (ttt = 0 or 1 means "immediately")
  * offsets  : optional per-cell individual offsets (dB) added to a cell's reading when it is a candidate,
               used for load-balancing bias (lab task 5)
With hyst_db = 0 and ttt = 0 the engine is the naive trigger. With several neighbours the strongest
candidate (after offsets) at each sample is the one that is tracked (lab task 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def make_series(n=300, seed=7, start_s=-80.0, slope_s=-0.16, start_n=-112.0, slope_n=0.16,
                walk_sd=0.15, noise_sd=1.2, flat=(100, 150), extra=None):
    """Serving cell falls, neighbour rises; between samples flat[0] and flat[1] both trends pause so
    the two curves stay close for a stretch (the device moves along the cell edge)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    # piecewise time axis: progress stops inside the flat stretch
    prog = t.copy()
    if flat:
        a, b = flat
        prog = np.where(t < a, t, np.where(t < b, a, t - (b - a)))
    trend_s = start_s + slope_s * prog                       # device moving away from A
    trend_n = start_n + slope_n * prog                       # ... and towards B
    s = trend_s + np.cumsum(rng.normal(0, walk_sd, n)) + rng.normal(0, noise_sd, n)
    m = trend_n + np.cumsum(rng.normal(0, walk_sd, n)) + rng.normal(0, noise_sd, n)
    out = {"A": s, "B": m}
    if extra:                                                # optional third cell: (start, slope, name)
        st, sl, name = extra
        out[name] = st + sl * prog + np.cumsum(rng.normal(0, walk_sd, n)) + rng.normal(0, noise_sd, n)
    return out


@dataclass
class HOResult:
    serving: np.ndarray                     # serving cell label per sample
    serving_rsrp: np.ndarray
    events: list = field(default_factory=list)   # (sample, from, to)

    @property
    def count(self):
        return len(self.events)

    def ping_pongs(self, window=10):
        """A handover back to the cell just left within `window` samples."""
        pp = 0
        for (t1, f1, to1), (t2, f2, to2) in zip(self.events, self.events[1:]):
            if to2 == f1 and t2 - t1 <= window:
                pp += 1
        return pp

    def min_serving(self):
        return float(self.serving_rsrp.min())

    def samples_below(self, thr):
        return int((self.serving_rsrp < thr).sum())

    def sustained_below(self, thr=-100.0, k=3):
        """True if serving RSRP stayed below thr for k consecutive samples (too-late handover risk).
        A single noisy sample below thr is not counted; a sustained drop is."""
        run = 0
        for v in self.serving_rsrp:
            run = run + 1 if v < thr else 0
            if run >= k:
                return True
        return False


def simulate(series: dict, hyst_db=0.0, ttt=0, offsets=None, start=None) -> HOResult:
    cells = list(series)
    offsets = offsets or {}
    n = len(next(iter(series.values())))
    serving = start or cells[0]
    streak, cand = 0, None
    serv_lab = []
    serv_val = np.empty(n)
    events = []
    need = max(int(ttt), 1)
    for t in range(n):
        s_val = series[serving][t] + offsets.get(serving, 0.0)
        best, best_val = None, -np.inf
        for c in cells:
            if c == serving:
                continue
            v = series[c][t] + offsets.get(c, 0.0)
            if v > s_val + hyst_db and v > best_val:          # entering condition (A3-like event)
                best, best_val = c, v
        if best is None:
            streak, cand = 0, None                            # condition broken: reset the timer
        else:
            streak = streak + 1 if best == cand else 1        # a different best candidate restarts TTT
            cand = best
            if streak >= need:
                events.append((t, serving, best))
                serving = best
                streak, cand = 0, None
        serv_lab.append(serving)
        serv_val[t] = series[serving][t]
    return HOResult(np.array(serv_lab), serv_val, events)
