#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry, v3: conditioned, recency-weighted empirical quantiles.

A price is a bounded probability, so the state space is its logit. The forecast for a market is the
empirical distribution of its own logit moves, conditioned on the market sitting where it sits now
(samples whose logit is far from the checkpoint's are about a different regime) and weighted toward
the recent past (yesterday's volatility is a better guide than three days ago). The quantiles are read
off that distribution directly, so a market that mostly sits still gets a band that mostly sits still
— an atom at the current price is a feature, not something to smooth away — while one that jumps gets
a fat band at exactly the horizon where jumps happen.

The outer levels are regularised toward a scaled normal, because 5% of a few hundred samples is a
noisy number and the tails are where a wrong band costs most.
"""
import json
import math
import os

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
HORIZON = 240
EPS = 1e-6

# --- tuned on the practice cases and held-out cuts of the same history ---
LAM = 0.3216
TAIL = 1.1673
SHRINK = 0.0941
DRIFT = 0.0
DELTA = 0.7944
DELTA_MIN_SAMPLES = 250
TAU = 1913.4461
DECAY = 1440.0       # weight decay beyond TAU (minutes)
REG = 0.4587
BASE_SIGMA = 0.0025  # the house baseline's price-space spread per square-root minute
SAMPLES = 30
STRIDE = 3          # sample every N minutes: the cap is 8 s a case, and the tail quantiles
                    # barely move when every third minute is used instead of every one
Z = {0.05: -1.644854, 0.25: -0.674490, 0.5: 0.0, 0.75: 0.674490, 0.95: 1.644854}
Z_OUT = {0.05: -1.644854, 0.95: 1.644854}
HORIZONS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 240]
LEVELS = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98]


def logit(p):
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(u):
    if u >= 0:
        return 1.0 / (1.0 + math.exp(-u))
    z = math.exp(u)
    return z / (1.0 + z)


def wquantile(vals_w, q):
    """Weighted quantile of (value, weight) pairs."""
    if not vals_w:
        return 0.0
    vals_w.sort()
    total = sum(w for _, w in vals_w)
    if total <= 0:
        return vals_w[len(vals_w) // 2][0]
    target = q * total
    acc = 0.0
    for v, w in vals_w:
        acc += w
        if acc >= target:
            return v
    return vals_w[-1][0]


def build_table(u, sel, h):
    n = len(u)
    d = []
    for i in sel:
        j = i + h
        if j < n:
            d.append((u[j] - u[i], 1.0))
    return d


def offsets(u, u0):
    n = len(u)
    sel = [i for i in range(0, n, STRIDE) if abs(u[i] - u0) <= DELTA]
    if len(sel) < DELTA_MIN_SAMPLES:
        sel = [i for i in range(0, n, STRIDE) if abs(u[i] - u0) <= 1.0]
    if len(sel) < DELTA_MIN_SAMPLES:
        sel = list(range(0, n, STRIDE))
    # recency weights: the last TAU minutes count fully, older minutes decay
    w = {}
    for i in sel:
        age = n - 1 - i
        w[i] = 1.0 if age <= TAU else math.exp(-(age - TAU) / DECAY)
    table = {}
    for h in HORIZONS:
        pairs = []
        for i in sel:
            j = i + h
            if j < n and w[i] > 1e-3:
                pairs.append((u[j] - u[i], w[i]))
        if len(pairs) < SAMPLES:
            continue
        table[h] = [wquantile(list(pairs), q) for q in LEVELS]
    if not table:
        return None
    hs = sorted(table)
    out = []
    for h in range(1, HORIZON + 1):
        if h in table:
            out.append(table[h])
            continue
        lo = max([x for x in hs if x <= h], default=hs[0])
        hi = min([x for x in hs if x >= h], default=hs[-1])
        if lo == hi:
            out.append(table[lo])
        else:
            t = (math.log(h) - math.log(lo)) / (math.log(hi) - math.log(lo))
            out.append([table[lo][k] + (table[hi][k] - table[lo][k]) * t for k in range(len(LEVELS))])
    return out


def level_value(o, tau):
    for k in range(len(LEVELS) - 1):
        if LEVELS[k] <= tau <= LEVELS[k + 1]:
            lo, hi = LEVELS[k], LEVELS[k + 1]
            if hi == lo:
                return o[k]
            t = (tau - lo) / (hi - lo)
            return o[k] + (o[k + 1] - o[k]) * t
    return o[-1] if tau > LEVELS[-1] else o[0]


def forecast_market(prices):
    x = prices[-1]
    u = [logit(p) for p in prices]
    u0 = u[-1]
    offs = offsets(u, u0)
    rows = []
    for h in range(1, HORIZON + 1):
        if offs is None:
            rows.append([min(max(x + Z[t] * BASE_SIGMA * math.sqrt(h), 0.0), 1.0)
                         for t in QUANTILES])
            continue
        o = offs[h - 1]
        iqr = o[LEVELS.index(0.75)] - o[LEVELS.index(0.25)]
        sig = iqr / 1.348980 if iqr > 0 else 0.0
        drift = DRIFT * (h / 60.0)
        row = []
        for tau in QUANTILES:
            e = level_value(o, tau)
            if REG > 0.0 and sig > 0.0 and tau != 0.5:
                e = (1.0 - REG) * e + REG * (Z[tau] * sig)
            scale = 1.0 if tau == 0.5 else (1.0 + (TAIL - 1.0) * (abs(tau - 0.5) / 0.45))
            q = sigmoid(u0 + drift + LAM * scale * e)
            if SHRINK > 0.0:
                b = min(max(x + Z[tau] * BASE_SIGMA * math.sqrt(h), 0.0), 1.0)
                q = q * (1.0 - SHRINK) + b * SHRINK
            row.append(min(max(q, 0.0), 1.0))
        row.sort()
        rows.append(row)
    return rows


def main():
    ind = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    outd = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(outd, exist_ok=True)
    for name in sorted(os.listdir(ind)):
        if not name.endswith(".in"):
            continue
        case = json.load(open(os.path.join(ind, name)))
        forecasts = []
        for m in case["markets"]:
            prices = [float(p) for p in m["prices"] if p is not None] or [0.5]
            rows = [[round(min(max(v, 0.0), 1.0), 6) for v in r] for r in forecast_market(prices)]
            forecasts.append({"id": m["id"], "q": rows})
        cid = case.get("case") or name[:-3]
        with open(os.path.join(outd, cid + ".out"), "w") as f:
            json.dump({"forecasts": forecasts}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
