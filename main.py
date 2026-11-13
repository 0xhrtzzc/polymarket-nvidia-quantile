#!/usr/bin/env python3
"""Polymarket NVIDIA CEO-forecast entry: pools every market's width-normalised logit moves into one shape, scales it back by each market's own width, and emits five quantiles per market per minute out to a four-hour horizon.
The two tails are scaled differently because each side carries its own multiplier (TAIL_DN, TAIL_UP) ramped up with distance from the median, so a fit that finds the upside and downside tails of unequal size can say so instead of forcing one symmetric factor on both.
"""
import json
import math
import os

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
HORIZON = 240
EPS = 1e-6

# --- tuned on the practice cases and held-out cuts of the same history ---
LAM = 0.32
MED = 0.0
TAIL_UP = 1.17
TAIL_DN = 1.17
REG = 0.46
SHRINK = 0.09
DRIFT = 0.0
POOL = 0.5
DELTA = 0.79
DELTA_MIN_SAMPLES = 250
TAU = 1913.0
DECAY = 1440.0       # weight decay beyond TAU (minutes)
STRIDE = 3           # sample every N minutes: the 8-second cap is a hard fail
BASE_SIGMA = 0.0025  # the house baseline's price-space spread per square-root minute
# Fewer moves than this in a market's conditioned sample and the horizon is dropped from its own table; a market left with no horizon falls back to the plain baseline spread.
SAMPLES = 30
Z = {0.05: -1.644854, 0.25: -0.674490, 0.5: 0.0, 0.75: 0.674490, 0.95: 1.644854}
HORIZONS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 240]
LEVELS = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98]
I_PROB = {0.02: -2.053749, 0.05: -1.644854, 0.10: -1.281552, 0.25: -0.674490, 0.50: 0.0,
          0.75: 0.674490, 0.90: 1.281552, 0.95: 1.644854, 0.98: 2.053749}


def logit(p):
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(u):
    if u >= 0:
        return 1.0 / (1.0 + math.exp(-u))
    z = math.exp(u)
    return z / (1.0 + z)


def wquantile(vals_w, q):
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


def market_offsets(prices):
    """Per-horizon logit-move quantiles for one market, conditioned and recency-weighted."""
    u = [logit(p) for p in prices]
    n = len(u)
    u0 = u[-1]
    for delta in (DELTA, 1.0, 1e9):
        sel = [i for i in range(0, n, STRIDE) if abs(u[i] - u0) <= delta]
        if len(sel) >= DELTA_MIN_SAMPLES or delta >= 1e9:
            break
    w = {}
    for i in sel:
        age = n - 1 - i
        w[i] = 1.0 if age <= TAU else math.exp(-(age - TAU) / DECAY)
    table, raw = {}, {}
    for h in HORIZONS:
        pairs = [(u[i + h] - u[i], w[i]) for i in sel if i + h < n and w[i] > 1e-3]
        if len(pairs) < SAMPLES:
            continue
        q = [wquantile(list(pairs), lv) for lv in LEVELS]
        width = q[LEVELS.index(0.75)] - q[LEVELS.index(0.25)]
        table[h] = {"q": q, "n": len(pairs), "w": width}
        if width > 1e-9:                      # normalise by this market's own width so a quiet
            raw[h] = [(v / width, wt) for v, wt in pairs]   # market and a jumpy one can pool
    return u0, table, raw


def pooled_shape(per_market):
    """The pooled shape: quantiles of every market's width-normalised move, per horizon."""
    pooled = {}
    for h in HORIZONS:
        pairs = []
        for _, _, raw in per_market:
            pairs.extend(raw.get(h, ()))
        if len(pairs) < 200:
            continue
        pooled[h] = [wquantile(list(pairs), lv) for lv in LEVELS]
    return pooled


def shape_value(shape, tau):
    """Interpolate a level table to a scored quantile level."""
    for k in range(len(LEVELS) - 1):
        if LEVELS[k] <= tau <= LEVELS[k + 1]:
            lo, hi = LEVELS[k], LEVELS[k + 1]
            if hi == lo:
                return shape[k]
            t = (tau - lo) / (hi - lo)
            return shape[k] + (shape[k + 1] - shape[k]) * t
    return shape[-1] if tau > LEVELS[-1] else shape[0]


def forecast_case(case):
    prices = {m["id"]: [float(p) for p in m["prices"] if p is not None] or [0.5]
              for m in case["markets"]}
    per = {mid: market_offsets(p) for mid, p in prices.items()}
    pooled = pooled_shape(list(per.values()))
    out = {}
    for mid, (u0, table, _raw) in per.items():
        rows = []
        for h in range(1, HORIZON + 1):
            hs = sorted(table)
            if not hs:
                rows.append([min(max(sigmoid(u0) + Z[t] * BASE_SIGMA * math.sqrt(h), 0.0), 1.0)
                             for t in QUANTILES])
                continue
            lo = max([x for x in hs if x <= h], default=hs[0])
            hi = min([x for x in hs if x >= h], default=hs[-1])
            if lo == hi:
                own, own_w = table[lo]["q"], table[lo]["w"]
            else:
                t = (math.log(h) - math.log(lo)) / (math.log(hi) - math.log(lo))
                own = [table[lo]["q"][k] + (table[hi]["q"][k] - table[lo]["q"][k]) * t
                       for k in range(len(LEVELS))]
                wl, wh = table[lo]["w"], table[hi]["w"]
                own_w = wl + (wh - wl) * t
            width = own_w
            # pooled shape for the same horizon, scaled back to this market's own width
            if pooled:
                phs = sorted(pooled)
                plo = max([x for x in phs if x <= h], default=phs[0])
                phi = min([x for x in phs if x >= h], default=phs[-1])
                if plo == phi:
                    ps = pooled[plo]
                else:
                    t = (math.log(h) - math.log(plo)) / (math.log(phi) - math.log(plo))
                    ps = [pooled[plo][k] + (pooled[phi][k] - pooled[plo][k]) * t for k in range(len(LEVELS))]
                off = [(1.0 - POOL) * own[k] + POOL * (ps[k] * width) for k in range(len(LEVELS))]
            else:
                off = own
            iqr = off[LEVELS.index(0.75)] - off[LEVELS.index(0.25)]
            sig = iqr / 1.348980 if iqr > 0 else 0.0
            drift = DRIFT * (h / 60.0)
            row = []
            for tau in QUANTILES:
                e = shape_value(off, tau) * (MED if tau == 0.5 else 1.0)
                if REG > 0.0 and sig > 0.0 and tau != 0.5:
                    e = (1.0 - REG) * e + REG * (Z[tau] * sig)
                if tau == 0.5:
                    sc = 1.0
                else:
                    tail = TAIL_UP if tau > 0.5 else TAIL_DN
                    sc = 1.0 + (tail - 1.0) * (abs(tau - 0.5) / 0.45)
                q = sigmoid(u0 + drift + LAM * sc * e)
                if SHRINK > 0.0:
                    b = min(max(sigmoid(u0) + Z[tau] * BASE_SIGMA * math.sqrt(h), 0.0), 1.0)
                    q = q * (1.0 - SHRINK) + b * SHRINK
                row.append(min(max(q, 0.0), 1.0))
            row.sort()
            rows.append(row)
        out[mid] = rows
    return out


def main():
    ind = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    outd = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(outd, exist_ok=True)
    for name in sorted(os.listdir(ind)):
        if not name.endswith(".in"):
            continue
        case = json.load(open(os.path.join(ind, name)))
        fc = forecast_case(case)
        forecasts = [{"id": mid, "q": [[round(min(max(v, 0.0), 1.0), 6) for v in r] for r in rows]}
                     for mid, rows in fc.items()]
        cid = case.get("case") or name[:-3]
        with open(os.path.join(outd, cid + ".out"), "w") as f:
            json.dump({"forecasts": forecasts}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
