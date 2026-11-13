#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry, v4: pooled shape, per-market width.

Same state space as v3 — the logit of the price — but the quantiles no longer come from one market's
own few hundred conditioned minutes alone. Each market's moves are normalised by its own interquartile
range, the normalised moves of every market of the case are pooled, and the pooled shape is scaled back
by each market's own width. A market with four thousand minutes still gets its own width (that is what
separates a quiet market from a jumpy one) while the *shape* of the tail comes from tens of thousands
of pooled minutes, which is the only way the 5% and 95% levels stop being the noisiest numbers in the
forecast. A market with too little of its own history falls back to the pooled shape alone.
When a market's conditioned sample comes up short of the sample floor at a horizon, no row of its own is built there and the pooled shape scaled by that market's own width carries the level instead, while a market that survives nowhere — or a case whose pooled table never reaches its own floor — is priced straight off the baseline spread.
"""
import json
import math
import os

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
HORIZON = 240
EPS = 1e-6

# --- tuned on the practice cases and held-out cuts of the same history ---
# per-horizon rows of quantile multipliers (short / mid / long), each row read 5,25,50,75,95; swept over the practice cases and held-out cuts of the same history
SCALES = [[0.2, 1.0, 0.1, 0.25, 0.2], [0.25, 0.6, 0.1, 0.1, 0.3], [0.5, 0.5, 0.1, 0.1, 0.45]]
                                         # fitted on the practice cases and held-out cuts
MED = 0.0
TAIL_UP = 1.0        # extra scale on the levels above the median (0.75 / 0.95)
TAIL_DN = 1.0        # extra scale on the levels below the median (0.05 / 0.25)
REG = 0.047
SHRINK = 0.0233
DRIFT = 0.0          # logit drift per hour
POOL = 0.4543
POOL_ADAPT = 0.0
REVERT = -0.0749
LOOKBACK = 120.0
DRIFT_LVL = 0.002
DELTA = 0.5223
DELTA_MIN_SAMPLES = 250
TAU = 720
DECAY = 1440.0       # weight decay beyond TAU (minutes)
STRIDE = 3           # sample every N minutes: the 8-second cap is a hard fail
BASE_SIGMA = 0.0025  # the house baseline's price-space spread per square-root minute
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
    lb = int(LOOKBACK)
    recent = u[-1] - (u[-1 - lb] if n > lb else u[0])
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
    return u0, table, raw, recent


def pooled_shape(per_market):
    """The pooled shape: quantiles of every market's width-normalised move, per horizon."""
    pooled = {}
    for h in HORIZONS:
        pairs = []
        for _, _, raw, _r in per_market:
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
    ws = sorted(v[1][max(v[1])]["w"] for v in per.values() if v[1])
    case_w = ws[len(ws) // 2] if ws else 0.0
    out = {}
    for mid, (u0, table, _raw, recent) in per.items():
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
                pw = POOL
                if POOL_ADAPT and width > 0 and case_w > 0:
                    pw = POOL * min(1.0, width / case_w)
                off = [(1.0 - pw) * own[k] + pw * (ps[k] * width) for k in range(len(LEVELS))]
            else:
                off = own
            iqr = off[LEVELS.index(0.75)] - off[LEVELS.index(0.25)]
            sig = iqr / 1.348980 if iqr > 0 else 0.0
            drift = DRIFT * (h / 60.0) + REVERT * recent + DRIFT_LVL * u0 * (h / 60.0)
            row = []
            for tau in QUANTILES:
                e = shape_value(off, tau) * (MED if tau == 0.5 else 1.0)
                if REG > 0.0 and sig > 0.0 and tau != 0.5:
                    e = (1.0 - REG) * e + REG * (Z[tau] * sig)
                bucket = 0 if h <= 80 else (1 if h <= 160 else 2)
                sc = SCALES[bucket][QUANTILES.index(tau)]
                q = sigmoid(u0 + drift + sc * e)
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
