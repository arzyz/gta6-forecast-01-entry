"""gta6-twitch-ccu-forecast-01 — Variant 3: similarity-weighted forecaster.

Like Variant 1, every comparable is aligned on its day_start and reduced
to log-ratios against its hour-1 viewers — but the pool is weighted by
similarity to the target case, and the spread is calibrated separately
from the centre:

- Similarity: each comparable and the target get ramp features from the
  hours before the day — A, hour 1 as a share of the pre-day peak the
  rolling unlock built; B, the pre-day tail mean (hours -6..-1) over
  hour 1; and the launch's scale, log10 of hour-1 viewers. A comparable's
  weight is a Gaussian kernel on the scale difference (the strong
  signal: big launches run flatter, small ones spike later) and, with a
  much wider width, on the log differences of A and B as a gentle shape
  prior, floored so the pool never collapses. (Widths calibrated on a
  leave-one-out backtest over the comparables.)
- Centre: the weighted median log-ratio per hour, blended slightly
  toward persistence.
- Spread: two-piece. The upper scale is the weighted 0.8413 quantile of
  the pooled ratios minus the centre, the lower scale the centre minus
  the weighted 0.1587 quantile; each is shrunk toward the baseline's
  0.15 * sqrt(h - 1) with its own shrinkage and used for its own side of
  the forecast, so a right-skewed pool widens the top more than the
  bottom.

Any failure falls back to the house baseline formula. Reads every
<id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output). Python 3.13, stdlib only,
deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
BASE_SIGMA = 0.15

KERNEL_A = 3.0       # kernel width on log A (gentle shape prior)
KERNEL_B = 3.0       # kernel width on log B (gentle shape prior)
KERNEL_SZ = 0.5      # kernel width on log10 hour-1 viewers (main signal)
W_FLOOR = 0.02
BETA = 0.05
SHRINK_SPREAD = 0.30
TAIL = 1.10
SPREAD_FLOOR = 0.10
SPREAD_CAP = 1.25
MIN_POOL = 5


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def baseline_rows(p1, hours):
    rows = []
    for h in hours:
        s = BASE_SIGMA * math.sqrt(h - 1)
        rows.append({"hour": h, "q": [round2(min(p1 * math.exp(z * s), 1e9)) for z in Z]})
    return rows


def hourly_series(points, time_key):
    out = {}
    for pt in points or []:
        try:
            t = parse_ts(pt[time_key])
            v = pt.get("viewers")
        except Exception:
            continue
        if v is not None:
            out[t] = float(v)
    return out


def ramp_features(series, day_start, base):
    """A: base over the pre-day peak (hours -14..-1). B: mean of the last
    six pre-day hours over base. Plus the scale, log10 of base."""
    pre = []
    for k in range(1, 15):
        v = series.get(day_start - timedelta(hours=k))
        if v is not None:
            pre.append(max(float(v), 0.0))
    if not pre or base <= 0:
        return None
    peak = max(pre)
    a = base / peak if peak > 0 else None
    tail = pre[:6]
    b = (sum(tail) / len(tail)) / base if tail else None
    return {"a": a, "b": b, "sz": math.log10(base)}


def comp_curve(comp):
    try:
        day_start = parse_ts(comp["day_start"])
        series = hourly_series(comp.get("hourly"), "from")
    except Exception:
        return None
    day = [series.get(day_start + timedelta(hours=i)) for i in range(24)]
    present = [v for v in day if v is not None]
    if len(present) < 20 or day[0] is None:
        return None
    base = day[0]
    if base is None or base < 1:
        early = sorted(v for v in day[:4] if v is not None)
        if not early:
            return None
        base = early[len(early) // 2]
    base = max(float(base), 1.0)
    log_base = math.log(base)
    lr = {}
    for h in range(2, 25):
        v = day[h - 1]
        if v is not None:
            lr[h] = math.log(max(float(v), 1.0)) - log_base
    return {"lr": lr, "feats": ramp_features(series, day_start, base)}


def target_features(case, p1):
    try:
        day_start = parse_ts(case["day_start"])
    except Exception:
        return None
    series = {}
    hourly = case.get("history_hourly")
    if hourly:
        series = hourly_series(hourly, "from")
    if not series:
        raw = hourly_series(case.get("history"), "at")
        buckets = {}
        for t, v in raw.items():
            hour = t.replace(minute=0, second=0, microsecond=0)
            buckets.setdefault(hour, []).append(v)
        series = {h: sum(vs) / len(vs) for h, vs in buckets.items()}
    return ramp_features(series, day_start, float(p1))


def weighted_median(pairs):
    pairs = sorted(pairs, key=lambda p: (p[0], p[1]))
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0] if pairs else 0.0
    acc = 0.0
    for value, w in pairs:
        acc += w
        if acc >= total / 2:
            return value
    return pairs[-1][0]


def weighted_quantile(pairs, tau):
    pairs = sorted(pairs, key=lambda p: (p[0], p[1]))
    total = sum(w for _, w in pairs)
    if total <= 0 or not pairs:
        return pairs[0][0] if pairs else 0.0
    target = tau * total
    acc = 0.0
    prev_v, prev_acc = pairs[0][0], 0.0
    for value, w in pairs:
        acc += w
        if acc >= target:
            if w <= 0:
                return value
            frac = (target - prev_acc) / w
            return prev_v + frac * (value - prev_v)
        prev_v, prev_acc = value, acc
    return pairs[-1][0]


def comp_weight(comp_feats, tgt_feats):
    w = 1.0
    if tgt_feats and comp_feats:
        for key, width in (("a", KERNEL_A), ("b", KERNEL_B)):
            tv = tgt_feats.get(key)
            cv = comp_feats.get(key)
            if tv and cv and tv > 0 and cv > 0:
                d = (math.log(cv) - math.log(tv)) / width
                w *= math.exp(-0.5 * d * d)
        tv = tgt_feats.get("sz")
        cv = comp_feats.get("sz")
        if tv is not None and cv is not None:
            d = (cv - tv) / KERNEL_SZ
            w *= math.exp(-0.5 * d * d)
    return max(w, W_FLOOR)


def forecast_case(case):
    hours = [row["hour"] for row in case["forecast"]]
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    curves = []
    for comp in case.get("comparables") or []:
        curve = comp_curve(comp)
        if curve is not None:
            curves.append(curve)
    if len(curves) < MIN_POOL:
        return {"hours": baseline_rows(p1, hours)}
    tgt_feats = target_features(case, p1)
    weighted = [(c, comp_weight(c["feats"], tgt_feats)) for c in curves]

    rows = []
    for h in hours:
        pairs = [(c["lr"][h], w) for c, w in weighted if h in c["lr"]]
        if len(pairs) < MIN_POOL:
            rows.extend(baseline_rows(p1, [h]))
            continue
        centre = weighted_median(pairs)
        base_s = BASE_SIGMA * math.sqrt(h - 1)

        def shrunk(s):
            s = math.sqrt((1 - SHRINK_SPREAD) * s * s + SHRINK_SPREAD * base_s * base_s)
            return min(max(s, SPREAD_FLOOR), SPREAD_CAP)

        s_up = shrunk(max(weighted_quantile(pairs, 0.8413) - centre, 0.01))
        s_dn = shrunk(max(centre - weighted_quantile(pairs, 0.1587), 0.01))
        loc = (1 - BETA) * centre
        qs = []
        for tau, z in zip(LEVELS, Z):
            zz = z * (TAIL if tau in (0.05, 0.95) else 1.0)
            q = p1 * math.exp(loc + zz * (s_up if zz >= 0 else s_dn))
            qs.append(round2(min(max(q, 0.01), 1e9)))
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            hours = [row["hour"] for row in case.get("forecast", [])]
            first = next((k for k in case.get("known", []) if k["hour"] == 1), None)
            p1 = max(1.0, float(first["viewers"])) if first else 1.0
            forecast = {"hours": baseline_rows(p1, hours)} if hours else {"hours": []}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
