#!/usr/bin/env python3
"""
JLST v2 Parameter Optimization via Grid Search
================================================
Uses BTC daily data to find optimal reversal-momentum parameters.
Train: 2017-08-17 to 2023-12-31 | Test: 2024-01-01 to present

Evaluates signal quality:
  - Long signal → forward N-day return (should be positive)
  - Short signal → forward N-day return (should be negative)
  - Combined hit rate + risk-adjusted return (Sharpe)

Output: data/v2_params.json
"""

import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_data import (
    rolling_mean, rolling_std, rolling_corr, zscore, clamp,
    true_range, rma,
    HALVING_DATES, HALV_WIN, HALV_BOOST,
    CORR_THR, PENALTY, SHOCK_K1, SHOCK_K2, SHOCK_WIN, SHOCK_ATTEN,
    ATR_LEN, ATR_SLOW_MUL, VOL_AVG_LEN, CLAMP_Z,
)

# ── Grid (reduced: 576 valid combos) ──
GRID = {
    "revLen":    [7, 14, 21, 30],
    "momLen":    [90, 180, 270, 330],
    "momSkip":   [7, 14, 21, 30],
    "normLen":   [60, 90, 120, 180],
    "entryThr":  [0.5, 0.75, 1.0],
    "noiseGain": [0.3, 0.5, 0.7],
}

DEAD_ZONE = 0.25
BARSYEAR = 365
FORWARD_DAYS = [7, 14, 30]

TRAIN_END = datetime(2023, 12, 31, tzinfo=timezone.utc).timestamp() * 1000
TEST_START = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000

# ---------------------------------------------------------------------------
# Precomputed caches (O(n²) work done once)
# ---------------------------------------------------------------------------

def precompute(close, high, low, volume, timestamps):
    """Compute all expensive rolling arrays once. Return a dict of caches."""
    n = len(close)
    log_ret = [None] * n
    for i in range(1, n):
        if close[i] > 0 and close[i - 1] > 0:
            log_ret[i] = math.log(close[i] / close[i - 1])

    # TR, ATR (fixed params — don't vary in grid)
    tr = true_range(high, low, close)
    atr_fast = rma(tr, ATR_LEN)
    atr_slow = rma(tr, ATR_LEN * ATR_SLOW_MUL)
    avg_vol = rolling_mean(volume, VOL_AVG_LEN)

    atr_pct = [None] * n
    vol_regime = [None] * n
    vol_ratio = [None] * n
    for i in range(n):
        if atr_fast[i] is not None and close[i] > 0:
            atr_pct[i] = atr_fast[i] / close[i]
        if atr_fast[i] is not None and atr_slow[i] is not None and atr_slow[i] > 0:
            vol_regime[i] = atr_fast[i] / atr_slow[i]
        if volume[i] is not None and avg_vol[i] is not None and avg_vol[i] > 0:
            vol_ratio[i] = volume[i] / avg_vol[i]

    has_vol = any(v > 0 for v in volume if v is not None)

    # Shock events (fixed params)
    shock_event = [False] * n
    for i in range(1, n):
        lr = abs(log_ret[i]) if log_ret[i] is not None else 0.0
        ap = atr_pct[i] if atr_pct[i] is not None else 999.0
        av = avg_vol[i] if avg_vol[i] is not None else 1.0
        vi = volume[i] if volume[i] is not None else 0.0
        shock_event[i] = (lr > SHOCK_K1 * ap if ap < 999 else False) and (vi > SHOCK_K2 * av if has_vol else True)

    post_shock = [False] * n
    last_shock = -9999
    for i in range(n):
        if shock_event[i]:
            last_shock = i
        if last_shock >= 0 and (i - last_shock) <= SHOCK_WIN:
            post_shock[i] = True

    # Halving
    post_halv = [False] * n
    for i in range(n):
        t_dt = datetime.fromtimestamp(timestamps[i] / 1000, tz=timezone.utc)
        for hd in reversed(HALVING_DATES):
            if t_dt >= hd:
                post_halv[i] = 0 <= (t_dt - hd).days <= HALV_WIN
                break

    # Pre-compute z-scores for noise components at each normLen
    norm_cache = {}
    for normLen in GRID["normLen"]:
        norm_cache[normLen] = {
            "nz_atr": zscore(atr_pct, normLen),
            "nz_vol": zscore(vol_ratio, normLen),
            "nz_regime": zscore(vol_regime, normLen),
            "vol_ann": _vol_ann(log_ret, normLen, BARSYEAR, n),
        }

    # Pre-compute rev_raw for each revLen
    rev_cache = {}
    for revLen in GRID["revLen"]:
        rev_raw = [None] * n
        for i in range(revLen, n):
            if close[i] > 0 and close[i - revLen] > 0:
                rev_raw[i] = -math.log(close[i] / close[i - revLen])
        rev_cache[revLen] = rev_raw

    # Pre-compute mom_raw for each (momLen, momSkip)
    mom_cache = {}
    for momLen in GRID["momLen"]:
        for momSkip in GRID["momSkip"]:
            if momSkip >= momLen:
                continue
            skip_total = momSkip + momLen
            mom_raw = [None] * n
            for i in range(skip_total, n):
                s1 = close[i - momSkip]
                s2 = close[i - skip_total]
                if s1 and s1 > 0 and s2 and s2 > 0:
                    mom_raw[i] = math.log(s1 / s2)
            mom_cache[(momLen, momSkip)] = mom_raw

    return {
        "n": n, "log_ret": log_ret, "close": close,
        "has_vol": has_vol, "post_shock": post_shock, "post_halv": post_halv,
        "norm_cache": norm_cache, "rev_cache": rev_cache, "mom_cache": mom_cache,
    }


def _vol_ann(log_ret, normLen, barsYear, n):
    arr = rolling_std(log_ret, normLen)
    out = [None] * n
    for i in range(n):
        if arr[i] is not None:
            out[i] = arr[i] * math.sqrt(barsYear)
    return out


# ---------------------------------------------------------------------------
# Fast signal computation using precomputed caches
# ---------------------------------------------------------------------------

def compute_signals_fast(cache, params):
    """Compute composite and signals using precomputed caches."""
    n = cache["n"]
    close = cache["close"]
    revLen = params["revLen"]
    momLen = params["momLen"]
    momSkip = params["momSkip"]
    normLen = params["normLen"]
    entryThr = params["entryThr"]
    noiseGain = params["noiseGain"]

    rev_raw = cache["rev_cache"][revLen]
    mom_raw = cache["mom_cache"].get((momLen, momSkip))
    if mom_raw is None:
        return None, [], []

    nc = cache["norm_cache"][normLen]
    vol_ann = nc["vol_ann"]

    rev_span = math.sqrt(revLen / BARSYEAR)
    mom_span = math.sqrt(momLen / BARSYEAR)

    # Scale & z-score
    rev_scaled = [None] * n
    mom_scaled = [None] * n
    for i in range(n):
        va = vol_ann[i] if vol_ann[i] and vol_ann[i] > 0 else 1.0
        if rev_raw[i] is not None:
            rev_scaled[i] = rev_raw[i] / (va * rev_span)
        if mom_raw[i] is not None:
            mom_scaled[i] = mom_raw[i] / (va * mom_span)

    rev_z = zscore(rev_scaled, normLen)
    mom_z = zscore(mom_scaled, normLen)
    rev_s = [clamp(v, -CLAMP_Z, CLAMP_Z) for v in rev_z]
    mom_s = [clamp(v, -CLAMP_Z, CLAMP_Z) for v in mom_z]

    # Noise → weights
    nz_atr = nc["nz_atr"]
    nz_vol = nc["nz_vol"]
    nz_regime = nc["nz_regime"]
    has_vol = cache["has_vol"]

    w_rev = [None] * n
    w_mom = [None] * n
    noise_z = [None] * n
    for i in range(n):
        a = nz_atr[i] if nz_atr[i] is not None else 0.0
        v = nz_vol[i] if nz_vol[i] is not None else 0.0
        r = nz_regime[i] if nz_regime[i] is not None else 0.0
        raw = 0.4 * a + 0.3 * v + 0.3 * r if has_vol else 0.5 * a + 0.5 * r
        nz = clamp(raw, -3.0, 3.0)
        noise_z[i] = nz

        wr = max(0.0, 0.50 * (1.0 + noiseGain * nz))
        wm = max(0.0, 0.50 * (1.0 - noiseGain * nz * 0.5))
        ws = max(wr + wm, 1e-6)
        w_rev[i] = wr / ws
        w_mom[i] = wm / ws

    # Shock attenuation
    post_shock = cache["post_shock"]
    rev_eff = [None] * n
    for i in range(n):
        if rev_s[i] is not None:
            rev_eff[i] = rev_s[i] * SHOCK_ATTEN if post_shock[i] else rev_s[i]

    # Halving boost
    post_halv = cache["post_halv"]
    mom_eff = [None] * n
    for i in range(n):
        if mom_s[i] is not None:
            mom_eff[i] = mom_s[i] * (HALV_BOOST if post_halv[i] else 1.0)

    # Regime
    corr_rm = rolling_corr(rev_s, mom_s, normLen)
    regime_f = [1.0] * n
    for i in range(n):
        cr = corr_rm[i]
        if cr is not None and cr > CORR_THR:
            regime_f[i] = PENALTY

    # Composite
    comp = [None] * n
    for i in range(n):
        me, re, wm, wr = mom_eff[i], rev_eff[i], w_mom[i], w_rev[i]
        if me is not None and re is not None and wm is not None and wr is not None:
            comp[i] = (wm * me + wr * re) * regime_f[i]

    # Signals
    long_sig = []
    short_sig = []
    for i in range(1, n):
        c_now, c_prev = comp[i], comp[i - 1]
        if c_now is None or c_prev is None:
            continue
        if abs(c_now) < DEAD_ZONE:
            continue
        if c_now >= entryThr and c_prev < entryThr:
            long_sig.append(i)
        if c_now <= -entryThr and c_prev > -entryThr:
            short_sig.append(i)

    return comp, long_sig, short_sig


def evaluate(close, timestamps, long_idx, short_idx, ts_start, ts_end):
    n = len(close)
    results = {}
    for fwd in FORWARD_DAYS:
        l_rets, s_rets = [], []
        for idx in long_idx:
            if timestamps[idx] < ts_start or timestamps[idx] > ts_end or idx + fwd >= n:
                continue
            if close[idx] > 0 and close[idx + fwd] > 0:
                l_rets.append(math.log(close[idx + fwd] / close[idx]))
        for idx in short_idx:
            if timestamps[idx] < ts_start or timestamps[idx] > ts_end or idx + fwd >= n:
                continue
            if close[idx] > 0 and close[idx + fwd] > 0:
                s_rets.append(-math.log(close[idx + fwd] / close[idx]))

        all_r = l_rets + s_rets
        ns = len(all_r)
        if ns == 0:
            results[f"{fwd}d"] = {"n": 0, "hit": 0.0, "avg_ret": 0.0, "sharpe": 0.0}
            continue
        hits = sum(1 for r in all_r if r > 0)
        avg = sum(all_r) / ns
        std = math.sqrt(sum((r - avg)**2 for r in all_r) / ns) if ns > 1 else 1.0
        sh = avg / std * math.sqrt(365 / fwd) if std > 0 else 0.0
        results[f"{fwd}d"] = {
            "n": ns, "n_long": len(l_rets), "n_short": len(s_rets),
            "hit": round(hits / ns, 4),
            "long_hit": round(sum(1 for r in l_rets if r > 0) / max(len(l_rets), 1), 4),
            "short_hit": round(sum(1 for r in s_rets if r > 0) / max(len(s_rets), 1), 4),
            "avg_ret": round(avg, 6),
            "sharpe": round(sh, 4),
        }
    return results


def score(ev):
    w = {"7d": 0.40, "14d": 0.35, "30d": 0.25}
    total = 0.0
    for h, wt in w.items():
        r = ev.get(h, {})
        ns = r.get("n", 0)
        if ns < 5:
            continue
        total += wt * r.get("hit", 0) * max(r.get("sharpe", 0), 0) * min(1.0, ns / 20.0)
    return total


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"

    print("=" * 60)
    print("JLST v2 Parameter Backtest")
    print("=" * 60)

    json_path = data_dir / "btc_daily.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run update_data.py first.")
        sys.exit(1)

    with open(json_path, "r") as f:
        existing = json.load(f)
    d = existing["data"]
    timestamps = d["timestamps"]
    close = [float(v) if v is not None else 0.0 for v in d["close"]]
    high = [float(v) if v is not None else 0.0 for v in d["high"]]
    low = [float(v) if v is not None else 0.0 for v in d["low"]]
    volume = [float(v) if v is not None else 0.0 for v in d["volume"]]
    n = len(timestamps)
    print(f"Loaded {n} bars")

    # Filter to 2017+ (Binance era)
    binance_start = datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000
    si = next(i for i in range(n) if timestamps[i] >= binance_start)
    ts = timestamps[si:]
    cl = close[si:]
    hi = high[si:]
    lo = low[si:]
    vo = volume[si:]
    print(f"Using {len(ts)} bars (2017+)")

    print("\nPrecomputing caches...")
    cache = precompute(cl, hi, lo, vo, ts)
    print("Done.")

    # Build valid combos
    keys = list(GRID.keys())
    combos = []
    for c in itertools.product(*GRID.values()):
        p = dict(zip(keys, c))
        if p["momSkip"] >= p["momLen"] and p["revLen"] <= p["normLen"]:
            continue
        combos.append(p)
    print(f"\nGrid search: {len(combos)} combinations")

    results = []
    best_score = -1

    for idx, params in enumerate(combos):
        comp, l_idx, s_idx = compute_signals_fast(cache, params)
        if comp is None:
            continue

        tr_ev = evaluate(cl, ts, l_idx, s_idx, 0, TRAIN_END)
        te_ev = evaluate(cl, ts, l_idx, s_idx, TEST_START, float("inf"))
        sc = score(tr_ev)
        results.append({"params": params, "train": tr_ev, "test": te_ev, "score": round(sc, 6)})

        if sc > best_score:
            best_score = sc

        if (idx + 1) % 500 == 0 or idx == len(combos) - 1:
            print(f"  [{(idx+1)/len(combos)*100:5.1f}%] {idx+1}/{len(combos)} | best={best_score:.4f}")

    results.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*60}")
    print("Top 5:")
    for i, r in enumerate(results[:5]):
        p = r["params"]
        print(f"  #{i+1} score={r['score']:.4f} | rev={p['revLen']} mom={p['momLen']} skip={p['momSkip']} norm={p['normLen']} thr={p['entryThr']} ng={p['noiseGain']}")
        for h in ["7d", "14d"]:
            t = r["train"].get(h, {})
            v = r["test"].get(h, {})
            print(f"      {h}: train hit={t.get('hit',0):.1%} sh={t.get('sharpe',0):.2f} n={t.get('n',0)} | test hit={v.get('hit',0):.1%} sh={v.get('sharpe',0):.2f} n={v.get('n',0)}")

    # Pick winner: best train score that also survives test
    winner = results[0]
    for r in results[:10]:
        ts_sc = score(r["test"])
        if ts_sc > 0 and r["score"] > best_score * 0.7:
            winner = r
            break

    wp = winner["params"]
    print(f"\nSELECTED: rev={wp['revLen']} mom={wp['momLen']} skip={wp['momSkip']} norm={wp['normLen']} thr={wp['entryThr']} ng={wp['noiseGain']}")

    v1 = {"revLen": 30, "momLen": 330, "momSkip": 30, "normLen": 180, "entryThr": 1.0, "noiseGain": 0.5}
    output = {
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "train_period": "2017-08-17 to 2023-12-31",
        "test_period": f"2024-01-01 to {datetime.fromtimestamp(ts[-1]/1000, tz=timezone.utc).strftime('%Y-%m-%d')}",
        "v1_params": v1,
        "best_params": {
            **wp,
            "barsYear": BARSYEAR,
            "corrLen": wp["normLen"],
            "corrThr": CORR_THR,
            "penalty": PENALTY,
            "deadZone": DEAD_ZONE,
            "shockK1": SHOCK_K1,
            "halvWin": HALV_WIN,
            "halvBoost": HALV_BOOST,
        },
        "train_stats": winner["train"],
        "test_stats": winner["test"],
        "top5": [{"params": r["params"], "score": r["score"], "train_7d": r["train"].get("7d",{}), "test_7d": r["test"].get("7d",{})} for r in results[:5]],
    }

    out_path = data_dir / "v2_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
