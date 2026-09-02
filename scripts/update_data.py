#!/usr/bin/env python3
"""
JLST Reversal–Momentum Composite Indicator — BTC Data Pipeline
===============================================================
Theory: Jegadeesh, Luo, Subrahmanyam & Titman (2025, RFS)
"Short-Term Reversals and Longer-Term Momentum around the World"

Implements the BTC-specialized version:
  - Log returns (high-volatility asset)
  - 24/7 calendar: 1 month = 30 bars, 1 year = 365 bars (daily)
  - Noise proxy: ATR% + volume ratio + volatility regime (ATR fast/slow)
  - Information shock module (replaces earnings for crypto)
  - Halving cycle modulation (heuristic/narrative)
  - Cascade detection for extreme liquidation events

Paper → Indicator mapping:
  Table 2 ρ₁<0        → Rev  = −log(src/src[revLen])
  Table 2 ρ₂≈0        → Dead zone (no signal when |comp| < 0.25)
  Table 2 ρ₃…ρ₁₂>0   → Mom  = log(src[skip]/src[skip+momLen])
  Proposition 2        → momSkip (skip 1 month enhances momentum)
  Prediction a         → Info shock attenuates Rev post-event
  Prediction b         → corr(Rev,Mom) < 0 normal; > corrThr → penalty
  Prediction c         → Noise↑ → wRev↑, wMom↓
"""

import csv
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BINANCE_BASE = "https://data-api.binance.vision/api/v3/klines"

# BTC halving dates (unix ms)
HALVING_DATES = [
    datetime(2012, 11, 28, tzinfo=timezone.utc),
    datetime(2016, 7, 9, tzinfo=timezone.utc),
    datetime(2020, 5, 11, tzinfo=timezone.utc),
    datetime(2024, 4, 20, tzinfo=timezone.utc),
    datetime(2028, 4, 20, tzinfo=timezone.utc),  # estimated
]

TIMEFRAME_PARAMS = {
    "daily": {
        "interval": "1d",
        "revLen": 30,
        "momLen": 330,
        "momSkip": 30,
        "barsYear": 365,
        "normLen": 180,
        "corrLen": 180,
        "label": "日线 Daily",
    },
    "weekly": {
        "interval": "1w",
        "revLen": 4,
        "momLen": 48,
        "momSkip": 4,
        "barsYear": 52,
        "normLen": 52,
        "corrLen": 52,
        "label": "周线 Weekly",
    },
    "monthly": {
        "interval": "1M",
        "revLen": 1,
        "momLen": 11,
        "momSkip": 1,
        "barsYear": 12,
        "normLen": 24,
        "corrLen": 24,
        "label": "月线 Monthly",
    },
}

# Fixed model parameters (from BTC spec)
REV_W = 0.50
MOM_W = 0.50
NOISE_GAIN = 0.50
ATR_LEN = 14
ATR_SLOW_MUL = 5
VOL_AVG_LEN = 50
USE_VOL_ADJ = True
CLAMP_Z = 3.0
CORR_THR = 0.20
PENALTY = 0.60
SHOCK_K1 = 2.5
SHOCK_K2 = 2.0
SHOCK_WIN = 10
SHOCK_ATTEN = 0.40
ENTRY_THR = 1.00
DEAD_ZONE = 0.25
HALV_WIN = 540
HALV_BOOST = 1.20
CASCADE_K = 3.5
CASCADE_VOL_K = 2.5
SMOOTH_LEN = 1

# v2 funding rate endpoint (futures API — works on GitHub Actions, may be blocked locally)
FAPI_BASE = "https://fapi.binance.com/fapi/v1"

# v2 noise weight allocation (with funding rate)
V2_NOISE_W = {"atr": 0.25, "vol": 0.25, "regime": 0.20, "funding": 0.30}
# v1 noise weights (for reference)
V1_NOISE_W = {"atr": 0.40, "vol": 0.30, "regime": 0.30}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_binance_klines(symbol: str, interval: str, limit: int = 1000) -> list:
    """Fetch all available klines by paginating backwards."""
    all_data = []
    end_time = None

    for _ in range(20):  # safety limit
        url = f"{BINANCE_BASE}?symbol={symbol}&interval={interval}&limit={limit}"
        if end_time is not None:
            url += f"&endTime={end_time}"

        req = urllib.request.Request(url)
        req.add_header("User-Agent", "JLST-BTC-Pipeline/1.0")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  [warn] fetch error: {e}, retrying in 2s...")
            time.sleep(2)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
            except Exception as e2:
                print(f"  [error] fetch failed: {e2}")
                break

        if not data:
            break

        all_data = data + all_data
        end_time = data[0][0] - 1

        if len(data) < limit:
            break

        time.sleep(0.3)  # rate limit courtesy

    return all_data


def parse_klines(raw: list) -> dict:
    """Parse Binance kline arrays into structured dict of arrays."""
    n = len(raw)
    ts = [0] * n
    o = [0.0] * n
    h = [0.0] * n
    lo = [0.0] * n
    c = [0.0] * n
    vol = [0.0] * n

    for i, k in enumerate(raw):
        ts[i] = int(k[0])
        o[i] = float(k[1])
        h[i] = float(k[2])
        lo[i] = float(k[3])
        c[i] = float(k[4])
        vol[i] = float(k[5])

    return {"ts": ts, "open": o, "high": h, "low": lo, "close": c, "volume": vol, "n": n}


def load_historical_csv(path: str) -> dict:
    """Load btc_historical.csv (semicolon-delimited, newest-first, OHLCV)."""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                dt_str = row["timeOpen"][:10]  # "2026-08-31"
                dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                ts_ms = int(dt.timestamp() * 1000)
                rows.append({
                    "ts": ts_ms,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
            except (ValueError, KeyError):
                continue
    # Sort ascending by timestamp
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    return {
        "ts": [r["ts"] for r in rows],
        "open": [r["open"] for r in rows],
        "high": [r["high"] for r in rows],
        "low": [r["low"] for r in rows],
        "close": [r["close"] for r in rows],
        "volume": [r["volume"] for r in rows],
        "n": n,
    }


def merge_data(csv_data: dict, binance_data: dict) -> dict:
    """Merge CSV base with Binance overlay. Binance wins on overlapping dates."""
    # Build date→index map for Binance data (keyed by date string to avoid ms mismatches)
    bin_map = {}
    for i in range(binance_data["n"]):
        day_key = datetime.fromtimestamp(binance_data["ts"][i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        bin_map[day_key] = i

    merged = {"ts": [], "open": [], "high": [], "low": [], "close": [], "volume": []}

    # First: add all CSV rows that DON'T exist in Binance
    seen_dates = set()
    for i in range(csv_data["n"]):
        day_key = datetime.fromtimestamp(csv_data["ts"][i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if day_key in bin_map:
            continue  # Binance will provide this date
        if day_key in seen_dates:
            continue
        seen_dates.add(day_key)
        merged["ts"].append(csv_data["ts"][i])
        merged["open"].append(csv_data["open"][i])
        merged["high"].append(csv_data["high"][i])
        merged["low"].append(csv_data["low"][i])
        merged["close"].append(csv_data["close"][i])
        merged["volume"].append(csv_data["volume"][i])

    # Then: add all Binance rows
    for i in range(binance_data["n"]):
        day_key = datetime.fromtimestamp(binance_data["ts"][i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if day_key in seen_dates:
            continue
        seen_dates.add(day_key)
        merged["ts"].append(binance_data["ts"][i])
        merged["open"].append(binance_data["open"][i])
        merged["high"].append(binance_data["high"][i])
        merged["low"].append(binance_data["low"][i])
        merged["close"].append(binance_data["close"][i])
        merged["volume"].append(binance_data["volume"][i])

    # Sort by timestamp
    indices = sorted(range(len(merged["ts"])), key=lambda i: merged["ts"][i])
    for key in merged:
        merged[key] = [merged[key][i] for i in indices]
    merged["n"] = len(merged["ts"])
    return merged


def fetch_funding_rate(symbol: str = "BTCUSDT") -> list:
    """Fetch full funding rate history from Binance Futures API.
    Returns list of {ts_ms, rate} sorted ascending.
    Each record is one 8h funding event."""
    all_data = []
    start_time = int(datetime(2019, 9, 10, tzinfo=timezone.utc).timestamp() * 1000)  # funding starts ~2019-09

    for _ in range(50):  # safety
        url = f"{FAPI_BASE}/fundingRate?symbol={symbol}&limit=1000&startTime={start_time}"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "JLST-BTC-Pipeline/2.0")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  [warn] funding rate fetch error: {e}")
            break

        if not data:
            break

        for item in data:
            all_data.append({
                "ts": int(item["fundingTime"]),
                "rate": float(item["fundingRate"]),
            })

        start_time = data[-1]["fundingTime"] + 1
        if len(data) < 1000:
            break
        time.sleep(0.3)

    all_data.sort(key=lambda x: x["ts"])
    return all_data


def aggregate_funding_daily(funding_records: list) -> dict:
    """Aggregate 8h funding rate records to daily.
    Returns dict: {"dates": {date_str: avg_rate}}."""
    from collections import defaultdict
    daily = defaultdict(list)
    for r in funding_records:
        day = datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        daily[day].append(r["rate"])
    return {day: sum(rates) / len(rates) for day, rates in daily.items()}


def align_funding_to_timestamps(funding_daily: dict, timestamps: list) -> list:
    """Align daily funding rates to OHLC timestamp array.
    Returns list same length as timestamps; None where no funding data."""
    out = [None] * len(timestamps)
    for i, ts in enumerate(timestamps):
        day = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        out[i] = funding_daily.get(day)
    return out


def load_funding_cache(path: str) -> list:
    """Load cached funding rate data from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_funding_cache(records: list, path: str):
    """Save funding rate records to JSON cache file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Indicator math helpers
# ---------------------------------------------------------------------------

def rolling_mean(arr: list, window: int) -> list:
    """Rolling mean using sliding window. O(n) with None-safe counting."""
    n = len(arr)
    out = [None] * n
    s = 0.0
    cnt = 0
    for i in range(n):
        if arr[i] is not None:
            s += arr[i]
            cnt += 1
        if i >= window:
            old = arr[i - window]
            if old is not None:
                s -= old
                cnt -= 1
        if i >= window - 1 and cnt > 0:
            out[i] = s / cnt
    return out


def rolling_std(arr: list, window: int) -> list:
    """Rolling standard deviation (population) using Welford-style sliding window. O(n)."""
    n = len(arr)
    out = [None] * n
    s = 0.0
    s2 = 0.0
    cnt = 0
    for i in range(n):
        if arr[i] is not None:
            s += arr[i]
            s2 += arr[i] * arr[i]
            cnt += 1
        if i >= window:
            old = arr[i - window]
            if old is not None:
                s -= old
                s2 -= old * old
                cnt -= 1
        if i >= window - 1 and cnt >= 2:
            mean = s / cnt
            var = s2 / cnt - mean * mean
            out[i] = math.sqrt(max(var, 0.0))
        elif i >= window - 1:
            out[i] = None
    return out


def rolling_corr(x: list, y: list, window: int) -> list:
    """Rolling Pearson correlation using sliding sums. O(n)."""
    n = len(x)
    out = [None] * n
    sx = 0.0; sy = 0.0; sxy = 0.0; sx2 = 0.0; sy2 = 0.0; cnt = 0
    for i in range(n):
        both = x[i] is not None and y[i] is not None
        if both:
            sx += x[i]; sy += y[i]
            sxy += x[i] * y[i]
            sx2 += x[i] * x[i]; sy2 += y[i] * y[i]
            cnt += 1
        if i >= window:
            ob = x[i - window] is not None and y[i - window] is not None
            if ob:
                sx -= x[i - window]; sy -= y[i - window]
                sxy -= x[i - window] * y[i - window]
                sx2 -= x[i - window] * x[i - window]
                sy2 -= y[i - window] * y[i - window]
                cnt -= 1
        if i >= window - 1 and cnt >= 10:
            mx = sx / cnt; my = sy / cnt
            var_x = sx2 / cnt - mx * mx
            var_y = sy2 / cnt - my * my
            cov = sxy / cnt - mx * my
            if var_x > 0 and var_y > 0:
                out[i] = cov / (math.sqrt(var_x) * math.sqrt(var_y))
    return out


def zscore(arr: list, window: int) -> list:
    """Rolling z-score."""
    mu = rolling_mean(arr, window)
    sd = rolling_std(arr, window)
    n = len(arr)
    out = [None] * n
    for i in range(n):
        if arr[i] is not None and mu[i] is not None and sd[i] is not None and sd[i] > 0:
            out[i] = (arr[i] - mu[i]) / sd[i]
    return out


def clamp(val, lo, hi):
    if val is None:
        return None
    return max(lo, min(hi, val))


def true_range(h: list, lo: list, c: list) -> list:
    """True Range series."""
    n = len(h)
    tr = [None] * n
    tr[0] = h[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
    return tr


def ema(arr: list, period: int) -> list:
    """Exponential moving average."""
    n = len(arr)
    out = [None] * n
    k = 2.0 / (period + 1)
    started = False
    prev = 0.0
    for i in range(n):
        if arr[i] is None:
            continue
        if not started:
            prev = arr[i]
            out[i] = prev
            started = True
        else:
            prev = arr[i] * k + prev * (1 - k)
            out[i] = prev
    return out


def rma(arr: list, period: int) -> list:
    """Wilder's RMA (used for ATR)."""
    n = len(arr)
    out = [None] * n
    # seed with SMA
    first_valid = []
    start = 0
    for i in range(n):
        if arr[i] is not None:
            first_valid.append(arr[i])
            if len(first_valid) == period:
                start = i
                break
    if len(first_valid) < period:
        return out
    out[start] = sum(first_valid) / period
    alpha = 1.0 / period
    for i in range(start + 1, n):
        if arr[i] is not None and out[i - 1] is not None:
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# JLST Indicator Computation
# ---------------------------------------------------------------------------

def compute_jlst(data: dict, params: dict) -> dict:
    """
    Compute all JLST reversal–momentum indicators for one timeframe.
    Returns dict of arrays + summary.
    """
    n = data["n"]
    opn = data["open"]
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    timestamps = data["ts"]

    revLen = params["revLen"]
    momLen = params["momLen"]
    momSkip = params["momSkip"]
    barsYear = params["barsYear"]
    normLen = params["normLen"]
    corrLen = params["corrLen"]

    # ── 1. Log returns ──
    log_ret = [None] * n
    for i in range(1, n):
        if close[i] > 0 and close[i - 1] > 0:
            log_ret[i] = math.log(close[i] / close[i - 1])

    # ── 2. Raw Rev & Mom ──
    rev_raw = [None] * n
    mom_raw = [None] * n
    for i in range(revLen, n):
        if close[i] > 0 and close[i - revLen] > 0:
            rev_raw[i] = -math.log(close[i] / close[i - revLen])  # ρ₁<0

    skip_total = momSkip + momLen
    for i in range(skip_total, n):
        src_skip = close[i - momSkip] if i - momSkip >= 0 else None
        src_back = close[i - skip_total] if i - skip_total >= 0 else None
        if src_skip and src_skip > 0 and src_back and src_back > 0:
            mom_raw[i] = math.log(src_skip / src_back)  # ρ₃…ρ₁₂>0

    # ── 3. Volatility & horizon scaling ──
    vol_ann_arr = rolling_std(log_ret, normLen)
    vol_ann = [None] * n
    for i in range(n):
        if vol_ann_arr[i] is not None:
            vol_ann[i] = vol_ann_arr[i] * math.sqrt(barsYear)

    rev_span = math.sqrt(revLen / barsYear)
    mom_span = math.sqrt(momLen / barsYear)

    rev_scaled = [None] * n
    mom_scaled = [None] * n

    if USE_VOL_ADJ:
        for i in range(n):
            va = vol_ann[i] if vol_ann[i] and vol_ann[i] > 0 else 1.0
            if rev_raw[i] is not None:
                rev_scaled[i] = rev_raw[i] / (va * rev_span)
            if mom_raw[i] is not None:
                mom_scaled[i] = mom_raw[i] / (va * mom_span)
    else:
        rev_scaled = rev_raw[:]
        mom_scaled = mom_raw[:]

    # z-score
    rev_z = zscore(rev_scaled, normLen)
    mom_z = zscore(mom_scaled, normLen)

    # clamp
    rev_s = [clamp(v, -CLAMP_Z, CLAMP_Z) for v in rev_z]
    mom_s = [clamp(v, -CLAMP_Z, CLAMP_Z) for v in mom_z]

    # ── 4. Noise / leverage proxy (Prediction c) ──
    tr = true_range(high, low, close)
    atr_fast = rma(tr, ATR_LEN)
    atr_slow = rma(tr, ATR_LEN * ATR_SLOW_MUL)

    atr_pct = [None] * n
    vol_regime = [None] * n
    vol_ratio = [None] * n
    avg_vol = rolling_mean(volume, VOL_AVG_LEN)

    for i in range(n):
        if atr_fast[i] is not None and close[i] > 0:
            atr_pct[i] = atr_fast[i] / close[i]
        if atr_fast[i] is not None and atr_slow[i] is not None and atr_slow[i] > 0:
            vol_regime[i] = atr_fast[i] / atr_slow[i]
        if volume[i] is not None and avg_vol[i] is not None and avg_vol[i] > 0:
            vol_ratio[i] = volume[i] / avg_vol[i]

    nz_atr = zscore(atr_pct, normLen)
    nz_vol = zscore(vol_ratio, normLen)
    nz_regime = zscore(vol_regime, normLen)

    has_vol = any(v > 0 for v in volume if v is not None)

    noise_z = [None] * n
    for i in range(n):
        a = nz_atr[i] if nz_atr[i] is not None else 0.0
        v = nz_vol[i] if nz_vol[i] is not None else 0.0
        r = nz_regime[i] if nz_regime[i] is not None else 0.0
        if has_vol:
            raw = 0.4 * a + 0.3 * v + 0.3 * r
        else:
            raw = 0.5 * a + 0.5 * r
        noise_z[i] = clamp(raw, -3.0, 3.0)

    # ── 5. Dynamic weights (Prediction c) ──
    w_rev = [None] * n
    w_mom = [None] * n
    for i in range(n):
        nz = noise_z[i] if noise_z[i] is not None else 0.0
        wr = max(0.0, REV_W * (1.0 + NOISE_GAIN * nz))
        wm = max(0.0, MOM_W * (1.0 - NOISE_GAIN * nz * 0.5))
        ws = max(wr + wm, 1e-6)
        w_rev[i] = wr / ws
        w_mom[i] = wm / ws

    # ── 6. Information shock (Prediction a — BTC version) ──
    shock_event = [False] * n
    post_shock = [False] * n
    for i in range(1, n):
        lr = abs(log_ret[i]) if log_ret[i] is not None else 0.0
        ap = atr_pct[i] if atr_pct[i] is not None else 999.0
        av = avg_vol[i] if avg_vol[i] is not None else 1.0
        vi = volume[i] if volume[i] is not None else 0.0
        move_ok = lr > SHOCK_K1 * ap if ap < 999 else False
        vol_ok = vi > SHOCK_K2 * av if has_vol else True
        shock_event[i] = move_ok and vol_ok

    # Mark post-shock windows
    bars_since_shock = [None] * n
    last_shock = -9999
    for i in range(n):
        if shock_event[i]:
            last_shock = i
        if last_shock >= 0:
            bars_since_shock[i] = i - last_shock
            post_shock[i] = bars_since_shock[i] <= SHOCK_WIN

    rev_eff = [None] * n
    for i in range(n):
        if rev_s[i] is not None:
            rev_eff[i] = rev_s[i] * SHOCK_ATTEN if post_shock[i] else rev_s[i]

    # ── 7. Halving cycle modulation ──
    post_halv = [False] * n
    days_since_halv = [None] * n
    for i in range(n):
        t_ms = timestamps[i]
        t_dt = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
        last_h = None
        for hd in reversed(HALVING_DATES):
            if t_dt >= hd:
                last_h = hd
                break
        if last_h is not None:
            days = (t_dt - last_h).days
            days_since_halv[i] = days
            post_halv[i] = 0 <= days <= HALV_WIN

    mom_eff = [None] * n
    for i in range(n):
        m = mom_s[i] if mom_s[i] is not None else None
        if m is not None:
            boost = HALV_BOOST if post_halv[i] else 1.0
            mom_eff[i] = m * boost

    # ── 8. Cascade detection ──
    cascade = [False] * n
    for i in range(1, n):
        lr = abs(log_ret[i]) if log_ret[i] is not None else 0.0
        ap = atr_pct[i] if atr_pct[i] is not None else 999.0
        av = avg_vol[i] if avg_vol[i] is not None else 1.0
        vi = volume[i] if volume[i] is not None else 0.0
        move_ok = lr > CASCADE_K * ap if ap < 999 else False
        vol_ok = vi > CASCADE_VOL_K * av if has_vol else True
        cascade[i] = move_ok and vol_ok

    # ── 9. Regime confidence (Prediction b) ──
    corr_rm = rolling_corr(rev_s, mom_s, corrLen)
    ac1_series = [None] * n
    log_ret_lag = [None] * n
    for i in range(1, n):
        log_ret_lag[i] = log_ret[i - 1]
    ac1_series = rolling_corr(log_ret, log_ret_lag, corrLen)

    regime_f = [1.0] * n
    regime_state = ["normal"] * n
    for i in range(n):
        cr = corr_rm[i]
        if cr is not None:
            if cr > CORR_THR:
                regime_f[i] = PENALTY
                regime_state[i] = "resonance"
            elif cr < -CORR_THR:
                regime_state[i] = "complementary"

    # ── 10. Composite score ──
    comp_raw = [None] * n
    for i in range(n):
        me = mom_eff[i]
        re = rev_eff[i]
        wm = w_mom[i]
        wr = w_rev[i]
        rf = regime_f[i]
        if me is not None and re is not None and wm is not None and wr is not None:
            comp_raw[i] = (wm * me + wr * re) * rf

    if SMOOTH_LEN > 1:
        comp = ema(comp_raw, SMOOTH_LEN)
    else:
        comp = comp_raw[:]

    # ── 11. Signals ──
    long_sig = [False] * n
    short_sig = [False] * n
    signal_state = ["watch"] * n

    for i in range(1, n):
        c_now = comp[i]
        c_prev = comp[i - 1]
        if c_now is None or c_prev is None:
            continue

        in_dead = abs(c_now) < DEAD_ZONE
        if c_now >= ENTRY_THR and c_prev < ENTRY_THR and not in_dead:
            long_sig[i] = True
        if c_now <= -ENTRY_THR and c_prev > -ENTRY_THR and not in_dead:
            short_sig[i] = True

        if c_now >= ENTRY_THR:
            signal_state[i] = "bullish"
        elif c_now <= -ENTRY_THR:
            signal_state[i] = "bearish"
        elif in_dead:
            signal_state[i] = "dead_zone"
        else:
            signal_state[i] = "watch"

    # ── 12. Summary / latest stats ──
    last_idx = n - 1
    summary = {
        "last_update": datetime.fromtimestamp(timestamps[last_idx] / 1000, tz=timezone.utc).isoformat(),
        "price": close[last_idx],
        "composite": _r(comp[last_idx]),
        "rev": _r(rev_eff[last_idx]),
        "mom": _r(mom_eff[last_idx]),
        "w_rev": _r(w_rev[last_idx]),
        "w_mom": _r(w_mom[last_idx]),
        "noise_z": _r(noise_z[last_idx]),
        "corr_rm": _r(corr_rm[last_idx]),
        "ac1": _r(ac1_series[last_idx]),
        "vol_regime": _r(vol_regime[last_idx]),
        "post_shock": post_shock[last_idx],
        "post_halv": post_halv[last_idx],
        "days_since_halv": days_since_halv[last_idx],
        "cascade": cascade[last_idx],
        "regime": regime_state[last_idx],
        "signal_state": signal_state[last_idx],
    }

    # Build output arrays (only include values from point where indicators are valid)
    # Find the first index where composite is not None
    start = 0
    for i in range(n):
        if comp[i] is not None:
            start = i
            break

    def _slice(arr, start):
        return [_r(v) for v in arr[start:]]

    def _slice_bool(arr, start):
        return arr[start:]

    def _slice_str(arr, start):
        return arr[start:]

    output = {
        "params": {
            "revLen": revLen,
            "momLen": momLen,
            "momSkip": momSkip,
            "barsYear": barsYear,
            "normLen": normLen,
            "corrLen": corrLen,
            "noiseGain": NOISE_GAIN,
            "corrThr": CORR_THR,
            "penalty": PENALTY,
            "entryThr": ENTRY_THR,
            "deadZone": DEAD_ZONE,
            "shockK1": SHOCK_K1,
            "halvWin": HALV_WIN,
            "halvBoost": HALV_BOOST,
        },
        "summary": summary,
        "data": {
            "timestamps": timestamps[start:],
            "open": [_r(v, 2) for v in opn[start:]],
            "high": [_r(v, 2) for v in high[start:]],
            "low": [_r(v, 2) for v in low[start:]],
            "close": [_r(v, 2) for v in close[start:]],
            "volume": [_r(v) for v in volume[start:]],
            "composite": _slice(comp, start),
            "rev": _slice(rev_eff, start),
            "mom": _slice(mom_eff, start),
            "rev_raw": _slice(rev_s, start),
            "mom_raw": _slice(mom_s, start),
            "w_rev": _slice(w_rev, start),
            "w_mom": _slice(w_mom, start),
            "noise_z": _slice(noise_z, start),
            "corr_rm": _slice(corr_rm, start),
            "ac1": _slice(ac1_series, start),
            "vol_regime": _slice(vol_regime, start),
            "regime_f": _slice(regime_f, start),
            "regime_state": _slice_str(regime_state, start),
            "post_shock": _slice_bool(post_shock, start),
            "post_halv": _slice_bool(post_halv, start),
            "days_since_halv": _slice(days_since_halv, start),
            "cascade": _slice_bool(cascade, start),
            "long_sig": _slice_bool(long_sig, start),
            "short_sig": _slice_bool(short_sig, start),
            "signal_state": _slice_str(signal_state, start),
        },
        "signals_history": _build_signal_history(
            timestamps, close, comp, rev_eff, mom_eff, long_sig, short_sig, cascade, signal_state, start
        ),
    }

    return output


def _r(v, digits=4):
    """Round a value for JSON output."""
    if v is None:
        return None
    return round(v, digits)


def _build_signal_history(ts, close, comp, rev, mom, long_sig, short_sig, cascade, state, start):
    """Extract a list of notable signal events for the history table."""
    events = []
    n = len(ts)
    for i in range(max(start, 1), n):
        if long_sig[i] or short_sig[i] or cascade[i]:
            dt = datetime.fromtimestamp(ts[i] / 1000, tz=timezone.utc)
            events.append({
                "date": dt.strftime("%Y-%m-%d"),
                "type": "long" if long_sig[i] else ("short" if short_sig[i] else "cascade"),
                "price": _r(close[i], 2),
                "composite": _r(comp[i]),
                "rev": _r(rev[i]),
                "mom": _r(mom[i]),
            })
    return events[-200:]  # keep last 200 events


# ---------------------------------------------------------------------------
# v2 Computation (BTC-adapted with funding rate + optimized params)
# ---------------------------------------------------------------------------

def compute_jlst_v2(data: dict, params: dict, funding_aligned: list) -> dict:
    """
    v2 indicator: same structure as compute_jlst() but with:
    - Funding rate in noise proxy
    - Parameters loaded from backtest optimization
    - Outputs funding_rate array for frontend chart
    """
    n = data["n"]
    opn = data["open"]
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    timestamps = data["ts"]

    revLen = params["revLen"]
    momLen = params["momLen"]
    momSkip = params["momSkip"]
    barsYear = params["barsYear"]
    normLen = params["normLen"]
    corrLen = params.get("corrLen", normLen)
    noiseGain = params.get("noiseGain", NOISE_GAIN)
    entryThr = params.get("entryThr", ENTRY_THR)
    deadZone = params.get("deadZone", DEAD_ZONE)

    # ── 1. Log returns ──
    log_ret = [None] * n
    for i in range(1, n):
        if close[i] > 0 and close[i - 1] > 0:
            log_ret[i] = math.log(close[i] / close[i - 1])

    # ── 2. Raw Rev & Mom ──
    rev_raw = [None] * n
    mom_raw = [None] * n
    for i in range(revLen, n):
        if close[i] > 0 and close[i - revLen] > 0:
            rev_raw[i] = -math.log(close[i] / close[i - revLen])

    skip_total = momSkip + momLen
    for i in range(skip_total, n):
        src_skip = close[i - momSkip] if i - momSkip >= 0 else None
        src_back = close[i - skip_total] if i - skip_total >= 0 else None
        if src_skip and src_skip > 0 and src_back and src_back > 0:
            mom_raw[i] = math.log(src_skip / src_back)

    # ── 3. Volatility & horizon scaling ──
    vol_ann_arr = rolling_std(log_ret, normLen)
    vol_ann = [None] * n
    for i in range(n):
        if vol_ann_arr[i] is not None:
            vol_ann[i] = vol_ann_arr[i] * math.sqrt(barsYear)

    rev_span = math.sqrt(revLen / barsYear)
    mom_span = math.sqrt(momLen / barsYear)

    rev_scaled = [None] * n
    mom_scaled = [None] * n

    if USE_VOL_ADJ:
        for i in range(n):
            va = vol_ann[i] if vol_ann[i] and vol_ann[i] > 0 else 1.0
            if rev_raw[i] is not None:
                rev_scaled[i] = rev_raw[i] / (va * rev_span)
            if mom_raw[i] is not None:
                mom_scaled[i] = mom_raw[i] / (va * mom_span)
    else:
        rev_scaled = rev_raw[:]
        mom_scaled = mom_raw[:]

    rev_z = zscore(rev_scaled, normLen)
    mom_z = zscore(mom_scaled, normLen)
    rev_s = [clamp(v, -CLAMP_Z, CLAMP_Z) for v in rev_z]
    mom_s = [clamp(v, -CLAMP_Z, CLAMP_Z) for v in mom_z]

    # ── 4. Noise proxy WITH funding rate (v2 difference) ──
    tr = true_range(high, low, close)
    atr_fast = rma(tr, ATR_LEN)
    atr_slow = rma(tr, ATR_LEN * ATR_SLOW_MUL)

    atr_pct = [None] * n
    vol_regime = [None] * n
    vol_ratio = [None] * n
    avg_vol = rolling_mean(volume, VOL_AVG_LEN)

    for i in range(n):
        if atr_fast[i] is not None and close[i] > 0:
            atr_pct[i] = atr_fast[i] / close[i]
        if atr_fast[i] is not None and atr_slow[i] is not None and atr_slow[i] > 0:
            vol_regime[i] = atr_fast[i] / atr_slow[i]
        if volume[i] is not None and avg_vol[i] is not None and avg_vol[i] > 0:
            vol_ratio[i] = volume[i] / avg_vol[i]

    nz_atr = zscore(atr_pct, normLen)
    nz_vol = zscore(vol_ratio, normLen)
    nz_regime = zscore(vol_regime, normLen)

    has_vol = any(v > 0 for v in volume if v is not None)

    # Funding rate noise component
    abs_funding = [abs(f) if f is not None else None for f in funding_aligned]
    nz_funding = zscore(abs_funding, normLen)
    has_funding = any(f is not None for f in funding_aligned)

    noise_z = [None] * n
    for i in range(n):
        a = nz_atr[i] if nz_atr[i] is not None else 0.0
        v = nz_vol[i] if nz_vol[i] is not None else 0.0
        r = nz_regime[i] if nz_regime[i] is not None else 0.0
        f = nz_funding[i] if nz_funding[i] is not None else 0.0

        if has_funding and nz_funding[i] is not None:
            # v2 weights with funding rate
            raw = V2_NOISE_W["atr"]*a + V2_NOISE_W["vol"]*v + V2_NOISE_W["regime"]*r + V2_NOISE_W["funding"]*f
        elif has_vol:
            # fallback to v1 weights
            raw = 0.4 * a + 0.3 * v + 0.3 * r
        else:
            raw = 0.5 * a + 0.5 * r
        noise_z[i] = clamp(raw, -3.0, 3.0)

    # ── 5. Dynamic weights ──
    w_rev = [None] * n
    w_mom = [None] * n
    for i in range(n):
        nz = noise_z[i] if noise_z[i] is not None else 0.0
        wr = max(0.0, REV_W * (1.0 + noiseGain * nz))
        wm = max(0.0, MOM_W * (1.0 - noiseGain * nz * 0.5))
        ws = max(wr + wm, 1e-6)
        w_rev[i] = wr / ws
        w_mom[i] = wm / ws

    # ── 6-12: Same as v1 (shock, halving, cascade, regime, composite, signals) ──
    shock_event = [False] * n
    post_shock = [False] * n
    for i in range(1, n):
        lr = abs(log_ret[i]) if log_ret[i] is not None else 0.0
        ap = atr_pct[i] if atr_pct[i] is not None else 999.0
        av = avg_vol[i] if avg_vol[i] is not None else 1.0
        vi = volume[i] if volume[i] is not None else 0.0
        move_ok = lr > SHOCK_K1 * ap if ap < 999 else False
        vol_ok = vi > SHOCK_K2 * av if has_vol else True
        shock_event[i] = move_ok and vol_ok

    bars_since_shock = [None] * n
    last_shock = -9999
    for i in range(n):
        if shock_event[i]:
            last_shock = i
        if last_shock >= 0:
            bars_since_shock[i] = i - last_shock
            post_shock[i] = bars_since_shock[i] <= SHOCK_WIN

    rev_eff = [None] * n
    for i in range(n):
        if rev_s[i] is not None:
            rev_eff[i] = rev_s[i] * SHOCK_ATTEN if post_shock[i] else rev_s[i]

    post_halv = [False] * n
    days_since_halv = [None] * n
    for i in range(n):
        t_ms = timestamps[i]
        t_dt = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
        last_h = None
        for hd in reversed(HALVING_DATES):
            if t_dt >= hd:
                last_h = hd
                break
        if last_h is not None:
            days = (t_dt - last_h).days
            days_since_halv[i] = days
            post_halv[i] = 0 <= days <= HALV_WIN

    mom_eff = [None] * n
    for i in range(n):
        m = mom_s[i] if mom_s[i] is not None else None
        if m is not None:
            boost = HALV_BOOST if post_halv[i] else 1.0
            mom_eff[i] = m * boost

    cascade = [False] * n
    for i in range(1, n):
        lr = abs(log_ret[i]) if log_ret[i] is not None else 0.0
        ap = atr_pct[i] if atr_pct[i] is not None else 999.0
        av = avg_vol[i] if avg_vol[i] is not None else 1.0
        vi = volume[i] if volume[i] is not None else 0.0
        move_ok = lr > CASCADE_K * ap if ap < 999 else False
        vol_ok = vi > CASCADE_VOL_K * av if has_vol else True
        cascade[i] = move_ok and vol_ok

    corr_rm = rolling_corr(rev_s, mom_s, corrLen)
    log_ret_lag = [None] * n
    for i in range(1, n):
        log_ret_lag[i] = log_ret[i - 1]
    ac1_series = rolling_corr(log_ret, log_ret_lag, corrLen)

    regime_f = [1.0] * n
    regime_state = ["normal"] * n
    corrThr = params.get("corrThr", CORR_THR)
    penalty = params.get("penalty", PENALTY)
    for i in range(n):
        cr = corr_rm[i]
        if cr is not None:
            if cr > corrThr:
                regime_f[i] = penalty
                regime_state[i] = "resonance"
            elif cr < -corrThr:
                regime_state[i] = "complementary"

    comp_raw = [None] * n
    for i in range(n):
        me = mom_eff[i]
        re = rev_eff[i]
        wm = w_mom[i]
        wr = w_rev[i]
        rf = regime_f[i]
        if me is not None and re is not None and wm is not None and wr is not None:
            comp_raw[i] = (wm * me + wr * re) * rf

    if SMOOTH_LEN > 1:
        comp = ema(comp_raw, SMOOTH_LEN)
    else:
        comp = comp_raw[:]

    long_sig = [False] * n
    short_sig = [False] * n
    signal_state = ["watch"] * n

    for i in range(1, n):
        c_now = comp[i]
        c_prev = comp[i - 1]
        if c_now is None or c_prev is None:
            continue
        in_dead = abs(c_now) < deadZone
        if c_now >= entryThr and c_prev < entryThr and not in_dead:
            long_sig[i] = True
        if c_now <= -entryThr and c_prev > -entryThr and not in_dead:
            short_sig[i] = True
        if c_now >= entryThr:
            signal_state[i] = "bullish"
        elif c_now <= -entryThr:
            signal_state[i] = "bearish"
        elif in_dead:
            signal_state[i] = "dead_zone"
        else:
            signal_state[i] = "watch"

    # Summary
    last_idx = n - 1
    summary = {
        "last_update": datetime.fromtimestamp(timestamps[last_idx] / 1000, tz=timezone.utc).isoformat(),
        "price": close[last_idx],
        "composite": _r(comp[last_idx]),
        "rev": _r(rev_eff[last_idx]),
        "mom": _r(mom_eff[last_idx]),
        "w_rev": _r(w_rev[last_idx]),
        "w_mom": _r(w_mom[last_idx]),
        "noise_z": _r(noise_z[last_idx]),
        "corr_rm": _r(corr_rm[last_idx]),
        "ac1": _r(ac1_series[last_idx]),
        "vol_regime": _r(vol_regime[last_idx]),
        "post_shock": post_shock[last_idx],
        "post_halv": post_halv[last_idx],
        "days_since_halv": days_since_halv[last_idx],
        "cascade": cascade[last_idx],
        "regime": regime_state[last_idx],
        "signal_state": signal_state[last_idx],
        "funding_rate": _r(funding_aligned[last_idx], 6) if funding_aligned[last_idx] is not None else None,
        "has_funding": has_funding,
    }

    start = 0
    for i in range(n):
        if comp[i] is not None:
            start = i
            break

    def _slice(arr, s):
        return [_r(v) for v in arr[s:]]
    def _slice_bool(arr, s):
        return arr[s:]
    def _slice_str(arr, s):
        return arr[s:]

    output = {
        "version": "v2",
        "params": {
            "revLen": revLen, "momLen": momLen, "momSkip": momSkip,
            "barsYear": barsYear, "normLen": normLen, "corrLen": corrLen,
            "noiseGain": noiseGain, "corrThr": corrThr, "penalty": penalty,
            "entryThr": entryThr, "deadZone": deadZone,
            "shockK1": SHOCK_K1, "halvWin": HALV_WIN, "halvBoost": HALV_BOOST,
            "hasFunding": has_funding,
        },
        "summary": summary,
        "data": {
            "timestamps": timestamps[start:],
            "open": [_r(v, 2) for v in opn[start:]],
            "high": [_r(v, 2) for v in high[start:]],
            "low": [_r(v, 2) for v in low[start:]],
            "close": [_r(v, 2) for v in close[start:]],
            "volume": [_r(v) for v in volume[start:]],
            "composite": _slice(comp, start),
            "rev": _slice(rev_eff, start),
            "mom": _slice(mom_eff, start),
            "rev_raw": _slice(rev_s, start),
            "mom_raw": _slice(mom_s, start),
            "w_rev": _slice(w_rev, start),
            "w_mom": _slice(w_mom, start),
            "noise_z": _slice(noise_z, start),
            "corr_rm": _slice(corr_rm, start),
            "ac1": _slice(ac1_series, start),
            "vol_regime": _slice(vol_regime, start),
            "regime_f": _slice(regime_f, start),
            "regime_state": _slice_str(regime_state, start),
            "post_shock": _slice_bool(post_shock, start),
            "post_halv": _slice_bool(post_halv, start),
            "days_since_halv": _slice(days_since_halv, start),
            "cascade": _slice_bool(cascade, start),
            "long_sig": _slice_bool(long_sig, start),
            "short_sig": _slice_bool(short_sig, start),
            "signal_state": _slice_str(signal_state, start),
            "funding_rate": [_r(v, 6) for v in funding_aligned[start:]],
        },
        "signals_history": _build_signal_history(
            timestamps, close, comp, rev_eff, mom_eff, long_sig, short_sig, cascade, signal_state, start
        ),
    }

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Force UTF-8 stdout on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("JLST BTC Momentum-Reversal Data Pipeline")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Load historical CSV for daily backfill
    csv_path = data_dir / "btc_historical.csv"
    csv_data = None
    if csv_path.exists():
        print(f"\n  Loading historical CSV: {csv_path.name}")
        csv_data = load_historical_csv(str(csv_path))
        print(f"  CSV: {csv_data['n']} rows")

    for tf_name, params in TIMEFRAME_PARAMS.items():
        print(f"\n>> Processing {params['label']} ({tf_name})...")

        # Fetch data
        print(f"  Fetching BTCUSDT {params['interval']} from Binance...")
        raw = fetch_binance_klines("BTCUSDT", params["interval"])
        if not raw:
            print(f"  [error] No data received for {tf_name}, skipping.")
            continue
        print(f"  Got {len(raw)} candles")

        data = parse_klines(raw)

        # Merge with historical CSV for daily timeframe
        if tf_name == "daily" and csv_data is not None:
            print(f"  Merging with historical CSV...")
            data = merge_data(csv_data, data)
            print(f"  Merged: {data['n']} total candles")

        # Compute indicators
        print(f"  Computing JLST indicators...")
        result = compute_jlst(data, params)

        # Write output
        out_path = data_dir / f"btc_{tf_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
        size_kb = out_path.stat().st_size / 1024
        print(f"  OK Written {out_path.name} ({size_kb:.0f} KB)")

        # Print summary
        s = result["summary"]
        print(f"  Price: ${s['price']:,.2f}")
        print(f"  Composite: {s['composite']}  |  Rev: {s['rev']}  |  Mom: {s['mom']}")
        print(f"  Weights: wRev={s['w_rev']} / wMom={s['w_mom']}")
        print(f"  Noise z: {s['noise_z']}  |  corr(R,M): {s['corr_rm']}  |  AC(1): {s['ac1']}")
        print(f"  Regime: {s['regime']}  |  State: {s['signal_state']}")
        print(f"  Halving: {s['days_since_halv']}d since last  |  Shock window: {s['post_shock']}")
        print(f"  Signals: {len(result['signals_history'])} historical events")

    # ── v2: Fetch funding rate + compute v2 indicators ──
    print(f"\n{'='*60}")
    print("JLST v2 — BTC Adapted Version")
    print(f"{'='*60}")

    # Load v2 optimized params
    v2_params_path = data_dir / "v2_params.json"
    v2_base_params = None
    if v2_params_path.exists():
        with open(v2_params_path, "r") as f:
            v2_meta = json.load(f)
        v2_base_params = v2_meta.get("best_params")
        print(f"  Loaded v2 params from {v2_params_path.name}")
    else:
        print(f"  [warn] {v2_params_path.name} not found, using v1 defaults for v2")

    # Fetch funding rate (graceful fallback)
    funding_cache_path = data_dir / "btc_funding.json"
    funding_records = load_funding_cache(str(funding_cache_path))
    print(f"  Funding cache: {len(funding_records)} records")

    try:
        print(f"  Fetching funding rate from Binance Futures API...")
        fresh = fetch_funding_rate("BTCUSDT")
        if fresh:
            # Merge with cache (dedup by timestamp)
            existing_ts = {r["ts"] for r in funding_records}
            new_count = 0
            for r in fresh:
                if r["ts"] not in existing_ts:
                    funding_records.append(r)
                    new_count += 1
            funding_records.sort(key=lambda x: x["ts"])
            save_funding_cache(funding_records, str(funding_cache_path))
            print(f"  Fetched {len(fresh)} records, {new_count} new. Total: {len(funding_records)}")
        else:
            print(f"  [warn] No fresh funding data, using cache")
    except Exception as e:
        print(f"  [warn] Funding rate fetch failed ({e}), using cache ({len(funding_records)} records)")

    funding_daily = aggregate_funding_daily(funding_records) if funding_records else {}
    print(f"  Daily funding rate entries: {len(funding_daily)}")

    # Produce v2 for each timeframe
    for tf_name, params in TIMEFRAME_PARAMS.items():
        print(f"\n>> v2 Processing {params['label']} ({tf_name})...")

        # Re-fetch or reuse data (already fetched above for v1)
        raw = fetch_binance_klines("BTCUSDT", params["interval"])
        if not raw:
            print(f"  [error] No data for {tf_name}, skipping v2.")
            continue
        data = parse_klines(raw)

        if tf_name == "daily" and csv_data is not None:
            data = merge_data(csv_data, data)

        # Build v2 params for this timeframe
        if v2_base_params and tf_name == "daily":
            tf_v2_params = dict(v2_base_params)
        elif v2_base_params and tf_name == "weekly":
            # Scale daily params to weekly
            scale = 7
            tf_v2_params = dict(v2_base_params)
            tf_v2_params["revLen"] = max(1, round(v2_base_params["revLen"] / scale))
            tf_v2_params["momLen"] = max(4, round(v2_base_params["momLen"] / scale))
            tf_v2_params["momSkip"] = max(1, round(v2_base_params["momSkip"] / scale))
            tf_v2_params["normLen"] = max(10, round(v2_base_params["normLen"] / scale))
            tf_v2_params["corrLen"] = tf_v2_params["normLen"]
            tf_v2_params["barsYear"] = 52
        elif v2_base_params and tf_name == "monthly":
            scale = 30
            tf_v2_params = dict(v2_base_params)
            tf_v2_params["revLen"] = max(1, round(v2_base_params["revLen"] / scale))
            tf_v2_params["momLen"] = max(2, round(v2_base_params["momLen"] / scale))
            tf_v2_params["momSkip"] = max(1, round(v2_base_params["momSkip"] / scale))
            tf_v2_params["normLen"] = max(6, round(v2_base_params["normLen"] / scale))
            tf_v2_params["corrLen"] = tf_v2_params["normLen"]
            tf_v2_params["barsYear"] = 12
        else:
            # No v2 params, use v1 params
            tf_v2_params = dict(params)

        # Align funding rate
        funding_aligned = align_funding_to_timestamps(funding_daily, data["ts"])

        print(f"  Computing JLST v2 indicators (revLen={tf_v2_params.get('revLen')}, momLen={tf_v2_params.get('momLen')})...")
        result = compute_jlst_v2(data, tf_v2_params, funding_aligned)

        out_path = data_dir / f"btc_v2_{tf_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
        size_kb = out_path.stat().st_size / 1024
        print(f"  OK Written {out_path.name} ({size_kb:.0f} KB)")

        s = result["summary"]
        print(f"  Composite v2: {s['composite']}  |  Rev: {s['rev']}  |  Mom: {s['mom']}")
        print(f"  Funding: {s.get('funding_rate', 'N/A')}  |  State: {s['signal_state']}")

    # Write metadata
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Binance BTCUSDT",
        "model": "JLST Reversal-Momentum Composite (BTC Specialized)",
        "paper": "Jegadeesh, Luo, Subrahmanyam & Titman (2025, RFS)",
        "timeframes": list(TIMEFRAME_PARAMS.keys()),
        "v2_available": v2_base_params is not None,
        "funding_records": len(funding_records),
    }
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Data written to {data_dir}")


if __name__ == "__main__":
    main()
