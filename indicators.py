import pandas as pd
import numpy as np


def _ts_ms(ts) -> int:
    """タイムスタンプをミリ秒に変換"""
    return int(ts.timestamp() * 1000)


def calc_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> dict:
    """ボリンジャーバンド"""
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std

    upper_out, middle_out, lower_out = [], [], []
    for ts in series.index:
        u = upper[ts]
        m = ma[ts]
        lo = lower[ts]
        if pd.notna(u) and pd.notna(m) and pd.notna(lo):
            t = _ts_ms(ts)
            upper_out.append({"x": t, "y": round(float(u), 4)})
            middle_out.append({"x": t, "y": round(float(m), 4)})
            lower_out.append({"x": t, "y": round(float(lo), 4)})

    return {"upper": upper_out, "middle": middle_out, "lower": lower_out}


def calc_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """一目均衡表"""
    def midpoint(h, l, period):
        return (h.rolling(period).max() + l.rolling(period).min()) / 2

    tenkan = midpoint(high, low, 9)
    kijun = midpoint(high, low, 26)
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = midpoint(high, low, 52).shift(26)
    chikou = close.shift(-26)

    def to_xy(s):
        out = []
        for ts, val in s.items():
            if pd.notna(val):
                out.append({"x": _ts_ms(ts), "y": round(float(val), 4)})
        return out

    return {
        "tenkan_sen":   to_xy(tenkan),
        "kijun_sen":    to_xy(kijun),
        "senkou_span_a": to_xy(senkou_a),
        "senkou_span_b": to_xy(senkou_b),
        "chikou_span":  to_xy(chikou),
    }


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> list:
    """ATR (Average True Range) — 損切り幅計算に使用"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    out = []
    for ts, val in atr.items():
        if pd.notna(val):
            out.append({"x": _ts_ms(ts), "y": round(float(val), 4)})
    return out


def calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                    k_period: int = 14, d_period: int = 3) -> dict:
    """Stochastic Oscillator"""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = highest_high - lowest_low
    k_raw = 100 * (close - lowest_low) / denom.replace(0, np.nan)
    k = k_raw.rolling(3).mean()   # Slow %K (3-period smoothing)
    d = k.rolling(d_period).mean()

    k_out, d_out = [], []
    for ts in close.index:
        kv = k[ts]
        dv = d[ts]
        t = _ts_ms(ts)
        if pd.notna(kv):
            k_out.append({"x": t, "y": round(float(kv), 2)})
        if pd.notna(dv):
            d_out.append({"x": t, "y": round(float(dv), 2)})

    return {"k": k_out, "d": d_out}


def calc_vwap(hist: pd.DataFrame) -> list:
    """VWAP (Volume Weighted Average Price) — 1日・5日足のみ有効"""
    typical = (hist["High"] + hist["Low"] + hist["Close"]) / 3
    vol = hist["Volume"].replace(0, np.nan)
    cum_tp_vol = (typical * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap = cum_tp_vol / cum_vol

    out = []
    for ts, val in vwap.items():
        if pd.notna(val):
            out.append({"x": _ts_ms(ts), "y": round(float(val), 4)})
    return out


def detect_candlestick_patterns(hist: pd.DataFrame) -> list:
    """ローソク足パターン認識"""
    results = []
    opens  = hist["Open"].values
    highs  = hist["High"].values
    lows   = hist["Low"].values
    closes = hist["Close"].values
    times  = [_ts_ms(ts) for ts in hist.index]
    n = len(closes)

    for i in range(2, n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        po, ph, pl, pc = opens[i-1], highs[i-1], lows[i-1], closes[i-1]
        ppo, pph, ppl, ppc = opens[i-2], highs[i-2], lows[i-2], closes[i-2]

        body = abs(c - o)
        candle_range = h - l if h != l else 1e-9
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        p_body = abs(pc - po)

        patterns_found = []

        # Doji: body が candle_range の10%以下
        if body / candle_range < 0.1:
            patterns_found.append(("doji", "neutral", 1))

        # Hammer: 下ひげ >= body*2、上ひげ小さい、前トレンド下落
        if (lower_wick >= body * 2 and upper_wick <= body * 0.5
                and pc > ppc):  # 前2本が下落トレンド
            patterns_found.append(("hammer", "buy", 2))

        # Inverted Hammer
        if (upper_wick >= body * 2 and lower_wick <= body * 0.5
                and c > o and pc < po):
            patterns_found.append(("inverted_hammer", "buy", 1))

        # Shooting Star: 上昇後の上ひげが長い陰線
        if (upper_wick >= body * 2 and lower_wick <= body * 0.3
                and c < o and pc > po):
            patterns_found.append(("shooting_star", "sell", 2))

        # Bullish Engulfing
        if (c > o and pc < po
                and o < pc and c > po
                and body > p_body * 1.0):
            patterns_found.append(("bullish_engulfing", "buy", 3))

        # Bearish Engulfing
        if (c < o and pc > po
                and o > pc and c < po
                and body > p_body * 1.0):
            patterns_found.append(("bearish_engulfing", "sell", 3))

        # Morning Star: 3本パターン — 下落大陰線、小体、大陽線
        ppo_body = abs(ppc - ppo)
        if (ppc < ppo           # 2日前: 陰線
                and p_body < ppo_body * 0.5   # 前日: 小体
                and c > o                      # 当日: 陽線
                and c > (ppo + ppc) / 2):      # 当日終値が2日前の中値超え
            patterns_found.append(("morning_star", "buy", 3))

        # Evening Star: 3本パターン — 上昇大陽線、小体、大陰線
        if (ppc > ppo           # 2日前: 陽線
                and p_body < ppo_body * 0.5   # 前日: 小体
                and c < o                      # 当日: 陰線
                and c < (ppo + ppc) / 2):      # 当日終値が2日前の中値割れ
            patterns_found.append(("evening_star", "sell", 3))

        for pat, signal, strength in patterns_found:
            results.append({
                "date": times[i],
                "pattern": pat,
                "signal": signal,
                "strength": strength,
            })

    return results


def find_support_resistance(hist: pd.DataFrame, n_levels: int = 5) -> dict:
    """サポート/レジスタンスライン自動検出"""
    closes = hist["Close"].values
    highs  = hist["High"].values
    lows   = hist["Low"].values
    n = len(closes)

    pivot_highs = []
    pivot_lows  = []

    # ローカル高値・安値を検出（前後2本比較）
    for i in range(2, n - 2):
        if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and \
           highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
            pivot_highs.append(highs[i])
        if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and \
           lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            pivot_lows.append(lows[i])

    def cluster_levels(prices, n_out):
        if not prices:
            return []
        prices = sorted(prices)
        price_range = max(prices) - min(prices)
        if price_range == 0:
            return [round(float(prices[0]), 2)]
        tolerance = price_range * 0.015   # 1.5% 許容幅でクラスタリング

        clusters = []
        current = [prices[0]]
        for p in prices[1:]:
            if p - current[-1] <= tolerance:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)

        # クラスタをヒット数でソートして上位 n_out を返す
        clusters.sort(key=lambda c: -len(c))
        result = []
        for c in clusters[:n_out]:
            result.append(round(float(np.mean(c)), 2))
        return sorted(result)

    support    = cluster_levels(pivot_lows,  n_levels)
    resistance = cluster_levels(pivot_highs, n_levels)

    return {"support": support, "resistance": resistance}
