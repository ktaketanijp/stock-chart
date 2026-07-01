#!/usr/bin/env python3
"""モメンタムスキャナー — 前日比上昇・出来高急増銘柄を検出"""

import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# ---------------------------------------------------------------------------
# デフォルト監視銘柄
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "GOOGL", "META", "AMD", "SMCI", "PLTR",
    "JPM", "BAC", "GS", "V", "MA",
    "XOM", "CVX", "LLY", "UNH", "COST",
    # ETF
    "QQQ", "SPY", "IWM",
]

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _get_canslim_score_safe(ticker: str) -> int:
    """CAN SLIMスコアを安全に取得。失敗時は50を返す。"""
    try:
        from fundamentals import get_canslim_score
        result = get_canslim_score(ticker)
        score = result.get("total_score")
        return int(score) if score is not None else 50
    except Exception:
        return 50


# ---------------------------------------------------------------------------
# scan_momentum
# ---------------------------------------------------------------------------

def _scan_single(ticker: str, strict_filter: bool = True):
    """1銘柄のモメンタムデータを取得。strict_filter=Trueの場合のみフィルタ条件を適用。"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2mo", interval="1d")
        if hist.empty or len(hist) < 6:
            return None

        closes = hist["Close"]
        volumes = hist["Volume"]

        # 前日比変化率
        prev_close = _safe_float(closes.iloc[-2])
        current_close = _safe_float(closes.iloc[-1])
        if prev_close is None or current_close is None or prev_close <= 0:
            return None
        change_pct = (current_close - prev_close) / prev_close * 100

        # 出来高比（本日÷20日平均）
        today_volume = _safe_float(volumes.iloc[-1])
        avg_volume_20d = _safe_float(volumes.iloc[-21:-1].mean()) if len(volumes) >= 21 else _safe_float(volumes.iloc[:-1].mean())
        if avg_volume_20d is None or avg_volume_20d <= 0:
            return None
        volume_ratio = today_volume / avg_volume_20d if today_volume else 0

        # フィルタ条件（strict_filterのみ適用）
        if strict_filter and (change_pct <= 1.0 or volume_ratio <= 1.5):
            return None

        # 52週高値からの位置
        week52_high = _safe_float(closes.max())
        pct_from_52h = ((current_close - week52_high) / week52_high * 100) if week52_high and week52_high > 0 else None

        # CAN SLIMスコア（簡易）— strict_filterのみ取得（速度優先）
        canslim_score = _get_canslim_score_safe(ticker) if strict_filter else None

        momentum_score = round(volume_ratio * change_pct, 3)

        return {
            "ticker": ticker,
            "current_price": round(current_close, 2),
            "change_pct": round(change_pct, 2),
            "volume_ratio": round(volume_ratio, 2),
            "today_volume": int(today_volume) if today_volume else 0,
            "avg_volume_20d": int(avg_volume_20d) if avg_volume_20d else 0,
            "pct_from_52w_high": round(pct_from_52h, 2) if pct_from_52h is not None else None,
            "canslim_score": canslim_score,
            "momentum_score": momentum_score,
        }

    except Exception:
        return None


def scan_momentum(tickers: list = None) -> list:
    """
    モメンタムスキャン。
    フィルタ条件:
      - 前日比変化率 > 1.0%
      - 出来高比（本日÷20日平均） > 1.5x
    結果をmomentum_score（出来高比×変化率）降順で最大20件返す。
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    results = []
    processed = set()
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(_scan_single, t): t for t in tickers}
        try:
            for future in as_completed(future_map, timeout=90):
                processed.add(id(future))
                try:
                    result = future.result(timeout=15)
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass
        except FuturesTimeoutError:
            for future in future_map:
                if id(future) not in processed and future.done():
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception:
                        pass

    results.sort(key=lambda x: x.get("momentum_score", 0), reverse=True)
    return results[:20]


# ---------------------------------------------------------------------------
# scan_top_movers（フィルタなし全銘柄ランキング）
# ---------------------------------------------------------------------------

def scan_top_movers(tickers: list = None, top_n: int = 20) -> list:
    """
    フィルタなしで全銘柄の変化率ランキングを返す。
    モメンタムスキャンが空の時のフォールバック用。
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    results = []
    processed = set()
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(_scan_single, t, False): t for t in tickers}
        try:
            for future in as_completed(future_map, timeout=60):
                processed.add(id(future))
                try:
                    result = future.result(timeout=15)
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass
        except FuturesTimeoutError:
            for future in future_map:
                if id(future) not in processed and future.done():
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception:
                        pass

    results.sort(key=lambda x: x.get("change_pct", 0), reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# scan_premarket
# ---------------------------------------------------------------------------

def _scan_premarket_single(ticker: str):
    """プレマーケットデータを1銘柄取得。"""
    try:
        stock = yf.Ticker(ticker)
        # prePostData=True でプレ/アフターマーケットを含む
        hist = stock.history(period="2d", interval="1m", prepost=True)
        if hist.empty:
            return None

        # 通常時間外のデータを抽出
        # yfinanceはprepost=Trueで時間外データも含む
        # 最新の終値（プレマーケット含む）
        current = _safe_float(hist["Close"].iloc[-1])
        if current is None:
            return None

        # 前日の正規市場終値を取得（比較用）
        regular_hist = stock.history(period="2d", interval="1d")
        if len(regular_hist) < 2:
            return None
        prev_regular_close = _safe_float(regular_hist["Close"].iloc[-2])
        if prev_regular_close is None or prev_regular_close <= 0:
            return None

        premarket_change_pct = (current - prev_regular_close) / prev_regular_close * 100
        volume = int(hist["Volume"].sum()) if not hist["Volume"].empty else 0

        return {
            "ticker": ticker,
            "premarket_price": round(current, 2),
            "prev_close": round(prev_regular_close, 2),
            "premarket_change_pct": round(premarket_change_pct, 2),
            "premarket_volume": volume,
        }

    except Exception:
        return None


def scan_premarket(tickers: list = None) -> list:
    """
    プレマーケット銘柄スキャン（yfinance prePostで取得）。
    前日終値比の変化率降順で返す。
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    results = []
    processed = set()
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(_scan_premarket_single, t): t for t in tickers}
        try:
            for future in as_completed(future_map, timeout=90):
                processed.add(id(future))
                try:
                    result = future.result(timeout=15)
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass
        except FuturesTimeoutError:
            for future in future_map:
                if id(future) not in processed and future.done():
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception:
                        pass

    results.sort(key=lambda x: x.get("premarket_change_pct", 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# RSI ヘルパー（シンプル実装）
# ---------------------------------------------------------------------------

def _calc_rsi(prices: list, period: int = 14) -> float:
    """終値リストからRSIを計算して返す（シンプル実装）。"""
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains  = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)


# ---------------------------------------------------------------------------
# scan_52w_breakout
# ---------------------------------------------------------------------------

def _scan_52w_breakout_single(ticker: str):
    """52週高値ブレイクアウト — 1銘柄処理。"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", interval="1d")
        if hist.empty or len(hist) < 10:
            return None

        closes  = hist["Close"]
        volumes = hist["Volume"]

        current_price = _safe_float(closes.iloc[-1])
        if current_price is None or current_price <= 0:
            return None

        week52_high = _safe_float(closes.max())
        if week52_high is None or week52_high <= 0:
            return None

        # 条件: 現在価格 >= 52週高値の98%
        if current_price < week52_high * 0.98:
            return None

        breakout_pct = round((current_price - week52_high) / week52_high * 100, 2)
        today_volume = _safe_float(volumes.iloc[-1])

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "week52_high": round(week52_high, 2),
            "breakout_pct": breakout_pct,
            "volume": int(today_volume) if today_volume else 0,
        }
    except Exception:
        return None


def scan_52w_breakout(tickers: list = None) -> list:
    """
    52週高値を更新した銘柄を探す。
    条件: 現在価格 >= 52週高値の98%以上
    返す情報: ticker, price, week52_high, breakout_pct, volume
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    results = []
    processed = set()
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(_scan_52w_breakout_single, t): t for t in tickers}
        try:
            for future in as_completed(future_map, timeout=90):
                processed.add(id(future))
                try:
                    result = future.result(timeout=15)
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass
        except FuturesTimeoutError:
            for future in future_map:
                if id(future) not in processed and future.done():
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception:
                        pass

    # breakout_pct 降順（高値に近いほど上位）
    results.sort(key=lambda x: x.get("breakout_pct", -999), reverse=True)
    return results


# ---------------------------------------------------------------------------
# scan_volume_surge
# ---------------------------------------------------------------------------

def _scan_volume_surge_single(ticker: str, multiplier: float):
    """出来高急増 — 1銘柄処理。"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2mo", interval="1d")
        if hist.empty or len(hist) < 6:
            return None

        closes  = hist["Close"]
        volumes = hist["Volume"]

        today_volume = _safe_float(volumes.iloc[-1])
        avg_volume = _safe_float(
            volumes.iloc[-21:-1].mean() if len(volumes) >= 21 else volumes.iloc[:-1].mean()
        )
        if avg_volume is None or avg_volume <= 0:
            return None

        volume_ratio = (today_volume / avg_volume) if today_volume else 0

        # 条件: 当日出来高 >= 20日平均出来高 × multiplier
        if volume_ratio < multiplier:
            return None

        prev_close    = _safe_float(closes.iloc[-2])
        current_price = _safe_float(closes.iloc[-1])
        if prev_close is None or current_price is None or prev_close <= 0:
            return None
        change_pct = round((current_price - prev_close) / prev_close * 100, 2)

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "change_pct": change_pct,
            "volume": int(today_volume) if today_volume else 0,
            "avg_volume": int(avg_volume) if avg_volume else 0,
            "volume_ratio": round(volume_ratio, 2),
        }
    except Exception:
        return None


def scan_volume_surge(tickers: list = None, multiplier: float = 2.0) -> list:
    """
    平均出来高の N倍以上の出来高を記録している銘柄。
    条件: 当日出来高 >= 20日平均出来高 × multiplier
    返す情報: ticker, price, change_pct, volume, avg_volume, volume_ratio
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    results = []
    processed = set()
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(_scan_volume_surge_single, t, multiplier): t for t in tickers}
        try:
            for future in as_completed(future_map, timeout=90):
                processed.add(id(future))
                try:
                    result = future.result(timeout=15)
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass
        except FuturesTimeoutError:
            for future in future_map:
                if id(future) not in processed and future.done():
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception:
                        pass

    results.sort(key=lambda x: x.get("volume_ratio", 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# scan_rsi_extreme
# ---------------------------------------------------------------------------

def _scan_rsi_extreme_single(ticker: str, oversold: float, overbought: float):
    """RSI極値スキャン — 1銘柄処理。"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo", interval="1d")
        if hist.empty or len(hist) < 20:
            return None

        closes = hist["Close"]
        prices = [_safe_float(c) for c in closes.tolist()]
        prices = [p for p in prices if p is not None]
        if len(prices) < 20:
            return None

        rsi = _calc_rsi(prices, period=14)

        if rsi > oversold and rsi < overbought:
            return None

        signal = "oversold" if rsi <= oversold else "overbought"

        current_price = prices[-1]
        prev_price    = prices[-2] if len(prices) >= 2 else current_price
        change_pct    = round((current_price - prev_price) / prev_price * 100, 2) if prev_price > 0 else 0

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "change_pct": change_pct,
            "rsi": rsi,
            "signal": signal,
        }
    except Exception:
        return None


def scan_rsi_extreme(tickers: list = None, oversold: float = 30, overbought: float = 70) -> list:
    """
    RSIが極端な値の銘柄。
    条件: RSI <= oversold (売られすぎ) または RSI >= overbought (買われすぎ)
    返す情報: ticker, price, change_pct, rsi, signal (oversold/overbought)
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    results = []
    processed = set()
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(_scan_rsi_extreme_single, t, oversold, overbought): t
            for t in tickers
        }
        try:
            for future in as_completed(future_map, timeout=90):
                processed.add(id(future))
                try:
                    result = future.result(timeout=15)
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass
        except FuturesTimeoutError:
            for future in future_map:
                if id(future) not in processed and future.done():
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception:
                        pass

    # oversold を上位（反発狙い向け）、次に overbought
    results.sort(key=lambda x: (x["signal"] == "overbought", x["rsi"]))
    return results


# ---------------------------------------------------------------------------
# CLI テスト
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    print("=== Momentum Scan ===")
    mo = scan_momentum(["AAPL", "NVDA", "TSLA"])
    print(json.dumps(mo, ensure_ascii=False, indent=2))
