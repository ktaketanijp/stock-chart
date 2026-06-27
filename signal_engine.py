#!/usr/bin/env python3
"""統合シグナルエンジン — CAN SLIM + SEPA + X/Twitter sentiment"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------

JST = timezone(timedelta(hours=9))

def _now_jst() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")


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


def _calc_rsi(series: pd.Series, period: int = 14):
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return _safe_float(val)


def _calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """(macd_line_last, signal_line_last) を返す。計算不可なら (None, None)"""  # noqa: E501
    if len(series) < slow + signal:
        return None, None
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return _safe_float(macd_line.iloc[-1]), _safe_float(signal_line.iloc[-1])


def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    if len(close) < period + 1:
        return None
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return _safe_float(atr.iloc[-1])


# ---------------------------------------------------------------------------
# Layer 1: Technical Score
# ---------------------------------------------------------------------------

def get_technical_score(ticker: str) -> dict:
    """テクニカルスコア（-100〜+100）を計算して返す。"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", interval="1d")
        if hist.empty or len(hist) < 30:
            return {"score": 0, "error": "データ不足", "details": {}}

        closes = hist["Close"]
        current_price = _safe_float(closes.iloc[-1])
        if current_price is None:
            return {"score": 0, "error": "現在価格取得失敗", "details": {}}

        score = 0
        details = {}
        reasons = []

        # MA20 / MA50 / MA200
        ma20 = _safe_float(closes.rolling(20).mean().iloc[-1])
        ma50 = _safe_float(closes.rolling(50).mean().iloc[-1])
        ma200 = _safe_float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None

        details["current_price"] = round(current_price, 2)
        details["ma20"] = round(ma20, 2) if ma20 else None
        details["ma50"] = round(ma50, 2) if ma50 else None
        details["ma200"] = round(ma200, 2) if ma200 else None

        # 価格 > MA200 → +30点
        if ma200 is not None:
            if current_price > ma200:
                score += 30
                reasons.append(f"価格がMA200以上（長期上昇トレンド）: {current_price:.2f} > {ma200:.2f}")
            else:
                reasons.append(f"価格がMA200未満（長期下落トレンド）: {current_price:.2f} < {ma200:.2f}")
        else:
            reasons.append("MA200計算不可（データ不足）")

        # MA50 > MA200 → +20点
        if ma50 is not None and ma200 is not None:
            if ma50 > ma200:
                score += 20
                reasons.append(f"MA50 > MA200（ゴールデンクロス状態）: {ma50:.2f} > {ma200:.2f}")
            else:
                reasons.append(f"MA50 < MA200（デッドクロス状態）: {ma50:.2f} < {ma200:.2f}")

        # RSI 40〜60 → +25点 / 40以下→過売り+15 / 70以上→買われすぎ+5
        rsi = _calc_rsi(closes)
        details["rsi"] = round(rsi, 2) if rsi is not None else None
        if rsi is not None:
            if 40 <= rsi <= 60:
                score += 25
                reasons.append(f"RSI中立ゾーン（健全な上昇余地）: {rsi:.1f}")
            elif rsi < 40:
                score += 15
                reasons.append(f"RSI過売りゾーン（反発期待）: {rsi:.1f}")
            elif rsi >= 70:
                score += 5
                reasons.append(f"RSI買われすぎゾーン（慎重）: {rsi:.1f}")
            else:
                reasons.append(f"RSI中間ゾーン: {rsi:.1f}")
        else:
            reasons.append("RSI計算不可")

        # MACDライン > シグナル → +25点
        macd_val, signal_val = _calc_macd(closes)
        details["macd"] = round(macd_val, 4) if macd_val is not None else None
        details["macd_signal"] = round(signal_val, 4) if signal_val is not None else None
        if macd_val is not None and signal_val is not None:
            if macd_val > signal_val:
                score += 25
                reasons.append(f"MACDがシグナル線を上回る（強気）: {macd_val:.4f} > {signal_val:.4f}")
            else:
                reasons.append(f"MACDがシグナル線を下回る（弱気）: {macd_val:.4f} < {signal_val:.4f}")
        else:
            reasons.append("MACD計算不可")

        score = max(-100, min(100, score))
        return {
            "score": score,
            "reasons": reasons,
            "details": details,
        }

    except Exception as e:
        return {"score": 0, "error": str(e), "details": {}}


# ---------------------------------------------------------------------------
# Layer 2: Fundamental Score
# ---------------------------------------------------------------------------

def get_fundamental_score(ticker: str) -> dict:
    """CAN SLIM + SEPA 平均スコアを返す（0〜100）。"""
    try:
        from fundamentals import get_canslim_score, get_sepa_score

        canslim_result = get_canslim_score(ticker)
        sepa_result = get_sepa_score(ticker)

        canslim_raw = canslim_result.get("total_score")
        # SEPA は conditions_met / total_conditions で正規化
        sepa_met = sepa_result.get("conditions_met", 0)
        sepa_total = sepa_result.get("total_conditions", 7)

        canslim_score = _safe_float(canslim_raw, 50)
        sepa_score_normalized = (sepa_met / sepa_total * 100) if sepa_total > 0 else 50

        avg_score = (canslim_score + sepa_score_normalized) / 2

        reasons = []
        if canslim_result.get("grade"):
            reasons.append(f"CAN SLIMグレード: {canslim_result['grade']} ({canslim_score:.0f}/100)")
        if sepa_result.get("conditions_met") is not None:
            reasons.append(f"SEPA条件 {sepa_met}/{sepa_total} クリア")

        return {
            "score": round(avg_score, 1),
            "canslim_score": round(canslim_score, 1),
            "sepa_score": round(sepa_score_normalized, 1),
            "reasons": reasons,
        }

    except Exception as e:
        return {
            "score": 50,
            "canslim_score": 50,
            "sepa_score": 50,
            "reasons": [f"ファンダメンタルズ取得失敗（フォールバック）: {str(e)[:80]}"],
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Layer 3: Sentiment Score
# ---------------------------------------------------------------------------

def get_sentiment_score_for_signal(ticker: str) -> dict:
    """X/Twitterセンチメントスコアを-100〜+100で返す。"""
    try:
        from sentiment import get_twitter_sentiment

        result = get_twitter_sentiment(ticker)
        score = result.get("sentiment_score", 0)
        score = _safe_float(score, 0)
        score = max(-100, min(100, score))

        reasons = []
        summary = result.get("summary", "")
        if summary:
            reasons.append(f"Xセンチメント: {summary}")
        velocity = result.get("velocity", "")
        if velocity:
            reasons.append(f"Xトレンド速度: {velocity}")
        if result.get("catalyst_detected"):
            reasons.append(f"触媒検出: {result.get('catalyst_type', 'unknown')}")

        return {
            "score": round(score),
            "reasons": reasons,
            "velocity": velocity,
            "catalyst_detected": result.get("catalyst_detected", False),
        }

    except Exception as e:
        return {
            "score": 0,
            "reasons": [f"センチメント取得失敗（フォールバック）: {str(e)[:80]}"],
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# シグナル判定ロジック
# ---------------------------------------------------------------------------

def _score_to_signal(total_score: float):
    """(signal_en, signal_ja) を返す"""
    if total_score >= 60:
        return "STRONG_BUY", "強い買いシグナル"
    elif total_score >= 30:
        return "BUY", "買いシグナル"
    elif total_score > -30:
        return "NEUTRAL", "中立"
    elif total_score > -60:
        return "SELL", "売りシグナル"
    else:
        return "STRONG_SELL", "強い売りシグナル"


# ---------------------------------------------------------------------------
# 統合シグナル生成
# ---------------------------------------------------------------------------

def generate_signal(ticker: str) -> dict:
    """3層を統合して総合シグナルを返す。"""
    ticker = ticker.upper().strip()

    technical_weight = 0.20
    fundamental_weight = 0.40
    sentiment_weight = 0.40

    # 各スコア取得
    tech = get_technical_score(ticker)
    fund = get_fundamental_score(ticker)
    sent = get_sentiment_score_for_signal(ticker)

    tech_score = _safe_float(tech.get("score"), 0)
    # fundamental は 0〜100 → -100〜+100 に変換
    fund_score_raw = _safe_float(fund.get("score"), 50)
    fund_score = (fund_score_raw - 50) * 2  # 50 → 0, 100 → +100, 0 → -100
    sent_score = _safe_float(sent.get("score"), 0)

    total_score = (
        tech_score * technical_weight
        + fund_score * fundamental_weight
        + sent_score * sentiment_weight
    )
    total_score = round(total_score, 1)

    signal_en, signal_ja = _score_to_signal(total_score)

    # 根拠リスト（日本語）
    reasons = []
    reasons.extend(tech.get("reasons", []))
    reasons.extend(fund.get("reasons", []))
    reasons.extend(sent.get("reasons", []))

    # 現在価格・ATR取得（エントリー/損切/利確計算用）
    current_price = None
    atr_val = None
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo", interval="1d")
        if not hist.empty:
            current_price = _safe_float(hist["Close"].iloc[-1])
            atr_val = _calc_atr(hist["High"], hist["Low"], hist["Close"])
    except Exception:
        pass

    entry_low = None
    entry_high = None
    stop_loss = None
    target = None
    risk_reward = None

    if current_price is not None:
        entry_low = round(current_price * 0.995, 2)
        entry_high = round(current_price * 1.005, 2)

        if atr_val is not None and atr_val > 0:
            sl_dist = atr_val * 1.5
            stop_loss = round(current_price - sl_dist, 2)
            target = round(current_price + sl_dist * 2, 2)
            risk = current_price - stop_loss
            reward = target - current_price
            risk_reward = round(reward / risk, 2) if risk > 0 else None
        else:
            # ATR取得失敗時: 現在値の2%を損切幅とする
            stop_loss = round(current_price * 0.98, 2)
            target = round(current_price * 1.04, 2)
            risk_reward = 2.0

    return {
        "ticker": ticker,
        "total_score": total_score,
        "signal": signal_en,
        "signal_ja": signal_ja,
        "technical_score": round(tech_score, 1),
        "fundamental_score": round(fund_score_raw, 1),
        "sentiment_score": round(sent_score, 1),
        "weights": {
            "technical": technical_weight,
            "fundamental": fundamental_weight,
            "sentiment": sentiment_weight,
        },
        "reasons": reasons,
        "current_price": round(current_price, 2) if current_price else None,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "target": target,
        "atr": round(atr_val, 2) if atr_val else None,
        "risk_reward": risk_reward,
        "updated_at": _now_jst(),
    }


# ---------------------------------------------------------------------------
# スキャン: BUY以上の銘柄を並列取得
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST = [
    # Mega cap tech
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
    # Semiconductors
    "AMD", "SMCI", "AVGO", "QCOM", "MU",
    # High growth / AI
    "PLTR", "CRWD", "NET", "SNOW", "DDOG",
    # Finance
    "JPM", "V", "MA",
    # Healthcare
    "LLY", "UNH",
]

BUY_SIGNALS = {"STRONG_BUY", "BUY"}


def scan_opportunities(tickers: list = None) -> list:
    """STRONG_BUYとBUYのシグナルを持つ銘柄をスコア降順で返す。タイムアウトは部分結果を返す。"""
    if tickers is None:
        tickers = DEFAULT_WATCHLIST

    results = []
    processed = set()
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(generate_signal, t): t for t in tickers}
        try:
            for future in as_completed(future_map, timeout=60):
                processed.add(id(future))
                try:
                    result = future.result(timeout=10)
                    if result.get("signal") in BUY_SIGNALS:
                        results.append(result)
                except Exception:
                    pass
        except FuturesTimeoutError:
            # タイムアウト時は未処理の完了分だけ追加
            for future in future_map:
                if id(future) not in processed and future.done():
                    try:
                        result = future.result()
                        if result.get("signal") in BUY_SIGNALS:
                            results.append(result)
                    except Exception:
                        pass

    results.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# CLI テスト
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    result = generate_signal("AAPL")
    print(json.dumps(result, ensure_ascii=False, indent=2))
