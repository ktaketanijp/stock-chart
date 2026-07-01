#!/usr/bin/env python3
"""デモトレード管理 — ¥100,000元手でシミュレーション（USD株価→JPY換算）"""

import json
import os
import time
import uuid
import threading
from datetime import datetime

import yfinance as yf

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "paper_trades.json")
_lock = threading.Lock()

TARGET_PNL = 10000  # 目標損益 ¥10,000

# ---------------------------------------------------------------------------
# USD/JPY 為替レート（1時間キャッシュ）
# ---------------------------------------------------------------------------

_forex_cache: dict = {}

def _get_usd_jpy_rate() -> float:
    """USD/JPY為替レートを取得（1時間キャッシュ）"""
    entry = _forex_cache.get("usd_jpy")
    if entry and time.time() - entry["ts"] < 3600:
        return entry["rate"]
    try:
        rate = float(yf.Ticker("JPY=X").fast_info.last_price)
        if rate and rate > 100:  # 合理的な範囲チェック
            _forex_cache["usd_jpy"] = {"ts": time.time(), "rate": round(rate, 2)}
            return round(rate, 2)
    except Exception:
        pass
    if entry:
        return entry["rate"]
    return 150.0  # フォールバック


# ---------------------------------------------------------------------------
# データ読み書き
# ---------------------------------------------------------------------------

def _default_store() -> dict:
    return {
        "capital_initial": 100000,
        "capital_cash": 100000,
        "positions": [],
        "trades": [],
        "daily_pnl": [],
    }


def _load() -> dict:
    if not os.path.exists(DATA_FILE):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        store = _default_store()
        _save(store)
        return store
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save(store: dict) -> None:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# yfinance 現在価格取得（USD）
# ---------------------------------------------------------------------------

def _get_current_price(ticker: str) -> float:
    """yfinanceで現在価格を取得（USD）。失敗した場合は 0.0 を返す。"""
    try:
        info = yf.Ticker(ticker).fast_info
        price = float(getattr(info, "last_price", 0) or 0)
        if price > 0:
            return price
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def get_portfolio() -> dict:
    """現在のポートフォリオ状態を返す（含み損益JPY換算込み）"""
    with _lock:
        store = _load()

    rate = _get_usd_jpy_rate()
    positions = store["positions"]
    enriched = []
    total_unrealized_jpy = 0

    for pos in positions:
        current_usd = _get_current_price(pos["ticker"])
        entry_usd = pos["entry_price"]

        if current_usd > 0:
            unrealized_jpy = round((current_usd - entry_usd) * pos["shares"] * rate)
            unrealized_pct = round((current_usd - entry_usd) / entry_usd * 100, 2) if entry_usd else None
            total_unrealized_jpy += unrealized_jpy
        else:
            unrealized_jpy = None
            unrealized_pct = None

        enriched.append({
            **pos,
            "current_price": round(current_usd, 2) if current_usd else None,
            "current_price_jpy": round(current_usd * rate) if current_usd else None,
            "unrealized_pnl": unrealized_jpy,
            "unrealized_pnl_pct": unrealized_pct,
        })

    realized_pnl = sum(t.get("realized_pnl", 0) for t in store["trades"])

    return {
        "capital_initial": store["capital_initial"],
        "capital_cash": store["capital_cash"],
        "positions": enriched,
        "trades": store["trades"],
        "daily_pnl": store["daily_pnl"],
        "total_unrealized_pnl": total_unrealized_jpy,
        "total_realized_pnl": round(realized_pnl),
        "total_pnl": round(total_unrealized_jpy + realized_pnl),
        "usd_jpy_rate": rate,
    }


def open_trade(
    ticker: str,
    shares: int,
    entry_price: float,
    reason: str,
    stop_loss: float,
    target: float,
) -> dict:
    """新規ポジションをオープン（entry_price は USD、コストは JPY換算）"""
    ticker = ticker.upper().strip()
    rate = _get_usd_jpy_rate()
    cost_jpy = round(shares * entry_price * rate)

    with _lock:
        store = _load()

        if store["capital_cash"] < cost_jpy:
            raise ValueError(
                f"現金不足: 必要 ¥{cost_jpy:,.0f} / 残高 ¥{store['capital_cash']:,.0f}"
            )

        position = {
            "id": str(uuid.uuid4()),
            "ticker": ticker,
            "shares": shares,
            "entry_price": entry_price,               # USD
            "entry_price_jpy": round(entry_price * rate),  # JPY/株
            "usd_jpy_rate": rate,                     # エントリー時レート
            "cost_jpy": cost_jpy,                     # 合計コスト JPY
            "entry_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "stop_loss": stop_loss,                   # USD
            "target": target,                         # USD
            "reason": reason,
            "memo": "",
        }

        store["positions"].append(position)
        store["capital_cash"] -= cost_jpy
        _save(store)

    return position


def close_trade(position_id: str, exit_price: float, memo: str = "") -> dict:
    """ポジションをクローズ（exit_price は USD、損益は JPY換算）"""
    rate = _get_usd_jpy_rate()

    with _lock:
        store = _load()

        pos = next((p for p in store["positions"] if p["id"] == position_id), None)
        if pos is None:
            raise ValueError(f"ポジションが見つかりません: {position_id}")

        entry_usd = pos["entry_price"]
        entry_rate = pos.get("usd_jpy_rate", rate)
        cost_jpy = pos.get("cost_jpy", round(pos["shares"] * entry_usd * entry_rate))

        # 損益をJPYで計算（決済時レートを使用）
        realized_pnl_jpy = round((exit_price - entry_usd) * pos["shares"] * rate)
        realized_pnl_pct = round(realized_pnl_jpy / cost_jpy * 100, 2) if cost_jpy else 0
        is_win = realized_pnl_jpy > 0

        # 決済額をJPYで資金に戻す
        exit_amount_jpy = round(pos["shares"] * exit_price * rate)

        trade_record = {
            "id": str(uuid.uuid4()),
            "position_id": position_id,
            "ticker": pos["ticker"],
            "shares": pos["shares"],
            "entry_price": entry_usd,                       # USD
            "entry_price_jpy": pos.get("entry_price_jpy"),  # JPY/株
            "usd_jpy_rate_entry": entry_rate,
            "entry_date": pos["entry_date"],
            "exit_price": exit_price,                       # USD
            "exit_price_jpy": round(exit_price * rate),     # JPY/株
            "usd_jpy_rate_exit": rate,
            "exit_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "realized_pnl": realized_pnl_jpy,               # JPY
            "realized_pnl_pct": realized_pnl_pct,
            "win": is_win,
            "reason": pos.get("reason", ""),
            "memo": memo,
        }

        store["trades"].append(trade_record)
        store["positions"] = [p for p in store["positions"] if p["id"] != position_id]
        store["capital_cash"] += exit_amount_jpy
        _save(store)

    return trade_record


def get_performance() -> dict:
    """パフォーマンスサマリー（すべてJPY）"""
    portfolio = get_portfolio()
    trades = portfolio["trades"]

    realized_pnl = portfolio["total_realized_pnl"]
    unrealized_pnl = portfolio["total_unrealized_pnl"]
    total_pnl = portfolio["total_pnl"]

    completed = trades
    wins = [t for t in completed if t.get("win")]
    win_rate = (len(wins) / len(completed) * 100) if completed else 0.0

    total_profit = sum(t["realized_pnl"] for t in completed if t["realized_pnl"] > 0)
    total_loss = abs(sum(t["realized_pnl"] for t in completed if t["realized_pnl"] < 0))
    profit_factor = (total_profit / total_loss) if total_loss > 0 else (float("inf") if total_profit > 0 else 0.0)

    daily_pnl = portfolio["daily_pnl"]
    max_drawdown = 0
    if daily_pnl:
        peak = daily_pnl[0]["pnl"]
        for dp in daily_pnl:
            if dp["pnl"] > peak:
                peak = dp["pnl"]
            dd = peak - dp["pnl"]
            if dd > max_drawdown:
                max_drawdown = dd

    achievement_rate = (total_pnl / TARGET_PNL * 100) if TARGET_PNL else 0.0

    return {
        "total_pnl": round(total_pnl),
        "unrealized_pnl": round(unrealized_pnl),
        "realized_pnl": round(realized_pnl),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
        "max_drawdown": round(max_drawdown),
        "trade_count": len(completed),
        "win_count": len(wins),
        "target_pnl": TARGET_PNL,
        "achievement_rate": round(achievement_rate, 2),
    }


def check_stop_losses() -> list:
    """損切りラインを割ったポジションを検出（価格はUSD、損失はJPY）"""
    rate = _get_usd_jpy_rate()
    alerts = []
    with _lock:
        store = _load()
    for pos in store["positions"]:
        sl = pos.get("stop_loss", 0)
        if not sl or sl <= 0:
            continue
        current_usd = _get_current_price(pos["ticker"])
        if current_usd <= 0:
            continue
        if current_usd <= sl:
            loss_jpy = round((current_usd - pos["entry_price"]) * pos["shares"] * rate)
            alerts.append({
                "position_id": pos["id"],
                "ticker": pos["ticker"],
                "shares": pos["shares"],
                "entry_price": pos["entry_price"],   # USD
                "stop_loss": sl,                     # USD
                "current_price": round(current_usd, 2),  # USD
                "unrealized_loss": loss_jpy,         # JPY
                "usd_jpy_rate": rate,
                "action": "損切り推奨",
            })
    return alerts


def calc_position_size(entry_price: float, stop_loss: float, risk_pct: float = 2.0) -> dict:
    """2%ルールに基づくポジションサイズ計算（USD価格→JPY換算）"""
    rate = _get_usd_jpy_rate()
    store = _load()
    capital_jpy = store["capital_cash"]

    max_risk_jpy = capital_jpy * (risk_pct / 100)
    risk_per_share_usd = entry_price - stop_loss
    risk_per_share_jpy = risk_per_share_usd * rate

    if risk_per_share_usd <= 0:
        return {"error": "損切り価格はエントリー価格より低くしてください", "shares": 0}

    shares = int(max_risk_jpy / risk_per_share_jpy)
    cost_jpy = round(shares * entry_price * rate)

    if cost_jpy > capital_jpy:
        shares = int(capital_jpy / (entry_price * rate))
        cost_jpy = round(shares * entry_price * rate)

    return {
        "shares": shares,
        "cost_jpy": cost_jpy,
        "cost": cost_jpy,
        "max_risk_jpy": round(shares * risk_per_share_jpy),
        "max_risk": round(shares * risk_per_share_jpy),
        "max_risk_pct": round(shares * risk_per_share_jpy / capital_jpy * 100, 2) if capital_jpy else 0,
        "capital_available": capital_jpy,
        "risk_per_share_usd": round(risk_per_share_usd, 2),
        "risk_per_share_jpy": round(risk_per_share_jpy),
        "usd_jpy_rate": rate,
    }


def update_position(position_id: str, stop_loss: float = None, target: float = None, memo: str = None) -> dict:
    """オープンポジションのSL/TP/メモを更新"""
    with _lock:
        store = _load()
        pos = next((p for p in store["positions"] if p["id"] == position_id), None)
        if pos is None:
            raise ValueError(f"ポジションが見つかりません: {position_id}")
        if stop_loss is not None:
            pos["stop_loss"] = stop_loss
        if target is not None:
            pos["target"] = target
        if memo is not None:
            pos["memo"] = memo
        _save(store)
    return pos


def reset_paper_trading() -> dict:
    """デモ口座を初期状態にリセット"""
    with _lock:
        store = _default_store()
        _save(store)
    return {"status": "reset", "capital": 100000}


def update_daily_pnl() -> dict:
    """現在の総損益を記録（1日1回呼ぶ）"""
    portfolio = get_portfolio()
    today = datetime.now().strftime("%Y-%m-%d")
    total_pnl = portfolio["total_pnl"]

    entry = {"date": today, "pnl": round(total_pnl)}

    with _lock:
        store = _load()
        store["daily_pnl"] = [d for d in store["daily_pnl"] if d["date"] != today]
        store["daily_pnl"].append(entry)
        store["daily_pnl"].sort(key=lambda x: x["date"])
        _save(store)

    return entry
