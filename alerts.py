"""価格アラート管理"""
import json
import os
import uuid
import threading
from datetime import datetime
import yfinance as yf

ALERTS_FILE = os.path.join(os.path.dirname(__file__), "data", "alerts.json")
_lock = threading.Lock()

_alert_log_file = os.path.join(os.path.dirname(__file__), "data", "alert_history.json")


def log_triggered_alert(alert: dict, triggered_value: float = None) -> None:
    """発動したアラートを履歴に記録"""
    with _lock:
        if os.path.exists(_alert_log_file):
            with open(_alert_log_file, encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []

        history.append({
            "id": alert["id"],
            "ticker": alert["ticker"],
            "condition": alert.get("condition", ""),
            "description": alert.get("description", ""),
            "type": alert.get("type", "price"),
            "triggered_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "triggered_value": triggered_value,
        })

        # 直近100件のみ保持
        history = history[-100:]

        os.makedirs(os.path.dirname(_alert_log_file), exist_ok=True)
        with open(_alert_log_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


def get_alert_history() -> list:
    """発動アラートの履歴を返す（新しい順）"""
    if not os.path.exists(_alert_log_file):
        return []
    with open(_alert_log_file, encoding="utf-8") as f:
        history = json.load(f)
    return list(reversed(history))


def _load():
    if not os.path.exists(ALERTS_FILE):
        return {"alerts": []}
    with open(ALERTS_FILE) as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                # 旧形式（リスト）から新形式へ自動変換
                return {"alerts": data}
            return data
        except Exception:
            return {"alerts": []}


def _save(data):
    os.makedirs(os.path.dirname(ALERTS_FILE), exist_ok=True)
    with open(ALERTS_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_alert(ticker: str, condition: str, price: float) -> dict:
    """アラート作成 condition: 'above' or 'below'"""
    alert = {
        "id": str(uuid.uuid4()),
        "ticker": ticker.upper().strip(),
        "condition": condition,
        "price": price,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "triggered": False,
        "triggered_at": None,
        "triggered_price": None,
    }
    with _lock:
        data = _load()
        data["alerts"].append(alert)
        _save(data)
    return alert


def delete_alert(alert_id: str) -> bool:
    """アラート削除"""
    with _lock:
        data = _load()
        before = len(data["alerts"])
        data["alerts"] = [a for a in data["alerts"] if str(a["id"]) != str(alert_id)]
        if len(data["alerts"]) < before:
            _save(data)
            return True
    return False


def get_alerts() -> list:
    """全アラート取得"""
    with _lock:
        data = _load()
    return data.get("alerts", [])


def check_alerts() -> list:
    """
    全アラートの価格チェック（スケジューラーから呼ぶ）
    トリガーされたアラートのリストを返す
    """
    triggered = []
    alerts = get_alerts()
    updates = {}

    for alert in alerts:
        if alert.get("triggered"):
            continue
        try:
            price = float(yf.Ticker(alert["ticker"]).fast_info.last_price or 0)
            if price <= 0:
                continue
            hit = (alert["condition"] == "above" and price >= alert["price"]) or \
                  (alert["condition"] == "below" and price <= alert["price"])
            if hit:
                updates[alert["id"]] = {
                    "triggered": True,
                    "triggered_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "triggered_price": round(price, 2),
                }
        except Exception:
            continue

    if updates:
        with _lock:
            data = _load()
            for alert in data["alerts"]:
                if alert["id"] in updates:
                    alert.update(updates[alert["id"]])
                    triggered.append(dict(alert))
            _save(data)
        for alert in triggered:
            log_triggered_alert(alert, alert.get("triggered_price"))

    return triggered


def create_technical_alert(ticker: str, condition: str, description: str = "") -> dict:
    """
    テクニカルアラート作成
    condition:
    - "golden_cross"   : MA20がMA50を上抜け（ゴールデンクロス）
    - "dead_cross"     : MA20がMA50を下抜け（デッドクロス）
    - "rsi_oversold"   : RSIが30以下（売られすぎ）
    - "rsi_overbought" : RSIが70以上（買われすぎ）
    """
    with _lock:
        data = _load()
        alert = {
            "id": str(uuid.uuid4()),
            "type": "technical",
            "ticker": ticker.upper(),
            "condition": condition,
            "description": description or _default_tech_description(condition),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "triggered": False,
            "triggered_at": None,
        }
        data["alerts"].append(alert)
        _save(data)
    return alert


def _default_tech_description(condition: str) -> str:
    return {
        "golden_cross": "ゴールデンクロス (MA20 > MA50)",
        "dead_cross": "デッドクロス (MA20 < MA50)",
        "rsi_oversold": "RSI 30以下（売られすぎ）",
        "rsi_overbought": "RSI 70以上（買われすぎ）",
    }.get(condition, condition)


def check_technical_alerts() -> list:
    """テクニカルアラートをチェック（スケジューラーから呼ぶ）"""
    triggered = []

    with _lock:
        data = _load()

    tech_alerts = [a for a in data["alerts"]
                   if a.get("type") == "technical" and not a.get("triggered")]
    if not tech_alerts:
        return []

    for alert in tech_alerts:
        try:
            ticker = alert["ticker"]
            hist = yf.Ticker(ticker).history(period="60d", interval="1d")
            if len(hist) < 55:
                continue

            close = hist["Close"]
            ma20 = close.rolling(20).mean()
            ma50 = close.rolling(50).mean()

            # RSI計算（簡易版）
            deltas = close.diff()
            gains = deltas.where(deltas > 0, 0).rolling(14).mean()
            losses = (-deltas.where(deltas < 0, 0)).rolling(14).mean()
            rsi = 100 - 100 / (1 + gains / losses)
            current_rsi = float(rsi.iloc[-1])

            condition = alert["condition"]
            hit = False

            if condition == "golden_cross":
                hit = (float(ma20.iloc[-2]) < float(ma50.iloc[-2]) and
                       float(ma20.iloc[-1]) >= float(ma50.iloc[-1]))
            elif condition == "dead_cross":
                hit = (float(ma20.iloc[-2]) > float(ma50.iloc[-2]) and
                       float(ma20.iloc[-1]) <= float(ma50.iloc[-1]))
            elif condition == "rsi_oversold":
                hit = current_rsi <= 30
            elif condition == "rsi_overbought":
                hit = current_rsi >= 70

            if hit:
                triggered_val = round(current_rsi, 1) if "rsi" in condition else None
                with _lock:
                    d2 = _load()
                    for a in d2["alerts"]:
                        if a["id"] == alert["id"]:
                            a["triggered"] = True
                            a["triggered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                            a["triggered_value"] = triggered_val
                    _save(d2)
                triggered.append(alert)
                log_triggered_alert(alert, triggered_val)
        except Exception:
            continue

    return triggered
