"""価格アラート管理"""
import json
import os
import uuid
import threading
from datetime import datetime
import yfinance as yf

ALERTS_FILE = os.path.join(os.path.dirname(__file__), "data", "alerts.json")
_lock = threading.Lock()


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

    return triggered
