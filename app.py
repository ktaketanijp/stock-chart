from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json, os, threading, time
from sentiment import get_twitter_sentiment, scan_breaking_catalysts, get_trending_stocks
from fundamentals import (
    get_fundamentals,
    get_canslim_score,
    get_sepa_score,
    get_earnings_calendar,
    get_news_with_summary,
    get_sector_rotation,
)
from indicators import (
    calc_bollinger_bands, calc_ichimoku, calc_atr,
    calc_stochastic, calc_vwap, detect_candlestick_patterns,
    find_support_resistance, calc_adx, calc_williams_r,
    detect_rsi_divergence, detect_ma_cross,
    calc_historical_volatility, calc_momentum_score,
)
import alerts as _alerts_mod

app = Flask(__name__)

# ---------------------------------------------------------------------------
# シンプルインメモリキャッシュ（重い scan_opportunities 向け）
# ---------------------------------------------------------------------------
_app_cache: dict = {}
_app_cache_lock = threading.Lock()

def _app_cache_get(key: str, ttl: int):
    with _app_cache_lock:
        entry = _app_cache.get(key)
        if entry and time.time() - entry["ts"] < ttl:
            return entry["data"]
    return None

def _app_cache_set(key: str, data):
    with _app_cache_lock:
        _app_cache[key] = {"ts": time.time(), "data": data}

PERIOD_MAP = {
    "1d": ("1d", "5m"),
    "5d": ("5d", "15m"),
    "1mo": ("1mo", "1h"),
    "3mo": ("3mo", "1d"),
    "6mo": ("6mo", "1d"),
    "1y": ("1y", "1d"),
    "2y": ("2y", "1wk"),
    "5y": ("5y", "1wk"),
}

def calc_ma(series, window):
    result = series.rolling(window).mean()
    out = []
    for ts, val in result.items():
        if pd.notna(val):
            out.append({"x": int(ts.timestamp() * 1000), "y": round(float(val), 2)})
    return out

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    out = []
    for ts, val in rsi.items():
        if pd.notna(val):
            out.append({"x": int(ts.timestamp() * 1000), "y": round(float(val), 2)})
    return out

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    macd_out, signal_out, hist_out = [], [], []
    for ts in macd_line.index:
        t = int(ts.timestamp() * 1000)
        m = macd_line[ts]
        s = signal_line[ts]
        h = histogram[ts]
        if pd.notna(m):
            macd_out.append({"x": t, "y": round(float(m), 4)})
        if pd.notna(s):
            signal_out.append({"x": t, "y": round(float(s), 4)})
        if pd.notna(h):
            hist_out.append({"x": t, "y": round(float(h), 4)})
    return macd_out, signal_out, hist_out


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.json")
_watchlist_lock = threading.Lock()

def _load_watchlist():
    if not os.path.exists(WATCHLIST_FILE):
        return {"tickers": []}
    with open(WATCHLIST_FILE) as f:
        return json.load(f)

def _save_watchlist(data):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f)


def run_backtest(ticker, strategy, period, params):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval="1d")
    if hist.empty:
        return None

    closes = hist["Close"]
    dates = [int(ts.timestamp() * 1000) for ts in hist.index]
    prices = closes.tolist()
    n = len(prices)

    capital = float(params.get("capital", 1000000))
    initial = capital
    position = 0
    entry_price = 0
    trades = []

    if strategy == "ma_cross":
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 50))
        ma_fast = closes.rolling(fast).mean()
        ma_slow = closes.rolling(slow).mean()
        for i in range(1, n):
            if pd.isna(ma_fast.iloc[i]) or pd.isna(ma_slow.iloc[i]):
                continue
            prev_diff = ma_fast.iloc[i-1] - ma_slow.iloc[i-1]
            curr_diff = ma_fast.iloc[i] - ma_slow.iloc[i]
            price = prices[i]
            if prev_diff <= 0 and curr_diff > 0 and position == 0:
                shares = int(capital / price)
                if shares > 0:
                    position = shares
                    entry_price = price
                    capital -= shares * price
                    trades.append({"date": dates[i], "type": "buy", "price": round(price, 2), "shares": shares})
            elif prev_diff >= 0 and curr_diff < 0 and position > 0:
                capital += position * price
                pnl = (price - entry_price) * position
                trades.append({"date": dates[i], "type": "sell", "price": round(price, 2), "shares": position, "pnl": round(pnl, 2)})
                position = 0

    elif strategy == "rsi":
        period_rsi = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30))
        overbought = float(params.get("overbought", 70))
        delta = closes.diff()
        gain = delta.clip(lower=0).ewm(com=period_rsi-1, min_periods=period_rsi).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period_rsi-1, min_periods=period_rsi).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        for i in range(1, n):
            if pd.isna(rsi.iloc[i]):
                continue
            price = prices[i]
            if rsi.iloc[i] < oversold and position == 0:
                shares = int(capital / price)
                if shares > 0:
                    position = shares
                    entry_price = price
                    capital -= shares * price
                    trades.append({"date": dates[i], "type": "buy", "price": round(price, 2), "shares": shares})
            elif rsi.iloc[i] > overbought and position > 0:
                capital += position * price
                pnl = (price - entry_price) * position
                trades.append({"date": dates[i], "type": "sell", "price": round(price, 2), "shares": position, "pnl": round(pnl, 2)})
                position = 0

    if position > 0:
        final_price = prices[-1]
        capital += position * final_price
        pnl = (final_price - entry_price) * position
        trades.append({"date": dates[-1], "type": "sell(final)", "price": round(final_price, 2), "shares": position, "pnl": round(pnl, 2)})
        position = 0

    total_return = (capital - initial) / initial * 100
    win_trades = [t for t in trades if t.get("type") in ("sell", "sell(final)") and t.get("pnl", 0) > 0]
    sell_trades = [t for t in trades if t.get("type") in ("sell", "sell(final)")]
    win_rate = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0

    equity = []
    eq_capital = initial
    eq_pos = 0
    eq_entry = 0
    trade_idx = 0
    for i, (d, p) in enumerate(zip(dates, prices)):
        while trade_idx < len(trades) and trades[trade_idx]["date"] == d:
            t = trades[trade_idx]
            if t["type"] == "buy":
                eq_capital -= t["shares"] * t["price"]
                eq_pos = t["shares"]
                eq_entry = t["price"]
            else:
                eq_capital += t["shares"] * t["price"]
                eq_pos = 0
            trade_idx += 1
        equity.append({"x": d, "y": round(eq_capital + eq_pos * p, 2)})

    return {
        "initial": initial,
        "final": round(capital, 2),
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "trade_count": len(sell_trades),
        "trades": trades,
        "equity": equity,
        "candles": [{"x": d, "o": round(float(hist["Open"].iloc[i]),2),
                     "h": round(float(hist["High"].iloc[i]),2),
                     "l": round(float(hist["Low"].iloc[i]),2),
                     "c": round(float(p),2)} for i,(d,p) in enumerate(zip(dates,prices))],
    }


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/guide")
def guide():
    return render_template("guide.html")

@app.route("/api/chart")
def chart():
    ticker = request.args.get("ticker", "").upper().strip()
    period = request.args.get("period", "3mo")

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    if period not in PERIOD_MAP:
        return jsonify({"error": "Invalid period"}), 400

    yf_period, interval = PERIOD_MAP[period]

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=yf_period, interval=interval)

        if hist.empty:
            return jsonify({"error": f"No data found for '{ticker}'"}), 404

        info = stock.fast_info
        try:
            name = stock.info.get("longName") or stock.info.get("shortName") or ticker
        except Exception:
            name = ticker
        currency = getattr(info, "currency", "USD") or "USD"
        current_price = float(getattr(info, "last_price", hist["Close"].iloc[-1]))
        prev_close = float(getattr(info, "previous_close", hist["Close"].iloc[-2] if len(hist) > 1 else current_price))
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        candles = []
        volumes = []
        for ts, row in hist.iterrows():
            t = int(ts.timestamp() * 1000)
            candles.append({
                "x": t,
                "o": round(float(row["Open"]), 2),
                "h": round(float(row["High"]), 2),
                "l": round(float(row["Low"]), 2),
                "c": round(float(row["Close"]), 2),
            })
            volumes.append({"x": t, "y": int(row["Volume"])})

        closes = hist["Close"]
        macd_data, signal_data, hist_data = calc_macd(closes)

        # Advanced indicators
        bb = calc_bollinger_bands(closes)
        ichimoku = calc_ichimoku(hist["High"], hist["Low"], hist["Close"])
        atr = calc_atr(hist["High"], hist["Low"], hist["Close"])
        stoch = calc_stochastic(hist["High"], hist["Low"], hist["Close"])
        patterns = detect_candlestick_patterns(hist)
        sr_levels = find_support_resistance(hist)

        # VWAPは短い足のみ（1d / 5d）
        vwap_data = []
        if period in ("1d", "5d"):
            vwap_data = calc_vwap(hist)

        return jsonify({
            "ticker": ticker,
            "name": name,
            "currency": currency,
            "current_price": round(current_price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "candles": candles,
            "volumes": volumes,
            "ma20": calc_ma(closes, 20),
            "ma50": calc_ma(closes, 50),
            "ma200": calc_ma(closes, 200),
            "rsi": calc_rsi(closes),
            "macd": macd_data,
            "macd_signal": signal_data,
            "macd_hist": hist_data,
            "bb": bb,
            "ichimoku": ichimoku,
            "atr": atr[-1]["y"] if atr else None,
            "atr_series": atr,
            "stochastic": stoch,
            "vwap": vwap_data,
            "patterns": patterns,
            "support_resistance": sr_levels,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()
    if not query or len(query) < 1:
        return jsonify([])

    popular = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
        {"symbol": "NVDA", "name": "NVIDIA Corporation"},
        {"symbol": "TSLA", "name": "Tesla Inc."},
        {"symbol": "META", "name": "Meta Platforms Inc."},
        {"symbol": "7203.T", "name": "トヨタ自動車"},
        {"symbol": "6758.T", "name": "ソニーグループ"},
        {"symbol": "9984.T", "name": "ソフトバンクグループ"},
        {"symbol": "6861.T", "name": "キーエンス"},
        {"symbol": "8306.T", "name": "三菱UFJフィナンシャル"},
        {"symbol": "BTC-USD", "name": "Bitcoin USD"},
        {"symbol": "ETH-USD", "name": "Ethereum USD"},
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF"},
        {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
        {"symbol": "^N225", "name": "日経平均株価"},
        {"symbol": "^GSPC", "name": "S&P 500"},
        {"symbol": "^DJI", "name": "Dow Jones"},
    ]

    q = query.upper()
    results = [s for s in popular if q in s["symbol"].upper() or q in s["name"].upper()]
    return jsonify(results[:8])


@app.route("/api/backtest", methods=["POST"])
def backtest():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    strategy = data.get("strategy", "ma_cross")
    period = data.get("period", "1y")
    params = data.get("params", {})
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    try:
        result = run_backtest(ticker, strategy, period, params)
        if result is None:
            return jsonify({"error": f"No data for '{ticker}'"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts", methods=["GET"])
def get_alerts_api():
    return jsonify({"alerts": _alerts_mod.get_alerts()})

@app.route("/api/alerts", methods=["POST"])
def create_alert_api():
    data = request.get_json() or {}
    ticker = data.get("ticker", "").upper().strip()
    condition = data.get("condition")
    try:
        price = float(data.get("price", 0))
    except (TypeError, ValueError):
        price = 0
    if not ticker or condition not in ("above", "below") or price <= 0:
        return jsonify({"error": "Invalid alert parameters"}), 400
    alert = _alerts_mod.create_alert(ticker, condition, price)
    return jsonify(alert), 201

@app.route("/api/alerts/check", methods=["POST"])
def check_alerts_api():
    triggered = _alerts_mod.check_alerts()
    return jsonify({"triggered": triggered, "alerts": _alerts_mod.get_alerts()})

@app.route("/api/alerts/technical", methods=["POST"])
def create_technical_alert_api():
    """{"ticker": "AAPL", "condition": "golden_cross"}"""
    data = request.get_json() or {}
    result = _alerts_mod.create_technical_alert(
        ticker=data.get("ticker", ""),
        condition=data.get("condition", ""),
    )
    return jsonify(result), 201

@app.route("/api/alerts/<alert_id>", methods=["DELETE"])
def delete_alert_api(alert_id):
    ok = _alerts_mod.delete_alert(alert_id)
    if ok:
        return jsonify({"ok": True})
    return jsonify({"error": "Alert not found"}), 404


@app.route("/api/sentiment")
def sentiment():
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    try:
        data = get_twitter_sentiment(ticker)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500

@app.route("/api/news/breaking")
def breaking_news():
    try:
        data = scan_breaking_catalysts()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/trending")
def trending():
    try:
        data = get_trending_stocks()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Fundamentals / CAN SLIM / SEPA routes
# ---------------------------------------------------------------------------

@app.route("/api/fundamentals")
def fundamentals_api():
    ticker = request.args.get("ticker", "AAPL").upper()
    try:
        return jsonify(get_fundamentals(ticker))
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500

@app.route("/api/canslim")
def canslim_api():
    ticker = request.args.get("ticker", "AAPL").upper()
    try:
        return jsonify(get_canslim_score(ticker))
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500

@app.route("/api/sepa")
def sepa_api():
    ticker = request.args.get("ticker", "AAPL").upper()
    try:
        return jsonify(get_sepa_score(ticker))
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500

@app.route("/api/earnings/calendar")
def earnings_calendar():
    try:
        days = int(request.args.get("days", 30))
        return jsonify(get_earnings_calendar(days))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/news")
def news_api():
    ticker = request.args.get("ticker", "AAPL").upper()
    try:
        return jsonify(get_news_with_summary(ticker))
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500

@app.route("/api/sector/rotation")
def sector_rotation():
    try:
        return jsonify(get_sector_rotation())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status_page():
    return render_template("status.html")

@app.route("/api/status/health")
def health_check():
    """システムヘルスチェック"""
    import subprocess
    checks = {}

    # yfinance疎通確認
    try:
        info = yf.Ticker("AAPL").fast_info
        price = float(getattr(info, "last_price", None) or 0)
        checks["yfinance"] = {"ok": price > 0, "value": f"AAPL ${price:.2f}"}
    except Exception as e:
        checks["yfinance"] = {"ok": False, "error": str(e)}

    # systemd サービス状態
    try:
        out = subprocess.check_output(
            ["systemctl", "is-active", "stock-chart.service"],
            text=True, timeout=5
        ).strip()
        checks["service"] = {"ok": out == "active", "value": out}
    except Exception as e:
        checks["service"] = {"ok": False, "error": str(e)}

    # データファイル確認
    for fname in ["paper_trades.json", "watchlist.json"]:
        fpath = os.path.join(DATA_DIR, fname)
        checks[fname] = {"ok": os.path.exists(fpath)}

    # 最終データ更新時刻（status.json）
    status_path = os.path.join(DATA_DIR, "status.json")
    if os.path.exists(status_path):
        mtime = os.path.getmtime(status_path)
        checks["last_update"] = {
            "ok": True,
            "value": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        checks["last_update"] = {"ok": False, "value": "status.json なし"}

    all_ok = all(v.get("ok", False) for v in checks.values())
    return jsonify({
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

@app.route("/api/status")
def status_api():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("update_status",
            os.path.join(os.path.dirname(__file__), "update_status.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return jsonify(mod.build_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/api/analysis/multi-timeframe/<ticker>")
def multi_timeframe_analysis(ticker):
    """1日足・1週足・1ヶ月足のテクニカル分析を返す"""
    ticker = ticker.upper().strip()
    results = {}
    timeframes = {
        "1d":  ("6mo", "1d"),    # 6ヶ月・日足
        "1w":  ("2y",  "1wk"),   # 2年・週足
        "1mo": ("5y",  "1mo"),   # 5年・月足
    }

    for tf_name, (period, interval) in timeframes.items():
        try:
            hist = yf.Ticker(ticker).history(period=period, interval=interval)
            if hist.empty or len(hist) < 20:
                results[tf_name] = {"error": "データ不足"}
                continue

            close = hist["Close"]
            high  = hist["High"]
            low   = hist["Low"]

            # RSI — 最終値だけ取り出す
            rsi_list = calc_rsi(close)
            rsi_val  = round(rsi_list[-1]["y"], 1) if rsi_list else None

            # MACD — 方向性を文字列で返す
            macd_list, sig_list, _ = calc_macd(close)
            if macd_list and sig_list:
                ml = macd_list[-1]["y"]
                sl = sig_list[-1]["y"]
                macd_signal = "bullish" if ml > sl else ("bearish" if ml < sl else "neutral")
            else:
                macd_signal = "neutral"

            # ADX — 新しく実装したものを使う
            adx_data = calc_adx(high, low, close)
            adx_val  = round(adx_data["adx"][-1]["y"], 1) if adx_data["adx"] else None

            # トレンド方向（MA20 vs MA50、差が1%未満は横ばい）
            ma20_list = calc_ma(close, 20)
            ma50_list = calc_ma(close, 50)
            if ma20_list and ma50_list:
                ma20_last = ma20_list[-1]["y"]
                ma50_last = ma50_list[-1]["y"]
                diff_pct  = (ma20_last - ma50_last) / ma50_last * 100 if ma50_last else 0
                trend = "uptrend" if diff_pct > 1 else ("downtrend" if diff_pct < -1 else "sideways")
            else:
                trend = "sideways"

            results[tf_name] = {
                "rsi":         rsi_val,
                "macd_signal": macd_signal,
                "adx":         adx_val,
                "trend":       trend,
                "close":       round(float(close.iloc[-1]), 2),
            }
        except Exception as e:
            results[tf_name] = {"error": str(e)}

    return jsonify({"ticker": ticker, "timeframes": results})


@app.route("/api/analysis/divergence/<ticker>")
def analysis_divergence(ticker):
    """RSIダイバージェンス・MAクロス・サポレジを返す"""
    ticker = ticker.upper().strip()
    try:
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d")
        if hist.empty:
            return jsonify({"error": "データ取得失敗"}), 404

        closes   = hist["Close"]
        rsi_data = calc_rsi(closes)

        divergence = detect_rsi_divergence(closes, rsi_data, lookback=20)
        ma_cross   = detect_ma_cross(closes, fast=20, slow=50)
        sr         = find_support_resistance(hist)

        current_price = float(closes.iloc[-1])

        # サポレジに現在価格との距離（%）を付加
        def enrich_levels(levels):
            result = []
            for lvl in levels:
                dist_pct = round((lvl - current_price) / current_price * 100, 2)
                result.append({"price": lvl, "dist_pct": dist_pct})
            return result

        return jsonify({
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "divergence": divergence,
            "ma_cross": ma_cross,
            "support_resistance": {
                "support":    enrich_levels(sr.get("support",    [])),
                "resistance": enrich_levels(sr.get("resistance", [])),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signal")
def signal():
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo", interval="1d")
        if hist.empty:
            return jsonify({"error": "No data"}), 404

        closes = hist["Close"]
        rsi_data = calc_rsi(closes)
        current_rsi = rsi_data[-1]["y"] if rsi_data else 50
        macd_line, sig_line, _ = calc_macd(closes)
        macd_val = macd_line[-1]["y"] if macd_line else 0
        sig_val = sig_line[-1]["y"] if sig_line else 0
        bb = calc_bollinger_bands(closes)
        current_price = float(closes.iloc[-1])
        atr_list = calc_atr(hist["High"], hist["Low"], hist["Close"])
        atr_val = atr_list[-1]["y"] if atr_list else 0

        score = 0
        reasons = []

        # RSI シグナル
        if current_rsi < 30:
            score += 30
            reasons.append(f"RSIが{current_rsi:.1f}と売られすぎゾーン（買いシグナル）")
        elif current_rsi > 70:
            score -= 30
            reasons.append(f"RSIが{current_rsi:.1f}と買われすぎゾーン（売りシグナル）")
        elif current_rsi < 50:
            score -= 10
            reasons.append(f"RSIが{current_rsi:.1f}と中立ゾーン下部")
        else:
            score += 10
            reasons.append(f"RSIが{current_rsi:.1f}と中立ゾーン上部")

        # MACD シグナル
        if macd_val > sig_val:
            score += 20
            reasons.append("MACDがシグナル線を上回る（強気）")
        else:
            score -= 20
            reasons.append("MACDがシグナル線を下回る（弱気）")

        # ボリンジャーバンド シグナル
        if bb.get("lower") and bb.get("upper"):
            bb_lower = bb["lower"][-1]["y"]
            bb_upper = bb["upper"][-1]["y"]
            bb_mid   = bb["middle"][-1]["y"]
            if current_price < bb_lower:
                score += 25
                reasons.append("株価がボリンジャーバンド下限を下回る（反発期待）")
            elif current_price > bb_upper:
                score -= 25
                reasons.append("株価がボリンジャーバンド上限を超過（過熱警戒）")
            elif current_price > bb_mid:
                score += 10
                reasons.append("株価がボリンジャーバンド中央線の上（強気）")
            else:
                score -= 10
                reasons.append("株価がボリンジャーバンド中央線の下（弱気）")

        # MA トレンド
        ma20 = calc_ma(closes, 20)
        ma50 = calc_ma(closes, 50)
        if ma20 and ma50:
            if ma20[-1]["y"] > ma50[-1]["y"]:
                score += 15
                reasons.append("MA20がMA50を上回るゴールデンクロス状態")
            else:
                score -= 15
                reasons.append("MA20がMA50を下回るデッドクロス状態")

        score = max(-100, min(100, score))

        if score >= 30:
            judgment = "買い"
        elif score <= -30:
            judgment = "売り"
        else:
            judgment = "中立"

        stop_loss   = round(current_price - atr_val * 1.5, 2)
        target      = round(current_price + atr_val * 3.0, 2)
        entry_low   = round(current_price * 0.99, 2)
        entry_high  = round(current_price * 1.01, 2)

        return jsonify({
            "ticker": ticker,
            "score": round(score),
            "judgment": judgment,
            "reasons": reasons,
            "current_price": round(current_price, 2),
            "entry_range": {"low": entry_low, "high": entry_high},
            "stop_loss": stop_loss,
            "target": target,
            "atr": round(atr_val, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


from signal_engine import generate_signal, scan_opportunities
from scanner import (
    scan_momentum, scan_premarket, scan_top_movers,
    scan_52w_breakout, scan_volume_surge, scan_rsi_extreme,
)

@app.route("/api/signal/advanced")
def signal_advanced_api():
    ticker = request.args.get("ticker", "AAPL").upper()
    try:
        cache_key = f"signal_advanced_{ticker}"
        cached = _app_cache_get(cache_key, ttl=300)  # 5分キャッシュ
        if cached is not None:
            return jsonify(cached)
        result = generate_signal(ticker)
        if not result.get("error"):
            _app_cache_set(cache_key, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500

@app.route("/api/scan/opportunities")
def opportunities_api():
    try:
        cached = _app_cache_get("opportunities", ttl=1800)
        if cached is not None:
            return jsonify(cached)
        result = scan_opportunities()
        _app_cache_set("opportunities", result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan/momentum")
def momentum_api():
    try:
        return jsonify(scan_momentum())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan/premarket")
def premarket_api():
    try:
        return jsonify(scan_premarket())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan/top-movers")
def top_movers_api():
    try:
        return jsonify(scan_top_movers())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan/52w-breakout")
def scan_52w_breakout_api():
    try:
        results = scan_52w_breakout()
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan/volume-surge")
def scan_volume_surge_api():
    try:
        results = scan_volume_surge()
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan/rsi-extreme")
def scan_rsi_extreme_api():
    try:
        results = scan_rsi_extreme()
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


from paper_trading import (
    get_portfolio, open_trade, close_trade, get_performance,
    calc_position_size, check_stop_losses, update_daily_pnl,
    update_position, reset_paper_trading,
)

@app.route("/journal")
def journal_page():
    return render_template("journal.html")

@app.route("/api/journal")
def journal_api():
    try:
        portfolio = get_portfolio()
        perf = get_performance()
        return jsonify({"portfolio": portfolio, "performance": perf})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/journal/open", methods=["POST"])
def journal_open():
    try:
        data = request.get_json()
        result = open_trade(
            data["ticker"], int(data["shares"]), float(data["entry_price"]),
            data.get("reason",""), float(data.get("stop_loss",0)), float(data.get("target",0))
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/journal/daily-pnl", methods=["POST"])
def journal_daily_pnl():
    """本日の損益を手動記録"""
    try:
        entry = update_daily_pnl()
        return jsonify(entry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/journal/check-stops")
def journal_check_stops():
    """損切りライン到達ポジションのアラート"""
    try:
        return jsonify(check_stop_losses())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/journal/size", methods=["POST"])
def journal_size():
    """2%ルールに基づくポジションサイズ計算"""
    try:
        data = request.get_json()
        result = calc_position_size(
            float(data["entry_price"]),
            float(data["stop_loss"]),
            float(data.get("risk_pct", 2.0))
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/journal/close", methods=["POST"])
def journal_close():
    try:
        data = request.get_json()
        result = close_trade(data["position_id"], float(data["exit_price"]), data.get("memo",""))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/journal/performance")
def get_performance_api():
    try:
        return jsonify(get_performance())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/journal/position/<position_id>", methods=["PATCH"])
def update_position_api(position_id):
    try:
        data = request.get_json()
        result = update_position(
            position_id,
            stop_loss=float(data["stop_loss"]) if data.get("stop_loss") is not None else None,
            target=float(data["target"]) if data.get("target") is not None else None,
            memo=data.get("memo"),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/journal/reset", methods=["POST"])
def reset_journal():
    try:
        return jsonify(reset_paper_trading())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/scanner")
def scanner_page():
    return render_template("scanner.html")


@app.route("/ptcg")
def ptcg():
    return render_template("ptcg.html")


# ---------------------------------------------------------------------------
# 市場概況
# ---------------------------------------------------------------------------

@app.route("/market")
def market_page():
    return render_template("market.html")


@app.route("/api/market/overview")
def market_overview():
    """主要指数・VIX"""
    INDICES = {
        "S&P 500":     "^GSPC",
        "NASDAQ":      "^IXIC",
        "Dow Jones":   "^DJI",
        "Russell 2000":"^RUT",
        "VIX":         "^VIX",
        "日経225":     "^N225",
    }

    indices_data = {}
    for name, symbol in INDICES.items():
        try:
            ticker = yf.Ticker(symbol)
            price = float(ticker.fast_info.last_price or 0)
            hist = ticker.history(period="2d", interval="1d")
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                change_pct = round((price - prev) / prev * 100, 2) if prev else 0
            else:
                change_pct = 0
            indices_data[name] = {"price": round(price, 2), "change_pct": change_pct}
        except Exception as e:
            indices_data[name] = {"price": None, "change_pct": 0, "error": str(e)}

    return jsonify({"indices": indices_data})


@app.route("/api/market/sectors")
def market_sectors():
    """セクター別パフォーマンス"""
    SECTORS = {
        "テクノロジー": "XLK",
        "ヘルスケア":   "XLV",
        "金融":         "XLF",
        "エネルギー":   "XLE",
        "素材":         "XLB",
        "工業":         "XLI",
        "公益":         "XLU",
        "生活必需品":   "XLP",
        "一般消費財":   "XLY",
        "不動産":       "XLRE",
        "通信":         "XLC",
    }

    sectors_data = {}
    for name, symbol in SECTORS.items():
        try:
            ticker = yf.Ticker(symbol)
            price = float(ticker.fast_info.last_price or 0)
            hist = ticker.history(period="2d", interval="1d")
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                change_pct = round((price - prev) / prev * 100, 2) if prev else 0
            else:
                change_pct = 0
            sectors_data[name] = {"symbol": symbol, "price": round(price, 2), "change_pct": change_pct}
        except Exception as e:
            sectors_data[name] = {"symbol": symbol, "price": None, "change_pct": 0, "error": str(e)}

    return jsonify({"sectors": sectors_data})


# ---------------------------------------------------------------------------
# ウォッチリスト
# ---------------------------------------------------------------------------

@app.route("/watchlist")
def watchlist_page():
    return render_template("watchlist.html")


@app.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    """ウォッチリスト取得（各銘柄の現在価格・変動率・AIシグナルも含む）"""
    with _watchlist_lock:
        wl = _load_watchlist()
    tickers = wl.get("tickers", [])

    result = []
    for ticker in tickers:
        entry = {"ticker": ticker, "price": None, "change_pct": None,
                 "signal": None, "score": None, "updated": None, "error": None}
        try:
            info = yf.Ticker(ticker).fast_info
            price = float(getattr(info, "last_price", None) or 0)
            prev = float(getattr(info, "previous_close", None) or 0)
            if price > 0:
                entry["price"] = round(price, 2)
            if price > 0 and prev > 0:
                entry["change_pct"] = round((price - prev) / prev * 100, 2)
        except Exception as e:
            entry["error"] = str(e)

        # シグナルはキャッシュ優先（TTL 5分）
        cache_key = f"watchlist_signal_{ticker}"
        sig_cached = _app_cache_get(cache_key, ttl=300)
        if sig_cached is not None:
            entry["signal"] = sig_cached.get("signal")
            entry["score"] = sig_cached.get("score")
        else:
            try:
                from signal_engine import generate_signal as _gen_sig
                sig = _gen_sig(ticker)
                entry["signal"] = sig.get("signal")
                entry["score"] = sig.get("total_score")
                _app_cache_set(cache_key, {"signal": entry["signal"], "score": entry["score"]})
            except Exception:
                pass

        entry["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        result.append(entry)

    return jsonify({"tickers": result})


@app.route("/api/watchlist", methods=["POST"])
def add_to_watchlist():
    """銘柄追加 {"ticker": "AAPL"}"""
    data = request.get_json() or {}
    ticker = data.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    with _watchlist_lock:
        wl = _load_watchlist()
        if ticker not in wl["tickers"]:
            wl["tickers"].append(ticker)
            _save_watchlist(wl)
    return jsonify({"ok": True, "tickers": wl["tickers"]})


@app.route("/api/watchlist/<ticker>", methods=["DELETE"])
def remove_from_watchlist(ticker):
    """銘柄削除"""
    ticker = ticker.upper().strip()
    with _watchlist_lock:
        wl = _load_watchlist()
        wl["tickers"] = [t for t in wl["tickers"] if t != ticker]
        _save_watchlist(wl)
    return jsonify({"ok": True, "tickers": wl["tickers"]})


# ---------------------------------------------------------------------------
# バックテスト: RSIリバーサル戦略
# ---------------------------------------------------------------------------

@app.route("/api/backtest/rsi-reversal/<ticker>")
def backtest_rsi_reversal(ticker):
    """RSIリバーサル戦略: RSI < 30 で買い / RSI > 70 で売り（期間: 1年）"""
    ticker = ticker.upper().strip()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", interval="1d")
        if hist.empty:
            return jsonify({"error": f"No data for '{ticker}'"}), 404

        closes = hist["Close"]
        dates = [int(ts.timestamp() * 1000) for ts in hist.index]
        prices = closes.tolist()
        n = len(prices)

        # RSI計算（期間14）
        delta = closes.diff()
        gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
        rsi = 100 - (100 / (1 + gain / loss))

        capital = 1_000_000.0
        initial = capital
        position = 0
        entry_price = 0.0
        trades = []

        for i in range(1, n):
            if pd.isna(rsi.iloc[i]):
                continue
            price = prices[i]
            rsi_val = float(rsi.iloc[i])

            if rsi_val < 30 and position == 0:
                shares = int(capital / price)
                if shares > 0:
                    position = shares
                    entry_price = price
                    capital -= shares * price
                    trades.append({
                        "date": dates[i], "type": "buy",
                        "price": round(price, 2), "shares": shares,
                        "rsi": round(rsi_val, 1),
                    })
            elif rsi_val > 70 and position > 0:
                capital += position * price
                pnl = (price - entry_price) * position
                trades.append({
                    "date": dates[i], "type": "sell",
                    "price": round(price, 2), "shares": position,
                    "pnl": round(pnl, 2), "rsi": round(rsi_val, 1),
                })
                position = 0

        # 残ポジションを最終日終値で清算
        if position > 0:
            final_price = prices[-1]
            capital += position * final_price
            pnl = (final_price - entry_price) * position
            trades.append({
                "date": dates[-1], "type": "sell(final)",
                "price": round(final_price, 2), "shares": position,
                "pnl": round(pnl, 2),
            })
            position = 0

        total_return_pct = (capital - initial) / initial * 100
        sell_trades = [t for t in trades if t["type"] in ("sell", "sell(final)")]
        win_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
        win_rate = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0

        # 最大ドローダウン計算
        eq_cap = initial
        eq_pos = 0
        trade_idx = 0
        peak = initial
        max_drawdown = 0.0
        for i, (d, p) in enumerate(zip(dates, prices)):
            while trade_idx < len(trades) and trades[trade_idx]["date"] == d:
                t = trades[trade_idx]
                if t["type"] == "buy":
                    eq_cap -= t["shares"] * t["price"]
                    eq_pos = t["shares"]
                else:
                    eq_cap += t["shares"] * t["price"]
                    eq_pos = 0
                trade_idx += 1
            equity_val = eq_cap + eq_pos * p
            if equity_val > peak:
                peak = equity_val
            dd = (equity_val - peak) / peak * 100
            if dd < max_drawdown:
                max_drawdown = dd

        return jsonify({
            "ticker": ticker,
            "strategy": "rsi-reversal",
            "total_return_pct": round(total_return_pct, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": len(sell_trades),
            "max_drawdown_pct": round(max_drawdown, 2),
            "trades": trades,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# バックテスト: MACDクロスオーバー戦略
# ---------------------------------------------------------------------------

@app.route("/api/backtest/macd-crossover/<ticker>")
def backtest_macd_crossover(ticker):
    """MACDクロスオーバー戦略: MACDがシグナル線を上抜け→買い / 下抜け→売り（期間: 1年）"""
    ticker = ticker.upper().strip()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", interval="1d")
        if hist.empty:
            return jsonify({"error": f"No data for '{ticker}'"}), 404

        closes = hist["Close"]
        dates = [int(ts.timestamp() * 1000) for ts in hist.index]
        prices = closes.tolist()
        n = len(prices)

        # MACD計算（12/26/9）
        ema_fast = closes.ewm(span=12, adjust=False).mean()
        ema_slow = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        capital = 1_000_000.0
        initial = capital
        position = 0
        entry_price = 0.0
        trades = []

        for i in range(1, n):
            if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
                continue
            price = prices[i]
            prev_diff = float(macd_line.iloc[i-1]) - float(signal_line.iloc[i-1])
            curr_diff = float(macd_line.iloc[i]) - float(signal_line.iloc[i])

            # 上抜けクロス: 買い
            if prev_diff <= 0 and curr_diff > 0 and position == 0:
                shares = int(capital / price)
                if shares > 0:
                    position = shares
                    entry_price = price
                    capital -= shares * price
                    trades.append({
                        "date": dates[i], "type": "buy",
                        "price": round(price, 2), "shares": shares,
                        "macd": round(float(macd_line.iloc[i]), 4),
                    })
            # 下抜けクロス: 売り
            elif prev_diff >= 0 and curr_diff < 0 and position > 0:
                capital += position * price
                pnl = (price - entry_price) * position
                trades.append({
                    "date": dates[i], "type": "sell",
                    "price": round(price, 2), "shares": position,
                    "pnl": round(pnl, 2),
                    "macd": round(float(macd_line.iloc[i]), 4),
                })
                position = 0

        # 残ポジションを最終日終値で清算
        if position > 0:
            final_price = prices[-1]
            capital += position * final_price
            pnl = (final_price - entry_price) * position
            trades.append({
                "date": dates[-1], "type": "sell(final)",
                "price": round(final_price, 2), "shares": position,
                "pnl": round(pnl, 2),
            })
            position = 0

        total_return_pct = (capital - initial) / initial * 100
        sell_trades = [t for t in trades if t["type"] in ("sell", "sell(final)")]
        win_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
        win_rate = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0

        # 最大ドローダウン計算
        eq_cap = initial
        eq_pos = 0
        trade_idx = 0
        peak = initial
        max_drawdown = 0.0
        for i, (d, p) in enumerate(zip(dates, prices)):
            while trade_idx < len(trades) and trades[trade_idx]["date"] == d:
                t = trades[trade_idx]
                if t["type"] == "buy":
                    eq_cap -= t["shares"] * t["price"]
                    eq_pos = t["shares"]
                else:
                    eq_cap += t["shares"] * t["price"]
                    eq_pos = 0
                trade_idx += 1
            equity_val = eq_cap + eq_pos * p
            if equity_val > peak:
                peak = equity_val
            dd = (equity_val - peak) / peak * 100
            if dd < max_drawdown:
                max_drawdown = dd

        return jsonify({
            "ticker": ticker,
            "strategy": "macd-crossover",
            "total_return_pct": round(total_return_pct, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": len(sell_trades),
            "max_drawdown_pct": round(max_drawdown, 2),
            "trades": trades,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# ウォッチリスト: 相関分析API
# ---------------------------------------------------------------------------

@app.route("/api/watchlist/correlation")
def watchlist_correlation():
    """ウォッチリスト銘柄間の相関係数行列（過去3ヶ月の日次リターン）"""
    with _watchlist_lock:
        wl = _load_watchlist()
    tickers = wl.get("tickers", [])
    if len(tickers) < 2:
        return jsonify({
            "error": "相関分析には2銘柄以上が必要です",
            "tickers": tickers,
            "matrix": [],
        }), 400

    try:
        price_dict = {}
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period="3mo", interval="1d")
                if not hist.empty and len(hist) > 5:
                    price_dict[t] = hist["Close"]
            except Exception:
                pass

        valid_tickers = list(price_dict.keys())
        if len(valid_tickers) < 2:
            return jsonify({"error": "有効な銘柄データが不足しています"}), 400

        prices_df = pd.DataFrame(price_dict).dropna()
        returns_df = prices_df.pct_change().dropna()
        corr = returns_df.corr()

        matrix = []
        for t1 in valid_tickers:
            row = []
            for t2 in valid_tickers:
                try:
                    val = corr.loc[t1, t2]
                    row.append(round(float(val), 4) if pd.notna(val) else None)
                except KeyError:
                    row.append(None)
            matrix.append(row)

        return jsonify({"tickers": valid_tickers, "matrix": matrix})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# 経済カレンダー
# ---------------------------------------------------------------------------

@app.route("/calendar")
def calendar_page():
    return render_template("calendar.html")


@app.route("/api/calendar/economic")
def economic_calendar():
    """主要経済イベントの予定（ハードコードされた2026年のイベント）"""
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    events = []

    # FOMC会合（2026年）
    fomc_dates = [
        ("2026-07-28", "2026-07-29"),
        ("2026-09-15", "2026-09-16"),
        ("2026-11-03", "2026-11-04"),
        ("2026-12-15", "2026-12-16"),
    ]
    for start_date, end_date in fomc_dates:
        if end_date >= today_str:
            events.append({
                "date": start_date,
                "event": "FOMC会合（1日目）",
                "importance": "HIGH",
                "category": "FRB",
                "description": "連邦公開市場委員会 - 金利決定前日",
            })
            events.append({
                "date": end_date,
                "event": "FOMC会合（金利発表）",
                "importance": "HIGH",
                "category": "FRB",
                "description": "連邦公開市場委員会 - 金利決定・議長会見",
            })

    # CPI（消費者物価指数）2026年残月の予定（米国BLS発表）
    cpi_dates = [
        ("2026-07-15", "6月CPI"),
        ("2026-08-12", "7月CPI"),
        ("2026-09-11", "8月CPI"),
        ("2026-10-14", "9月CPI"),
        ("2026-11-13", "10月CPI"),
        ("2026-12-11", "11月CPI"),
    ]
    for date, label in cpi_dates:
        if date >= today_str:
            events.append({
                "date": date,
                "event": f"CPI（消費者物価指数）- {label}",
                "importance": "HIGH",
                "category": "CPI",
                "description": "米国消費者物価指数 - インフレ指標（BLS発表）",
            })

    # 雇用統計（毎月第1金曜日）
    jobs_dates = [
        ("2026-07-02", "6月雇用統計"),
        ("2026-08-07", "7月雇用統計"),
        ("2026-09-04", "8月雇用統計"),
        ("2026-10-02", "9月雇用統計"),
        ("2026-11-06", "10月雇用統計"),
        ("2026-12-04", "11月雇用統計"),
    ]
    for date, label in jobs_dates:
        if date >= today_str:
            events.append({
                "date": date,
                "event": f"雇用統計 - {label}",
                "importance": "HIGH",
                "category": "雇用",
                "description": "米国非農業部門雇用者数・失業率（BLS発表）",
            })

    # PCE（個人消費支出物価指数）
    pce_dates = [
        ("2026-07-31", "5月PCE"),
        ("2026-08-28", "7月PCE"),
        ("2026-09-25", "8月PCE"),
        ("2026-10-30", "9月PCE"),
        ("2026-11-25", "10月PCE"),
        ("2026-12-23", "11月PCE"),
    ]
    for date, label in pce_dates:
        if date >= today_str:
            events.append({
                "date": date,
                "event": f"PCE物価指数 - {label}",
                "importance": "HIGH",
                "category": "PCE",
                "description": "個人消費支出物価指数 - FRBが重視するインフレ指標",
            })

    # GDP速報値（四半期）
    gdp_dates = [
        ("2026-07-30", "Q2 2026 GDP速報値"),
        ("2026-10-29", "Q3 2026 GDP速報値"),
    ]
    for date, label in gdp_dates:
        if date >= today_str:
            events.append({
                "date": date,
                "event": f"GDP速報値 - {label}",
                "importance": "MEDIUM",
                "category": "GDP",
                "description": "米国GDP速報値（商務省発表）",
            })

    # ISM製造業景況感指数
    ism_dates = [
        ("2026-07-01", "6月ISM製造業"),
        ("2026-08-03", "7月ISM製造業"),
        ("2026-09-01", "8月ISM製造業"),
        ("2026-10-01", "9月ISM製造業"),
        ("2026-11-02", "10月ISM製造業"),
        ("2026-12-01", "11月ISM製造業"),
    ]
    for date, label in ism_dates:
        if date >= today_str:
            events.append({
                "date": date,
                "event": f"ISM製造業景況感指数 - {label}",
                "importance": "MEDIUM",
                "category": "景況感",
                "description": "米国製造業景況感指数（50以上で拡大）",
            })

    # 日銀金融政策決定会合（2026年）
    boj_dates = [
        ("2026-07-30", "2026-07-31"),
        ("2026-09-18", "2026-09-19"),
        ("2026-10-29", "2026-10-30"),
        ("2026-12-18", "2026-12-19"),
    ]
    for start_date, end_date in boj_dates:
        if end_date >= today_str:
            events.append({
                "date": end_date,
                "event": "日銀金融政策決定会合（結果発表）",
                "importance": "HIGH",
                "category": "日銀",
                "description": "日本銀行 金融政策決定会合・政策金利発表",
            })

    events.sort(key=lambda x: x["date"])
    return jsonify({"events": events, "as_of": today_str})


@app.route("/api/calendar/earnings")
def earnings_calendar_api():
    """ウォッチリスト銘柄の決算日（今後90日）"""
    with _watchlist_lock:
        wl = _load_watchlist()
    tickers = wl.get("tickers", [])

    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    cutoff = (today + timedelta(days=90)).strftime("%Y-%m-%d")

    results = []
    for ticker in tickers:
        found = False
        # calendar から試みる
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is not None and not cal.empty:
                for col in ["Earnings Date", "earnings_date"]:
                    if col in cal.columns:
                        for val in cal[col]:
                            try:
                                date_str = pd.Timestamp(val).strftime("%Y-%m-%d")
                                if today_str <= date_str <= cutoff:
                                    results.append({
                                        "ticker": ticker,
                                        "date": date_str,
                                        "description": "決算発表予定",
                                    })
                                    found = True
                            except Exception:
                                pass
                        break
        except Exception:
            pass

        # earnings_dates からも試みる
        if not found:
            try:
                t = yf.Ticker(ticker)
                ed = t.earnings_dates
                if ed is not None and not ed.empty:
                    for idx in ed.index:
                        try:
                            date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")
                            if today_str <= date_str <= cutoff:
                                results.append({
                                    "ticker": ticker,
                                    "date": date_str,
                                    "description": "決算発表予定（予測含む）",
                                })
                                break
                        except Exception:
                            pass
            except Exception:
                pass

    results.sort(key=lambda x: x["date"])
    return jsonify({"earnings": results, "as_of": today_str})


@app.route("/api/analysis/patterns/<ticker>")
def analysis_patterns(ticker):
    """ローソク足パターン検出"""
    ticker = ticker.upper().strip()
    try:
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d")
        if hist.empty:
            return jsonify({"error": "データ取得失敗"}), 404

        patterns = detect_candlestick_patterns(hist)
        # 直近の10件に絞る
        recent = [p for p in patterns if p][-10:]
        # 表示用の日付文字列を付加
        for p in recent:
            p["date_str"] = datetime.fromtimestamp(p["date"] / 1000).strftime("%Y-%m-%d")
        return jsonify({"ticker": ticker, "patterns": recent, "count": len(recent)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare")
def compare_page():
    return render_template("compare.html")


@app.route("/api/compare")
def compare_stocks():
    """
    複数銘柄の騰落率比較
    クエリパラメータ: tickers=AAPL,NVDA,MSFT&period=6mo
    """
    tickers_param = request.args.get("tickers", "")
    period = request.args.get("period", "6mo")

    result = {}
    for ticker in tickers_param.split(",")[:4]:  # 最大4銘柄
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        try:
            hist = yf.Ticker(ticker).history(period=period, interval="1d")
            if hist.empty:
                continue
            closes = hist["Close"].tolist()
            dates = [str(d.date()) for d in hist.index]
            base = closes[0]
            # 基準日からの騰落率に正規化
            returns = [round((c - base) / base * 100, 2) for c in closes]
            result[ticker] = {
                "dates": dates,
                "returns": returns,
                "current_return": returns[-1] if returns else 0,
                "max_return": round(max(returns), 2) if returns else 0,
                "min_return": round(min(returns), 2) if returns else 0,
            }
        except Exception:
            continue

    return jsonify({"data": result, "period": period})


@app.route("/api/analysis/earnings-history/<ticker>")
def earnings_history(ticker):
    """四半期EPS（純利益）・売上の推移（過去8四半期）"""
    t = yf.Ticker(ticker.upper())
    result = {
        "ticker": ticker.upper(),
        "quarterly_eps": [],
        "quarterly_revenue": [],
    }

    try:
        qf = t.quarterly_financials
        if qf is not None and not qf.empty:
            if "Net Income" in qf.index:
                net_income = qf.loc["Net Income"]
                for col in list(net_income.index)[:8]:
                    val = net_income[col]
                    if pd.notna(val):
                        result["quarterly_eps"].append({
                            "period": str(col.date()),
                            "value": round(float(val) / 1e6, 1),
                            "label": "Net Income($M)",
                        })
    except Exception:
        pass

    try:
        qf = t.quarterly_financials
        if qf is not None and not qf.empty:
            revenue_keys = ["Total Revenue", "Revenue", "Sales"]
            for key in revenue_keys:
                if key in qf.index:
                    rev = qf.loc[key]
                    for col in list(rev.index)[:8]:
                        val = rev[col]
                        if pd.notna(val):
                            result["quarterly_revenue"].append({
                                "period": str(col.date()),
                                "value": round(float(val) / 1e9, 2),
                                "label": "Revenue($B)",
                            })
                    break
    except Exception:
        pass

    return jsonify(result)


# ---------------------------------------------------------------------------
# カスタムスクリーナー
# ---------------------------------------------------------------------------

@app.route("/api/screen", methods=["POST"])
def custom_screen():
    """
    カスタムスクリーニング
    POSTボディ: {
        "tickers": ["AAPL", "NVDA", ...],  # 省略可能（省略時はデフォルトリスト）
        "filters": {
            "rsi_min": 30,          # RSI最小値
            "rsi_max": 70,          # RSI最大値
            "change_pct_min": 1.0,  # 当日変動率最小（%）
            "change_pct_max": 10.0, # 当日変動率最大（%）
            "price_min": 10.0,      # 株価最小
            "price_max": 500.0,     # 株価最大
            "volume_min": 1000000,  # 出来高最小（3ヶ月平均）
        }
    }
    """
    from scanner import DEFAULT_TICKERS, _calc_rsi as _scanner_calc_rsi

    data = request.get_json() or {}
    filters = data.get("filters", {})
    tickers = data.get("tickers") or DEFAULT_TICKERS

    results = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            price = float(t.fast_info.last_price or 0)
            if price <= 0:
                continue

            # 当日変動率
            hist = t.history(period="2d", interval="1d")
            if len(hist) < 2:
                continue
            prev = float(hist["Close"].iloc[-2])
            change_pct = round((price - prev) / prev * 100, 2) if prev else 0

            # RSI（14日）
            hist30 = t.history(period="30d", interval="1d")
            rsi = None
            if len(hist30) >= 15:
                rsi = _scanner_calc_rsi(hist30["Close"].tolist())

            # 出来高（3ヶ月平均）
            try:
                volume = int(t.fast_info.three_month_average_volume or 0)
            except Exception:
                volume = 0

            # フィルター適用
            f = filters
            if f.get("rsi_min") is not None and rsi is not None and rsi < f["rsi_min"]:
                continue
            if f.get("rsi_max") is not None and rsi is not None and rsi > f["rsi_max"]:
                continue
            if f.get("change_pct_min") is not None and change_pct < f["change_pct_min"]:
                continue
            if f.get("change_pct_max") is not None and change_pct > f["change_pct_max"]:
                continue
            if f.get("price_min") is not None and price < f["price_min"]:
                continue
            if f.get("price_max") is not None and price > f["price_max"]:
                continue
            if f.get("volume_min") is not None and volume < f["volume_min"]:
                continue

            results.append({
                "ticker": ticker,
                "price": round(price, 2),
                "change_pct": change_pct,
                "rsi": rsi,
                "volume": volume,
            })
        except Exception:
            continue

    return jsonify({"results": results, "count": len(results), "filters_applied": filters})


# ---------------------------------------------------------------------------
# 配当情報
# ---------------------------------------------------------------------------

@app.route("/api/analysis/dividend/<ticker>")
def dividend_info(ticker):
    """配当情報（配当利回り・配当金・権利落ち日）"""
    t = yf.Ticker(ticker.upper())

    try:
        info = t.info
        # yfinance の dividendYield はバージョンによって返し方が異なる
        # fast_info.last_price を使って実測値を計算する
        div_rate = float(info.get("dividendRate", 0) or 0)
        try:
            price = float(t.fast_info.last_price or 0)
            div_yield = round(div_rate / price * 100, 2) if price > 0 and div_rate > 0 else 0
        except Exception:
            raw_yield = float(info.get("dividendYield", 0) or 0)
            # 0.005 形式（小数）なら *100、0.5 形式（%）ならそのまま
            div_yield = round(raw_yield * 100 if raw_yield < 0.1 else raw_yield, 2)

        # 権利落ち日: Unix タイムスタンプ → 日付文字列
        ex_div_ts = info.get("exDividendDate")
        if ex_div_ts:
            try:
                ex_div_str = datetime.fromtimestamp(int(ex_div_ts)).strftime("%Y-%m-%d")
            except Exception:
                ex_div_str = str(ex_div_ts)
        else:
            ex_div_str = None

        result = {
            "ticker": ticker.upper(),
            "dividend_yield": div_yield,
            "dividend_rate": round(div_rate, 2),
            "ex_dividend_date": ex_div_str,
            "payout_ratio": round(float(info.get("payoutRatio", 0) or 0) * 100, 1),
            "five_year_avg_yield": round(float(info.get("fiveYearAvgDividendYield", 0) or 0), 2),
        }
    except Exception as e:
        result = {"error": str(e), "ticker": ticker.upper()}

    return jsonify(result)


# ---------------------------------------------------------------------------
# FX・暗号通貨
# ---------------------------------------------------------------------------

@app.route("/api/market/forex")
def market_forex():
    """主要FXレート"""
    PAIRS = {
        "USD/JPY": "JPY=X",
        "EUR/JPY": "EURJPY=X",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/CNY": "CNY=X",
    }
    result = {}
    for name, symbol in PAIRS.items():
        try:
            ticker = yf.Ticker(symbol)
            price = float(ticker.fast_info.last_price or 0)
            hist = ticker.history(period="2d", interval="1d")
            change_pct = 0
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                change_pct = round((price - prev) / prev * 100, 3) if prev else 0
            result[name] = {"rate": round(price, 4), "change_pct": change_pct}
        except Exception:
            result[name] = {"rate": None}
    return jsonify({"forex": result})


@app.route("/api/market/crypto")
def market_crypto():
    """主要暗号通貨"""
    CRYPTOS = {
        "Bitcoin":  "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana":   "SOL-USD",
        "XRP":      "XRP-USD",
    }
    result = {}
    for name, symbol in CRYPTOS.items():
        try:
            ticker = yf.Ticker(symbol)
            price = float(ticker.fast_info.last_price or 0)
            hist = ticker.history(period="2d", interval="1d")
            change_pct = 0
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                change_pct = round((price - prev) / prev * 100, 2) if prev else 0
            result[name] = {"price": round(price, 2), "change_pct": change_pct, "symbol": symbol}
        except Exception:
            result[name] = {"price": None, "symbol": symbol}
    return jsonify({"crypto": result})


@app.route("/news")
def news_page():
    return render_template("news.html")


@app.route("/api/news/feed")
def news_feed():
    """
    ウォッチリスト全銘柄のニュースを集約
    最新20件を日時降順で返す
    """
    watchlist_data = _load_watchlist()
    tickers = watchlist_data.get("tickers", [])

    # デフォルト銘柄（ウォッチリストが空の場合）
    if not tickers:
        tickers = ["AAPL", "NVDA", "MSFT", "TSLA"]

    all_news = []

    for ticker in tickers[:6]:  # 最大6銘柄（API負荷軽減）
        try:
            t = yf.Ticker(ticker)
            news = t.news or []
            for article in news[:5]:  # 各銘柄5件
                # タイムスタンプを日時に変換
                content = article.get("content", {})
                timestamp = (
                    article.get("providerPublishTime")
                    or content.get("pubDate")
                    or 0
                )
                # pubDate が文字列の場合は数値に変換（ISO 8601 / RFC 2822 両対応）
                if isinstance(timestamp, str):
                    try:
                        # ISO 8601: "2026-07-01T13:50:00Z"
                        from datetime import timezone
                        ts_str = timestamp.replace("Z", "+00:00")
                        dt_parsed = datetime.fromisoformat(ts_str)
                        timestamp = int(dt_parsed.timestamp())
                    except Exception:
                        try:
                            from email.utils import parsedate_to_datetime
                            timestamp = int(parsedate_to_datetime(timestamp).timestamp())
                        except Exception:
                            timestamp = 0

                if timestamp:
                    dt = datetime.fromtimestamp(int(timestamp))
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                else:
                    date_str = "不明"

                title = (
                    article.get("title")
                    or content.get("title", "")
                )
                publisher = (
                    article.get("publisher")
                    or content.get("provider", {}).get("displayName", "")
                )
                link = (
                    article.get("link")
                    or content.get("canonicalUrl", {}).get("url", "")
                    or content.get("clickThroughUrl", {}).get("url", "")
                )

                if not title:
                    continue

                all_news.append({
                    "ticker": ticker,
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "date": date_str,
                    "timestamp": int(timestamp) if timestamp else 0,
                })
        except Exception:
            continue

    # 日時降順でソート
    all_news.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    return jsonify({"news": all_news[:20], "count": len(all_news[:20]),
                    "tickers": tickers})


@app.route("/portfolio")
def portfolio_page():
    return render_template("portfolio.html")


@app.route("/api/portfolio/backtest", methods=["POST"])
def portfolio_backtest():
    """
    ポートフォリオバックテスト
    POST: {
        "holdings": {"AAPL": 25, "NVDA": 25, "MSFT": 25, "AMZN": 25},
        "period": "1y",
        "initial_amount": 1000000
    }
    """
    import math

    data = request.get_json() or {}
    holdings = data.get("holdings", {})
    period = data.get("period", "1y")
    initial_jpy = float(data.get("initial_amount", 1000000))

    if not holdings:
        return jsonify({"error": "ポートフォリオが空です"})

    # 比率を正規化（合計100%に）
    total_weight = sum(holdings.values())
    weights = {k: v / total_weight for k, v in holdings.items()}

    # 各銘柄の日次リターンを取得
    returns_data = {}
    for ticker in holdings:
        try:
            hist = yf.Ticker(ticker.upper()).history(period=period, interval="1d")
            if hist.empty:
                continue
            closes = hist["Close"].tolist()
            daily_returns = [(closes[i] - closes[i-1]) / closes[i-1]
                             for i in range(1, len(closes))]
            returns_data[ticker] = {
                "closes": closes,
                "dates": [str(d.date()) for d in hist.index],
                "daily_returns": daily_returns,
                "total_return": (closes[-1] - closes[0]) / closes[0] * 100,
            }
        except Exception:
            continue

    if not returns_data:
        return jsonify({"error": "データ取得失敗"})

    # ポートフォリオの日次リターンを計算（加重平均）
    min_len = min(len(r["daily_returns"]) for r in returns_data.values())
    dates = list(returns_data.values())[0]["dates"][1:min_len+1]

    portfolio_returns = []
    for i in range(min_len):
        daily_ret = sum(
            weights.get(ticker, 0) * returns_data[ticker]["daily_returns"][i]
            for ticker in returns_data
        )
        portfolio_returns.append(daily_ret)

    # 累積リターン
    cumulative = [1.0]
    for r in portfolio_returns:
        cumulative.append(cumulative[-1] * (1 + r))

    # パフォーマンス指標計算
    total_return_pct = (cumulative[-1] - 1) * 100

    # 最大ドローダウン
    peak = cumulative[0]
    max_drawdown = 0
    for val in cumulative:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_drawdown:
            max_drawdown = dd

    # シャープレシオ（年率、リスクフリーレート4.5%と仮定）
    avg_daily_return = sum(portfolio_returns) / len(portfolio_returns) if portfolio_returns else 0
    std_daily = (sum((r - avg_daily_return)**2 for r in portfolio_returns) / len(portfolio_returns))**0.5 if portfolio_returns else 0
    annual_return = avg_daily_return * 252
    annual_std = std_daily * (252**0.5)
    risk_free = 0.045 / 252
    sharpe = (avg_daily_return - risk_free) / std_daily * (252**0.5) if std_daily > 0 else 0

    # 最終資産額（JPY換算）
    final_jpy = round(initial_jpy * cumulative[-1])

    return jsonify({
        "portfolio": weights,
        "period": period,
        "initial_jpy": initial_jpy,
        "final_jpy": final_jpy,
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "annual_volatility_pct": round(annual_std * 100, 2),
        "dates": dates,
        "cumulative_returns": [round((v - 1) * 100, 2) for v in cumulative[1:]],
        "individual_returns": {t: round(r["total_return"], 2) for t, r in returns_data.items()},
    })


# ---------------------------------------------------------------------------
# ショートインタレスト・インサイダー取引
# ---------------------------------------------------------------------------

@app.route("/api/analysis/short-interest/<ticker>")
def short_interest(ticker):
    """
    空売り比率・ショートインタレスト
    yfinanceの info から取得できる情報を活用
    """
    t = yf.Ticker(ticker.upper())

    try:
        info = t.info
        result = {
            "ticker": ticker.upper(),
            "short_ratio": round(float(info.get("shortRatio", 0) or 0), 2),
            "short_percent_of_float": round(float(info.get("shortPercentOfFloat", 0) or 0) * 100, 2),
            "shares_short": int(info.get("sharesShort", 0) or 0),
            "shares_short_prior_month": int(info.get("sharesShortPriorMonth", 0) or 0),
            "float_shares": int(info.get("floatShares", 0) or 0),
        }

        # ショートスクイーズリスク評価
        short_pct = result["short_percent_of_float"]
        short_ratio = result["short_ratio"]

        if short_pct > 20 and short_ratio > 5:
            result["squeeze_risk"] = "HIGH"
            result["squeeze_desc"] = f"空売り比率{short_pct}%・日数カバー{short_ratio}日 — ショートスクイーズリスク高"
        elif short_pct > 10:
            result["squeeze_risk"] = "MEDIUM"
            result["squeeze_desc"] = f"空売り比率{short_pct}% — 注意"
        else:
            result["squeeze_risk"] = "LOW"
            result["squeeze_desc"] = f"空売り比率{short_pct}% — 正常"

    except Exception as e:
        result = {"error": str(e), "ticker": ticker.upper()}

    return jsonify(result)


@app.route("/api/analysis/insider/<ticker>")
def insider_trading(ticker):
    """
    インサイダー取引情報（yfinanceから取得）
    """
    t = yf.Ticker(ticker.upper())

    try:
        # yfinanceのinsider_transactions
        insider = t.insider_transactions
        if insider is None or insider.empty:
            return jsonify({"ticker": ticker.upper(), "transactions": [], "summary": "データなし"})

        transactions = []
        for _, row in insider.head(10).iterrows():
            try:
                # yfinanceのカラム名はバージョンにより異なるため両方試す
                insider_name = str(row.get("Insider") or row.get("Insider Trading") or "")
                position     = str(row.get("Position") or row.get("Relationship") or "")
                # Transactionカラムが空の場合はTextから取引タイプを判定
                tx_type = str(row.get("Transaction") or "")
                text    = str(row.get("Text") or "")
                if not tx_type.strip():
                    if "sale" in text.lower() or "sell" in text.lower():
                        tx_type = "Sale"
                    elif "purchase" in text.lower() or "buy" in text.lower():
                        tx_type = "Purchase"
                    elif "gift" in text.lower():
                        tx_type = "Gift"
                    else:
                        tx_type = text[:40] if text else "Unknown"
                start_date = row.get("Start Date", row.name)
                date_str   = str(start_date)[:10] if start_date is not None else ""
                transactions.append({
                    "date": date_str,
                    "insider": insider_name,
                    "position": position,
                    "transaction": tx_type,
                    "shares": int(row.get("Shares", 0) or 0),
                    "value": int(row.get("Value", 0) or 0),
                })
            except Exception:
                continue

        # 売買バランス
        buys = sum(1 for tx in transactions if "buy" in tx["transaction"].lower() or "purchase" in tx["transaction"].lower())
        sells = sum(1 for tx in transactions if "sell" in tx["transaction"].lower() or "sale" in tx["transaction"].lower())

        return jsonify({
            "ticker": ticker.upper(),
            "transactions": transactions,
            "buy_count": buys,
            "sell_count": sells,
            "summary": "買い優勢" if buys > sells else "売り優勢" if sells > buys else "中立",
        })
    except Exception as e:
        return jsonify({"ticker": ticker.upper(), "transactions": [], "error": str(e)})


# ---------------------------------------------------------------------------
# リスク指標: ベータ値・ヒストリカルボラティリティ・モメンタム
# ---------------------------------------------------------------------------

@app.route("/api/analysis/risk/<ticker>")
def risk_analysis(ticker):
    """
    リスク指標: ベータ値・ヒストリカルボラティリティ・モメンタム
    """
    t = yf.Ticker(ticker.upper())

    result = {"ticker": ticker.upper()}

    try:
        # ベータ値（yfinanceのinfo.betaから取得）
        info = t.info
        result["beta"] = round(float(info.get("beta", 1.0) or 1.0), 2)
    except Exception:
        result["beta"] = None

    try:
        # 1年データでボラティリティ・モメンタム計算
        hist = t.history(period="1y", interval="1d")
        close = hist["Close"]

        result["volatility_20d"] = calc_historical_volatility(close, 20)
        result["volatility_60d"] = calc_historical_volatility(close, 60)
        result["momentum"] = calc_momentum_score(close)

        # ベータ値（S&P500と比較、手動計算 — infoから取得できなかった場合）
        if not result.get("beta"):
            sp500 = yf.Ticker("^GSPC").history(period="1y", interval="1d")["Close"]
            min_len = min(len(close), len(sp500))
            if min_len > 60:
                stock_ret = close.pct_change().dropna().iloc[-60:]
                sp500_ret = sp500.pct_change().dropna().iloc[-60:]
                min_len2 = min(len(stock_ret), len(sp500_ret))
                if min_len2 > 20:
                    cov = float(stock_ret.iloc[:min_len2].cov(sp500_ret.iloc[:min_len2]))
                    var = float(sp500_ret.iloc[:min_len2].var())
                    result["beta"] = round(cov / var, 2) if var else 1.0
    except Exception as e:
        result["error"] = str(e)

    # リスク評価
    beta = result.get("beta", 1.0) or 1.0
    vol = result.get("volatility_20d", 20) or 20

    if beta > 1.5 or vol > 40:
        result["risk_level"] = "HIGH"
    elif beta > 1.0 or vol > 25:
        result["risk_level"] = "MEDIUM"
    else:
        result["risk_level"] = "LOW"

    return jsonify(result)


# ---------------------------------------------------------------------------
# セクターヒートマップ
# ---------------------------------------------------------------------------

@app.route("/api/market/heatmap")
def market_heatmap():
    """
    S&P500主要銘柄のセクター別ヒートマップデータ
    （各銘柄の前日比を返す）
    """
    HEATMAP_TICKERS = {
        "テクノロジー": ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN"],
        "金融":         ["JPM", "BAC", "GS", "MS", "BRK-B"],
        "ヘルスケア":   ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
        "エネルギー":   ["XOM", "CVX", "COP", "SLB"],
        "自動車/EV":    ["TSLA", "GM", "F"],
        "消費者":       ["WMT", "COST", "MCD", "SBUX"],
    }

    result = {}
    for sector, tickers in HEATMAP_TICKERS.items():
        result[sector] = []
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                price = float(t.fast_info.last_price or 0)
                hist = t.history(period="2d", interval="1d")
                change_pct = 0
                if len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                    change_pct = round((price - prev) / prev * 100, 2) if prev else 0
                market_cap = float(t.fast_info.market_cap or 0) / 1e9
                result[sector].append({
                    "ticker": ticker,
                    "price": round(price, 2),
                    "change_pct": change_pct,
                    "market_cap_b": round(market_cap, 0),
                })
            except Exception:
                continue

    return jsonify({"heatmap": result})


# ---------------------------------------------------------------------------
# マーケットセンチメント指標
# ---------------------------------------------------------------------------

@app.route("/api/market/sentiment")
def market_sentiment():
    """
    マーケット全体のセンチメント指標
    VIX・市場パフォーマンスから算出する簡易Fear&Greed
    """
    score = 50
    components = {}

    try:
        vix = float(yf.Ticker("^VIX").fast_info.last_price or 20)
        if vix < 15:
            vix_score = 80
        elif vix < 20:
            vix_score = 60
        elif vix < 25:
            vix_score = 40
        elif vix < 30:
            vix_score = 25
        else:
            vix_score = 10
        components["vix"] = {"value": round(vix, 1), "score": vix_score}
        score = vix_score
    except Exception:
        pass

    try:
        sp500 = yf.Ticker("^GSPC").history(period="1mo", interval="1d")
        if len(sp500) >= 5:
            recent_return = (
                float(sp500["Close"].iloc[-1]) - float(sp500["Close"].iloc[-5])
            ) / float(sp500["Close"].iloc[-5]) * 100
            if recent_return > 3:
                momentum_score = 80
            elif recent_return > 1:
                momentum_score = 65
            elif recent_return > -1:
                momentum_score = 50
            elif recent_return > -3:
                momentum_score = 35
            else:
                momentum_score = 20
            components["momentum"] = {"value": round(recent_return, 2), "score": momentum_score}
            score = round((score + momentum_score) / 2)
    except Exception:
        pass

    if score >= 75:
        label = "極度の強欲"
        color = "#dc2626"
    elif score >= 55:
        label = "強欲"
        color = "#f97316"
    elif score >= 45:
        label = "中立"
        color = "#eab308"
    elif score >= 25:
        label = "恐怖"
        color = "#3b82f6"
    else:
        label = "極度の恐怖"
        color = "#1e40af"

    return jsonify({
        "score": score,
        "label": label,
        "color": color,
        "components": components,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
