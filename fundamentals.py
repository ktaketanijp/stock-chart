import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from dotenv import load_dotenv

load_dotenv("/home/ec2-user/.env")

# ---------------------------------------------------------------------------
# インメモリキャッシュ（RS スコア・セクターローテーション用）
# ---------------------------------------------------------------------------
_cache: dict = {}

def _cache_get(key: str, ttl: int):
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < ttl:
        return entry["data"]
    return None

def _cache_set(key: str, data):
    _cache[key] = {"ts": time.time(), "data": data}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val, default=None):
    """None / NaN ガード"""
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def get_rs_score(ticker: str, period: str = "6mo") -> int:
    """相対強度スコア（0-99）: S&P500比の騰落率パーセンタイル。結果を1時間キャッシュ。"""
    cache_key = f"rs:{ticker}:{period}"
    cached = _cache_get(cache_key, ttl=3600)
    if cached is not None:
        return cached

    sp500_sample = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
        "LLY", "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "CVX",
        "MRK", "ABBV", "COST", "PEP", "KO", "WMT", "BAC", "CSCO", "CRM",
        "MCD", "NFLX", "AMD", "INTC", "TMO", "ACN", "ORCL", "ABT", "TXN",
        "DHR", "NKE", "QCOM", "PM", "LIN", "UPS", "HON", "AMGN", "RTX",
        "IBM", "CAT", "GE", "NOW", "SPGI",
    ]

    # ベースライン（S&P500全体）をまとめてダウンロードして API 呼び出しを最小化
    base_key = f"rs_base:{period}"
    perfs_base = _cache_get(base_key, ttl=3600)
    if perfs_base is None:
        try:
            tickers_str = " ".join(sp500_sample)
            data = yf.download(tickers_str, period=period, progress=False, auto_adjust=True)["Close"]
            perfs_base = {}
            for t in sp500_sample:
                if t in data.columns:
                    col = data[t].dropna()
                    if len(col) >= 2:
                        perfs_base[t] = float((col.iloc[-1] - col.iloc[0]) / col.iloc[0])
        except Exception:
            perfs_base = {}
        _cache_set(base_key, perfs_base)

    def _perf_single(t):
        try:
            h = yf.Ticker(t).history(period=period)
            if len(h) < 2:
                return None
            return float((h["Close"].iloc[-1] - h["Close"].iloc[0]) / h["Close"].iloc[0])
        except Exception:
            return None

    target_perf = perfs_base.get(ticker) or _perf_single(ticker)
    if target_perf is None:
        return 50

    perfs = [v for k, v in perfs_base.items() if k != ticker]
    if not perfs:
        return 50

    perfs_arr = np.array(perfs)
    percentile = float(np.sum(perfs_arr < target_perf)) / len(perfs_arr) * 99
    result = int(round(percentile))
    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Task 1: get_fundamentals
# ---------------------------------------------------------------------------

def get_fundamentals(ticker: str) -> dict:
    """財務指標の取得"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        per = _safe(info.get("trailingPE"))
        pbr = _safe(info.get("priceToBook"))
        roe = _safe(info.get("returnOnEquity"))
        if roe is not None:
            roe = round(roe * 100, 2)

        # EPS成長率（前年比）
        eps_ttm = _safe(info.get("trailingEps"))
        eps_fwd = _safe(info.get("forwardEps"))
        eps_growth_yoy = None
        if eps_ttm and eps_fwd and eps_ttm != 0:
            eps_growth_yoy = round((eps_fwd - eps_ttm) / abs(eps_ttm) * 100, 2)

        # 売上成長率
        revenue_growth_yoy = _safe(info.get("revenueGrowth"))
        if revenue_growth_yoy is not None:
            revenue_growth_yoy = round(revenue_growth_yoy * 100, 2)

        # trailingAnnualDividendYieldは小数（0.0035 = 0.35%）で正確
        # dividendYieldは誤った値を返すことがある
        dividend_yield = _safe(info.get("trailingAnnualDividendYield")) or _safe(info.get("dividendYield"))
        if dividend_yield is not None:
            dividend_yield = round(float(dividend_yield) * 100, 4)

        market_cap = _safe(info.get("marketCap"))
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")

        week52_high = _safe(info.get("fiftyTwoWeekHigh"))
        week52_low = _safe(info.get("fiftyTwoWeekLow"))
        current_price = _safe(info.get("currentPrice")) or _safe(info.get("regularMarketPrice"))

        price_vs_52w_high = None
        if week52_high and current_price and week52_high > 0:
            price_vs_52w_high = round((current_price - week52_high) / week52_high * 100, 2)

        inst_ownership = _safe(info.get("heldPercentInstitutions"))
        if inst_ownership is not None:
            inst_ownership = round(inst_ownership * 100, 2)

        short_ratio = _safe(info.get("shortRatio"))

        # 決算日
        next_earnings_date = None
        try:
            cal = stock.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed_list = cal.get("Earnings Date") or []
                    if ed_list:
                        ed = ed_list[0] if isinstance(ed_list, list) else ed_list
                        if ed:
                            next_earnings_date = str(ed) if isinstance(ed, str) else str(ed)
                elif isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.columns:
                    ed = cal["Earnings Date"].iloc[0]
                    if pd.notna(ed):
                        next_earnings_date = str(ed.date()) if hasattr(ed, "date") else str(ed)
        except Exception:
            next_earnings_date = None

        analyst_target_price = _safe(info.get("targetMeanPrice"))
        analyst_recommendation = info.get("recommendationKey", "N/A")

        return {
            "ticker": ticker,
            "per": _safe(per, None),
            "pbr": _safe(pbr, None),
            "roe": roe,
            "eps_growth_yoy": eps_growth_yoy,
            "revenue_growth_yoy": revenue_growth_yoy,
            "dividend_yield": dividend_yield,
            "market_cap": market_cap,
            "sector": sector,
            "industry": industry,
            "week52_high": week52_high,
            "week52_low": week52_low,
            "current_price": current_price,
            "price_vs_52w_high": price_vs_52w_high,
            "inst_ownership": inst_ownership,
            "short_ratio": short_ratio,
            "next_earnings_date": next_earnings_date,
            "analyst_target_price": analyst_target_price,
            "analyst_recommendation": analyst_recommendation,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ---------------------------------------------------------------------------
# Task 2: get_canslim_score
# ---------------------------------------------------------------------------

def get_canslim_score(ticker: str) -> dict:
    """CAN SLIM スコアリング（0-100点）"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="2y")

        scores = {}

        # --- C: Current Quarterly Earnings (配点 14点) ---
        eps_qtr_growth = _safe(info.get("earningsQuarterlyGrowth"))
        if eps_qtr_growth is not None:
            eps_qtr_growth_pct = eps_qtr_growth * 100
        else:
            eps_qtr_growth_pct = None

        if eps_qtr_growth_pct is not None and eps_qtr_growth_pct >= 25:
            c_score = 14
            c_pass = True
            c_detail = f"直近四半期EPS成長率 {eps_qtr_growth_pct:.1f}% (基準: 25%以上)"
        elif eps_qtr_growth_pct is not None:
            # 0-25%は部分点
            c_score = int(14 * max(0, eps_qtr_growth_pct) / 25)
            c_pass = False
            c_detail = f"直近四半期EPS成長率 {eps_qtr_growth_pct:.1f}% (基準: 25%以上 — 未達)"
        else:
            c_score = 0
            c_pass = False
            c_detail = "直近四半期EPS成長率データなし"
        scores["C"] = {"score": c_score, "pass": c_pass, "detail": c_detail}

        # --- A: Annual Earnings Growth (配点 14点) ---
        # revenueGrowth / earningsGrowth を使用
        annual_earnings_growth = _safe(info.get("earningsGrowth"))
        if annual_earnings_growth is not None:
            aeg_pct = annual_earnings_growth * 100
        else:
            aeg_pct = None

        if aeg_pct is not None and aeg_pct >= 25:
            a_score = 14
            a_pass = True
            a_detail = f"年間EPS成長率 {aeg_pct:.1f}% (基準: 25%以上)"
        elif aeg_pct is not None:
            a_score = int(14 * max(0, aeg_pct) / 25)
            a_pass = False
            a_detail = f"年間EPS成長率 {aeg_pct:.1f}% (基準: 25%以上 — 未達)"
        else:
            a_score = 0
            a_pass = False
            a_detail = "年間EPS成長率データなし"
        scores["A"] = {"score": a_score, "pass": a_pass, "detail": a_detail}

        # --- N: New Product / Near 52-week High Breakout (配点 14点) ---
        week52_high = _safe(info.get("fiftyTwoWeekHigh"))
        current_price = _safe(info.get("currentPrice")) or _safe(info.get("regularMarketPrice"))
        if not current_price and not hist.empty:
            current_price = float(hist["Close"].iloc[-1])

        if week52_high and current_price:
            pct_from_high = (current_price - week52_high) / week52_high * 100
            if pct_from_high >= -5:  # 高値の5%以内
                n_score = 14
                n_pass = True
                n_detail = f"52週高値の {pct_from_high:.1f}% 水準 — ブレイクアウト圏内"
            elif pct_from_high >= -15:
                n_score = 7
                n_pass = False
                n_detail = f"52週高値の {pct_from_high:.1f}% 水準 — やや離れている"
            else:
                n_score = 0
                n_pass = False
                n_detail = f"52週高値の {pct_from_high:.1f}% 水準 — 高値から大きく下落"
        else:
            n_score = 0
            n_pass = False
            n_detail = "価格データなし"
        scores["N"] = {"score": n_score, "pass": n_pass, "detail": n_detail}

        # --- S: Supply/Demand (配点 14点) ---
        # 出来高トレンド: 直近20日平均 vs 前20日平均
        if len(hist) >= 40:
            recent_vol = hist["Volume"].iloc[-20:].mean()
            prev_vol = hist["Volume"].iloc[-40:-20].mean()
            vol_ratio = recent_vol / prev_vol if prev_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        float_shares = _safe(info.get("floatShares"))
        shares_outstanding = _safe(info.get("sharesOutstanding"))
        float_ratio = None
        if float_shares and shares_outstanding and shares_outstanding > 0:
            float_ratio = float_shares / shares_outstanding

        if vol_ratio >= 1.1:
            s_vol_score = 7
            s_vol_msg = f"出来高増加 {vol_ratio:.2f}x"
        else:
            s_vol_score = int(7 * min(vol_ratio, 1.0))
            s_vol_msg = f"出来高減少 {vol_ratio:.2f}x"

        if float_ratio is not None and float_ratio < 0.3:
            s_float_score = 7
            s_float_msg = f"浮動株比率 {float_ratio*100:.1f}% (少なく良好)"
        elif float_ratio is not None:
            s_float_score = int(7 * (1 - min(float_ratio, 1.0)))
            s_float_msg = f"浮動株比率 {float_ratio*100:.1f}%"
        else:
            s_float_score = 3
            s_float_msg = "浮動株データなし"

        s_score = s_vol_score + s_float_score
        s_pass = s_score >= 10
        s_detail = f"{s_vol_msg} / {s_float_msg}"
        scores["S"] = {"score": s_score, "pass": s_pass, "detail": s_detail}

        # --- L: Leader (相対強度) (配点 15点) ---
        rs = get_rs_score(ticker, "6mo")
        if rs >= 80:
            l_score = 15
            l_pass = True
        elif rs >= 70:
            l_score = 12
            l_pass = True
        elif rs >= 50:
            l_score = 7
            l_pass = False
        else:
            l_score = 3
            l_pass = False
        l_detail = f"相対強度スコア {rs}/99 (基準: 70以上)"
        scores["L"] = {"score": l_score, "pass": l_pass, "detail": l_detail}

        # --- I: Institutional Sponsorship (配点 15点) ---
        inst_pct = _safe(info.get("heldPercentInstitutions"))
        inst_increase = _safe(info.get("institutionsPercentHeld"))  # 同値のフォールバック

        if inst_pct is not None:
            inst_pct_val = inst_pct * 100
        elif inst_increase is not None:
            inst_pct_val = inst_increase * 100
        else:
            inst_pct_val = None

        if inst_pct_val is not None and inst_pct_val >= 40:
            i_score = 15
            i_pass = True
            i_detail = f"機関投資家保有率 {inst_pct_val:.1f}% (良好)"
        elif inst_pct_val is not None and inst_pct_val >= 20:
            i_score = 10
            i_pass = True
            i_detail = f"機関投資家保有率 {inst_pct_val:.1f}% (普通)"
        elif inst_pct_val is not None:
            i_score = 5
            i_pass = False
            i_detail = f"機関投資家保有率 {inst_pct_val:.1f}% (低い)"
        else:
            i_score = 5
            i_pass = False
            i_detail = "機関投資家データなし"
        scores["I"] = {"score": i_score, "pass": i_pass, "detail": i_detail}

        # --- M: Market Direction (配点 14点) ---
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="3mo")
            if len(spy_hist) >= 60:
                spy_ma50 = spy_hist["Close"].iloc[-50:].mean()
                spy_current = float(spy_hist["Close"].iloc[-1])
                spy_1m_ago = float(spy_hist["Close"].iloc[-22])
                if spy_current > spy_ma50 and spy_current > spy_1m_ago:
                    m_score = 14
                    m_pass = True
                    m_detail = "S&P500 上昇トレンド (MA50上・1ヶ月前比上昇)"
                elif spy_current > spy_ma50:
                    m_score = 10
                    m_pass = True
                    m_detail = "S&P500 MA50上だが勢い弱め"
                else:
                    m_score = 4
                    m_pass = False
                    m_detail = "S&P500 MA50下 — 市場調整局面"
            else:
                m_score = 7
                m_pass = True
                m_detail = "市場データ不足 — 中立評価"
        except Exception:
            m_score = 7
            m_pass = True
            m_detail = "市場データ取得失敗 — 中立評価"
        scores["M"] = {"score": m_score, "pass": m_pass, "detail": m_detail}

        # --- 合計 ---
        total_score = sum(v["score"] for v in scores.values())
        if total_score >= 80:
            grade = "A"
        elif total_score >= 65:
            grade = "B"
        elif total_score >= 50:
            grade = "C"
        elif total_score >= 35:
            grade = "D"
        else:
            grade = "F"

        passed = sum(1 for v in scores.values() if v["pass"])
        summary = (
            f"{ticker} の CAN SLIM スコアは {total_score}/100点（グレード{grade}）。"
            f"7項目中 {passed}項目をクリア。"
        )

        return {
            "ticker": ticker,
            "total_score": total_score,
            "grade": grade,
            "C": scores["C"],
            "A": scores["A"],
            "N": scores["N"],
            "S": scores["S"],
            "L": scores["L"],
            "I": scores["I"],
            "M": scores["M"],
            "summary": summary,
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ---------------------------------------------------------------------------
# Task 3: get_sepa_score
# ---------------------------------------------------------------------------

def get_sepa_score(ticker: str) -> dict:
    """SEPA（Minervini）条件チェック"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if hist.empty or len(hist) < 50:
            return {"ticker": ticker, "error": "データ不足"}

        closes = hist["Close"]
        current_price = float(closes.iloc[-1])

        # 移動平均
        ma50 = float(closes.iloc[-50:].mean()) if len(closes) >= 50 else None
        ma150 = float(closes.iloc[-150:].mean()) if len(closes) >= 150 else None
        ma200 = float(closes.iloc[-200:].mean()) if len(closes) >= 200 else None

        # MA200 トレンド: 現在 vs 1ヶ月前
        ma200_trend_up = False
        if len(closes) >= 220:
            ma200_now = float(closes.iloc[-200:].mean())
            ma200_1m = float(closes.iloc[-220:-20].mean())
            ma200_trend_up = ma200_now > ma200_1m

        # 52週高値・安値
        week52_high = float(closes.max())
        week52_low = float(closes.min())

        # 相対強度 vs SPY
        rs = get_rs_score(ticker, "6mo")

        conditions = []

        # 1. Price > MA200
        if ma200:
            cond1 = current_price > ma200
            conditions.append({
                "name": "株価 > MA200",
                "pass": cond1,
                "value": f"株価 {current_price:.2f} / MA200 {ma200:.2f}",
            })
        else:
            conditions.append({
                "name": "株価 > MA200",
                "pass": False,
                "value": "データ不足（200日未満）",
            })

        # 2. Price > MA150
        if ma150:
            cond2 = current_price > ma150
            conditions.append({
                "name": "株価 > MA150",
                "pass": cond2,
                "value": f"株価 {current_price:.2f} / MA150 {ma150:.2f}",
            })
        else:
            conditions.append({
                "name": "株価 > MA150",
                "pass": False,
                "value": "データ不足",
            })

        # 3. Price > MA50
        if ma50:
            cond3 = current_price > ma50
            conditions.append({
                "name": "株価 > MA50",
                "pass": cond3,
                "value": f"株価 {current_price:.2f} / MA50 {ma50:.2f}",
            })
        else:
            conditions.append({
                "name": "株価 > MA50",
                "pass": False,
                "value": "データ不足",
            })

        # 4. MA200 上昇トレンド
        conditions.append({
            "name": "MA200 上昇トレンド（1ヶ月比）",
            "pass": ma200_trend_up,
            "value": "上昇中" if ma200_trend_up else "下降または横ばい",
        })

        # 5. 株価が52週高値の25%以内
        pct_from_52h = (current_price - week52_high) / week52_high * 100
        cond5 = pct_from_52h >= -25
        conditions.append({
            "name": "52週高値から25%以内",
            "pass": cond5,
            "value": f"高値比 {pct_from_52h:.1f}%",
        })

        # 6. 株価が52週安値の30%以上上
        pct_from_52l = (current_price - week52_low) / week52_low * 100
        cond6 = pct_from_52l >= 30
        conditions.append({
            "name": "52週安値から30%以上上昇",
            "pass": cond6,
            "value": f"安値比 +{pct_from_52l:.1f}%",
        })

        # 7. RS >= 70
        cond7 = rs >= 70
        conditions.append({
            "name": "相対強度 (RS) >= 70",
            "pass": cond7,
            "value": f"RS = {rs}/99",
        })

        conditions_met = sum(1 for c in conditions if c["pass"])
        sepa_qualified = conditions_met == 7

        if sepa_qualified:
            summary = f"{ticker} はMinervini SEPA条件をすべてクリア。強気セットアップ候補。"
        elif conditions_met >= 5:
            summary = f"{ticker} はSEPA条件 {conditions_met}/7 クリア。条件近傍だが未達あり。"
        elif conditions_met >= 3:
            summary = f"{ticker} はSEPA条件 {conditions_met}/7 クリア。トレンド発展途上。"
        else:
            summary = f"{ticker} はSEPA条件 {conditions_met}/7 のみクリア。現時点では買い候補外。"

        return {
            "ticker": ticker,
            "sepa_qualified": sepa_qualified,
            "conditions_met": conditions_met,
            "total_conditions": 7,
            "conditions": conditions,
            "rs_score": rs,
            "summary": summary,
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ---------------------------------------------------------------------------
# Task 4: get_earnings_calendar
# ---------------------------------------------------------------------------

def get_earnings_calendar(days_ahead: int = 30) -> list:
    """決算カレンダー（今後N日）"""
    watchlist = [
        "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META",
        "7203.T", "6758.T", "9984.T",
    ]
    today = datetime.now().date()
    cutoff = today + timedelta(days=days_ahead)
    results = []

    for ticker in watchlist:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            company = info.get("longName") or info.get("shortName") or ticker

            # 決算日取得
            cal = stock.calendar
            earnings_date = None
            if cal is not None:
                import datetime as _dt
                if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.columns:
                    raw = cal["Earnings Date"].iloc[0]
                    if pd.notna(raw):
                        if isinstance(raw, _dt.datetime):
                            earnings_date = raw.date()
                        elif isinstance(raw, _dt.date):
                            earnings_date = raw
                elif isinstance(cal, dict):
                    raw_list = cal.get("Earnings Date") or []
                    if raw_list:
                        raw = raw_list[0] if isinstance(raw_list, list) else raw_list
                        if raw and pd.notna(raw):
                            if isinstance(raw, _dt.datetime):
                                earnings_date = raw.date()
                            elif isinstance(raw, _dt.date):
                                earnings_date = raw

            if earnings_date is None:
                continue
            if earnings_date < today or earnings_date > cutoff:
                continue

            days_until = (earnings_date - today).days
            est_eps = _safe(info.get("forwardEps"))
            prev_eps = _safe(info.get("trailingEps"))

            results.append({
                "ticker": ticker,
                "company": company,
                "earnings_date": str(earnings_date),
                "days_until": days_until,
                "est_eps": est_eps,
                "prev_eps": prev_eps,
                "surprise_history_avg_pct": None,  # yfinance から取得困難
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["days_until"])
    return results


# ---------------------------------------------------------------------------
# Task 5: get_news_with_summary (Groq AI要約)
# ---------------------------------------------------------------------------

def get_news_with_summary(ticker: str) -> dict:
    """ニュース取得 + AI要約"""
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

        stock = yf.Ticker(ticker)
        raw_news = stock.news
        if not raw_news:
            return {"ticker": ticker, "articles": [], "ai_summary": "ニュースなし", "sentiment": "neutral"}

        news_items = raw_news[:5]
        articles = []
        titles_text = []

        for item in news_items:
            content = item.get("content", {})
            title = content.get("title", item.get("title", "タイトル不明"))
            pub_date = content.get("pubDate", item.get("providerPublishTime", ""))
            provider = content.get("provider", {})
            if isinstance(provider, dict):
                source = provider.get("displayName", "不明")
            else:
                source = str(provider)
            url = ""
            click_through = content.get("clickThroughUrl", {})
            if isinstance(click_through, dict):
                url = click_through.get("url", "")
            if not url:
                url = item.get("link", "")

            articles.append({
                "title": title,
                "source": source,
                "pub_date": str(pub_date),
                "url": url,
            })
            titles_text.append(f"- {title}")

        # Groq でAI要約
        prompt = (
            f"{ticker} に関する最新ニュース5件を以下に示します。"
            f"日本語で100〜150文字の簡潔なサマリーを作成し、"
            f"センチメント（positive/negative/neutral）を判断してください。\n\n"
            f"ニュース:\n" + "\n".join(titles_text) + "\n\n"
            f"回答形式:\n要約: <要約文>\nセンチメント: <positive|negative|neutral>"
        )

        chat = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        reply = chat.choices[0].message.content.strip()

        # パース
        ai_summary = reply
        sentiment = "neutral"
        for line in reply.split("\n"):
            if line.startswith("要約:"):
                ai_summary = line.replace("要約:", "").strip()
            if line.startswith("センチメント:"):
                raw_sent = line.replace("センチメント:", "").strip().lower()
                if "positive" in raw_sent:
                    sentiment = "positive"
                elif "negative" in raw_sent:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

        return {
            "ticker": ticker,
            "articles": articles,
            "ai_summary": ai_summary,
            "sentiment": sentiment,
        }

    except Exception as e:
        return {"ticker": ticker, "articles": [], "ai_summary": f"エラー: {str(e)}", "sentiment": "neutral"}


# ---------------------------------------------------------------------------
# Task 6: get_sector_rotation
# ---------------------------------------------------------------------------

SECTOR_ETFS = {
    "XLK": "テクノロジー",
    "XLV": "ヘルスケア",
    "XLF": "金融",
    "XLY": "一般消費財",
    "XLP": "生活必需品",
    "XLE": "エネルギー",
    "XLU": "公益事業",
    "XLI": "資本財",
    "XLB": "素材",
    "XLRE": "不動産",
    "XLC": "通信サービス",
}


def get_sector_rotation() -> dict:
    """セクターローテーション分析。結果を30分キャッシュ。"""
    cache_key = "sector_rotation"
    cached = _cache_get(cache_key, ttl=1800)
    if cached is not None:
        return cached

    try:
        performances = {}
        etf_list = list(SECTOR_ETFS.keys())

        # 3ヶ月分を一括ダウンロード（1ヶ月分は先頭を除いて算出）
        try:
            raw = yf.download(" ".join(etf_list), period="3mo", progress=False, auto_adjust=True)["Close"]
        except Exception:
            raw = None

        for etf, name in SECTOR_ETFS.items():
            try:
                if raw is not None and etf in raw.columns:
                    col = raw[etf].dropna()
                else:
                    col = yf.Ticker(etf).history(period="3mo")["Close"].dropna()
                if len(col) < 5:
                    continue
                perf_3m = float((col.iloc[-1] - col.iloc[0]) / col.iloc[0] * 100)
                # 1ヶ月分: 末尾21営業日
                col_1m = col.iloc[-21:] if len(col) >= 21 else col
                perf_1m = float((col_1m.iloc[-1] - col_1m.iloc[0]) / col_1m.iloc[0] * 100)
                performances[etf] = {
                    "name": name,
                    "perf_1m": round(perf_1m, 2),
                    "perf_3m": round(perf_3m, 2),
                }
            except Exception:
                continue

        if not performances:
            return {"error": "セクターデータ取得失敗"}

        sorted_by_1m = sorted(performances.items(), key=lambda x: x[1]["perf_1m"], reverse=True)

        leading = [f"{v['name']}({k})" for k, v in sorted_by_1m[:3]]
        lagging = [f"{v['name']}({k})" for k, v in sorted_by_1m[-3:]]

        # Risk On/Off 判定: テクノロジー・一般消費財が上位 = リスクオン
        top3_etfs = [k for k, _ in sorted_by_1m[:3]]
        bottom3_etfs = [k for k, _ in sorted_by_1m[-3:]]

        risk_on_etfs = {"XLK", "XLY", "XLC", "XLI"}
        risk_off_etfs = {"XLU", "XLP", "XLV", "XLRE"}

        risk_on_count = len(risk_on_etfs & set(top3_etfs))
        risk_off_count = len(risk_off_etfs & set(top3_etfs))

        if risk_on_count >= 2:
            rotation_signal = "risk_on"
        elif risk_off_count >= 2:
            rotation_signal = "risk_off"
        else:
            rotation_signal = "neutral"

        result = {
            "sectors": performances,
            "leading": leading,
            "lagging": lagging,
            "rotation_signal": rotation_signal,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        _cache_set(cache_key, result)
        return result

    except Exception as e:
        return {"error": str(e)}
