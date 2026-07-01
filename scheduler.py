#!/usr/bin/env python3
"""
自動スケジューラー — 毎日定時に情報収集・レポート生成
実行方法: python3 scheduler.py
tmuxセッションで常時起動推奨
"""
import schedule
import time
import json
import os
from datetime import datetime, timezone, timedelta

def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))

def morning_scan():
    """07:30 JST: プレマーケット・X速報収集"""
    print(f"[{jst_now().strftime('%H:%M JST')}] 朝のスキャン開始...")
    try:
        import requests
        r = requests.get("http://localhost:5000/api/news/breaking", timeout=30)
        data = r.json()
        save_report("morning_breaking", data if isinstance(data, dict) else {"catalysts": data})
        catalysts = data.get('catalysts', []) if isinstance(data, dict) else data
        print(f"  速報ニュース: {len(catalysts)}件")
    except Exception as e:
        print(f"  エラー: {e}")

def morning_research():
    """08:00 JST: 決算・経済指標・セクターまとめ"""
    print(f"[{jst_now().strftime('%H:%M JST')}] ファンダ調査開始...")
    try:
        import requests
        r = requests.get("http://localhost:5000/api/earnings/calendar", timeout=30)
        earnings = r.json()
        r2 = requests.get("http://localhost:5000/api/sector/rotation", timeout=30)
        sector = r2.json()
        save_report("morning_research", {"earnings": earnings, "sector": sector})
        events = earnings.get('events', []) if isinstance(earnings, dict) else earnings
        print(f"  決算予定: {len(events)}件")
    except Exception as e:
        print(f"  エラー: {e}")

def morning_signals():
    """08:30 JST: モメンタムスキャン・シグナル生成"""
    print(f"[{jst_now().strftime('%H:%M JST')}] シグナルスキャン開始...")
    try:
        import requests
        r = requests.get("http://localhost:5000/api/scan/momentum", timeout=60)
        momentum = r.json()
        save_report("morning_signals", {"momentum": momentum[:5] if isinstance(momentum, list) else momentum})
        print(f"  モメンタム銘柄: {len(momentum) if isinstance(momentum, list) else 0}件")
    except Exception as e:
        print(f"  エラー: {e}")

def afternoon_update():
    """15:30 JST: 日本株引け後のアップデート"""
    print(f"[{jst_now().strftime('%H:%M JST')}] 午後レポート生成...")
    try:
        from update_status import build_status
        status = build_status()
        print(f"  進捗: {status['progress_pct']}%")
    except Exception as e:
        print(f"  エラー: {e}")

def evening_sentiment():
    """21:00 JST: 米国市場センチメント収集"""
    print(f"[{jst_now().strftime('%H:%M JST')}] 夜間センチメント収集...")
    watchlist = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    try:
        import requests
        results = []
        for ticker in watchlist:
            r = requests.get(f"http://localhost:5000/api/sentiment?ticker={ticker}", timeout=30)
            results.append(r.json())
        save_report("evening_sentiment", results)
        print(f"  センチメント収集完了: {len(results)}銘柄")
    except Exception as e:
        print(f"  エラー: {e}")

def record_daily_pnl():
    """21:30 JST: デモトレード日次損益を記録"""
    print(f"[{jst_now().strftime('%H:%M JST')}] 日次損益記録...")
    try:
        from paper_trading import update_daily_pnl
        entry = update_daily_pnl()
        print(f"  損益記録: {entry['date']} → ¥{entry['pnl']:+,.0f}")
    except Exception as e:
        print(f"  エラー: {e}")

def evening_report():
    """22:00 JST: 日次レポート生成・Obsidian更新"""
    print(f"[{jst_now().strftime('%H:%M JST')}] 日次レポート生成...")
    try:
        from update_status import build_status
        status = build_status()
        # Obsidianに日次レポート保存
        today = jst_now().strftime("%Y-%m-%d")
        report_path = f"/home/ec2-user/obsidian-vault/daily/{today}.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 日次レポート {today}\n\n")
            f.write(f"## システム進捗\n- Phase: {status['phase']}\n- 進捗: {status['progress_pct']}%\n\n")
            f.write(f"## タスク状況\n- 完了: {len(status['tasks'].get('completed', []))}件\n")
            f.write(f"- 進行中: {len(status['tasks'].get('in_progress', []))}件\n")
            f.write(f"- 待機中: {len(status['tasks'].get('pending', []))}件\n")
        print(f"  日次レポート保存: {report_path}")
    except Exception as e:
        print(f"  エラー: {e}")

def hourly_update():
    """毎時: シグナル・センチメント・ステータスを更新"""
    print(f"[{jst_now().strftime('%H:%M JST')}] 毎時更新開始...")
    import requests

    # シグナルスキャン
    try:
        r = requests.get("http://localhost:5000/api/scan/opportunities", timeout=60)
        opps = r.json()
        count = len(opps) if isinstance(opps, list) else 0
        save_report("hourly_signals", {"opportunities": opps})
        print(f"  シグナル: {count}銘柄")
    except Exception as e:
        print(f"  シグナルエラー: {e}")

    # センチメント（主要5銘柄）
    try:
        watchlist = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]
        results = []
        for ticker in watchlist:
            r = requests.get(f"http://localhost:5000/api/sentiment?ticker={ticker}", timeout=20)
            results.append(r.json())
        save_report("hourly_sentiment", results)
        print(f"  センチメント: {len(results)}銘柄")
    except Exception as e:
        print(f"  センチメントエラー: {e}")

    # ステータス更新
    try:
        from update_status import build_status
        build_status()
        print(f"  ステータス更新完了")
    except Exception as e:
        print(f"  ステータスエラー: {e}")


def check_price_alerts():
    """5分ごと: 価格アラートチェック"""
    try:
        import sys
        sys.path.insert(0, "/home/ec2-user/stock-chart")
        from alerts import check_alerts
        triggered = check_alerts()
        if triggered:
            print(f"[{jst_now().strftime('%H:%M JST')}] アラート発動: {len(triggered)}件")
            for a in triggered:
                cond = "以上" if a["condition"] == "above" else "以下"
                print(f"  {a['ticker']} ${a['triggered_price']} (設定: ${a['price']} {cond})")
        else:
            print(f"[{jst_now().strftime('%H:%M JST')}] アラートチェック完了（発動なし）")
    except Exception as e:
        print(f"  アラートチェックエラー: {e}")


def save_report(name: str, data: dict):
    path = f"/home/ec2-user/stock-chart/data/reports/{name}_{jst_now().strftime('%Y%m%d')}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    # スケジュール登録
    schedule.every(5).minutes.do(check_price_alerts)
    schedule.every(1).hours.do(hourly_update)
    schedule.every().day.at("07:30").do(morning_scan)
    schedule.every().day.at("08:00").do(morning_research)
    schedule.every().day.at("08:30").do(morning_signals)
    schedule.every().day.at("15:30").do(afternoon_update)
    schedule.every().day.at("21:00").do(evening_sentiment)
    schedule.every().day.at("21:30").do(record_daily_pnl)
    schedule.every().day.at("22:00").do(evening_report)

    # 起動時に登録済みスケジュール一覧を表示
    print("=" * 50)
    print(f"スケジューラー起動 — {jst_now().strftime('%Y-%m-%d %H:%M JST')}")
    print("=" * 50)
    print("登録済みスケジュール:")
    for job in schedule.get_jobs():
        print(f"  {job}")
    print("=" * 50)
    print("Ctrl+C で停止")
    print()

    # メインループ
    while True:
        schedule.run_pending()
        time.sleep(30)
