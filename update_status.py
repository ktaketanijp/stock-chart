#!/usr/bin/env python3
"""
毎時ステータス更新スクリプト
cron または tmux ループから呼び出す
"""
import json, os, subprocess
from datetime import datetime

JST_OFFSET = 9  # UTC+9

def jst_now():
    from datetime import timezone, timedelta
    return datetime.now(timezone(timedelta(hours=JST_OFFSET))).strftime("%Y-%m-%d %H:%M JST")

def get_tmux_sessions():
    try:
        out = subprocess.check_output(["tmux", "list-sessions"], text=True)
        return [line.split(":")[0] for line in out.strip().split("\n") if line]
    except Exception:
        return []

def get_obsidian_tasks():
    vault = os.path.expanduser("~/obsidian-vault/tasks")
    result = {"pending": [], "in_progress": [], "completed": []}
    for state in result:
        d = os.path.join(vault, state)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".md"):
                    result[state].append(f.replace(".md","").split("-", 4)[-1] if "-" in f else f)
    return result

def get_routes():
    try:
        import requests
        r = requests.get("http://localhost:5000/api/status_raw", timeout=3)
        return r.json().get("routes", [])
    except Exception:
        # フォールバック: app.pyを読む
        try:
            with open("/home/ec2-user/stock-chart/app.py") as f:
                return [l.strip() for l in f if l.strip().startswith("@app.route")]
        except Exception:
            return []

def check_files():
    base = "/home/ec2-user/stock-chart"
    files = {
        "sentiment.py": "X/Twitter センチメントエンジン",
        "fundamentals.py": "ファンダメンタルズ分析",
        "signal_engine.py": "AIシグナルエンジン",
        "indicators.py": "追加インジケーター（BB/Ichimoku/ATR）",
        "scanner.py": "モメンタムスキャナー",
        "paper_trading.py": "デモトレード管理",
        "scheduler.py": "自動スケジューラー",
        "templates/analysis.html": "分析ダッシュボード",
        "templates/scanner.html": "スキャナーページ",
        "templates/journal.html": "トレード日誌",
        "templates/guide.html": "ガイドページ",
    }
    result = []
    for path, label in files.items():
        full = os.path.join(base, path)
        exists = os.path.exists(full)
        size = os.path.getsize(full) if exists else 0
        result.append({"path": path, "label": label, "done": exists, "size": size})
    return result

def build_status():
    tasks = get_obsidian_tasks()
    sessions = get_tmux_sessions()
    file_status = check_files()
    done_count = sum(1 for f in file_status if f["done"])
    total_count = len(file_status)
    pct = int(done_count / total_count * 100)

    # フェーズ判定
    if pct < 40:
        phase = "Phase 1A: 情報収集基盤構築中"
        phase_color = "blue"
    elif pct < 80:
        phase = "Phase 1B: 統合・UI構築中"
        phase_color = "yellow"
    elif pct < 100:
        phase = "Phase 1: 仕上げ中"
        phase_color = "orange"
    else:
        phase = "Phase 2: デモトレード進行中 — ¥100,000元手 → 目標+¥10,000"
        phase_color = "green"

    status = {
        "updated_at": jst_now(),
        "phase": phase,
        "phase_color": phase_color,
        "progress_pct": pct,
        "files": file_status,
        "tasks": tasks,
        "tmux_sessions": sessions,
        "done_count": done_count,
        "total_count": total_count,
    }

    out = "/home/ec2-user/stock-chart/data/status.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    print(f"[{jst_now()}] Status updated: {pct}% ({done_count}/{total_count})")
    return status

if __name__ == "__main__":
    build_status()
