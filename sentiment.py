import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/home/ec2-user/.env")

client = OpenAI(
    api_key=os.environ["GROK_API_KEY"],
    base_url="https://api.x.ai/v1"
)

# 5分間キャッシュ
_cache = {}
CACHE_TTL = 300  # seconds

def _cache_get(key):
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]
        del _cache[key]
    return None

def _cache_set(key, data):
    _cache[key] = {"ts": time.time(), "data": data}

def _extract_json(raw: str) -> str:
    """Grok応答から ```json ... ``` ブロックを除去してJSONだけ返す。"""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] が "json\n{...}" または "{...}"
        inner = parts[1]
        if inner.startswith("json"):
            inner = inner[4:]
        raw = inner.strip()
    return raw


def _call_grok(prompt: str, model: str = "grok-3") -> str:
    """Grok API呼び出し（LiveSearch有効）。モデルフォールバック付き。"""
    models_to_try = [model, "grok-2-1212", "grok-beta"]
    for m in models_to_try:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial sentiment analyst with access to real-time X/Twitter data. "
                            "Always respond with valid JSON only, no markdown code blocks, no extra text."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            # モデルが存在しない場合は次を試す
            if "model" in err.lower() or "not found" in err.lower() or "does not exist" in err.lower():
                continue
            # それ以外はそのまま raise
            raise
    raise RuntimeError(f"All models failed: {models_to_try}")


def get_twitter_sentiment(ticker: str) -> dict:
    """X/Twitterのリアルタイムセンチメント分析"""
    cache_key = f"sentiment:{ticker}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    prompt = f"""Search X/Twitter right now for posts about stock ticker ${ticker}.
Analyze the last 1-2 hours of posts and return a JSON object with these exact fields:

{{
  "ticker": "{ticker}",
  "sentiment_score": <integer from -100 (very bearish) to +100 (very bullish)>,
  "trending_keywords": [<list of 5-8 trending words/phrases related to {ticker} on X>],
  "notable_posts": [
    {{"text": "<post text>", "time": "<approximate time e.g. '30 minutes ago'>"}},
    {{"text": "<post text>", "time": "<approximate time>"}},
    {{"text": "<post text>", "time": "<approximate time>"}}
  ],
  "catalyst_detected": <true or false>,
  "catalyst_type": "<one of: earnings_beat, earnings_miss, fda_approval, fda_reject, merger_acquisition, analyst_upgrade, analyst_downgrade, guidance_raised, guidance_lowered, none>",
  "velocity": "<one of: 急増, 安定, 急減>",
  "summary": "<1-2 sentence Japanese summary of current X sentiment for {ticker}>"
}}

Base this on actual real-time X/Twitter data. Return only valid JSON."""

    try:
        raw = _call_grok(prompt)
        # JSONブロックの抽出（念のため ```json ... ``` を除去）
        raw = _extract_json(raw)
        data = json.loads(raw)
        # 必須フィールドの検証・デフォルト補完
        data.setdefault("ticker", ticker)
        data.setdefault("sentiment_score", 0)
        data.setdefault("trending_keywords", [])
        data.setdefault("notable_posts", [])
        data.setdefault("catalyst_detected", False)
        data.setdefault("catalyst_type", "none")
        data.setdefault("velocity", "安定")
        data.setdefault("summary", "データ取得中")
        _cache_set(cache_key, data)
        return data
    except Exception as e:
        # フォールバック: モックデータを返す（アプリをクラッシュさせない）
        fallback = {
            "ticker": ticker,
            "sentiment_score": 0,
            "trending_keywords": [],
            "notable_posts": [],
            "catalyst_detected": False,
            "catalyst_type": "none",
            "velocity": "安定",
            "summary": f"センチメントデータの取得に失敗しました: {str(e)[:100]}",
            "error": str(e)
        }
        return fallback


def scan_breaking_catalysts() -> list:
    """市場全体の速報触媒をスキャン（過去1時間）"""
    cache_key = "breaking_catalysts"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    prompt = """Search X/Twitter right now for breaking financial catalyst events in the last 1 hour.
Look for these keywords on X: "earnings beat", "earnings miss", "FDA approval", "FDA reject",
"merger", "acquisition", "analyst upgrade", "analyst downgrade", "guidance raised",
"guidance lowered", "record revenue", "buyback", "dividend".

Return a JSON array of up to 10 catalyst events found. Each element:
{
  "ticker": "<stock ticker symbol, e.g. AAPL>",
  "catalyst_type": "<one of: earnings_beat, earnings_miss, fda_approval, fda_reject, merger_acquisition, analyst_upgrade, analyst_downgrade, guidance_raised, guidance_lowered, other>",
  "summary": "<1 sentence English summary of the catalyst>",
  "urgency": "<one of: high, medium, low>",
  "time": "<approximate time e.g. '15 minutes ago'>"
}

Return only a valid JSON array. If no catalysts found, return [].
Base this on actual real-time X/Twitter data."""

    try:
        raw = _call_grok(prompt)
        raw = _extract_json(raw)
        data = json.loads(raw)
        if not isinstance(data, list):
            data = []
        _cache_set(cache_key, data)
        return data
    except Exception as e:
        return [{"error": str(e), "ticker": "N/A", "catalyst_type": "none",
                 "summary": f"データ取得失敗: {str(e)[:100]}", "urgency": "low", "time": "unknown"}]


def get_trending_stocks() -> list:
    """X/Twitterでトレンド中の銘柄トップ10"""
    cache_key = "trending_stocks"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    prompt = """Search X/Twitter right now for the most discussed stock tickers in the last 1-2 hours.
Identify the top 10 most mentioned stock symbols (e.g. $AAPL, $TSLA, $NVDA etc.) on X/Twitter.

Return a JSON array of exactly 10 elements sorted by mentions descending:
{
  "ticker": "<stock ticker symbol without $>",
  "mentions": <estimated number of mentions in last hour as integer>,
  "sentiment": <integer -100 to +100 reflecting overall X sentiment>,
  "reason": "<brief English reason why this stock is trending on X>"
}

Return only a valid JSON array based on real-time X/Twitter data."""

    try:
        raw = _call_grok(prompt)
        raw = _extract_json(raw)
        data = json.loads(raw)
        if not isinstance(data, list):
            data = []
        _cache_set(cache_key, data)
        return data
    except Exception as e:
        return [{"ticker": "N/A", "mentions": 0, "sentiment": 0,
                 "reason": f"データ取得失敗: {str(e)[:100]}", "error": str(e)}]
