# Stock Analysis & Paper Trading System

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

An AI-powered stock analysis web application featuring a three-layer signal engine that integrates technical analysis, fundamental scoring (CAN SLIM + SEPA), and real-time social sentiment via Grok AI. Includes a full paper trading journal with USD/JPY conversion, multi-mode stock scanner, backtesting engine, and price alerts — all accessible through a 6-page web UI.

## Features

- **3-Layer AI Signal Engine** — Weighted composite score: Technical 20% + Fundamental 40% + Sentiment 40%
- **Real-time stock analysis** — BUY / SELL / HOLD signals (STRONG_BUY → STRONG_SELL) with entry range, stop-loss, and target price
- **CAN SLIM + SEPA scoring** — Fundamental grading using William O'Neil's CAN SLIM method and Mark Minervini's SEPA criteria
- **Sentiment analysis** — Powered by Grok AI (xAI), scanning X/Twitter for catalysts and velocity
- **Paper Trading Journal** — ¥100,000 virtual capital, open/close trades, daily P&L tracking, USD/JPY conversion
- **Position size calculator** — 2% risk rule based on entry price and stop-loss distance
- **Stock Scanner** — Momentum scan, pre-market scan, top movers, and AI opportunity scan
- **Sector rotation analysis** — Track fund flow across 11 GICS sectors
- **Backtesting Engine** — MA crossover and RSI strategies with equity curve visualization
- **Price Alerts** — Set above/below triggers with auto-check endpoint
- **6-page Web UI** — Analysis, Scanner, Journal, Guide, Status (systemd health)
- **Live demo**: https://libra.lunasoph.com

## Screenshots

<!-- Screenshots -->

## Tech Stack

| Category | Technology |
|----------|------------|
| Backend | Python 3, Flask 3.x |
| Market Data | yfinance |
| AI / Analysis | Grok AI (xAI), Custom 3-layer signal engine |
| Fundamental | CAN SLIM, SEPA (Minervini criteria) |
| Technical Indicators | RSI, MACD, Bollinger Bands, Ichimoku, ATR, VWAP, Stochastic |
| Frontend | HTML5, CSS3, JavaScript |
| Deployment | AWS EC2, systemd, nginx |
| Scheduler | schedule (Python) |

## Installation

```bash
git clone https://github.com/ktaketanijp/stock-chart.git
cd stock-chart
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python app.py
```

The app runs on `http://localhost:5000` by default.

## Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```
GROK_API_KEY=      # xAI Grok — required for sentiment analysis
GROQ_API_KEY=      # Groq — fast LLM inference (optional fallback)
GEMINI_API_KEY=    # Google Gemini (optional)
OPENAI_API_KEY=    # OpenAI (optional)
```

## API Endpoints

### Chart & Signals

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/chart` | OHLCV candles + all technical indicators (`?ticker=AAPL&period=3mo`) |
| GET | `/api/signal` | Basic technical signal with score and reasons |
| GET | `/api/signal/advanced` | 3-layer AI signal (Technical + Fundamental + Sentiment) |

### Fundamentals

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/fundamentals` | Raw fundamental data (P/E, EPS growth, revenue, etc.) |
| GET | `/api/canslim` | CAN SLIM score and grade |
| GET | `/api/sepa` | SEPA (Minervini) conditions check |
| GET | `/api/earnings/calendar` | Upcoming earnings calendar |
| GET | `/api/news` | News headlines with AI summary |
| GET | `/api/sector/rotation` | Sector ETF performance comparison |

### Scanner

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/scan/momentum` | Momentum-based stock scan |
| GET | `/api/scan/premarket` | Pre-market movers scan |
| GET | `/api/scan/top-movers` | Top gainers and losers |
| GET | `/api/scan/opportunities` | AI-scored BUY/STRONG_BUY opportunities (cached 30 min) |

### Sentiment

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sentiment` | X/Twitter sentiment score via Grok AI (`?ticker=AAPL`) |
| GET | `/api/news/breaking` | Breaking catalyst scan |
| GET | `/api/trending` | Trending tickers |

### Paper Trading Journal

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/journal` | Portfolio positions and performance summary |
| POST | `/api/journal/open` | Open a new paper trade |
| POST | `/api/journal/close` | Close an existing position |
| POST | `/api/journal/size` | Calculate position size (2% risk rule) |
| GET | `/api/journal/check-stops` | Check if any positions hit stop-loss |
| POST | `/api/journal/daily-pnl` | Record daily P&L snapshot |

### Alerts & Backtest

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/alerts` | List all price alerts |
| POST | `/api/alerts` | Create a new price alert |
| DELETE | `/api/alerts/<id>` | Delete an alert |
| GET | `/api/alerts/check` | Check and trigger alerts |
| POST | `/api/backtest` | Run backtest (MA crossover or RSI strategy) |

### Search & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/search` | Ticker search (`?q=AAPL`) |
| GET | `/api/status` | System health and scheduler status |

## Project Structure

```
stock-chart/
├── app.py              # Flask main app, all route definitions
├── signal_engine.py    # 3-layer AI signal engine (Technical / Fundamental / Sentiment)
├── fundamentals.py     # CAN SLIM scoring, SEPA scoring, earnings calendar
├── indicators.py       # Bollinger Bands, Ichimoku, ATR, Stochastic, VWAP, patterns
├── sentiment.py        # Grok AI sentiment analysis, catalyst detection
├── scanner.py          # Momentum, pre-market, top movers scan
├── paper_trading.py    # Paper trade management, USD/JPY conversion, 2% risk rule
├── scheduler.py        # Periodic background scheduler
├── update_status.py    # System status builder
├── requirements.txt
├── .env.example
└── templates/          # Jinja2 HTML templates
    ├── index.html      # Chart viewer (home)
    ├── analysis.html   # Full AI analysis page
    ├── scanner.html    # Stock scanner
    ├── journal.html    # Paper trading journal
    ├── guide.html      # User guide
    └── status.html     # System status
```

## License

MIT License
