# Indian Stock Market AI Agent

A Python-based AI agent that performs **technical analysis** on Indian stock market (NSE) data using a **local LLM** via Ollama. Accessible through both a CLI and a Streamlit web dashboard.

> **Disclaimer**: This tool is for educational and personal analysis purposes only. It does not constitute financial advice.

## Features

- **Technical Analysis**: RSI, MACD, SMA, EMA, Bollinger Bands, crossover detection, support/resistance
- **Interactive Charts**: Plotly candlestick charts with indicator overlays
- **AI Chat**: Ask natural-language questions about stocks and get analysis-backed answers
- **CLI + Web UI**: Use from the terminal or a browser-based dashboard
- **NSE India Data**: Fetches data directly from NSE India — no third-party finance APIs
- **100% Local & Free**: Runs Ollama locally — no API keys, no cloud, no cost

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — install from [ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull llama3
```

## Setup

```bash
cd indian-stock-agent
python3 -m pip install -r requirements.txt
```

Optionally edit `.env` to change the model or Ollama URL.

## Usage

### CLI

```bash
# Analyze a stock
python cli.py analyze RELIANCE

# Ask the agent a question
python cli.py ask "Is TCS showing bullish signals?"

# Compare multiple stocks
python cli.py compare TCS INFY WIPRO

# Market overview
python cli.py market
```

### Web Dashboard

```bash
streamlit run app.py
```

Opens a browser with a chat interface + interactive charts.

## Project Structure

```
indian-stock-agent/
├── requirements.txt          # Dependencies
├── config.py                 # Configuration
├── cli.py                    # CLI entry point
├── app.py                    # Streamlit web dashboard
├── agents/
│   ├── stock_agent.py        # LangChain agent with Ollama
│   └── tools.py              # Agent tool definitions
├── data/
│   └── fetcher.py            # Stock data fetching (NSE India APIs)
├── analysis/
│   └── technical.py          # Technical indicator computation
└── charts/
    └── plotter.py            # Plotly chart generation
```
