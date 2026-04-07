from __future__ import annotations

import json
from langchain_core.tools import tool

from data.fetcher import get_stock_data, get_live_quote, get_nifty50_stocks, get_stock_info
from analysis.technical import generate_analysis_summary


@tool
def fetch_stock_data(symbol: str, period: str = "6mo") -> str:
    """Fetch recent historical price data for an Indian stock.

    Args:
        symbol: NSE stock symbol, e.g. RELIANCE, TCS, INFY
        period: How far back to look — 1mo, 3mo, 6mo, 1y, 2y, 5y
    """
    try:
        df = get_stock_data(symbol, period=period)
        recent = df.tail(5)
        rows = []
        for idx, row in recent.iterrows():
            date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)
            rows.append(
                f"{date_str}: Open={row['Open']:.2f} High={row['High']:.2f} "
                f"Low={row['Low']:.2f} Close={row['Close']:.2f} Vol={int(row['Volume']):,}"
            )
        return f"Last 5 trading days for {symbol}:\n" + "\n".join(rows)
    except Exception as e:
        return f"Error fetching data for {symbol}: {e}"


@tool
def analyze_technicals(symbol: str, period: str = "6mo") -> str:
    """Run full technical analysis on an Indian stock and return signals.

    Args:
        symbol: NSE stock symbol, e.g. RELIANCE, TCS, INFY
        period: How far back to look — 1mo, 3mo, 6mo, 1y, 2y
    """
    try:
        df = get_stock_data(symbol, period=period)
        summary = generate_analysis_summary(df, symbol)
        lines = [
            f"=== Technical Analysis: {summary['symbol']} ===",
            f"Price: ₹{summary['price']}  Change: {summary['change']} ({summary['pct_change']}%)",
            f"RSI (14): {summary['rsi']}",
            f"MACD: {summary['macd']}",
            f"SMA Trend: {summary['sma_trend']}",
            f"Bollinger Bands: {summary['bollinger']}",
            f"Support: ₹{summary['support']}  |  Resistance: ₹{summary['resistance']}",
            f"Overall Signal: {summary['overall_signal']}",
        ]
        if summary["crossovers"]:
            lines.append("Recent Crossovers:")
            for c in summary["crossovers"]:
                lines.append(f"  {c['date']}: {c['type']} — {c['detail']}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error analyzing {symbol}: {e}"


@tool
def compare_stocks(symbols: str) -> str:
    """Compare technical signals across multiple Indian stocks.

    Args:
        symbols: Comma-separated NSE stock symbols, e.g. "TCS,INFY,WIPRO"
    """
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        results = []
        for sym in symbol_list:
            df = get_stock_data(sym, period="6mo")
            summary = generate_analysis_summary(df, sym)
            results.append(
                f"{summary['symbol']}: ₹{summary['price']} | RSI: {summary['rsi']} | "
                f"MACD: {summary['macd']} | Trend: {summary['sma_trend']} | "
                f"Signal: {summary['overall_signal']}"
            )
        return "=== Stock Comparison ===\n" + "\n".join(results)
    except Exception as e:
        return f"Error comparing stocks: {e}"


@tool
def get_market_overview() -> str:
    """Get an overview of the Indian stock market — Nifty 50 constituents list and general info."""
    stocks = get_nifty50_stocks()
    return (
        "Nifty 50 Constituents:\n"
        + ", ".join(stocks)
        + "\n\nUse analyze_technicals on any symbol above for detailed analysis."
    )


@tool
def get_stock_details(symbol: str) -> str:
    """Get basic information about a stock — name, sector, market cap, P/E ratio.

    Args:
        symbol: NSE stock symbol, e.g. RELIANCE, TCS
    """
    try:
        info = get_stock_info(symbol)
        lines = [
            f"=== {info['name']} ({info['symbol']}) ===",
            f"Sector: {info['sector']}",
            f"Industry: {info['industry']}",
            f"Market Cap: ₹{info['market_cap']:,.0f}" if info['market_cap'] else "Market Cap: N/A",
            f"P/E Ratio: {info['pe_ratio']:.2f}" if info['pe_ratio'] else "P/E Ratio: N/A",
            f"52-Week High: ₹{info['52w_high']}" if info['52w_high'] else "52W High: N/A",
            f"52-Week Low: ₹{info['52w_low']}" if info['52w_low'] else "52W Low: N/A",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching info for {symbol}: {e}"


ALL_TOOLS = [
    fetch_stock_data,
    analyze_technicals,
    compare_stocks,
    get_market_overview,
    get_stock_details,
]
