#!/usr/bin/env python3
"""CLI interface for the Indian Stock Market AI Agent."""
from __future__ import annotations

import sys
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from data.fetcher import get_stock_data, get_nifty50_stocks
from analysis.technical import generate_analysis_summary
from charts.plotter import create_candlestick_chart

app = typer.Typer(
    name="indian-stock-agent",
    help="AI-powered Indian stock market technical analysis agent.",
)
console = Console()


def _print_analysis(summary: dict) -> None:
    """Pretty-print an analysis summary using Rich."""
    signal_color = {
        "Bullish": "green",
        "Bearish": "red",
        "Neutral": "yellow",
    }.get(summary["overall_signal"], "white")

    table = Table(title=f"Technical Analysis — {summary['symbol']}", show_header=False, border_style="cyan")
    table.add_column("Indicator", style="bold")
    table.add_column("Value")

    table.add_row("Price", f"₹{summary['price']}")
    table.add_row("Change", f"{summary['change']} ({summary['pct_change']}%)")
    table.add_row("RSI (14)", summary["rsi"])
    table.add_row("MACD", summary["macd"])
    table.add_row("SMA Trend", summary["sma_trend"])
    table.add_row("Bollinger Bands", summary["bollinger"])
    table.add_row("Support", f"₹{summary['support']}")
    table.add_row("Resistance", f"₹{summary['resistance']}")
    table.add_row("Overall Signal", f"[bold {signal_color}]{summary['overall_signal']}[/]")

    console.print(table)

    if summary["crossovers"]:
        console.print("\n[bold]Recent Crossovers:[/]")
        for c in summary["crossovers"]:
            icon = "[green]▲[/]" if c["type"] == "Golden Cross" else "[red]▼[/]"
            console.print(f"  {icon} {c['date']}: {c['type']} — {c['detail']}")


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="NSE stock symbol, e.g. RELIANCE, TCS"),
    period: str = typer.Option("6mo", help="Data period: 1mo, 3mo, 6mo, 1y, 2y, 5y"),
    chart: bool = typer.Option(False, "--chart", "-c", help="Open an interactive chart in the browser"),
):
    """Run technical analysis on a stock."""
    console.print(f"\n[cyan]Fetching data for {symbol.upper()}...[/]")
    try:
        df = get_stock_data(symbol, period=period)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)

    summary = generate_analysis_summary(df, symbol)
    _print_analysis(summary)

    if chart:
        console.print("\n[cyan]Opening chart in browser...[/]")
        fig = create_candlestick_chart(df, symbol.upper())
        fig.show()

    console.print(
        "\n[dim]Disclaimer: This analysis is for educational purposes only "
        "and does not constitute financial advice.[/]\n"
    )


@app.command()
def compare(
    symbols: list[str] = typer.Argument(..., help="Stock symbols to compare, e.g. TCS INFY WIPRO"),
    period: str = typer.Option("6mo", help="Data period"),
):
    """Compare technical signals across multiple stocks."""
    table = Table(title="Stock Comparison", border_style="cyan")
    table.add_column("Symbol", style="bold")
    table.add_column("Price (₹)")
    table.add_column("RSI")
    table.add_column("MACD")
    table.add_column("Trend")
    table.add_column("Signal", justify="center")

    for sym in symbols:
        try:
            df = get_stock_data(sym, period=period)
            s = generate_analysis_summary(df, sym)
            color = {"Bullish": "green", "Bearish": "red", "Neutral": "yellow"}.get(s["overall_signal"], "white")
            table.add_row(
                s["symbol"], str(s["price"]), s["rsi"], s["macd"],
                s["sma_trend"], f"[{color}]{s['overall_signal']}[/]",
            )
        except Exception as e:
            table.add_row(sym.upper(), "[red]Error[/]", "-", "-", "-", str(e))

    console.print(table)
    console.print("\n[dim]Disclaimer: For educational purposes only.[/]\n")


@app.command()
def market():
    """Show Nifty 50 stock list."""
    stocks = get_nifty50_stocks()
    console.print(Panel(", ".join(stocks), title="Nifty 50 Constituents", border_style="cyan"))
    console.print("[dim]Use [bold]analyze <SYMBOL>[/bold] for detailed analysis.[/]\n")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question for the AI agent"),
):
    """Ask the AI agent a question about Indian stocks."""
    console.print(f"\n[cyan]Thinking...[/]\n")
    try:
        from agents.stock_agent import create_agent, chat
        agent = create_agent()
        response = chat(agent, question)
        console.print(Markdown(response))
    except Exception as e:
        console.print(f"[red]Agent error: {e}[/]")
        console.print("[dim]Make sure Ollama is running (ollama serve) with the correct model pulled.[/]")
    console.print()


@app.command()
def chat_mode():
    """Start an interactive chat session with the AI agent."""
    console.print(Panel(
        "[bold cyan]Indian Stock Market AI Agent[/]\n"
        "Type your questions about Indian stocks. Type [bold]quit[/bold] to exit.",
        border_style="cyan",
    ))

    try:
        from agents.stock_agent import create_agent, chat
        agent = create_agent()
    except Exception as e:
        console.print(f"[red]Failed to start agent: {e}[/]")
        console.print("[dim]Make sure Ollama is running.[/]")
        raise typer.Exit(1)

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/] ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        if not user_input.strip():
            continue

        console.print("[bold cyan]Agent:[/] ", end="")
        try:
            response = chat(agent, user_input)
            console.print(Markdown(response))
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")

    console.print("\n[dim]Goodbye![/]\n")


if __name__ == "__main__":
    app()
