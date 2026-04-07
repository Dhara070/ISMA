from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.technical import add_all_indicators


def create_candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    show_sma: bool = True,
    show_ema: bool = False,
    show_bollinger: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
    show_volume: bool = True,
) -> go.Figure:
    """Build an interactive Plotly candlestick chart with optional indicator overlays."""
    df = add_all_indicators(df.copy())

    row_heights = [0.50]
    rows = 1
    subplot_titles = [f"{symbol} Price"]

    if show_volume:
        rows += 1
        row_heights.append(0.14)
        subplot_titles.append("Volume")
    if show_rsi:
        rows += 1
        row_heights.append(0.14)
        subplot_titles.append("RSI (14)")
    if show_macd:
        rows += 1
        row_heights.append(0.18)
        subplot_titles.append("MACD")

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Price candles
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#22c55e",
            increasing_fillcolor="#22c55e",
            decreasing_line_color="#ef4444",
            decreasing_fillcolor="#ef4444",
        ),
        row=1,
        col=1,
    )

    # SMA overlays
    if show_sma:
        sma_styles = [
            ("SMA_20", "#60a5fa", 1.0, None),
            ("SMA_50", "#f59e0b", 1.2, None),
            ("SMA_200", "#10b981", 1.5, None),
        ]
        for col, color, width, dash in sma_styles:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], name=col,
                               line=dict(width=width, color=color, dash=dash)),
                    row=1, col=1,
                )

    # EMA overlays
    if show_ema:
        for col, color in [("EMA_9", "#a78bfa"), ("EMA_21", "#f472b6")]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], name=col,
                               line=dict(width=1.2, color=color, dash="dot")),
                    row=1, col=1,
                )

    # Bollinger Bands
    if show_bollinger:
        bbu, bbl, bbm = "BBU_20_2.0", "BBL_20_2.0", "BBM_20_2.0"
        if bbu in df.columns and bbl in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[bbu], name="BB Upper",
                           line=dict(width=0.8, color="rgba(147,197,253,0.5)")),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df[bbl], name="BB Lower",
                           line=dict(width=0.8, color="rgba(147,197,253,0.5)"),
                           fill="tonexty", fillcolor="rgba(147,197,253,0.08)"),
                row=1, col=1,
            )
        if bbm in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[bbm], name="BB Mid",
                           line=dict(width=0.7, color="rgba(147,197,253,0.6)", dash="dot")),
                row=1, col=1,
            )

    current_row = 2

    # Volume bars
    if show_volume:
        colors = ["#ef4444" if c < o else "#22c55e" for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume",
                   marker_color=colors, opacity=0.7, showlegend=False),
            row=current_row, col=1,
        )
        current_row += 1

    # RSI
    if show_rsi and "RSI_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI_14"], name="RSI 14",
                       line=dict(color="#a855f7", width=1.5)),
            row=current_row, col=1,
        )
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.08)",
                      line_width=0, row=current_row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.08)",
                      line_width=0, row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                      line_width=0.8, opacity=0.6, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e",
                      line_width=0.8, opacity=0.6, row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#94a3b8",
                      line_width=0.5, opacity=0.4, row=current_row, col=1)
        current_row += 1

    # MACD
    if show_macd:
        macd_col = "MACD_12_26_9"
        signal_col = "MACDs_12_26_9"
        hist_col = "MACDh_12_26_9"
        if macd_col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[macd_col], name="MACD",
                           line=dict(color="#3b82f6", width=1.5)),
                row=current_row, col=1,
            )
        if signal_col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[signal_col], name="Signal",
                           line=dict(color="#f59e0b", width=1.5)),
                row=current_row, col=1,
            )
        if hist_col in df.columns:
            hist_colors = ["#ef4444" if v < 0 else "#22c55e" for v in df[hist_col]]
            fig.add_trace(
                go.Bar(x=df.index, y=df[hist_col], name="Histogram",
                       marker_color=hist_colors, opacity=0.6, showlegend=False),
                row=current_row, col=1,
            )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=180 + rows * 200,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafafa",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=20, t=50, b=30),
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="#f1f5f9"),
    )

    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor="#f1f5f9", row=i, col=1)
        fig.update_yaxes(gridcolor="#f1f5f9", row=i, col=1)

    return fig
