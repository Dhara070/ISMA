"""Streamlit web dashboard for the Indian Stock Market AI Agent."""
from __future__ import annotations

import streamlit as st

from config import NIFTY50_SYMBOLS
from data.fetcher import (
    get_stock_data, get_fundamentals, get_stock_news,
    get_index_data, get_nifty_top_movers,
)
from analysis.technical import (
    generate_analysis_summary,
    generate_recommendation,
    generate_fno_advisory,
)
from charts.plotter import create_candlestick_chart


@st.cache_data(ttl=180, show_spinner="Loading top movers…")
def _cached_nifty_top_movers() -> dict:
    return get_nifty_top_movers(max_symbols=6)


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock AI Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stCheckbox label {
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] hr { border-color: #334155 !important; }

/* ── KPI cards ───────────────────────────── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}
.kpi-delta {
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 4px;
}
.kpi-delta.positive { color: #16a34a; }
.kpi-delta.negative { color: #dc2626; }

/* ── Section cards ───────────────────────── */
.section-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Action badge ────────────────────────── */
.action-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 12px 24px;
    border-radius: 10px;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 0.02em;
}
.action-meta {
    font-size: 0.85rem;
    color: #64748b;
    margin-left: 12px;
    font-weight: 500;
}

/* ── Signal pill ─────────────────────────── */
.signal-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px 2px;
}
.pill-bullish { background: #dcfce7; color: #166534; }
.pill-bearish { background: #fee2e2; color: #991b1b; }
.pill-neutral { background: #f1f5f9; color: #475569; }

/* ── Trade card ──────────────────────────── */
.trade-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.9rem;
}
.trade-row:last-child { border-bottom: none; }
.trade-label { color: #64748b; font-weight: 500; }
.trade-value { color: #0f172a; font-weight: 700; }

/* ── News card ───────────────────────────── */
.news-item {
    padding: 10px 14px;
    border-left: 3px solid #3b82f6;
    background: #f8fafc;
    border-radius: 0 8px 8px 0;
    margin-bottom: 8px;
}
.news-item .title { font-weight: 600; font-size: 0.88rem; color: #0f172a; }
.news-item .meta { font-size: 0.75rem; color: #94a3b8; margin-top: 2px; }

/* ── Gauge bar ───────────────────────────── */
.gauge-container {
    background: #f1f5f9;
    border-radius: 8px;
    height: 14px;
    overflow: hidden;
    position: relative;
    margin: 8px 0;
}
.gauge-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    margin-top: 4px;
}

/* ── Misc ────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    padding: 10px 24px;
    font-weight: 600;
    border-radius: 8px 8px 0 0;
}
div[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

/* ── Period pill selector (TradingView-style) ── */
.period-bar [role="radiogroup"] {
    display: flex !important;
    gap: 0 !important;
    background: #f1f5f9;
    border-radius: 8px;
    padding: 3px;
    width: fit-content !important;
}
.period-bar [role="radiogroup"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 5px 14px !important;
    border-radius: 6px !important;
    border: none !important;
    cursor: pointer;
    min-height: unset !important;
    transition: all 0.15s;
    color: #64748b !important;
    background: transparent !important;
}
.period-bar [role="radiogroup"] label[data-checked="true"],
.period-bar [role="radiogroup"] label:has(input:checked) {
    background: #ffffff !important;
    color: #0f172a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}
.period-bar [role="radiogroup"] label span { padding: 0 !important; }
.period-bar [data-testid="stWidgetLabel"] { display: none !important; }
.period-bar { margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_error" not in st.session_state:
    st.session_state.agent_error = None
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 Stock Analyzer")
    st.caption("NSE / BSE Technical Analysis & F&O Advisory")
    st.markdown("---")

    selected_symbol = st.selectbox(
        "Stock Symbol",
        options=[""] + sorted(NIFTY50_SYMBOLS),
        index=0,
        help="Select a Nifty 50 stock or type a custom symbol below",
    )
    custom_symbol = st.text_input("Or enter custom symbol", placeholder="e.g. TATAMOTORS")
    symbol = (custom_symbol.strip().upper() or selected_symbol).strip()

    period = st.radio(
        "Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2, horizontal=True,
    )

    st.markdown("---")
    st.markdown("**Chart Overlays**")
    show_sma = st.checkbox("SMA (20, 50, 200)", value=True)
    show_ema = st.checkbox("EMA (9, 21)", value=False)
    show_bollinger = st.checkbox("Bollinger Bands", value=True)
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_volume = st.checkbox("Volume", value=True)

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.session_state.refresh_counter += 1
        st.rerun()

    st.markdown("---")
    st.caption(
        "**Disclaimer:** For educational and personal analysis only. "
        "Not financial advice. Consult a SEBI-registered advisor."
    )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _kpi_card(label: str, value: str, delta: str = "", delta_positive: bool | None = None):
    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else "negative" if delta_positive is False else ""
        arrow = "▲" if delta_positive else "▼" if delta_positive is False else ""
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""


def _signal_pill(text: str, signal: str):
    cls = {"bullish": "pill-bullish", "bearish": "pill-bearish"}.get(signal, "pill-neutral")
    dot = {"bullish": "🟢", "bearish": "🔴"}.get(signal, "⚪")
    return f'<span class="signal-pill {cls}">{dot} {text}</span>'


def _gauge_bar(bullish: int, bearish: int):
    total = bullish + bearish or 1
    pct = int((bullish / total) * 100)
    if pct > 60:
        color = "#22c55e"
    elif pct < 40:
        color = "#ef4444"
    else:
        color = "#eab308"
    return f"""
    <div class="gauge-container">
        <div class="gauge-fill" style="width:{pct}%;background:{color};"></div>
    </div>
    <div class="gauge-labels">
        <span>Bearish ({bearish} pts)</span>
        <span>Bullish ({bullish} pts)</span>
    </div>"""


def _trade_row(label: str, value: str):
    return f'<div class="trade-row"><span class="trade-label">{label}</span><span class="trade-value">{value}</span></div>'


def _format_mcap(mc):
    if mc is None:
        return "N/A"
    if mc >= 1e12:
        return f"₹{mc/1e12:.2f}L Cr"
    if mc >= 1e7:
        return f"₹{mc/1e7:,.0f} Cr"
    return f"₹{mc:,.0f}"


# ──────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────
st.markdown("## Indian Stock Market AI Agent")

tab_nifty, tab_dashboard, tab_chat = st.tabs(["🏛️ Nifty Market & F&O", "📊 Stock Dashboard", "💬 AI Chat"])

# ──────────────────────────────────────────────
# Tab 1: Nifty Market & F&O
# ──────────────────────────────────────────────
with tab_nifty:
    _ = st.session_state.refresh_counter  # force re-fetch on refresh

    with st.container():
        st.markdown('<div class="period-bar">', unsafe_allow_html=True)
        nifty_period = st.radio(
            "Period", ["1mo", "3mo", "6mo", "1y"],
            index=2, horizontal=True, key="nifty_period", label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Loading Nifty market data..."):
        try:
            nifty_df = get_index_data("NIFTY 50", period=nifty_period)
            nifty_summary = generate_analysis_summary(nifty_df, "NIFTY 50")
        except Exception as e:
            st.error(f"Could not fetch Nifty 50 data: {e}")
            st.stop()

        _nifty_src = nifty_df.attrs.get("data_source", "")
        if "fallback" in _nifty_src.lower():
            st.caption(f"⚠️ Data source: {_nifty_src} — NSE India blocked this server's IP. Prices may differ slightly.")

        nifty_fno = generate_fno_advisory(nifty_summary)

        snapshot: dict = {}
        try:
            # Bank Nifty
            bn_df = get_index_data("BANK NIFTY", period="1mo")
            bn_lat, bn_prv = bn_df.iloc[-1], bn_df.iloc[-2] if len(bn_df) > 1 else bn_df.iloc[-1]
            snapshot["banknifty"] = {
                "close": round(float(bn_lat["Close"]), 2),
                "change": round(float(bn_lat["Close"] - bn_prv["Close"]), 2),
                "pct_change": round(((bn_lat["Close"] - bn_prv["Close"]) / bn_prv["Close"]) * 100, 2),
            }
        except Exception:
            snapshot["banknifty"] = None
        try:
            vix_df = get_index_data("INDIA VIX", period="1mo")
            vix_lat = vix_df.iloc[-1]
            vix_prv = vix_df.iloc[-2] if len(vix_df) > 1 else vix_lat
            snapshot["vix"] = {
                "value": round(float(vix_lat["Close"]), 2),
                "change": round(float(vix_lat["Close"] - vix_prv["Close"]), 2),
            }
        except Exception:
            snapshot["vix"] = None

    # ── Header KPI row ─────────────────────────
    st.markdown(
        '<div style="margin-bottom:4px;">'
        '<span style="font-size:1.5rem;font-weight:800;color:#0f172a;">NIFTY 50 Index</span>'
        '<span style="font-size:0.95rem;color:#64748b;margin-left:10px;">NSE · Market Overview & F&O Advisory</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    n_delta = f"{nifty_summary['change']} ({nifty_summary['pct_change']}%)"
    n_pos = nifty_summary['change'] >= 0
    sig_map = {"Bullish": ("🟢 Bullish", True), "Bearish": ("🔴 Bearish", False), "Neutral": ("🟡 Neutral", None)}
    sig_t, _ = sig_map.get(nifty_summary['overall_signal'], ("🟡 Neutral", None))

    nk1, nk2, nk3, nk4, nk5, nk6 = st.columns(6)
    nk1.markdown(_kpi_card("Nifty 50", f"{nifty_summary['price']:,.2f}", n_delta, n_pos), unsafe_allow_html=True)

    bn = snapshot.get("banknifty")
    with nk2:
        if bn:
            bn_dstr = f"{bn['change']:+,.2f} ({bn['pct_change']:+.2f}%)"
            st.markdown(
                _kpi_card("Bank Nifty", f"{bn['close']:,.2f}", bn_dstr, bn["change"] >= 0),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(_kpi_card("Bank Nifty", "N/A"), unsafe_allow_html=True)

    vix = snapshot.get("vix")
    if vix:
        vix_d = f"{vix['change']:+.2f}"
        nk3.markdown(_kpi_card("India VIX", f"{vix['value']:.2f}", vix_d, vix['change'] <= 0), unsafe_allow_html=True)
    else:
        nk3.markdown(_kpi_card("India VIX", "N/A"), unsafe_allow_html=True)

    nk4.markdown(_kpi_card("Support", f"{nifty_summary['support']:,.2f}"), unsafe_allow_html=True)
    nk5.markdown(_kpi_card("Resistance", f"{nifty_summary['resistance']:,.2f}"), unsafe_allow_html=True)
    nk6.markdown(_kpi_card("Signal", sig_t), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Nifty Chart ────────────────────────────
    nifty_fig = create_candlestick_chart(
        nifty_df, "NIFTY 50",
        show_sma=True, show_ema=True, show_bollinger=True,
        show_rsi=True, show_macd=True, show_volume=True,
    )
    st.plotly_chart(nifty_fig, use_container_width=True)

    # ── Nifty F&O Advisory ─────────────────────
    nfno_col, ndet_col = st.columns(2)

    with nfno_col:
        direction = nifty_fno["direction"]
        dir_colors = {
            "Buy Call (CE)": ("#059669", "#ecfdf5"),
            "Buy Put (PE)": ("#dc2626", "#fef2f2"),
            "Neutral / Avoid": ("#ca8a04", "#fefce8"),
        }
        dc, dbg = dir_colors.get(direction, ("#64748b", "#f8fafc"))
        dir_icon = {"Buy Call (CE)": "📈", "Buy Put (PE)": "📉"}.get(direction, "⚖️")
        sg = nifty_fno["strike_guidance"]

        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">🎯 Nifty F&O Advisory</div>
            <div class="action-badge" style="background:{dbg};color:{dc};border:2px solid {dc};">
                {dir_icon} {direction}
            </div>
            <span class="action-meta">Strategy: {nifty_fno['strategy']}</span>
            <div style="margin-top:16px;">
                {_gauge_bar(nifty_fno['bullish_points'], nifty_fno['bearish_points'])}
            </div>
            <div style="margin-top:14px;">
                {_trade_row("ATM Strike", f"{sg['atm_strike']:,.0f}")}
                {_trade_row("Target", f"{sg['target']:,.2f}" if sg.get('target') else "—")}
                {_trade_row("Stop Loss", f"{sg['stoploss']:,.2f}" if sg.get('stoploss') else "—")}
                {_trade_row("Expected Move (2×ATR)", f"±{sg['expected_move']:,.2f}" if sg.get('expected_move') else "—")}
                {_trade_row("Support / Resistance", f"{nifty_summary['support']:,.2f} / {nifty_summary['resistance']:,.2f}")}
            </div>
        </div>""", unsafe_allow_html=True)

        with st.expander("📋 F&O Rationale & Risks", expanded=False):
            for r in nifty_fno["rationale"]:
                st.markdown(f"- {r}")
            st.markdown("---")
            for note in nifty_fno["risk_notes"]:
                st.caption(f"⚠️ {note}")

    with ndet_col:
        # VIX context card
        vix_level = "N/A"
        vix_note = ""
        if vix:
            v = vix["value"]
            if v < 15:
                vix_level = "Low"
                vix_note = "Market complacent — cheap premiums, good for option buying"
                vix_color = "#16a34a"
            elif v < 20:
                vix_level = "Moderate"
                vix_note = "Normal volatility — balanced for both buyers and sellers"
                vix_color = "#ca8a04"
            elif v < 25:
                vix_level = "Elevated"
                vix_note = "Heightened fear — premiums are expensive, favour option selling or spreads"
                vix_color = "#ea580c"
            else:
                vix_level = "High"
                vix_note = "Extreme fear — very expensive premiums, expect large swings"
                vix_color = "#dc2626"

            st.markdown(f"""
            <div class="section-card">
                <div class="section-title">📊 Volatility Context</div>
                <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px;">
                    <span style="font-size:2rem;font-weight:800;color:{vix_color};">{v:.2f}</span>
                    <span style="font-size:1rem;font-weight:700;color:{vix_color};">{vix_level} Volatility</span>
                </div>
                <div style="font-size:0.88rem;color:#475569;">{vix_note}</div>
                <div style="margin-top:12px;">
                    {_trade_row("India VIX", f"{v:.2f}")}
                    {_trade_row("VIX Change", f"{vix['change']:+.2f}")}
                    {_trade_row("Regime", vix_level)}
                </div>
            </div>""", unsafe_allow_html=True)

        # Technical signals
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">📐 Nifty Technical Signals</div>
            {_trade_row("RSI (14)", nifty_summary['rsi'])}
            {_trade_row("MACD", nifty_summary['macd'])}
            {_trade_row("SMA Trend", nifty_summary['sma_trend'])}
            {_trade_row("Bollinger", nifty_summary['bollinger'])}
            {_trade_row("ATR (14)", f"₹{nifty_summary['atr']}" if nifty_summary.get('atr') else "N/A")}
            {_trade_row("EMA 9 / 21", f"{nifty_summary.get('ema_9', 'N/A')} / {nifty_summary.get('ema_21', 'N/A')}")}
        </div>""", unsafe_allow_html=True)

    # ── Top Movers (fast chart API only; cached 3 min) ───────────────
    with st.expander("📊 Nifty 50 Top Movers (sampled)", expanded=False):
        st.caption("Quick sample of Nifty 50 constituents — live data from NSE India.")
        try:
            market = _cached_nifty_top_movers()
        except Exception:
            market = {"top_gainers": [], "top_losers": []}

        mv1, mv2 = st.columns(2)
        with mv1:
            st.markdown("**Top Gainers**")
            for g in market.get("top_gainers", []):
                st.markdown(
                    f'<div class="trade-row">'
                    f'<span class="trade-label">{g["symbol"]}</span>'
                    f'<span class="trade-value" style="color:#16a34a;">₹{g["price"]:,.2f} &nbsp; ▲{g["pct"]:+.2f}%</span>'
                    f'</div>', unsafe_allow_html=True)
            if not market.get("top_gainers"):
                st.caption("No gainers data available.")
        with mv2:
            st.markdown("**Top Losers**")
            for l in market.get("top_losers", []):
                st.markdown(
                    f'<div class="trade-row">'
                    f'<span class="trade-label">{l["symbol"]}</span>'
                    f'<span class="trade-value" style="color:#dc2626;">₹{l["price"]:,.2f} &nbsp; ▼{l["pct"]:.2f}%</span>'
                    f'</div>', unsafe_allow_html=True)
            if not market.get("top_losers"):
                st.caption("No losers data available.")

    # ── Nifty News ─────────────────────────────
    try:
        nifty_news = get_stock_news("NIFTY")
    except Exception:
        nifty_news = []
    if nifty_news:
        with st.expander("📰 Market News", expanded=False):
            for n in nifty_news[:6]:
                pub = n.get("publisher", "")
                url = n.get("url", "")
                title = n.get("title", "No title")
                date = n.get("date", "")[:10]
                link = f'<a href="{url}" target="_blank" style="text-decoration:none;color:inherit;">' if url else ""
                link_end = "</a>" if url else ""
                st.markdown(f"""
                <div class="news-item">
                    {link}<div class="title">{title}</div>{link_end}
                    <div class="meta">{pub} · {date}</div>
                </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Tab 2: Stock Dashboard
# ──────────────────────────────────────────────
with tab_dashboard:
    _ = st.session_state.refresh_counter  # force re-fetch on refresh

    if not symbol:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
            <div style="font-size:3rem;margin-bottom:12px;">📈</div>
            <h3 style="color:#334155;margin-bottom:8px;">Select a Stock to Begin</h3>
            <p style="color:#94a3b8;max-width:400px;margin:auto;">
                Choose a Nifty 50 stock from the sidebar or enter any NSE/BSE symbol
                to view technical analysis, buy/sell advice, and F&O recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                df = get_stock_data(symbol, period=period)
                summary = generate_analysis_summary(df, symbol)
            except Exception as e:
                st.error(f"Could not fetch data for **{symbol}**: {e}")
                st.stop()

            _stock_src = df.attrs.get("data_source", "")
            if "fallback" in _stock_src.lower():
                st.caption(f"⚠️ Data source: {_stock_src} — NSE India blocked this server's IP. Prices may differ slightly.")

            fundamentals = None
            news = []
            try:
                fundamentals = get_fundamentals(symbol)
            except Exception:
                pass
            try:
                news = get_stock_news(symbol)
            except Exception:
                pass

            recommendation = generate_recommendation(summary, fundamentals, news)
            fno = generate_fno_advisory(summary, fundamentals)

        # ── Company header ────────────────────────
        comp_name = fundamentals.get("name", symbol) if fundamentals else symbol
        sector = fundamentals.get("sector", "") if fundamentals else ""
        header_sub = f"&nbsp;·&nbsp;{sector}" if sector and sector != "N/A" else ""
        st.markdown(
            f'<div style="margin-bottom:6px;">'
            f'<span style="font-size:1.5rem;font-weight:800;color:#0f172a;">{comp_name}</span>'
            f'<span style="font-size:0.95rem;color:#64748b;margin-left:10px;">NSE: {symbol}{header_sub}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── KPI row ───────────────────────────────
        delta_str = f"{summary['change']} ({summary['pct_change']}%)"
        delta_pos = summary['change'] >= 0

        signal_map = {"Bullish": ("🟢 Bullish", True), "Bearish": ("🔴 Bearish", False), "Neutral": ("🟡 Neutral", None)}
        sig_text, sig_pos = signal_map.get(summary['overall_signal'], ("🟡 Neutral", None))

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(_kpi_card("Price", f"₹{summary['price']:,.2f}", delta_str, delta_pos), unsafe_allow_html=True)
        k2.markdown(_kpi_card("RSI (14)", summary["rsi"]), unsafe_allow_html=True)
        k3.markdown(_kpi_card("Support", f"₹{summary['support']:,.2f}"), unsafe_allow_html=True)
        k4.markdown(_kpi_card("Resistance", f"₹{summary['resistance']:,.2f}"), unsafe_allow_html=True)
        k5.markdown(_kpi_card("Signal", sig_text), unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Chart ─────────────────────────────────
        fig = create_candlestick_chart(
            df, symbol,
            show_sma=show_sma, show_ema=show_ema, show_bollinger=show_bollinger,
            show_rsi=show_rsi, show_macd=show_macd, show_volume=show_volume,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Recommendation + F&O side by side ─────
        rec_col, fno_col = st.columns(2)

        # ── Buy / Sell / Hold ─────────────────────
        with rec_col:
            action = recommendation["action"]
            act_colors = {
                "Strong Buy": ("#059669", "#ecfdf5"), "Buy": ("#16a34a", "#f0fdf4"),
                "Hold": ("#ca8a04", "#fefce8"), "Sell": ("#dc2626", "#fef2f2"),
                "Strong Sell": ("#991b1b", "#fef2f2"),
            }
            ac, abg = act_colors.get(action, ("#64748b", "#f8fafc"))

            st.markdown(f"""
            <div class="section-card">
                <div class="section-title">📊 Recommendation</div>
                <div class="action-badge" style="background:{abg};color:{ac};border:2px solid {ac};">
                    {action}
                </div>
                <span class="action-meta">
                    Confidence: {recommendation['confidence']}% &nbsp;·&nbsp; Score: {recommendation['score']}
                </span>
            </div>""", unsafe_allow_html=True)

            # Technical + Fundamental signals
            with st.expander("🔍 Signal Details", expanded=True):
                st.markdown("**Technical Signals**")
                pills = ""
                for r in recommendation["technical_reasons"]:
                    pills += _signal_pill(f"{r['factor']}: {r['detail']}", r["signal"])
                st.markdown(pills, unsafe_allow_html=True)

                if recommendation["fundamental_reasons"]:
                    st.markdown("**Fundamental Signals**")
                    pills = ""
                    for r in recommendation["fundamental_reasons"]:
                        pills += _signal_pill(f"{r['factor']}: {r['detail']}", r["signal"])
                    st.markdown(pills, unsafe_allow_html=True)

                st.markdown(f"**News Sentiment:** {recommendation['news_sentiment']}")

        # ── F&O Advisory ──────────────────────────
        with fno_col:
            direction = fno["direction"]
            dir_colors = {
                "Buy Call (CE)": ("#059669", "#ecfdf5"),
                "Buy Put (PE)": ("#dc2626", "#fef2f2"),
                "Neutral / Avoid": ("#ca8a04", "#fefce8"),
            }
            dc, dbg = dir_colors.get(direction, ("#64748b", "#f8fafc"))
            dir_icon = {"Buy Call (CE)": "📈", "Buy Put (PE)": "📉"}.get(direction, "⚖️")

            sg = fno["strike_guidance"]
            st.markdown(f"""
            <div class="section-card">
                <div class="section-title">🎯 F&O Advisory</div>
                <div class="action-badge" style="background:{dbg};color:{dc};border:2px solid {dc};">
                    {dir_icon} {direction}
                </div>
                <span class="action-meta">Strategy: {fno['strategy']}</span>
                <div style="margin-top:16px;">
                    {_gauge_bar(fno['bullish_points'], fno['bearish_points'])}
                </div>
                <div style="margin-top:14px;">
                    {_trade_row("ATM Strike", f"₹{sg['atm_strike']:,.0f}")}
                    {_trade_row("Target", f"₹{sg['target']:,.2f}" if sg.get('target') else "—")}
                    {_trade_row("Stop Loss", f"₹{sg['stoploss']:,.2f}" if sg.get('stoploss') else "—")}
                    {_trade_row("Expected Move (2×ATR)", f"±₹{sg['expected_move']:,.2f}" if sg.get('expected_move') else "—")}
                    {_trade_row("Support / Resistance", f"₹{summary['support']:,.2f} / ₹{summary['resistance']:,.2f}")}
                </div>
            </div>""", unsafe_allow_html=True)

            with st.expander("📋 Rationale & Risks", expanded=False):
                for r in fno["rationale"]:
                    st.markdown(f"- {r}")
                st.markdown("---")
                for note in fno["risk_notes"]:
                    st.caption(f"⚠️ {note}")

        # ── Company Fundamentals ──────────────────
        if fundamentals:
            with st.expander("🏢 Company Fundamentals", expanded=False):
                f1, f2, f3, f4, f5, f6 = st.columns(6)
                pe = fundamentals.get("trailing_pe")
                fwd_pe = fundamentals.get("forward_pe")
                eg = fundamentals.get("earnings_growth")
                rg = fundamentals.get("revenue_growth")
                dte = fundamentals.get("debt_to_equity")
                dy = fundamentals.get("dividend_yield")
                beta = fundamentals.get("beta")
                mc = fundamentals.get("market_cap")

                f1.metric("Market Cap", _format_mcap(mc))
                f2.metric("P/E (TTM)", f"{pe:.1f}" if pe else "N/A")
                f3.metric("Fwd P/E", f"{fwd_pe:.1f}" if fwd_pe else "N/A")
                f4.metric("Earnings Gr.", f"{eg*100:.1f}%" if eg else "N/A")
                f5.metric("Revenue Gr.", f"{rg*100:.1f}%" if rg else "N/A")
                f6.metric("Debt/Equity", f"{dte:.0f}%" if dte else "N/A")

                f7, f8, f9, f10, f11, f12 = st.columns(6)
                f7.metric("Beta", f"{beta:.2f}" if beta else "N/A")
                dy_pct = (dy if dy and dy > 1 else dy * 100) if dy else None
                f8.metric("Div Yield", f"{dy_pct:.1f}%" if dy_pct else "N/A")
                h52 = fundamentals.get("52w_high")
                l52 = fundamentals.get("52w_low")
                f9.metric("52W High", f"₹{h52:,.0f}" if h52 else "N/A")
                f10.metric("52W Low", f"₹{l52:,.0f}" if l52 else "N/A")
                eps = fundamentals.get("trailing_eps")
                f11.metric("EPS (TTM)", f"₹{eps:.1f}" if eps else "N/A")
                pb = fundamentals.get("price_to_book")
                f12.metric("P/B Ratio", f"{pb:.2f}" if pb else "N/A")

        # ── News Headlines ────────────────────────
        if news:
            with st.expander("📰 Recent News", expanded=False):
                for n in news[:6]:
                    pub = n.get("publisher", "")
                    url = n.get("url", "")
                    title = n.get("title", "No title")
                    date = n.get("date", "")[:10]
                    link = f'<a href="{url}" target="_blank" style="text-decoration:none;color:inherit;">' if url else ""
                    link_end = "</a>" if url else ""
                    st.markdown(f"""
                    <div class="news-item">
                        {link}<div class="title">{title}</div>{link_end}
                        <div class="meta">{pub} · {date}</div>
                    </div>""", unsafe_allow_html=True)

        # ── Detailed Technical Signals ─────────────
        with st.expander("📐 Detailed Technical Signals", expanded=False):
            det_col1, det_col2 = st.columns(2)
            with det_col1:
                st.markdown(f"**MACD:** {summary['macd']}")
                st.markdown(f"**SMA Trend:** {summary['sma_trend']}")
                if summary.get("atr"):
                    st.markdown(f"**ATR (14):** ₹{summary['atr']}")
            with det_col2:
                st.markdown(f"**Bollinger Bands:** {summary['bollinger']}")
                if summary.get("ema_9") and summary.get("ema_21"):
                    st.markdown(f"**EMA 9 / 21:** ₹{summary['ema_9']} / ₹{summary['ema_21']}")
                if summary["crossovers"]:
                    st.markdown("**Recent Crossovers:**")
                    for c in summary["crossovers"]:
                        icon = "🟢" if c["type"] == "Golden Cross" else "🔴"
                        st.markdown(f"- {icon} **{c['date']}**: {c['type']} — {c['detail']}")
                else:
                    st.markdown("**Crossovers:** None in this period")

# ──────────────────────────────────────────────
# Tab 2: AI Chat
# ──────────────────────────────────────────────
with tab_chat:
    st.markdown("""
    <div style="padding:8px 0 16px;">
        <span style="font-size:1.1rem;font-weight:700;color:#0f172a;">Chat with the AI Agent</span>
        <br><span style="font-size:0.85rem;color:#94a3b8;">
        Ask questions about Indian stocks — the agent fetches live data and analyzes it for you.</span>
    </div>
    """, unsafe_allow_html=True)

    def _init_agent():
        if st.session_state.agent is None and st.session_state.agent_error is None:
            try:
                from agents.stock_agent import create_agent
                st.session_state.agent = create_agent()
            except Exception as e:
                st.session_state.agent_error = str(e)

    _init_agent()

    if st.session_state.agent_error:
        st.warning(
            f"Could not connect to Ollama: {st.session_state.agent_error}\n\n"
            "Make sure Ollama is running (`ollama serve`) and you have pulled a model (`ollama pull llama3`).\n\n"
            "The Dashboard tab still works without Ollama."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about any Indian stock..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.agent is None:
                st.error("Agent not available. Please ensure Ollama is running.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        from agents.stock_agent import chat as agent_chat
                        response = agent_chat(st.session_state.agent, prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
