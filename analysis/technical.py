from __future__ import annotations

import numpy as np
import pandas as pd

# Indicators are implemented with pandas/numpy only. pandas_ta pulls in numba, which
# can raise RuntimeError on import when packages live in site-packages ("no locator
# available" for @njit(cache=True)).


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig
    return macd_line, sig, hist


def _stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    smooth_k: int = 3,
    d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k, min_periods=k).min()
    highest = high.rolling(k, min_periods=k).max()
    denom = (highest - lowest).replace(0, np.nan)
    raw_k = 100 * (close - lowest) / denom
    k_line = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d_line = k_line.rolling(d, min_periods=d).mean()
    return k_line, d_line


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().to_numpy())
    direction[0] = 0
    return pd.Series(direction * volume.to_numpy(), index=close.index).cumsum()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all standard technical indicators to an OHLCV DataFrame (in-place)."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # --- Trend ---
    df["SMA_20"] = c.rolling(20, min_periods=20).mean()
    df["SMA_50"] = c.rolling(50, min_periods=50).mean()
    df["SMA_200"] = c.rolling(200, min_periods=200).mean()
    df["EMA_9"] = _ema(c, 9)
    df["EMA_21"] = _ema(c, 21)

    macd_line, macd_sig, macd_hist = _macd(c)
    df["MACD_12_26_9"] = macd_line
    df["MACDs_12_26_9"] = macd_sig
    df["MACDh_12_26_9"] = macd_hist

    # --- Momentum ---
    df["RSI_14"] = _rsi(c, 14)
    sk, sd = _stoch(h, l, c)
    df["STOCHk_14_3_3"] = sk
    df["STOCHd_14_3_3"] = sd

    # --- Volatility (Bollinger names match pandas_ta / plotter) ---
    mid = c.rolling(20, min_periods=20).mean()
    std = c.rolling(20, min_periods=20).std()
    df["BBM_20_2.0"] = mid
    df["BBU_20_2.0"] = mid + 2 * std
    df["BBL_20_2.0"] = mid - 2 * std
    df["BBB_20_2.0"] = (df["BBU_20_2.0"] - df["BBL_20_2.0"]) / mid.replace(0, np.nan)

    df["ATR_14"] = _atr(h, l, c, 14)

    # --- Volume ---
    df["OBV"] = _obv(c, v)
    typical = (h + l + c) / 3
    df["VWAP"] = (typical * v).cumsum() / v.cumsum().replace(0, np.nan)

    return df


def detect_crossovers(df: pd.DataFrame) -> list[dict]:
    """Detect Golden Cross (SMA50 crosses above SMA200) and Death Cross (opposite)."""
    signals: list[dict] = []
    if "SMA_50" not in df.columns or "SMA_200" not in df.columns:
        return signals

    sma50 = df["SMA_50"]
    sma200 = df["SMA_200"]
    for i in range(1, len(df)):
        if pd.isna(sma50.iloc[i]) or pd.isna(sma200.iloc[i]):
            continue
        if pd.isna(sma50.iloc[i - 1]) or pd.isna(sma200.iloc[i - 1]):
            continue

        prev_diff = sma50.iloc[i - 1] - sma200.iloc[i - 1]
        curr_diff = sma50.iloc[i] - sma200.iloc[i]

        if prev_diff <= 0 < curr_diff:
            signals.append({
                "date": str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i]),
                "type": "Golden Cross",
                "detail": "SMA 50 crossed above SMA 200 — bullish signal",
            })
        elif prev_diff >= 0 > curr_diff:
            signals.append({
                "date": str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i]),
                "type": "Death Cross",
                "detail": "SMA 50 crossed below SMA 200 — bearish signal",
            })
    return signals


def find_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """Estimate support and resistance levels from rolling min/max."""
    recent = df.tail(window)
    return {
        "support": round(float(recent["Low"].min()), 2),
        "resistance": round(float(recent["High"].max()), 2),
    }


def interpret_rsi(rsi_value: float | None) -> str:
    if rsi_value is None or pd.isna(rsi_value):
        return "N/A"
    if rsi_value >= 70:
        return f"Overbought ({rsi_value:.1f})"
    if rsi_value <= 30:
        return f"Oversold ({rsi_value:.1f})"
    return f"Neutral ({rsi_value:.1f})"


def interpret_macd(macd_val: float | None, signal_val: float | None) -> str:
    if macd_val is None or signal_val is None:
        return "N/A"
    if pd.isna(macd_val) or pd.isna(signal_val):
        return "N/A"
    if macd_val > signal_val:
        return "Bullish (MACD above signal)"
    return "Bearish (MACD below signal)"


def generate_analysis_summary(df: pd.DataFrame, symbol: str) -> dict:
    """Run full technical analysis and return a structured summary dict."""
    df = add_all_indicators(df)

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    price = round(float(latest["Close"]), 2)
    change = round(float(latest["Close"] - prev["Close"]), 2)
    pct_change = round((change / float(prev["Close"])) * 100, 2) if float(prev["Close"]) != 0 else 0.0

    rsi_val = latest.get("RSI_14")
    macd_val = latest.get("MACD_12_26_9")
    macd_signal = latest.get("MACDs_12_26_9")

    crossovers = detect_crossovers(df)
    sr = find_support_resistance(df)

    sma_trend = "N/A"
    if not pd.isna(latest.get("SMA_50", float("nan"))) and not pd.isna(latest.get("SMA_200", float("nan"))):
        sma_trend = "Bullish (SMA50 > SMA200)" if latest["SMA_50"] > latest["SMA_200"] else "Bearish (SMA50 < SMA200)"

    bb_position = "N/A"
    bbu = latest.get("BBU_20_2.0")
    bbl = latest.get("BBL_20_2.0")
    if bbu is not None and bbl is not None and not pd.isna(bbu) and not pd.isna(bbl):
        if price >= float(bbu):
            bb_position = "At/above upper band — potentially overbought"
        elif price <= float(bbl):
            bb_position = "At/below lower band — potentially oversold"
        else:
            bb_position = "Within bands"

    overall_signals: list[str] = []
    if rsi_val is not None and not pd.isna(rsi_val):
        if rsi_val >= 70:
            overall_signals.append("bearish")
        elif rsi_val <= 30:
            overall_signals.append("bullish")
    if macd_val is not None and macd_signal is not None and not pd.isna(macd_val) and not pd.isna(macd_signal):
        overall_signals.append("bullish" if macd_val > macd_signal else "bearish")
    if sma_trend.startswith("Bullish"):
        overall_signals.append("bullish")
    elif sma_trend.startswith("Bearish"):
        overall_signals.append("bearish")

    bullish_count = overall_signals.count("bullish")
    bearish_count = overall_signals.count("bearish")
    if bullish_count > bearish_count:
        overall = "Bullish"
    elif bearish_count > bullish_count:
        overall = "Bearish"
    else:
        overall = "Neutral"

    return {
        "symbol": symbol.upper(),
        "price": price,
        "change": change,
        "pct_change": pct_change,
        "rsi": interpret_rsi(rsi_val),
        "rsi_value": float(rsi_val) if rsi_val is not None and not pd.isna(rsi_val) else None,
        "macd": interpret_macd(macd_val, macd_signal),
        "macd_value": float(macd_val) if macd_val is not None and not pd.isna(macd_val) else None,
        "macd_signal_value": float(macd_signal) if macd_signal is not None and not pd.isna(macd_signal) else None,
        "sma_trend": sma_trend,
        "bollinger": bb_position,
        "support": sr["support"],
        "resistance": sr["resistance"],
        "crossovers": crossovers[-3:] if crossovers else [],
        "overall_signal": overall,
        "atr": round(float(latest.get("ATR_14", 0)), 2) if not pd.isna(latest.get("ATR_14", float("nan"))) else None,
        "ema_9": round(float(latest.get("EMA_9", 0)), 2) if not pd.isna(latest.get("EMA_9", float("nan"))) else None,
        "ema_21": round(float(latest.get("EMA_21", 0)), 2) if not pd.isna(latest.get("EMA_21", float("nan"))) else None,
        "sma_20": round(float(latest.get("SMA_20", 0)), 2) if not pd.isna(latest.get("SMA_20", float("nan"))) else None,
        "sma_50": round(float(latest.get("SMA_50", 0)), 2) if not pd.isna(latest.get("SMA_50", float("nan"))) else None,
        "volume": int(latest.get("Volume", 0)),
    }


# ────────────────────────────────────────────────────────────
# Buy / Sell / Hold recommendation engine
# ────────────────────────────────────────────────────────────

def _score_technicals(summary: dict) -> tuple[float, list[dict]]:
    """Score technical indicators on a -100..+100 scale.

    Returns (score, reasons) where each reason is
    {"factor": str, "signal": "bullish"|"bearish"|"neutral", "detail": str, "weight": float}.
    """
    reasons: list[dict] = []
    score = 0.0

    # --- RSI (weight 20) ---
    rsi = summary.get("rsi_value")
    if rsi is not None:
        if rsi <= 25:
            s = 20
            reasons.append({"factor": "RSI", "signal": "bullish", "detail": f"Deeply oversold at {rsi:.1f} — strong bounce potential", "weight": s})
        elif rsi <= 35:
            s = 12
            reasons.append({"factor": "RSI", "signal": "bullish", "detail": f"Oversold zone at {rsi:.1f} — buying interest likely", "weight": s})
        elif rsi >= 80:
            s = -20
            reasons.append({"factor": "RSI", "signal": "bearish", "detail": f"Extremely overbought at {rsi:.1f} — correction likely", "weight": s})
        elif rsi >= 70:
            s = -12
            reasons.append({"factor": "RSI", "signal": "bearish", "detail": f"Overbought at {rsi:.1f} — caution advised", "weight": s})
        elif 45 <= rsi <= 55:
            s = 0
            reasons.append({"factor": "RSI", "signal": "neutral", "detail": f"Neutral at {rsi:.1f}", "weight": s})
        elif rsi > 55:
            s = 5
            reasons.append({"factor": "RSI", "signal": "bullish", "detail": f"Positive momentum at {rsi:.1f}", "weight": s})
        else:
            s = -5
            reasons.append({"factor": "RSI", "signal": "bearish", "detail": f"Weak momentum at {rsi:.1f}", "weight": s})
        score += s

    # --- MACD (weight 20) ---
    macd_v = summary.get("macd_value")
    macd_s = summary.get("macd_signal_value")
    if macd_v is not None and macd_s is not None:
        diff = macd_v - macd_s
        if diff > 0 and macd_v > 0:
            s = 20
            reasons.append({"factor": "MACD", "signal": "bullish", "detail": "MACD above signal and positive — strong uptrend", "weight": s})
        elif diff > 0:
            s = 10
            reasons.append({"factor": "MACD", "signal": "bullish", "detail": "MACD above signal — momentum turning up", "weight": s})
        elif diff < 0 and macd_v < 0:
            s = -20
            reasons.append({"factor": "MACD", "signal": "bearish", "detail": "MACD below signal and negative — strong downtrend", "weight": s})
        else:
            s = -10
            reasons.append({"factor": "MACD", "signal": "bearish", "detail": "MACD below signal — momentum weakening", "weight": s})
        score += s

    # --- SMA trend (weight 15) ---
    sma_trend = summary.get("sma_trend", "N/A")
    if sma_trend.startswith("Bullish"):
        s = 15
        reasons.append({"factor": "SMA Trend", "signal": "bullish", "detail": "SMA 50 > SMA 200 — long-term uptrend intact", "weight": s})
    elif sma_trend.startswith("Bearish"):
        s = -15
        reasons.append({"factor": "SMA Trend", "signal": "bearish", "detail": "SMA 50 < SMA 200 — long-term downtrend", "weight": s})
    if sma_trend != "N/A":
        score += s

    # --- EMA short-term (weight 10) ---
    ema9 = summary.get("ema_9")
    ema21 = summary.get("ema_21")
    price = summary.get("price", 0)
    if ema9 is not None and ema21 is not None:
        if ema9 > ema21 and price > ema9:
            s = 10
            reasons.append({"factor": "EMA Crossover", "signal": "bullish", "detail": f"Price (₹{price}) above EMA 9 (₹{ema9}) > EMA 21 (₹{ema21})", "weight": s})
        elif ema9 < ema21 and price < ema9:
            s = -10
            reasons.append({"factor": "EMA Crossover", "signal": "bearish", "detail": f"Price (₹{price}) below EMA 9 (₹{ema9}) < EMA 21 (₹{ema21})", "weight": s})
        else:
            s = 0
            reasons.append({"factor": "EMA Crossover", "signal": "neutral", "detail": "EMAs are mixed — no clear short-term trend", "weight": s})
        score += s

    # --- Bollinger Bands (weight 10) ---
    bb = summary.get("bollinger", "N/A")
    if "overbought" in bb.lower():
        s = -10
        reasons.append({"factor": "Bollinger Bands", "signal": "bearish", "detail": "Price at/above upper band — stretched, pullback likely", "weight": s})
    elif "oversold" in bb.lower():
        s = 10
        reasons.append({"factor": "Bollinger Bands", "signal": "bullish", "detail": "Price at/below lower band — bounce likely", "weight": s})
    elif bb == "Within bands":
        s = 0
        reasons.append({"factor": "Bollinger Bands", "signal": "neutral", "detail": "Price within bands — no extreme", "weight": s})
    if bb != "N/A":
        score += s

    # --- Support / Resistance proximity (weight 10) ---
    support = summary.get("support", 0)
    resistance = summary.get("resistance", 0)
    if support and resistance and price:
        sr_range = resistance - support
        if sr_range > 0:
            position = (price - support) / sr_range
            if position <= 0.2:
                s = 10
                reasons.append({"factor": "Support Proximity", "signal": "bullish", "detail": f"Price near support ₹{support} — good entry zone", "weight": s})
            elif position >= 0.8:
                s = -10
                reasons.append({"factor": "Resistance Proximity", "signal": "bearish", "detail": f"Price near resistance ₹{resistance} — could face selling pressure", "weight": s})
            else:
                s = 0
                reasons.append({"factor": "S/R Position", "signal": "neutral", "detail": f"Price in mid-range between ₹{support} and ₹{resistance}", "weight": s})
            score += s

    # --- Recent crossovers (weight 15) ---
    crossovers = summary.get("crossovers", [])
    if crossovers:
        last = crossovers[-1]
        if last["type"] == "Golden Cross":
            s = 15
            reasons.append({"factor": "Crossover", "signal": "bullish", "detail": f"Golden Cross on {last['date']} — major bullish signal", "weight": s})
        else:
            s = -15
            reasons.append({"factor": "Crossover", "signal": "bearish", "detail": f"Death Cross on {last['date']} — major bearish signal", "weight": s})
        score += s

    return score, reasons


def _score_fundamentals(fundamentals: dict | None) -> tuple[float, list[dict]]:
    """Score fundamental data on a -100..+100 scale."""
    if not fundamentals:
        return 0.0, []

    reasons: list[dict] = []
    score = 0.0

    # --- P/E valuation (weight 15) ---
    pe = fundamentals.get("trailing_pe")
    fwd_pe = fundamentals.get("forward_pe")
    if pe is not None:
        if pe < 15:
            s = 15
            reasons.append({"factor": "P/E Ratio", "signal": "bullish", "detail": f"Trailing P/E {pe:.1f} — attractively valued", "weight": s})
        elif pe < 25:
            s = 5
            reasons.append({"factor": "P/E Ratio", "signal": "bullish", "detail": f"Trailing P/E {pe:.1f} — fairly valued", "weight": s})
        elif pe < 40:
            s = -5
            reasons.append({"factor": "P/E Ratio", "signal": "bearish", "detail": f"Trailing P/E {pe:.1f} — slightly expensive", "weight": s})
        else:
            s = -15
            reasons.append({"factor": "P/E Ratio", "signal": "bearish", "detail": f"Trailing P/E {pe:.1f} — richly valued", "weight": s})
        score += s

    if pe and fwd_pe and fwd_pe < pe:
        s = 5
        reasons.append({"factor": "Forward P/E", "signal": "bullish", "detail": f"Forward P/E {fwd_pe:.1f} < Trailing {pe:.1f} — earnings expected to grow", "weight": s})
        score += s

    # --- Earnings growth (weight 15) ---
    eg = fundamentals.get("earnings_growth")
    if eg is not None:
        pct = eg * 100
        if pct > 20:
            s = 15
            reasons.append({"factor": "Earnings Growth", "signal": "bullish", "detail": f"Earnings growing at {pct:.0f}% — strong growth", "weight": s})
        elif pct > 5:
            s = 8
            reasons.append({"factor": "Earnings Growth", "signal": "bullish", "detail": f"Earnings growing at {pct:.0f}% — moderate growth", "weight": s})
        elif pct > -5:
            s = 0
            reasons.append({"factor": "Earnings Growth", "signal": "neutral", "detail": f"Earnings flat at {pct:.0f}%", "weight": s})
        else:
            s = -15
            reasons.append({"factor": "Earnings Growth", "signal": "bearish", "detail": f"Earnings declining at {pct:.0f}%", "weight": s})
        score += s

    # --- Revenue growth (weight 10) ---
    rg = fundamentals.get("revenue_growth")
    if rg is not None:
        pct = rg * 100
        if pct > 15:
            s = 10
            reasons.append({"factor": "Revenue Growth", "signal": "bullish", "detail": f"Revenue growing at {pct:.0f}%", "weight": s})
        elif pct > 0:
            s = 3
            reasons.append({"factor": "Revenue Growth", "signal": "neutral", "detail": f"Revenue growing at {pct:.0f}%", "weight": s})
        else:
            s = -10
            reasons.append({"factor": "Revenue Growth", "signal": "bearish", "detail": f"Revenue declining at {pct:.0f}%", "weight": s})
        score += s

    # --- Debt-to-equity (weight 10) ---
    dte = fundamentals.get("debt_to_equity")
    if dte is not None:
        if dte < 50:
            s = 10
            reasons.append({"factor": "Debt/Equity", "signal": "bullish", "detail": f"D/E {dte:.0f}% — low leverage, healthy balance sheet", "weight": s})
        elif dte < 100:
            s = 0
            reasons.append({"factor": "Debt/Equity", "signal": "neutral", "detail": f"D/E {dte:.0f}% — moderate leverage", "weight": s})
        else:
            s = -10
            reasons.append({"factor": "Debt/Equity", "signal": "bearish", "detail": f"D/E {dte:.0f}% — high leverage, risky", "weight": s})
        score += s

    # --- Dividend yield (weight 5) ---
    dy = fundamentals.get("dividend_yield")
    if dy is not None and dy > 0:
        pct = dy if dy > 1 else dy * 100
        if pct >= 2:
            s = 5
            reasons.append({"factor": "Dividend Yield", "signal": "bullish", "detail": f"{pct:.1f}% yield — good income", "weight": s})
        else:
            s = 2
            reasons.append({"factor": "Dividend Yield", "signal": "neutral", "detail": f"{pct:.1f}% yield", "weight": s})
        score += s

    # --- 52-week position (weight 10) ---
    high52 = fundamentals.get("52w_high")
    low52 = fundamentals.get("52w_low")
    if high52 and low52:
        w_range = high52 - low52
        if w_range > 0:
            # we don't have the live price in fundamentals, so skip this
            pass

    return score, reasons


def generate_recommendation(
    summary: dict,
    fundamentals: dict | None = None,
    news: list[dict] | None = None,
) -> dict:
    """Produce a Buy / Sell / Hold recommendation with reasoning.

    Returns a dict with:
      action: "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell"
      confidence: 0-100
      score: -100..+100 composite score
      technical_reasons: list of scored factor dicts
      fundamental_reasons: list of scored factor dicts
      news_sentiment: str summary of recent news tone
      summary_text: human-readable recommendation paragraph
    """
    tech_score, tech_reasons = _score_technicals(summary)
    fund_score, fund_reasons = _score_fundamentals(fundamentals)

    # Blend: 60 % technical, 40 % fundamental
    if fund_reasons:
        composite = tech_score * 0.6 + fund_score * 0.4
    else:
        composite = tech_score

    # Determine action
    if composite >= 40:
        action = "Strong Buy"
    elif composite >= 15:
        action = "Buy"
    elif composite > -15:
        action = "Hold"
    elif composite > -40:
        action = "Sell"
    else:
        action = "Strong Sell"

    confidence = min(100, int(abs(composite)))

    # News sentiment (simple keyword scan)
    news_sentiment = "No recent news available"
    if news:
        pos_words = {"growth", "profit", "surge", "beat", "strong", "upgrade", "bull", "rally",
                     "record", "positive", "expand", "outperform", "rises", "gains", "boom",
                     "recovery", "optimistic", "breakout"}
        neg_words = {"loss", "decline", "fall", "downgrade", "bear", "crash", "weak", "miss",
                     "debt", "negative", "risk", "warning", "drops", "slump", "cut", "fraud",
                     "concern", "slowdown", "default"}
        pos = neg = 0
        for n in news:
            text = (n.get("title", "") + " " + n.get("summary", "")).lower()
            pos += sum(1 for w in pos_words if w in text)
            neg += sum(1 for w in neg_words if w in text)
        if pos > neg * 1.5:
            news_sentiment = "Positive — recent news is predominantly optimistic"
        elif neg > pos * 1.5:
            news_sentiment = "Negative — recent news highlights concerns"
        elif pos + neg > 0:
            news_sentiment = "Mixed — news has both positive and negative signals"
        else:
            news_sentiment = "Neutral — no strong sentiment in recent news"

    # Build summary paragraph
    price = summary.get("price", "N/A")
    symbol = summary.get("symbol", "")
    bull_factors = [r["detail"] for r in tech_reasons + fund_reasons if r["signal"] == "bullish"]
    bear_factors = [r["detail"] for r in tech_reasons + fund_reasons if r["signal"] == "bearish"]

    parts = [f"**{action}** recommendation for **{symbol}** at ₹{price} (confidence: {confidence}%)."]
    if bull_factors:
        parts.append("**Bullish factors:** " + "; ".join(bull_factors[:3]) + ".")
    if bear_factors:
        parts.append("**Bearish factors:** " + "; ".join(bear_factors[:3]) + ".")
    parts.append(f"**News sentiment:** {news_sentiment}.")
    summary_text = " ".join(parts)

    return {
        "action": action,
        "confidence": confidence,
        "score": round(composite, 1),
        "technical_reasons": tech_reasons,
        "fundamental_reasons": fund_reasons,
        "news_sentiment": news_sentiment,
        "summary_text": summary_text,
    }


# ────────────────────────────────────────────────────────────
# F&O (Futures & Options) trade advisory
# ────────────────────────────────────────────────────────────

def generate_fno_advisory(summary: dict, fundamentals: dict | None = None) -> dict:
    """Generate an F&O trading advisory — Call vs Put recommendation.

    Returns a dict with:
      direction: "Buy Call (CE)" | "Buy Put (PE)" | "Neutral / Avoid"
      strategy: suggested option strategy name
      confidence: 0-100
      rationale: list of reasoning strings
      risk_notes: list of risk/caveat strings
      strike_guidance: dict with entry/target/stoploss levels
    """
    price = summary.get("price", 0)
    support = summary.get("support", 0)
    resistance = summary.get("resistance", 0)
    atr = summary.get("atr")
    rsi = summary.get("rsi_value")
    macd_v = summary.get("macd_value")
    macd_s = summary.get("macd_signal_value")
    ema9 = summary.get("ema_9")
    ema21 = summary.get("ema_21")
    overall = summary.get("overall_signal", "Neutral")
    crossovers = summary.get("crossovers", [])

    bullish_pts = 0
    bearish_pts = 0
    rationale: list[str] = []

    # --- Trend alignment ---
    if overall == "Bullish":
        bullish_pts += 2
        rationale.append("Overall technical signal is bullish")
    elif overall == "Bearish":
        bearish_pts += 2
        rationale.append("Overall technical signal is bearish")

    # --- RSI momentum ---
    if rsi is not None:
        if rsi <= 30:
            bullish_pts += 2
            rationale.append(f"RSI {rsi:.0f} — oversold, likely reversal up")
        elif rsi <= 40:
            bullish_pts += 1
            rationale.append(f"RSI {rsi:.0f} — approaching oversold")
        elif rsi >= 70:
            bearish_pts += 2
            rationale.append(f"RSI {rsi:.0f} — overbought, likely reversal down")
        elif rsi >= 60:
            bearish_pts += 0
            bullish_pts += 1
            rationale.append(f"RSI {rsi:.0f} — bullish momentum")

    # --- MACD ---
    if macd_v is not None and macd_s is not None:
        if macd_v > macd_s:
            bullish_pts += 1
            rationale.append("MACD above signal line — bullish momentum")
        else:
            bearish_pts += 1
            rationale.append("MACD below signal line — bearish momentum")

    # --- EMA ---
    if ema9 is not None and ema21 is not None:
        if ema9 > ema21 and price > ema9:
            bullish_pts += 1
            rationale.append("Price above rising EMA 9/21 — short-term uptrend")
        elif ema9 < ema21 and price < ema9:
            bearish_pts += 1
            rationale.append("Price below falling EMA 9/21 — short-term downtrend")

    # --- Support/Resistance ---
    if support and resistance and price:
        sr_range = resistance - support
        if sr_range > 0:
            if (price - support) / sr_range <= 0.25:
                bullish_pts += 1
                rationale.append(f"Price near support ₹{support} — good call entry")
            elif (price - support) / sr_range >= 0.75:
                bearish_pts += 1
                rationale.append(f"Price near resistance ₹{resistance} — good put entry")

    # --- Crossovers ---
    if crossovers:
        last = crossovers[-1]
        if last["type"] == "Golden Cross":
            bullish_pts += 2
            rationale.append(f"Recent Golden Cross ({last['date']}) — strong bullish signal")
        else:
            bearish_pts += 2
            rationale.append(f"Recent Death Cross ({last['date']}) — strong bearish signal")

    # --- Determine direction ---
    total = bullish_pts + bearish_pts
    if bullish_pts > bearish_pts + 1:
        direction = "Buy Call (CE)"
        confidence = min(95, int((bullish_pts / max(total, 1)) * 100))
    elif bearish_pts > bullish_pts + 1:
        direction = "Buy Put (PE)"
        confidence = min(95, int((bearish_pts / max(total, 1)) * 100))
    else:
        direction = "Neutral / Avoid"
        confidence = max(20, 50 - abs(bullish_pts - bearish_pts) * 10)

    # --- Strategy suggestion ---
    if direction == "Buy Call (CE)":
        if rsi and rsi <= 35:
            strategy = "Bull Call Spread"
        elif bullish_pts >= 6:
            strategy = "Naked Call (aggressive)"
        else:
            strategy = "Bull Call Spread"
    elif direction == "Buy Put (PE)":
        if rsi and rsi >= 65:
            strategy = "Bear Put Spread"
        elif bearish_pts >= 6:
            strategy = "Naked Put (aggressive)"
        else:
            strategy = "Bear Put Spread"
    else:
        strategy = "Iron Condor / Straddle (range-bound play)"

    # --- Strike price guidance ---
    atr_val = atr if atr else price * 0.02
    if direction == "Buy Call (CE)":
        entry_strike = _round_strike(price, price)
        target = round(price + atr_val * 2, 2)
        stoploss = round(max(support, price - atr_val * 1.5), 2)
    elif direction == "Buy Put (PE)":
        entry_strike = _round_strike(price, price)
        target = round(price - atr_val * 2, 2)
        stoploss = round(min(resistance, price + atr_val * 1.5), 2)
    else:
        entry_strike = _round_strike(price, price)
        target = None
        stoploss = None

    strike_guidance = {
        "atm_strike": entry_strike,
        "target": target,
        "stoploss": stoploss,
        "expected_move": round(atr_val * 2, 2) if atr_val else None,
    }

    # --- Risk notes ---
    risk_notes = [
        "F&O trading involves substantial risk of loss and is not suitable for all investors.",
        "Options lose value over time (theta decay) — manage positions with strict stop-losses.",
    ]
    beta = fundamentals.get("beta") if fundamentals else None
    if beta is not None:
        if beta > 1.3:
            risk_notes.append(f"High beta ({beta:.2f}) — stock is more volatile than the market.")
        elif beta < 0.7:
            risk_notes.append(f"Low beta ({beta:.2f}) — stock moves less than the market; option premiums may be low.")

    if direction == "Neutral / Avoid":
        risk_notes.append("Signals are mixed — avoid large directional bets. Consider hedged strategies.")

    return {
        "direction": direction,
        "strategy": strategy,
        "confidence": confidence,
        "bullish_points": bullish_pts,
        "bearish_points": bearish_pts,
        "rationale": rationale,
        "risk_notes": risk_notes,
        "strike_guidance": strike_guidance,
    }


def _round_strike(price: float, base: float) -> float:
    """Round to the nearest standard option strike interval."""
    if price >= 5000:
        interval = 100
    elif price >= 1000:
        interval = 50
    elif price >= 500:
        interval = 20
    elif price >= 100:
        interval = 10
    else:
        interval = 5
    return round(base / interval) * interval
