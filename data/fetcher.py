"""Data-fetching layer — NSE India APIs first, yfinance fallback for cloud."""
from __future__ import annotations

import datetime as dt
import logging
from urllib.parse import quote

import numpy as np
import pandas as pd

from config import NIFTY50_SYMBOLS, DEFAULT_PERIOD, DEFAULT_INTERVAL

log = logging.getLogger(__name__)

# ── Optional imports (graceful degradation) ───────────────────
try:
    from curl_cffi import requests as cffi_requests
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

try:
    from nsetools import Nse
    _HAS_NSETOOLS = True
except ImportError:
    _HAS_NSETOOLS = False

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

_PERIOD_START_DAYS: dict[str, int] = {
    "1d": 5, "5d": 10, "1mo": 40, "3mo": 100, "6mo": 200,
    "1y": 380, "2y": 760, "5y": 1900, "10y": 3800, "ytd": 400,
    "max": 365 * 80,
}

_NSE_CHUNK_DAYS = 90

_NSE_INDEX_TYPES: dict[str, str] = {
    "NIFTY 50": "NIFTY 50",
    "BANK NIFTY": "NIFTY BANK",
    "INDIA VIX": "INDIA VIX",
}

INDEX_SYMBOLS = {
    "NIFTY 50": "NIFTY 50",
    "BANK NIFTY": "NIFTY BANK",
    "INDIA VIX": "INDIA VIX",
}


# ── NSE session (curl_cffi with Chrome impersonation) ────────

_cached_session = None


def _nse_session():
    """Create a curl_cffi session impersonating Chrome, with NSE cookies."""
    if not _HAS_CURL_CFFI:
        return None
    s = cffi_requests.Session(impersonate="chrome")
    s.headers.update({
        "Referer": "https://www.nseindia.com/",
        "Accept-Language": "en-US,en;q=0.9",
    })
    s.get("https://www.nseindia.com/", timeout=20)
    return s


def _get_session():
    global _cached_session
    if _cached_session is None:
        _cached_session = _nse_session()
    return _cached_session


def _reset_session():
    global _cached_session
    _cached_session = None


def _nse_get_json(path: str) -> dict | None:
    """GET JSON from NSE, retrying with a fresh session on failure."""
    if not _HAS_CURL_CFFI:
        return None
    for attempt in range(2):
        try:
            s = _get_session()
            if s is None:
                return None
            r = s.get(f"https://www.nseindia.com{path}", timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403):
                _reset_session()
                continue
            log.debug("NSE API HTTP %s for %s", r.status_code, path)
        except Exception as exc:
            log.debug("NSE API failed (attempt %d): %s", attempt, exc)
            _reset_session()
    return None


# ── Common parsing helpers ───────────────────────────────────

def _parse_nse_date(raw: str) -> pd.Timestamp | None:
    """Parse NSE date strings which come in varying formats:
    - YYYY-MM-DD  (equity CH_TIMESTAMP)
    - DD-MMM-YYYY (index EOD_TIMESTAMP, e.g. 07-APR-2026)
    - DD-MM-YYYY
    """
    if not raw:
        return None
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y", "%d-%B-%Y"):
        try:
            return pd.Timestamp(dt.datetime.strptime(raw, fmt)).normalize()
        except ValueError:
            continue
    ts = pd.to_datetime(raw, errors="coerce")
    return ts.normalize() if pd.notna(ts) else None


def _nse_parse_float(val) -> float:
    if val in (None, "", "-"):
        return float("nan")
    try:
        return float(str(val).replace(",", ""))
    except ValueError:
        return float("nan")


def _nse_row_get(row: dict, *candidates: str):
    up = {str(k).upper(): v for k, v in row.items()}
    for c in candidates:
        if c.upper() in up:
            return up[c.upper()]
    return None


# ── NSE Index data ───────────────────────────────────────────

def _nse_index_type_variants(app_key: str) -> tuple[str, ...]:
    if app_key == "INDIA VIX":
        return ("INDIA VIX", "INDIAVIX")
    name = _NSE_INDEX_TYPES.get(app_key)
    return (name,) if name else ()


def _nse_index_row_to_point(row: dict):
    raw_d = None
    for key in ("EOD_TIMESTAMP", "CH_TIMESTAMP", "TIMESTAMP", "DATE", "HistoricalDate"):
        raw_d = _nse_row_get(row, key)
        if raw_d:
            break
    ts = _parse_nse_date(raw_d)
    if ts is None:
        return None
    o = _nse_parse_float(_nse_row_get(row, "EOD_OPEN_INDEX_VAL", "OPEN_INDEX_VAL", "OPEN"))
    h = _nse_parse_float(_nse_row_get(row, "EOD_HIGH_INDEX_VAL", "HIGH_INDEX_VAL", "HIGH"))
    low_v = _nse_parse_float(_nse_row_get(row, "EOD_LOW_INDEX_VAL", "LOW_INDEX_VAL", "LOW"))
    c = _nse_parse_float(
        _nse_row_get(row, "EOD_CLOSE_INDEX_VAL", "CLOSING_INDEX_VAL", "CLOSE_INDEX_VAL", "CLOSE")
    )
    if all(pd.isna(x) for x in (o, h, low_v, c)):
        return None
    v_raw = _nse_row_get(row, "EOD_INDEX_VOLUMES", "TOTALTRADEDVOLUME", "TOTTRDQTY", "VOLUME")
    vol = _nse_parse_float(v_raw)
    if pd.isna(vol):
        vol = 0.0
    return (ts, o, h, low_v, c, int(vol))


def _records_to_df(points: list[tuple]) -> pd.DataFrame:
    if not points:
        return pd.DataFrame()
    df = pd.DataFrame(points, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _nse_fetch_index_chunk(index_type: str, from_dmY: str, to_dmY: str) -> pd.DataFrame:
    enc = quote(index_type)
    path = f"/api/historicalOR/indicesHistory?indexType={enc}&from={from_dmY}&to={to_dmY}"
    data = _nse_get_json(path)
    if not data:
        return pd.DataFrame()

    records = data.get("data", {})
    if isinstance(records, dict):
        records = records.get("indexCloseOnlineRecords", [])
    if not isinstance(records, list):
        return pd.DataFrame()

    points = [pt for row in records if isinstance(row, dict) and (pt := _nse_index_row_to_point(row)) is not None]
    return _records_to_df(points)


def _nse_index_daily_ohlc(index_type: str, period: str) -> pd.DataFrame:
    end = dt.date.today()
    days_back = _PERIOD_START_DAYS.get(period.lower(), 200)
    start = end - dt.timedelta(days=days_back)
    parts: list[pd.DataFrame] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=_NSE_CHUNK_DAYS - 1), end)
        from_s = cur.strftime("%d-%m-%Y")
        to_s = chunk_end.strftime("%d-%m-%Y")
        chunk_df = _nse_fetch_index_chunk(index_type, from_s, to_s)
        if not chunk_df.empty:
            parts.append(chunk_df)
        cur = chunk_end + dt.timedelta(days=1)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


# ── NSE Equity historical data ──────────────────────────────

def _nse_equity_row_to_point(row: dict):
    raw_d = None
    for key in ("CH_TIMESTAMP", "mTIMESTAMP", "TIMESTAMP", "DATE"):
        raw_d = _nse_row_get(row, key)
        if raw_d:
            break
    ts = _parse_nse_date(raw_d)
    if ts is None:
        return None
    o = _nse_parse_float(_nse_row_get(row, "CH_OPENING_PRICE", "OPEN_PRICE", "OPEN"))
    h = _nse_parse_float(_nse_row_get(row, "CH_TRADE_HIGH_PRICE", "HIGH_PRICE", "HIGH"))
    low_v = _nse_parse_float(_nse_row_get(row, "CH_TRADE_LOW_PRICE", "LOW_PRICE", "LOW"))
    c = _nse_parse_float(_nse_row_get(row, "CH_CLOSING_PRICE", "CLOSE_PRICE", "CLOSE"))
    if all(pd.isna(x) for x in (o, h, low_v, c)):
        return None
    v_raw = _nse_row_get(row, "CH_TOT_TRADED_QTY", "TTL_TRD_QNTY", "VOLUME", "TOTTRDQTY")
    vol = _nse_parse_float(v_raw)
    if pd.isna(vol):
        vol = 0.0
    return (ts, o, h, low_v, c, int(vol))


def _nse_fetch_equity_chunk(symbol: str, from_dmY: str, to_dmY: str) -> pd.DataFrame:
    enc_sym = quote(symbol)
    path = (
        f"/api/historicalOR/cm/equity"
        f"?symbol={enc_sym}&series=[%22EQ%22]&from={from_dmY}&to={to_dmY}"
    )
    data = _nse_get_json(path)
    if not data:
        return pd.DataFrame()

    records = data.get("data", [])
    if not isinstance(records, list):
        return pd.DataFrame()

    points = [pt for row in records if isinstance(row, dict) and (pt := _nse_equity_row_to_point(row)) is not None]
    return _records_to_df(points)


def _nse_equity_daily_ohlc(symbol: str, period: str) -> pd.DataFrame:
    end = dt.date.today()
    days_back = _PERIOD_START_DAYS.get(period.lower(), 200)
    start = end - dt.timedelta(days=days_back)
    parts: list[pd.DataFrame] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=_NSE_CHUNK_DAYS - 1), end)
        from_s = cur.strftime("%d-%m-%Y")
        to_s = chunk_end.strftime("%d-%m-%Y")
        chunk_df = _nse_fetch_equity_chunk(symbol, from_s, to_s)
        if not chunk_df.empty:
            parts.append(chunk_df)
        cur = chunk_end + dt.timedelta(days=1)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


# ── nsetools client ──────────────────────────────────────────

_nse_client = None
_nse_init_failed = False


def _get_nse():
    global _nse_client, _nse_init_failed
    if not _HAS_NSETOOLS or _nse_init_failed:
        return None
    if _nse_client is None:
        try:
            _nse_client = Nse()
        except Exception as exc:
            log.warning("NSE client unavailable: %s", exc)
            _nse_init_failed = True
            return None
    return _nse_client


# ── yfinance fallback helpers ────────────────────────────────

_YF_INDEX_MAP: dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "BANK NIFTY": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX",
}


def _yf_download(ticker: str, period: str) -> pd.DataFrame:
    """Download OHLCV via yfinance and normalise column names."""
    if not _HAS_YFINANCE:
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    except Exception as exc:
        log.debug("yfinance download failed for %s: %s", ticker, exc)
        return pd.DataFrame()
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "Date"
    return df


# ── Public API ───────────────────────────────────────────────

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]
    return symbol


def get_stock_data(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Fetch historical OHLCV data — NSE India first, yfinance fallback."""
    symbol = normalize_symbol(symbol)
    df = _nse_equity_daily_ohlc(symbol, period)
    if not df.empty:
        df.index.name = "Date"
        return df
    log.info("NSE direct failed for %s, trying yfinance fallback", symbol)
    df = _yf_download(f"{symbol}.NS", period)
    if not df.empty:
        return df
    raise ValueError(
        f"No data returned for {symbol}. "
        "NSE India may be unreachable and yfinance did not return data either."
    )


def get_live_quote(symbol: str) -> dict:
    symbol = normalize_symbol(symbol)
    path = f"/api/quote-equity?symbol={quote(symbol)}"
    data = _nse_get_json(path)
    if data:
        price_info = data.get("priceInfo", {})
        return {
            "symbol": symbol,
            "last_price": price_info.get("lastPrice"),
            "change": price_info.get("change"),
            "pct_change": price_info.get("pChange"),
            "open": price_info.get("open"),
            "high": price_info.get("intraDayHighLow", {}).get("max"),
            "low": price_info.get("intraDayHighLow", {}).get("min"),
            "close": price_info.get("previousClose"),
            "volume": data.get("securityWiseDP", {}).get("quantityTraded"),
            "52w_high": price_info.get("weekHighLow", {}).get("max"),
            "52w_low": price_info.get("weekHighLow", {}).get("min"),
        }
    client = _get_nse()
    if client:
        q = client.get_quote(symbol)
        if q:
            return {
                "symbol": symbol,
                "last_price": q.get("lastPrice"),
                "change": q.get("change"),
                "pct_change": q.get("pChange"),
                "open": q.get("open"),
                "high": q.get("dayHigh"),
                "low": q.get("dayLow"),
                "close": q.get("previousClose"),
                "volume": q.get("totalTradedVolume"),
                "52w_high": q.get("high52"),
                "52w_low": q.get("low52"),
            }
    if _HAS_YFINANCE:
        try:
            t = yf.Ticker(f"{symbol}.NS")
            info = t.info or {}
            return {
                "symbol": symbol,
                "last_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "change": info.get("regularMarketChange"),
                "pct_change": info.get("regularMarketChangePercent"),
                "open": info.get("regularMarketOpen"),
                "high": info.get("regularMarketDayHigh"),
                "low": info.get("regularMarketDayLow"),
                "close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
                "volume": info.get("regularMarketVolume"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception:
            pass
    raise ValueError(f"Could not fetch live quote for {symbol}.")


def get_nifty50_stocks() -> list[str]:
    return list(NIFTY50_SYMBOLS)


def get_stock_info(symbol: str) -> dict:
    symbol = normalize_symbol(symbol)
    path = f"/api/quote-equity?symbol={quote(symbol)}"
    data = _nse_get_json(path)
    if data:
        info = data.get("info", {})
        meta = data.get("metadata", {})
        price_info = data.get("priceInfo", {})
        return {
            "symbol": symbol,
            "name": info.get("companyName") or meta.get("companyName", symbol),
            "sector": meta.get("industry", "N/A"),
            "industry": meta.get("industry", "N/A"),
            "market_cap": None,
            "pe_ratio": meta.get("pdSymbolPe"),
            "52w_high": price_info.get("weekHighLow", {}).get("max"),
            "52w_low": price_info.get("weekHighLow", {}).get("min"),
            "avg_volume": None,
        }
    if _HAS_YFINANCE:
        try:
            info = yf.Ticker(f"{symbol}.NS").info or {}
            return {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName", symbol),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
            }
        except Exception:
            pass
    return {
        "symbol": symbol, "name": symbol, "sector": "N/A", "industry": "N/A",
        "market_cap": None, "pe_ratio": None, "52w_high": None, "52w_low": None,
        "avg_volume": None,
    }


def get_fundamentals(symbol: str) -> dict:
    symbol = normalize_symbol(symbol)
    path = f"/api/quote-equity?symbol={quote(symbol)}"
    data = _nse_get_json(path)
    if data:
        info = data.get("info", {})
        meta = data.get("metadata", {})
        price_info = data.get("priceInfo", {})
        return {
            "symbol": symbol,
            "name": info.get("companyName") or meta.get("companyName", symbol),
            "sector": meta.get("industry", "N/A"),
            "industry": meta.get("industry", "N/A"),
            "market_cap": None,
            "trailing_pe": _safe_float(meta.get("pdSymbolPe")),
            "forward_pe": None,
            "trailing_eps": None,
            "forward_eps": None,
            "price_to_book": None,
            "debt_to_equity": None,
            "return_on_equity": None,
            "revenue_growth": None,
            "earnings_growth": None,
            "dividend_yield": None,
            "beta": None,
            "52w_high": price_info.get("weekHighLow", {}).get("max"),
            "52w_low": price_info.get("weekHighLow", {}).get("min"),
            "avg_volume": None,
        }
    if _HAS_YFINANCE:
        try:
            info = yf.Ticker(f"{symbol}.NS").info or {}
            return {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName", symbol),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap"),
                "trailing_pe": _safe_float(info.get("trailingPE")),
                "forward_pe": _safe_float(info.get("forwardPE")),
                "trailing_eps": _safe_float(info.get("trailingEps")),
                "forward_eps": _safe_float(info.get("forwardEps")),
                "price_to_book": _safe_float(info.get("priceToBook")),
                "debt_to_equity": _safe_float(info.get("debtToEquity")),
                "return_on_equity": _safe_float(info.get("returnOnEquity")),
                "revenue_growth": _safe_float(info.get("revenueGrowth")),
                "earnings_growth": _safe_float(info.get("earningsGrowth")),
                "dividend_yield": _safe_float(info.get("dividendYield")),
                "beta": _safe_float(info.get("beta")),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
            }
        except Exception:
            pass
    return {"symbol": symbol, "name": symbol, "sector": "N/A", "industry": "N/A"}


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def get_index_data(
    index_name: str = "NIFTY 50",
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical OHLCV data — NSE India first, yfinance fallback."""
    key = index_name.upper()
    if interval != "1d":
        raise ValueError(f"Only daily interval is supported for index data (got {interval!r}).")

    for nse_type in _nse_index_type_variants(key):
        df = _nse_index_daily_ohlc(nse_type, period)
        if not df.empty:
            df.index.name = "Date"
            return df

    yf_ticker = _YF_INDEX_MAP.get(key)
    if yf_ticker:
        log.info("NSE direct failed for %s, trying yfinance (%s)", key, yf_ticker)
        df = _yf_download(yf_ticker, period)
        if not df.empty:
            return df

    raise ValueError(
        f"No data returned for index {index_name}. "
        "NSE India may be unreachable and yfinance did not return data either."
    )


def get_nifty_top_movers(max_symbols: int = 6) -> dict[str, list[dict]]:
    data = _nse_get_json("/api/equity-stockIndices?index=NIFTY%2050")
    gainers: list[dict] = []
    losers: list[dict] = []
    if data:
        for item in (data.get("data") or [])[:max_symbols + 1]:
            sym = item.get("symbol", "")
            if sym == "NIFTY 50":
                continue
            price = _safe_float(item.get("lastPrice"))
            pct = _safe_float(item.get("pChange"))
            chg = _safe_float(item.get("change"))
            if price is None or pct is None:
                continue
            entry = {
                "symbol": sym,
                "price": round(price, 2),
                "change": round(chg, 2) if chg else 0,
                "pct": round(pct, 2),
            }
            if pct >= 0:
                gainers.append(entry)
            else:
                losers.append(entry)
    if not gainers and not losers and _HAS_YFINANCE:
        sample = NIFTY50_SYMBOLS[:max_symbols]
        tickers = " ".join(f"{s}.NS" for s in sample)
        try:
            df = yf.download(tickers, period="2d", interval="1d", progress=False, auto_adjust=True, group_by="ticker")
            for sym in sample:
                tk = f"{sym}.NS"
                try:
                    sub = df[tk] if tk in df.columns.get_level_values(0) else None
                except Exception:
                    sub = None
                if sub is None or sub.empty or len(sub) < 2:
                    continue
                prev_c, last_c = float(sub["Close"].iloc[-2]), float(sub["Close"].iloc[-1])
                if pd.isna(prev_c) or pd.isna(last_c) or prev_c == 0:
                    continue
                chg = round(last_c - prev_c, 2)
                pct = round(((last_c - prev_c) / prev_c) * 100, 2)
                entry = {"symbol": sym, "price": round(last_c, 2), "change": chg, "pct": pct}
                if pct >= 0:
                    gainers.append(entry)
                else:
                    losers.append(entry)
        except Exception:
            pass
    gainers.sort(key=lambda x: x["pct"], reverse=True)
    losers.sort(key=lambda x: x["pct"])
    return {"top_gainers": gainers[:5], "top_losers": losers[:5]}


def get_market_snapshot(period: str = "6mo") -> dict:
    result: dict = {}
    for idx_name, key in [("NIFTY 50", "nifty"), ("BANK NIFTY", "banknifty")]:
        try:
            df = get_index_data(idx_name, period="1mo" if key != "nifty" else period)
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            result[key] = {
                "close": round(float(latest["Close"]), 2),
                "change": round(float(latest["Close"] - prev["Close"]), 2),
                "pct_change": round(((latest["Close"] - prev["Close"]) / prev["Close"]) * 100, 2),
            }
            if key == "nifty":
                result[key]["high"] = round(float(latest["High"]), 2)
                result[key]["low"] = round(float(latest["Low"]), 2)
        except Exception:
            result[key] = None

    try:
        vix_df = get_index_data("INDIA VIX", period="1mo")
        latest = vix_df.iloc[-1]
        prev = vix_df.iloc[-2] if len(vix_df) > 1 else latest
        result["vix"] = {
            "value": round(float(latest["Close"]), 2),
            "change": round(float(latest["Close"] - prev["Close"]), 2),
        }
    except Exception:
        result["vix"] = None

    result["top_gainers"] = []
    result["top_losers"] = []
    return result


def get_stock_news(symbol: str, max_items: int = 8) -> list[dict]:
    """News is not available without a third-party provider."""
    return []
