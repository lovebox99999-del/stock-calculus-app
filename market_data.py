"""market_data.py — 주가(일별 종가) 데이터 가져오기.

1순위: yfinance 라이브러리
2순위: 야후 파이낸스 차트 주소를 직접 호출 (yfinance 가 막혔을 때)
둘 다 실패하면 오류 메시지를 돌려주고, 화면에서 '예시 데이터'로 계산 과정을 볼 수 있게 한다.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def ticker_candidates(raw: str) -> list[str]:
    """입력을 야후 파이낸스 코드로 바꾼다. 6자리 숫자는 한국 종목으로 본다."""
    t = (raw or "").strip().upper().replace(" ", "")
    if not t:
        return []
    if re.fullmatch(r"\d{6}", t):
        return [t + ".KS", t + ".KQ"]         # 코스피 먼저, 안 되면 코스닥
    return [t]


def _tidy(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[close > 0]
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.name = "close"
    return close


def _from_yfinance(ticker: str, period: str):
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period=period, interval="1d", auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist:
        raise ValueError("빈 응답")
    meta = {}
    try:
        info = tk.history_metadata or {}
        meta = {"currency": info.get("currency", ""), "name": info.get("longName") or info.get("shortName") or ""}
    except Exception:
        pass
    return _tidy(hist["Close"]), meta


def _from_yahoo_chart(ticker: str, period: str):
    resp = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        params={"range": period, "interval": "1d", "events": "div,splits"},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    stamps = result["timestamp"]
    indicators = result["indicators"]
    adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    values = adj or indicators["quote"][0]["close"]
    meta_raw = result.get("meta", {})
    offset = int(meta_raw.get("gmtoffset", 0) or 0)       # 거래소 현지 날짜로 맞춘다
    index = pd.to_datetime(np.asarray(stamps, dtype="int64") + offset, unit="s")
    meta = {
        "currency": meta_raw.get("currency", ""),
        "name": meta_raw.get("longName") or meta_raw.get("shortName") or "",
    }
    return _tidy(pd.Series(values, index=index, dtype="float64")), meta


def fetch_prices(raw_ticker: str, period: str = "5y") -> dict:
    """종가 시계열을 가져온다.

    반환: {"ok": bool, "ticker": 실제 사용한 코드, "close": Series, "source": 출처,
           "currency": 통화, "name": 회사명, "errors": [실패 기록]}
    """
    errors = []
    for ticker in ticker_candidates(raw_ticker):
        for source, loader in (("yfinance", _from_yfinance), ("야후 파이낸스 직접 호출", _from_yahoo_chart)):
            try:
                close, meta = loader(ticker, period)
                if len(close) < 40:
                    raise ValueError(f"데이터가 {len(close)}일뿐입니다")
                return {"ok": True, "ticker": ticker, "close": close, "source": source, "errors": errors, **meta}
            except Exception as exc:
                errors.append(f"{ticker} / {source}: {type(exc).__name__}: {str(exc)[:120]}")
    return {"ok": False, "ticker": (raw_ticker or "").upper(), "close": None, "source": "", "errors": errors,
            "currency": "", "name": ""}


def sample_prices(n: int = 504, seed: int = 11) -> pd.Series:
    """실제 주가가 아닌 '가상' 데이터. 데이터를 못 받았을 때 계산 과정을 보여 주는 용도.

    느리게 변하는 추세(국면) 위에 매일의 잡음을 얹어 만든다.
    """
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    for i in range(1, n):
        drift[i] = 0.97 * drift[i - 1] + rng.normal(0, 0.0011)
    returns = drift + rng.normal(0, 0.014, n)
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=index, name="close")
