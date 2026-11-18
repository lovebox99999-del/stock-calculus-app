"""build.py — 1시간마다 GitHub Actions 가 실행하는 계산 스크립트.

하는 일
  1) 네 언론사 헤드라인을 모으고, Gemini(키가 있으면) 또는 키워드로 호재 점수를 매긴다.
  2) 종목 사전의 모든 종목 + 호재가 뜬 종목의 주가를 받아 미분·적분·예측·검증을 계산한다.
  3) 결과를 site/data/*.json 으로 저장하고, index.html 을 site/ 에 복사한다.
웹페이지(index.html)는 이 JSON 을 읽어 그래프를 그린다.

실행:  python build.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ai_judge
import config
import engine
import market_data
import news

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
DATA = SITE / "data"

WINDOW = 21            # 평활 창 (거래일)
HORIZON = 10           # 예측 기간 (거래일)
TAU = 10.0             # 추세 지속 시간
BETA = 0.02            # 호재 점수 +100 일 때 최대 반영 2%
HALF_LIFE = 36.0       # 뉴스 반감기 (시간)
SERIES_DAYS = 253      # 그래프에 넘길 기간 (약 1년)
AI_MAX_HEADLINES = 80
EXTRA_NEWS_TICKERS = 10   # 종목 이름으로 따로 기사를 찾아볼 상위 종목 수
MAX_AI_TICKERS = 15       # AI 가 사전 밖에서 추가한 종목 중 계산할 최대 개수


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def fid(ticker: str) -> str:
    """파일 이름으로 쓸 수 있는 종목 코드 (005930.KS → 005930_KS)."""
    return ticker.replace(".", "_").replace("/", "_")


def num(x, digits=6):
    """JSON 에 넣을 수 있는 숫자로. NaN·inf 는 None."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, digits) if math.isfinite(x) else None


def nums(arr, digits=6):
    return [num(v, digits) for v in np.asarray(arr, dtype=float)]


def iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ----------------------------------------------------------------------------
# 1. 뉴스
# ----------------------------------------------------------------------------
def build_radar(key: str):
    headlines, report = news.collect_headlines(max_age_days=3)
    log(f"헤드라인 {len(headlines)}건")
    ai_results, ai_model, ai_error = {}, None, None
    if key and headlines:
        matched = [(h, [m["ticker"] for m in news.match_companies(h["title"])]) for h in headlines]
        ordered = [x for x in matched if x[1]] + [x for x in matched if not x[1]]
        items = [{"title": h["title"], "outlet": h["outlet"], "candidates": c} for h, c in ordered[:AI_MAX_HEADLINES]]
        ai_results, ai_model, ai_error = ai_judge.judge_headlines(items, key)
        log(f"AI 판단 {len(ai_results)}건, 모델 {ai_model}, 오류 {ai_error}")
    elif not key:
        ai_error = "GEMINI_API_KEY 가 설정되지 않아 키워드 점수만 사용했습니다."
    mentions = news.build_mentions(headlines, ai_results)
    return headlines, report, mentions, ai_results, ai_model, ai_error


def extra_news(tickers: list[str], key: str, use_ai: bool):
    """상위 종목은 종목 이름으로 기사를 한 번 더 찾아 본다."""
    frames, all_items = [], {}
    for t in tickers:
        try:
            all_items[t] = news.search_company_news(t)[:20]
        except Exception as exc:
            log("기사 검색 실패", t, exc)
    ai_results = {}
    if use_ai and key:
        items, seen = [], set()
        for t, hs in all_items.items():
            for h in hs:
                if h["title"] not in seen:
                    seen.add(h["title"])
                    items.append({"title": h["title"], "outlet": h["outlet"], "candidates": [t]})
        if items:
            ai_results, _, err = ai_judge.judge_headlines(items[:120], key)
            if err:
                log("추가 기사 AI 판단 오류:", err)
    for t, hs in all_items.items():
        f = news.build_mentions(hs, ai_results, force_ticker=t)
        if len(f):
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else None


# ----------------------------------------------------------------------------
# 2. 주가
# ----------------------------------------------------------------------------
def download_prices(tickers: list[str]) -> dict:
    """yfinance 로 한꺼번에 받고, 빠진 종목은 하나씩 다시 시도한다."""
    out = {}
    try:
        import yfinance as yf

        df = yf.download(tickers, period="5y", interval="1d", auto_adjust=True, group_by="ticker",
                         threads=True, progress=False)
        for t in tickers:
            try:
                close = df[t]["Close"] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
                close = market_data._tidy(close)
                if len(close) >= 120:
                    out[t] = {"close": close, "source": "yfinance", "name": "",
                              "currency": "KRW" if t.endswith((".KS", ".KQ")) else "USD"}
            except Exception:
                pass
        log(f"yfinance 일괄 다운로드: {len(out)}/{len(tickers)}")
    except Exception as exc:
        log("yfinance 일괄 다운로드 실패:", exc)

    missing = [t for t in tickers if t not in out]

    def one(t):
        time.sleep(0.2)
        return t, market_data.fetch_prices(t, "5y")

    with ThreadPoolExecutor(max_workers=4) as pool:
        for t, res in pool.map(one, missing):
            if res["ok"]:
                out[t] = res
    log(f"주가 확보: {len(out)}/{len(tickers)}")
    return out


# ----------------------------------------------------------------------------
# 3. 종목 하나 계산
# ----------------------------------------------------------------------------
def analyze_ticker(ticker: str, price: dict, score: float, mentions: pd.DataFrame) -> dict:
    close = price["close"]
    df = engine.analyze(close, WINDOW)
    x = df["log"].to_numpy()
    sigma = engine.volatility(x)
    tail = df.iloc[-min(SERIES_DAYS, len(df)):]

    periods = {}
    for name, days in engine.PERIODS.items():
        seg = df.iloc[-min(days + 1, len(df)):]
        if len(seg) < 10:
            continue
        s = engine.period_summary(seg)
        tp = engine.turning_points(seg)
        legs = engine.find_legs(seg)
        periods[name] = {
            "start": len(tail) - len(seg),
            "net_actual": num(s["net_pct_actual"], 3),
            "net_integral": num(s["net_pct_integral"], 3),
            "net_trend": num(s["net_pct_trend"], 3),
            "up": num(s["up_pct"], 3),
            "down": num(s["down_pct"], 3),
            "efficiency": num(s["efficiency"], 4),
            "up_days": s["up_days"],
            "down_days": s["down_days"],
            "cum_up": nums(s["cum_up"] * 100, 4),
            "cum_down": nums(s["cum_down"] * 100, 4),
            "peaks": tp["peaks"],
            "troughs": tp["troughs"],
            "inflections": tp["inflections"],
            "legs": [
                {"start": iso(r["시작"]), "end": iso(r["끝"]), "dir": r["방향"], "days": int(r["기간(거래일)"]),
                 "pct": num(r["변화율(%)"], 2), "per_day": num(r["하루 평균(%)"], 3)}
                for _, r in legs.iterrows()
            ],
        }

    bt = engine.backtest(close, WINDOW, HORIZON, TAU, True)
    fc = engine.forecast(close, WINDOW, HORIZON, TAU, True, k=bt["k_now"], news_score=score, beta=BETA,
                         band_scale=bt.get("band_scale", 1.0))
    label, desc = engine.phase(fc["v0"], fc["a0"], fc["sigma"])

    backtest = {"ok": bool(bt.get("ok")), "n_eval": bt["n_eval"], "k_now": num(bt["k_now"], 3)}
    if bt.get("ok"):
        t = bt["table"]
        step = max(1, len(t) // 400)
        t = t.iloc[::step]
        backtest.update({
            "hit_rate": num(bt["hit_rate"], 4), "base_up": num(bt["base_up"], 4), "skill": num(bt["skill"], 4),
            "coverage68": num(bt["coverage68"], 4), "n_eff": num(bt["n_eff"], 1), "se": num(bt["se"], 4),
            "significant": bt["significant"], "corr": num(bt["corr"], 3),
            "points": {"d": [iso(d) for d in t.index], "pred": nums(np.expm1(t["pred"]) * 100, 3),
                       "actual": nums(np.expm1(t["actual"]) * 100, 3)},
        })

    news_rows = []
    if mentions is not None and len(mentions):
        for _, m in mentions.sort_values("published", ascending=False).head(10).iterrows():
            news_rows.append({
                "outlet": m["outlet"], "title": m["title"], "title_ko": m["title_ko"], "link": m["link"],
                "published": pd.Timestamp(m["published"]).isoformat(), "score": num(m["score"], 3),
                "method": m["method"], "why": m["why"],
            })

    company = config.BY_TICKER.get(ticker, {})
    return {
        "ticker": ticker,
        "name": company.get("ko") or price.get("name") or ticker,
        "sector": company.get("sector", "AI 추가"),
        "currency": price.get("currency", ""),
        "source": price.get("source", ""),
        "last_date": iso(close.index[-1]),
        "last": num(close.iloc[-1], 4),
        "n_days": len(close),
        "chg5": num(math.expm1(x[-1] - x[-6]) * 100, 3) if len(x) > 6 else None,
        "series": {
            "d": [iso(d) for d in tail.index],
            "close": nums(tail["close"], 4),
            "trend": nums(tail["trend"], 4),
            "v": nums(tail["v"] * 100, 5),
            "a": nums(tail["a"] * 100, 6),
        },
        "periods": periods,
        "phase": label,
        "phase_desc": desc,
        "v0": num(fc["v0"] * 100, 4),
        "a0": num(fc["a0"] * 100, 5),
        "sigma": num(fc["sigma"] * 100, 4),
        "score": num(score, 1),
        "forecast": {
            "dates": [iso(d) for d in fc["dates"]],
            "center": nums(fc["center"], 4), "lo68": nums(fc["lo68"], 4), "hi68": nums(fc["hi68"], 4),
            "lo95": nums(fc["lo95"], 4), "hi95": nums(fc["hi95"], 4),
            "total_pct": num(fc["total_pct"], 3), "trend_pct": num(fc["trend_pct"], 3),
            "news_pct": num(fc["news_pct"], 3), "raw_trend_pct": num(fc["raw_trend_pct"], 3),
            "band68": [num(fc["band68_pct"][0], 2), num(fc["band68_pct"][1], 2)],
            "p_up": num(fc["p_up"], 4), "k": num(fc["k"], 3), "discount": num(fc["discount"], 3),
            "recent3": num(fc["recent3_pct"], 2), "band_scale": num(bt.get("band_scale", 1.0), 3),
        },
        "backtest": backtest,
        "news": news_rows,
    }


# ----------------------------------------------------------------------------
# 4. AI 브리핑
# ----------------------------------------------------------------------------
def ai_brief(key: str, radar_rows: list[dict], results: dict) -> tuple[str, str]:
    if not key or not radar_rows:
        return "", ""
    top = []
    for r in radar_rows[:8]:
        t = results.get(r["ticker"])
        top.append({
            "종목": f'{r["name"]} ({r["ticker"]})', "호재점수": r["score"], "대표헤드라인": r["headline"],
            "국면": t["phase"] if t else None, "10일예상_%": t["forecast"]["total_pct"] if t else None,
            "과거적중률_%": round(t["backtest"]["hit_rate"] * 100, 1) if t and t["backtest"].get("ok") else None,
        })
    prompt = (
        "아래는 오늘 주요 경제지 헤드라인에서 호재·악재 점수가 높게 나온 종목과 각 종목의 주가 분석 결과다(JSON).\n"
        "이것으로 한국어 '오늘의 브리핑'을 써 줘. 전체 흐름 2~3문장, 눈에 띄는 종목 3~4개를 한두 문장씩, "
        "마지막에 예측의 한계 한 문장. 300~450자. 마크다운 제목 없이 문단만.\n\n"
        + json.dumps(top, ensure_ascii=False)
    )
    try:
        return ai_judge.generate(prompt, key, system=ai_judge.COMMENT_SYSTEM)
    except ai_judge.GeminiError as exc:
        log("AI 브리핑 실패:", exc)
        return "", ""


# ----------------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------------
def main():
    started = time.time()
    key = config.get_gemini_key()
    if SITE.exists():
        shutil.rmtree(SITE)
    (DATA / "t").mkdir(parents=True)

    headlines, report, mentions, ai_results, ai_model, ai_error = build_radar(key)
    table = news.aggregate(mentions, HALF_LIFE)

    top_abs = table.reindex(table["호재 점수"].abs().sort_values(ascending=False).index)
    extra_for = [t for t in top_abs["ticker"].tolist() if t in config.BY_TICKER][:EXTRA_NEWS_TICKERS]
    extra = extra_news(extra_for, key, use_ai=bool(ai_results))
    all_mentions = mentions if extra is None else pd.concat([mentions, extra], ignore_index=True)
    all_mentions = all_mentions.drop_duplicates(subset=["ticker", "outlet", "title"])
    scores_table = news.aggregate(all_mentions, HALF_LIFE)
    scores = dict(zip(scores_table["ticker"], scores_table["호재 점수"]))

    ai_added = [t for t in table["ticker"] if t not in config.BY_TICKER][:MAX_AI_TICKERS]
    tickers = [c["ticker"] for c in config.COMPANIES] + ai_added
    prices = download_prices(tickers)

    results, failed = {}, []
    for t in tickers:
        if t not in prices:
            failed.append(t)
            continue
        try:
            m = all_mentions[all_mentions["ticker"] == t]
            results[t] = analyze_ticker(t, prices[t], float(scores.get(t, 0.0)), m)
            with open(DATA / "t" / f"{fid(t)}.json", "w", encoding="utf-8") as f:
                json.dump(results[t], f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except Exception:
            failed.append(t)
            log("계산 실패", t)
            traceback.print_exc()
    log(f"계산 완료 {len(results)}개, 실패 {len(failed)}개")

    radar_rows = []
    for _, r in table.iterrows():
        radar_rows.append({
            "ticker": r["ticker"], "name": r["종목"], "sector": r["분야"], "score": num(r["호재 점수"], 1),
            "n": int(r["기사 수"]), "pos": int(r["호재"]), "neg": int(r["악재"]), "outlets": r["언론사"],
            "latest": pd.Timestamp(r["최근 기사"]).isoformat(), "headline": r["대표 헤드라인"], "link": r["link"],
            "has_data": r["ticker"] in results,
        })
    radar_mentions = []
    for _, m in mentions.sort_values("published", ascending=False).iterrows():
        radar_mentions.append({
            "ticker": m["ticker"], "outlet": m["outlet"], "title": m["title"], "title_ko": m["title_ko"],
            "link": m["link"], "published": pd.Timestamp(m["published"]).isoformat(), "score": num(m["score"], 3),
            "method": m["method"], "why": m["why"],
        })

    brief, brief_model = ai_brief(key, radar_rows, results)
    outlet_counts = pd.Series([h["outlet"] for h in headlines]).value_counts().to_dict() if headlines else {}

    summary = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "settings": {"window": WINDOW, "horizon": HORIZON, "tau": TAU, "beta_pct": BETA * 100, "half_life": HALF_LIFE},
        "headline_count": len(headlines),
        "outlet_counts": outlet_counts,
        "ai": {"model": ai_model, "error": ai_error, "count": len(ai_results)},
        "brief": brief, "brief_model": brief_model,
        "feed_report": [{k: v for k, v in r.items() if k != "주소"} for r in report],
        "radar": radar_rows,
        "mentions": radar_mentions[:400],
        "tickers": [
            {"ticker": t, "file": fid(t), "name": r["name"], "sector": r["sector"], "last": r["last"],
             "currency": r["currency"], "chg5": r["chg5"], "phase": r["phase"], "score": r["score"],
             "fc": r["forecast"]["total_pct"], "p_up": r["forecast"]["p_up"],
             "hit": r["backtest"].get("hit_rate")}
            for t, r in results.items()
        ],
        "failed": failed,
        "elapsed_sec": round(time.time() - started, 1),
    }
    with open(DATA / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    shutil.copy(ROOT / "index.html", SITE / "index.html")
    (SITE / ".nojekyll").write_text("")
    log(f"끝. {summary['elapsed_sec']}초")
    if not results:
        log("주가를 하나도 받지 못했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
