"""news.py — 뉴스 헤드라인 수집, 종목 연결, 키워드 호재 점수.

흐름
  1) 블룸버그·WSJ·로이터·FT 헤드라인을 RSS 로 받는다(제목과 링크만 사용).
  2) 헤드라인에 등장하는 회사 이름을 종목 사전(config.COMPANIES)과 맞춘다.
  3) 금융 뉴스에서 자주 쓰는 표현으로 호재(+)·악재(-) 점수를 매긴다.
  4) 종목별로 모으되, 최근 기사일수록 더 큰 비중을 준다(반감기).
"""
from __future__ import annotations

import html
import math
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import pandas as pd
import requests

import config

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
}
MAX_FEED_BYTES = 5_000_000


# ----------------------------------------------------------------------------
# 1. RSS 수집
# ----------------------------------------------------------------------------
def google_news_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_date(text: str | None):
    if not text:
        return None
    text = text.strip()
    try:
        dt = parsedate_to_datetime(text)                 # RSS 형식 (RFC 822)
    except Exception:
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))   # Atom 형식 (ISO 8601)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clean_title(title: str, source: str | None = None) -> str:
    title = re.sub(r"\s+", " ", html.unescape(title or "")).strip()
    if source and title.endswith(" - " + source):        # 구글 뉴스는 제목 끝에 ' - 언론사'를 붙인다
        title = title[: -len(source) - 3].rstrip()
    return title


def parse_feed(content: bytes) -> list[dict]:
    """RSS 2.0 과 Atom 을 모두 읽어 [{title, link, published}] 로 돌려준다."""
    root = ET.fromstring(content)
    items = []
    for node in root.iter():
        if _local(node.tag) not in ("item", "entry"):
            continue
        title = link = date = source = None
        for child in node:
            name = _local(child.tag)
            if name == "title":
                title = "".join(child.itertext())
            elif name == "link" and not link:
                link = (child.text or "").strip() or child.attrib.get("href")
            elif name in ("pubDate", "published", "updated", "date") and not date:
                date = child.text
            elif name == "source":
                source = (child.text or "").strip()
        title = clean_title(title or "", source)
        if title and link:
            items.append({"title": title, "link": link, "published": _parse_date(date)})
    return items


def fetch_feed(url: str, timeout: int = 12) -> list[dict]:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return parse_feed(resp.content[:MAX_FEED_BYTES])


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def collect_headlines(max_age_days: int = 3, use_google: bool = True):
    """네 언론사의 최근 헤드라인을 모은다.

    반환: (헤드라인 목록, 출처별 수집 결과)
      헤드라인 = {outlet, title, link, published(UTC), via}
    """
    jobs = []
    for outlet, info in config.OUTLETS.items():
        for url in info["feeds"]:
            jobs.append((outlet, url, "공식 RSS"))
        if use_google:
            site = f'site:{info["domain"]}'
            jobs.append((outlet, google_news_url(f"when:{max_age_days}d {site} {config.GOOGLE_NEWS_KEYWORDS}"), "구글 뉴스"))
            jobs.append((outlet, google_news_url(f"when:1d {site}"), "구글 뉴스"))

    def run(job):
        outlet, url, via = job
        try:
            return job, fetch_feed(url), None
        except Exception as exc:                         # 한 곳이 막혀도 나머지는 계속
            return job, [], f"{type(exc).__name__}: {exc}"[:160]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, jobs))

    now = datetime.now(timezone.utc)
    oldest = now - timedelta(days=max_age_days)
    seen, headlines, report = set(), [], []
    for (outlet, url, via), items, error in results:
        kept = 0
        for it in items:
            published = it["published"] or (now - timedelta(hours=12))   # 날짜 없는 기사는 12시간 전으로 가정
            if published < oldest or published > now + timedelta(hours=1):
                continue
            key = (outlet, _norm(it["title"]))
            if key in seen:
                continue
            seen.add(key)
            headlines.append(
                {"outlet": outlet, "title": it["title"], "link": it["link"], "published": published, "via": via}
            )
            kept += 1
        report.append({"언론사": outlet, "경로": via, "받은 기사": len(items), "새 기사": kept, "오류": error or "", "주소": url})
    headlines.sort(key=lambda h: h["published"], reverse=True)
    return headlines, report


def search_company_news(ticker: str, days: int = 7) -> list[dict]:
    """한 종목에 대한 네 언론사의 기사만 구글 뉴스 RSS 로 찾아 온다."""
    company = config.BY_TICKER.get(ticker)
    names = company["aliases"][:2] if company else [ticker.split(".")[0]]
    who = " OR ".join(f'"{n}"' for n in names)
    out, seen = [], set()
    now = datetime.now(timezone.utc)
    for outlet, info in config.OUTLETS.items():
        try:
            items = fetch_feed(google_news_url(f'({who}) site:{info["domain"]} when:{days}d'))
        except Exception:
            continue
        for it in items:
            key = (outlet, _norm(it["title"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "outlet": outlet,
                    "title": it["title"],
                    "link": it["link"],
                    "published": it["published"] or (now - timedelta(hours=12)),
                    "via": "구글 뉴스",
                }
            )
    out.sort(key=lambda h: h["published"], reverse=True)
    return out


# ----------------------------------------------------------------------------
# 2. 헤드라인 → 종목
# ----------------------------------------------------------------------------
def _build_matchers():
    matchers = []
    for c in config.COMPANIES:
        names = sorted(c["aliases"], key=len, reverse=True)
        pattern = r"(?<![A-Za-z0-9])(?:%s)(?![A-Za-z0-9])" % "|".join(re.escape(n) for n in names)
        matchers.append((c["ticker"], re.compile(pattern)))
    return matchers


_MATCHERS = _build_matchers()


def match_companies(title: str) -> list[dict]:
    """헤드라인에 나온 종목과 그 위치를 찾는다.

    'Samsung Biologics' 안의 'Samsung' 처럼 더 긴 이름에 포함된 것은 버린다.
    """
    spans = []
    for ticker, pattern in _MATCHERS:
        for m in pattern.finditer(title):
            spans.append((m.start(), m.end(), ticker))
    found = {}
    for s, e, ticker in spans:
        inside_longer = any(
            t2 != ticker and s2 <= s and e <= e2 and (e2 - s2) > (e - s) for s2, e2, t2 in spans
        )
        if not inside_longer and ticker not in found:
            found[ticker] = {"ticker": ticker, "start": s, "end": e}
    return list(found.values())


# ----------------------------------------------------------------------------
# 3. 키워드 호재 점수
#    (표현, 가중치). 위에 있는 긴 표현부터 찾고, 찾은 부분은 지워서 중복 계산을 막는다.
# ----------------------------------------------------------------------------
_EST = r"(?:estimates?|expectations?|forecasts?|views?|targets?)"
_GUIDE = r"(?:guidance|forecasts?|outlook|targets?|dividends?|buybacks?)"
LEXICON = [
    # --- 호재: 긴 표현 ---
    (rf"beats? (?:[\w-]+ ){{0,2}}{_EST}", 2.0),
    (rf"tops? (?:[\w-]+ ){{0,2}}{_EST}", 2.0),
    (r"better[- ]than[- ]expected", 2.0),
    (rf"(?:raises?|raised|lifts?|lifted|boosts?|boosted|hikes?|hiked) (?:[\w-]+ ){{0,3}}{_GUIDE}", 2.0),
    (r"(?:record|all[- ]time) highs?", 2.0),
    (r"record (?:[\w-]+ )?(?:profits?|revenue|sales|earnings|quarter|deliveries|orders)", 2.0),
    (r"(?:wins?|won|secures?|secured|lands?|landed|awarded) (?:[\w-]+ ){0,4}(?:approval|contracts?|orders?|deal|clearance)", 1.8),
    (r"(?:fda|eu|regulatory|regulators?) (?:approval|approves?|approved|clearance|clears?|cleared|nod)", 1.8),
    (r"(?:share |stock )?buybacks?", 1.5),
    (r"strong (?:demand|sales|earnings|results|growth|quarter|outlook)", 1.5),
    # --- 악재: 긴 표현 ---
    (rf"miss(?:es|ed)? (?:[\w-]+ ){{0,2}}{_EST}", -2.0),
    (r"falls? short|fell short", -2.0),
    (r"worse[- ]than[- ]expected", -2.0),
    (rf"(?:cuts?|lowers?|lowered|slash(?:es|ed)?|trims?|trimmed|reduces?|reduced|withdraws?|withdrew) (?:[\w-]+ ){{0,3}}{_GUIDE}", -2.0),
    (r"profit warning", -2.0),
    (r"(?:record|all[- ]time|multi[- ]year) lows?", -2.0),
    (r"job cuts|lay(?:s|ing)? off|layoffs?", -0.6),
    (r"short[- ]sell(?:ers?|ing)", -1.3),
    (r"sell[- ]?off", -1.5),
    # --- 호재: 단어 ---
    (r"soar(?:s|ed|ing)?|skyrocket(?:s|ed|ing)?", 2.0),
    (r"surg(?:e|es|ed|ing)", 1.8),
    (r"jump(?:s|ed|ing)?|rall(?:y|ies|ied|ying)", 1.5),
    (r"upgrad(?:e|es|ed)|bullish|breakthrough", 1.5),
    (r"beat(?:s|ing)?|outperform(?:s|ed)?|doubles?|doubled|triples?|tripled|approv(?:al|es|ed)", 1.2),
    (r"climb(?:s|ed|ing)?|ris(?:e|es|ing)|rose|gain(?:s|ed|ing)?|rebound(?:s|ed|ing)?|recover(?:s|ed|y|ing)?", 1.0),
    (r"milestone|optimis(?:m|tic)|boom(?:s|ing)?", 1.0),
    (r"advanc(?:e|es|ed)|boost(?:s|ed)?|wins?|partner(?:s|ship|ed)?", 0.8),
    (r"expan(?:d|ds|ded|sion)|strong(?:er|est)?", 0.6),
    (r"launch(?:es|ed)?|growth", 0.5),
    # --- 악재: 단어 ---
    (r"plung(?:e|es|ed|ing)|plummet(?:s|ed|ing)?|crash(?:es|ed|ing)?|collaps(?:e|es|ed)|fraud|bankruptc(?:y|ies)", -2.0),
    (r"tumbl(?:e|es|ed|ing)", -1.8),
    (r"slump(?:s|ed|ing)?|sink(?:s|ing)?|sank|sunk|downgrad(?:e|es|ed)|bearish|recall(?:s|ed)?", -1.5),
    (r"scandal|default(?:s|ed)?|shortfall|disappoint(?:s|ed|ing|ment)?", -1.5),
    (r"prob(?:e|es|ed)|investigat(?:ion|es|ed|ing)|warn(?:s|ed|ing)?|fail(?:s|ed|ure)?", -1.3),
    (r"ban(?:s|ned)?|hack(?:s|ed)?|breach(?:es|ed)?", -1.3),
    (r"lawsuit|sued|fined|penalt(?:y|ies)|underperform(?:s|ed)?|miss(?:es|ed)?", -1.2),
    (r"slid(?:e|es|ing)?|drop(?:s|ped|ping)?|fall(?:s|ing)?|fell|declin(?:e|es|ed|ing)", -1.0),
    (r"los(?:s|ses|es|ing)|lost|weak(?:er|ens|ening|ness)?|delay(?:s|ed)?|halt(?:s|ed)?|suspend(?:s|ed)?", -1.0),
    (r"reject(?:s|ed)?|struggl(?:e|es|ed|ing)|glut", -1.0),
    (r"antitrust", -0.8),
    (r"dip(?:s|ped)?|concerns?|worr(?:y|ies|ied)|fears?|slow(?:s|ed|ing|down)?", -0.6),
    (r"tariffs?", -0.5),
    (r"pressure", -0.4),
]
_LEXICON = [(re.compile(rf"\b(?:{p})\b"), w) for p, w in LEXICON]
_NEGATORS = re.compile(r"(?:\bnot\b|\bno\b|\bnever\b|\bwithout\b|n't\b|\bfail(?:s|ed)? to\b|\bunlikely to\b)\s+(?:\w+\s+){0,2}$")
_SPLIT = re.compile(r"\s+(?:as|while|but|though|although|despite|even as)\s+|[;:,]\s+|\s[—–-]\s")
_SPECULATIVE = re.compile(r"\b(?:could|may|might|reportedly|rumou?rs?|considers?|weighs?|mulls?)\b")


def _clause_score(text: str):
    """한 절(clause)의 점수와, 점수를 만든 표현들을 돌려준다."""
    work = text.lower()
    total, hits = 0.0, []
    for pattern, weight in _LEXICON:
        while True:
            m = pattern.search(work)
            if not m:
                break
            w = weight
            if _NEGATORS.search(work[: m.start()]):      # 'fails to beat', 'no growth' → 뒤집기
                w = -0.6 * w
            total += w
            hits.append(f"{m.group(0)}({w:+.1f})")
            work = work[: m.start()] + " " * (m.end() - m.start()) + work[m.end():]
    return total, hits


def keyword_score(title: str, start: int | None = None, end: int | None = None):
    """헤드라인의 호재 점수(-1 ~ +1)와 근거 표현.

    'Nvidia surges as AMD slumps' 같은 문장을 위해 절 단위로 나눈다.
    회사 이름이 들어 있는 절은 100%, 다른 절은 40%만 반영한다.
    """
    bounds, last = [], 0
    for m in _SPLIT.finditer(title):
        bounds.append((last, m.start()))
        last = m.end()
    bounds.append((last, len(title)))

    own_total, other_total, hits = 0.0, 0.0, []
    for s, e in bounds:
        score, h = _clause_score(title[s:e])
        is_own = start is None or (s <= start < e)
        if is_own:
            own_total += score
        else:
            other_total += score
        hits += h
    total = own_total + (0.4 if own_total != 0 else 0.6) * other_total
    if title.rstrip().endswith("?"):                      # 물음표로 끝나는 제목은 불확실
        total *= 0.6
    if _SPECULATIVE.search(title.lower()):               # '~할 수도 있다' 류
        total *= 0.7
    return math.tanh(total / 2), hits


# ----------------------------------------------------------------------------
# 4. 종목별 집계
# ----------------------------------------------------------------------------
_KNOWN = {t.upper().replace(".", "-"): t for t in config.BY_TICKER}


def _canonical(ticker: str) -> str:
    """AI 가 준 종목 코드를 사전의 표기로 맞춘다 (예: BRK.B → BRK-B)."""
    ticker = str(ticker).strip().upper()
    return _KNOWN.get(ticker.replace(".", "-"), ticker)


def build_mentions(
    headlines: list[dict], ai_results: dict | None = None, force_ticker: str | None = None
) -> pd.DataFrame:
    """(헤드라인, 종목) 쌍마다 한 줄. AI 판단이 있으면 그것을, 없으면 키워드 점수를 쓴다.

    force_ticker 를 주면(종목을 지정해 검색해 온 기사), 제목에 회사 이름이 없어도
    그 종목의 기사로 보고 그 종목에 대한 줄만 만든다.
    """
    ai_results = ai_results or {}
    rows = []
    for h in headlines:
        ai = ai_results.get(h["title"])
        matched = {m["ticker"]: m for m in match_companies(h["title"])}
        ai_items = {_canonical(it["ticker"]): it for it in (ai or {}).get("items", [])}
        tickers = [force_ticker] if force_ticker else list(dict.fromkeys(list(matched) + list(ai_items)))
        for ticker in tickers:
            m = matched.get(ticker)
            kw, hits = keyword_score(h["title"], m["start"], m["end"]) if m else keyword_score(h["title"])
            if ai is not None:
                if ticker not in ai_items:
                    continue                      # AI 가 이 종목 기사가 아니라고 판단
                item = ai_items[ticker]
                score, method, why, event = item["score"], "AI", item.get("why", ""), item.get("event", "")
            else:
                score, method, why, event = kw, "키워드", ", ".join(hits), ""
            rows.append(
                {
                    "ticker": ticker,
                    "outlet": h["outlet"],
                    "title": h["title"],
                    "title_ko": (ai or {}).get("ko", ""),
                    "link": h["link"],
                    "published": h["published"],
                    "score": float(score),
                    "keyword_score": float(kw),
                    "method": method,
                    "event": event,
                    "why": why,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["ticker", "outlet", "title", "title_ko", "link", "published", "score",
                 "keyword_score", "method", "event", "why"],
    )


def aggregate(mentions: pd.DataFrame, half_life_hours: float = 36.0, now=None) -> pd.DataFrame:
    """종목별 호재 점수(-100 ~ +100).

    기사 하나의 비중 w = 0.5^(경과 시간 / 반감기)
    평균 논조 = Σ(기사 점수 × w) / Σw            (-1 ~ +1)
    확신도   = 1 - e^(-Σw / 1.5)                 (기사가 쌓일수록 1에 가까워진다)
    종목 점수 = 100 × 평균 논조 × 확신도
    → 강한 호재 기사 하나면 약 +45, 같은 방향 기사가 서너 개 겹치면 +80 안팎.
      기사가 아무리 많아도 호재와 악재가 섞여 있으면 점수는 낮게 나온다.
    """
    cols = ["ticker", "종목", "분야", "호재 점수", "기사 수", "호재", "악재", "언론사", "최근 기사", "대표 헤드라인", "link"]
    if mentions.empty:
        return pd.DataFrame(columns=cols)
    now = now or datetime.now(timezone.utc)
    m = mentions.copy()
    age_h = (now - m["published"]).dt.total_seconds().clip(lower=0) / 3600
    m["weight"] = 0.5 ** (age_h / half_life_hours)
    m["contrib"] = m["score"] * m["weight"]
    rows = []
    for ticker, g in m.groupby("ticker", sort=False):
        top = g.loc[g["contrib"].abs().idxmax()]
        total_w = float(g["weight"].sum())
        company = config.BY_TICKER.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "종목": company.get("ko", ticker),
                "분야": company.get("sector", "AI 추가"),
                "호재 점수": round(
                    100 * (g["contrib"].sum() / total_w) * (1 - math.exp(-total_w / 1.5)) if total_w > 0 else 0.0, 1
                ),
                "기사 수": len(g),
                "호재": int((g["score"] > 0.15).sum()),
                "악재": int((g["score"] < -0.15).sum()),
                "언론사": ", ".join(sorted(g["outlet"].unique())),
                "최근 기사": g["published"].max(),
                "대표 헤드라인": top["title_ko"] or top["title"],
                "link": top["link"],
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["호재 점수", "기사 수"], ascending=[False, False]).reset_index(drop=True)
