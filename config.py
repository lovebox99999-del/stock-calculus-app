"""config.py — 직접 고쳐 쓰는 설정 파일.

키, 뉴스 출처, 종목 사전이 모두 여기에 있습니다.
"""
from __future__ import annotations

import os

# ----------------------------------------------------------------------------
# 1. Gemini API 키 (AI 호재 판단·AI 브리핑에 사용)
#    키는 코드에 적지 않습니다. GitHub 저장소의 Settings > Secrets and variables > Actions 에
#    GEMINI_API_KEY 라는 이름으로 넣어 두면, 1시간마다 도는 자동 작업이 그 값을 읽어 씁니다.
#    키가 없으면 AI 판단 대신 키워드 점수로 계산합니다.
# ----------------------------------------------------------------------------
# 위에서부터 차례로 시도하고, 모두 실패하면 사용 가능한 flash 모델을 자동으로 찾습니다.
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-flash-latest"]


def get_gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


# ----------------------------------------------------------------------------
# 2. 뉴스 출처
#    feeds  : 언론사가 직접 제공하는 RSS
#    domain : 구글 뉴스 RSS 에서 그 언론사 기사만 골라 올 때 쓰는 주소
#    (로이터는 2020년에 공식 RSS 를 닫아서 구글 뉴스 경로만 씁니다.)
# ----------------------------------------------------------------------------
OUTLETS = {
    "Bloomberg": {
        "domain": "bloomberg.com",
        "feeds": [
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://feeds.bloomberg.com/technology/news.rss",
            "https://feeds.bloomberg.com/industries/news.rss",
        ],
    },
    "WSJ": {
        "domain": "wsj.com",
        "feeds": [
            "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
            "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness",
            "https://feeds.content.dowjones.io/public/rss/RSSWSJD",
        ],
    },
    "Reuters": {"domain": "reuters.com", "feeds": []},
    "Financial Times": {
        "domain": "ft.com",
        "feeds": [
            "https://www.ft.com/markets?format=rss",
            "https://www.ft.com/companies?format=rss",
            "https://www.ft.com/technology?format=rss",
        ],
    },
}

# 구글 뉴스에서 종목 관련 기사를 더 잘 건지기 위한 검색어
GOOGLE_NEWS_KEYWORDS = "(shares OR stock OR earnings OR profit OR forecast)"

# ----------------------------------------------------------------------------
# 3. 종목 사전: 헤드라인에 aliases 중 하나가 나오면 그 종목 기사로 본다.
#    (대소문자를 구분합니다. 'Target', 'Visa' 처럼 일반 단어와 헷갈리는 이름은 뺐습니다.)
#    종목을 추가하려면 같은 형식으로 한 줄 더 쓰면 됩니다.
# ----------------------------------------------------------------------------
def _c(ticker, ko, sector, *aliases):
    return {"ticker": ticker, "ko": ko, "sector": sector, "aliases": list(aliases)}


COMPANIES = [
    # 빅테크·소프트웨어
    _c("AAPL", "애플", "빅테크", "Apple"),
    _c("MSFT", "마이크로소프트", "빅테크", "Microsoft"),
    _c("GOOGL", "알파벳(구글)", "빅테크", "Alphabet", "Google"),
    _c("AMZN", "아마존", "빅테크", "Amazon"),
    _c("META", "메타", "빅테크", "Meta Platforms", "Meta", "Facebook"),
    _c("NFLX", "넷플릭스", "빅테크", "Netflix"),
    _c("ORCL", "오라클", "소프트웨어", "Oracle"),
    _c("CRM", "세일즈포스", "소프트웨어", "Salesforce"),
    _c("ADBE", "어도비", "소프트웨어", "Adobe"),
    _c("IBM", "IBM", "소프트웨어", "IBM"),
    _c("PLTR", "팔란티어", "소프트웨어", "Palantir"),
    _c("SNOW", "스노우플레이크", "소프트웨어", "Snowflake"),
    _c("CRWD", "크라우드스트라이크", "소프트웨어", "CrowdStrike"),
    _c("PANW", "팔로알토네트웍스", "소프트웨어", "Palo Alto Networks"),
    _c("SHOP", "쇼피파이", "소프트웨어", "Shopify"),
    _c("UBER", "우버", "플랫폼", "Uber"),
    _c("ABNB", "에어비앤비", "플랫폼", "Airbnb"),
    _c("SPOT", "스포티파이", "플랫폼", "Spotify"),
    # 반도체·하드웨어
    _c("NVDA", "엔비디아", "반도체", "Nvidia", "NVIDIA"),
    _c("AMD", "AMD", "반도체", "AMD", "Advanced Micro Devices"),
    _c("INTC", "인텔", "반도체", "Intel"),
    _c("AVGO", "브로드컴", "반도체", "Broadcom"),
    _c("QCOM", "퀄컴", "반도체", "Qualcomm"),
    _c("MU", "마이크론", "반도체", "Micron"),
    _c("TSM", "TSMC", "반도체", "TSMC", "Taiwan Semiconductor"),
    _c("ASML", "ASML", "반도체", "ASML"),
    _c("ARM", "Arm", "반도체", "Arm Holdings"),
    _c("AMAT", "어플라이드머티어리얼즈", "반도체", "Applied Materials"),
    _c("TXN", "텍사스인스트루먼트", "반도체", "Texas Instruments"),
    _c("SMCI", "슈퍼마이크로", "반도체", "Super Micro", "Supermicro"),
    _c("DELL", "델", "하드웨어", "Dell"),
    # 자동차
    _c("TSLA", "테슬라", "자동차", "Tesla"),
    _c("F", "포드", "자동차", "Ford Motor", "Ford"),
    _c("GM", "GM", "자동차", "General Motors", "GM"),
    _c("RIVN", "리비안", "자동차", "Rivian"),
    _c("TM", "도요타", "자동차", "Toyota"),
    _c("NIO", "니오", "자동차", "Nio", "NIO"),
    # 금융
    _c("JPM", "JP모건", "금융", "JPMorgan", "JP Morgan"),
    _c("GS", "골드만삭스", "금융", "Goldman Sachs", "Goldman"),
    _c("MS", "모건스탠리", "금융", "Morgan Stanley"),
    _c("BAC", "뱅크오브아메리카", "금융", "Bank of America"),
    _c("C", "씨티그룹", "금융", "Citigroup", "Citi"),
    _c("WFC", "웰스파고", "금융", "Wells Fargo"),
    _c("BLK", "블랙록", "금융", "BlackRock"),
    _c("BRK-B", "버크셔해서웨이", "금융", "Berkshire Hathaway", "Berkshire"),
    _c("MA", "마스터카드", "금융", "Mastercard"),
    _c("PYPL", "페이팔", "금융", "PayPal"),
    _c("COIN", "코인베이스", "금융", "Coinbase"),
    # 바이오·헬스케어
    _c("LLY", "일라이릴리", "바이오", "Eli Lilly", "Lilly"),
    _c("NVO", "노보노디스크", "바이오", "Novo Nordisk", "Novo"),
    _c("PFE", "화이자", "바이오", "Pfizer"),
    _c("MRNA", "모더나", "바이오", "Moderna"),
    _c("JNJ", "존슨앤드존슨", "바이오", "Johnson & Johnson", "J&J"),
    _c("MRK", "머크", "바이오", "Merck"),
    _c("ABBV", "애브비", "바이오", "AbbVie"),
    _c("AMGN", "암젠", "바이오", "Amgen"),
    _c("GILD", "길리어드", "바이오", "Gilead"),
    _c("REGN", "리제네론", "바이오", "Regeneron"),
    _c("VRTX", "버텍스", "바이오", "Vertex Pharmaceuticals"),
    _c("AZN", "아스트라제네카", "바이오", "AstraZeneca"),
    _c("BNTX", "바이오엔테크", "바이오", "BioNTech"),
    _c("NVS", "노바티스", "바이오", "Novartis"),
    _c("SNY", "사노피", "바이오", "Sanofi"),
    _c("GSK", "GSK", "바이오", "GSK", "GlaxoSmithKline"),
    _c("UNH", "유나이티드헬스", "헬스케어", "UnitedHealth"),
    _c("ISRG", "인튜이티브서지컬", "헬스케어", "Intuitive Surgical"),
    # 에너지·친환경
    _c("XOM", "엑슨모빌", "에너지", "Exxon Mobil", "ExxonMobil", "Exxon"),
    _c("CVX", "셰브론", "에너지", "Chevron"),
    _c("BP", "BP", "에너지", "BP"),
    _c("OXY", "옥시덴털", "에너지", "Occidental"),
    _c("NEE", "넥스트에라에너지", "친환경에너지", "NextEra"),
    _c("FSLR", "퍼스트솔라", "친환경에너지", "First Solar"),
    _c("ENPH", "엔페이즈", "친환경에너지", "Enphase"),
    _c("PLUG", "플러그파워", "친환경에너지", "Plug Power"),
    _c("BE", "블룸에너지", "친환경에너지", "Bloom Energy"),
    _c("CEG", "컨스텔레이션에너지", "친환경에너지", "Constellation Energy"),
    _c("GEV", "GE버노바", "친환경에너지", "GE Vernova"),
    # 소비재
    _c("WMT", "월마트", "소비재", "Walmart"),
    _c("COST", "코스트코", "소비재", "Costco"),
    _c("NKE", "나이키", "소비재", "Nike"),
    _c("SBUX", "스타벅스", "소비재", "Starbucks"),
    _c("MCD", "맥도날드", "소비재", "McDonald's", "McDonald’s", "McDonalds"),
    _c("KO", "코카콜라", "소비재", "Coca-Cola"),
    _c("PEP", "펩시코", "소비재", "PepsiCo", "Pepsi"),
    _c("DIS", "디즈니", "소비재", "Disney"),
    _c("HD", "홈디포", "소비재", "Home Depot"),
    _c("LULU", "룰루레몬", "소비재", "Lululemon"),
    # 산업재·항공·방산
    _c("BA", "보잉", "산업재", "Boeing"),
    _c("CAT", "캐터필러", "산업재", "Caterpillar"),
    _c("GE", "GE에어로스페이스", "산업재", "GE Aerospace", "General Electric"),
    _c("LMT", "록히드마틴", "산업재", "Lockheed Martin", "Lockheed"),
    _c("RTX", "RTX", "산업재", "RTX", "Raytheon"),
    _c("UPS", "UPS", "산업재", "UPS"),
    _c("FDX", "페덱스", "산업재", "FedEx"),
    _c("DAL", "델타항공", "산업재", "Delta Air Lines"),
    _c("UAL", "유나이티드항공", "산업재", "United Airlines"),
    # 아시아
    _c("BABA", "알리바바", "아시아", "Alibaba"),
    _c("BIDU", "바이두", "아시아", "Baidu"),
    _c("SONY", "소니", "아시아", "Sony"),
    # 한국 (야후 파이낸스 코드: 코스피 .KS / 코스닥 .KQ)
    _c("005930.KS", "삼성전자", "한국", "Samsung Electronics", "Samsung"),
    _c("000660.KS", "SK하이닉스", "한국", "SK Hynix", "SK hynix"),
    _c("005380.KS", "현대차", "한국", "Hyundai Motor", "Hyundai"),
    _c("000270.KS", "기아", "한국", "Kia"),
    _c("373220.KS", "LG에너지솔루션", "한국", "LG Energy Solution", "LG Energy"),
    _c("207940.KS", "삼성바이오로직스", "한국", "Samsung Biologics"),
    _c("068270.KS", "셀트리온", "한국", "Celltrion"),
    _c("035420.KS", "네이버", "한국", "Naver"),
    _c("035720.KS", "카카오", "한국", "Kakao"),
    _c("051910.KS", "LG화학", "한국", "LG Chem"),
    _c("006400.KS", "삼성SDI", "한국", "Samsung SDI"),
    _c("005490.KS", "포스코홀딩스", "한국", "POSCO", "Posco"),
    _c("066570.KS", "LG전자", "한국", "LG Electronics"),
    _c("012450.KS", "한화에어로스페이스", "한국", "Hanwha Aerospace"),
    _c("034020.KS", "두산에너빌리티", "한국", "Doosan Enerbility"),
    _c("329180.KS", "HD현대중공업", "한국", "HD Hyundai Heavy"),
]

BY_TICKER = {c["ticker"]: c for c in COMPANIES}


def display_name(ticker: str) -> str:
    c = BY_TICKER.get(ticker)
    return f'{c["ko"]} ({ticker})' if c else ticker
