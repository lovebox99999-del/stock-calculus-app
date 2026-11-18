# 미적분 주가 분석기 (GitHub Pages 버전)

뉴스에서 호재가 뜬 종목을 찾고, 주가를 미분·적분해서 흐름을 읽고, 앞으로의 범위를 예측한 뒤,
같은 방법이 과거에 얼마나 맞았는지까지 스스로 채점하는 웹페이지입니다.

서버 없이 GitHub만으로 돌아갑니다.
- **GitHub Actions**가 1시간마다 `build.py`를 실행해 뉴스·주가를 받아 계산하고, 결과를 `gh-pages` 브랜치에 올립니다.
- **GitHub Pages**가 그 결과를 웹페이지로 보여 줍니다. 링크는 잠들지 않고 늘 켜져 있습니다.

## 처음 한 번 해야 할 설정

1. **API 키 넣기**: 저장소 Settings → Secrets and variables → Actions → New repository secret
   이름 `GEMINI_API_KEY`, 값에 Gemini 키를 넣습니다. 키가 없으면 AI 판단 대신 키워드 점수로 계산합니다.
2. **첫 실행**: Actions 탭 → '데이터 업데이트' → Run workflow. 5~10분쯤 걸립니다.
3. **Pages 켜기**: Settings → Pages → Source 를 'Deploy from a branch', Branch 를 `gh-pages` / `(root)` 로 저장합니다.
4. 1~2분 뒤 Settings → Pages 위쪽에 나오는 주소(`https://아이디.github.io/저장소이름/`)가 페이지 링크입니다.

## 파일 구성

| 파일 | 내용 |
|---|---|
| `build.py` | 1시간마다 실행되는 계산 스크립트 (뉴스 → 호재 점수 → 주가 → 미분·적분·예측·검증 → JSON) |
| `index.html` | 웹페이지. 계산 결과 JSON 을 읽어 그래프를 그립니다 |
| `engine.py` | 미분·적분·예측·검증 계산 |
| `news.py` | 헤드라인 수집, 종목 연결, 키워드 호재 점수 |
| `ai_judge.py` | Gemini 로 호재 판단·브리핑 |
| `market_data.py` | 주가 데이터 받기 |
| `config.py` | 뉴스 출처, 종목 사전 (종목을 추가하려면 여기에 한 줄) |
| `.github/workflows/update.yml` | 1시간마다 도는 자동 작업 설정 |

## 알아 둘 점

- 화면의 숫자는 맨 위 '마지막 업데이트' 시각 기준입니다. GitHub 의 예약 실행은 붐비면 몇십 분 늦어질 수 있습니다.
- 종목 사전에 있는 종목과, 그때그때 AI 가 호재 종목으로 추가한 종목만 볼 수 있습니다.
- 공개 저장소에서 60일 동안 코드 변경이 없으면 GitHub 가 예약 실행을 멈출 수 있습니다. 그런 메일이 오면 Actions 탭에서 다시 켜면 됩니다.
- 주가의 단기 방향은 원래 맞히기 어렵습니다. 검증 화면의 적중률이 50% 근처인 종목이 많은 것이 정상입니다.
- 호재가 주가를 얼마나 움직이는지는 가정값이며 검증되지 않았습니다. 이 페이지의 결과는 투자 권유가 아닙니다.
