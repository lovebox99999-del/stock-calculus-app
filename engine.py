"""engine.py — 미분과 적분으로 주가를 읽는 계산 엔진.

이 파일은 화면(Streamlit)과 완전히 분리된 순수 계산 코드입니다.
핵심 아이디어는 네 가지입니다.

1) 주가 P 대신 로그 주가 x = ln P 를 쓴다.
   dx/dt = P'/P 이므로, x 의 기울기가 곧 "하루 수익률"이 된다.
2) 주가는 잡음이 많아서 그대로 미분하면 잡음만 커진다.
   그래서 창(window) 안의 점들에 2차 다항식을 맞춘 뒤(평활화) 그 다항식을 미분한다.
   1차 미분 v = 속도(추세의 방향과 세기), 2차 미분 a = 가속도(추세가 강해지는지 약해지는지)
3) 속도를 적분하면 변화량이 된다(미적분의 기본정리).
   v 가 양수인 부분만 적분 = 오른 정도, 음수인 부분만 적분 = 내린 정도
4) 예측은 "앞으로의 속도"를 가정하고 그것을 다시 적분해서 얻는다.
   그리고 그 예측이 과거에 실제로 맞았는지 스스로 검증(백테스트)한다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs, savgol_filter

POLY = 2  # 창 안에서 맞추는 다항식 차수 (2차: 위치·속도·가속도까지 표현)
PERIODS = {"1개월": 21, "3개월": 63, "6개월": 126, "1년": 252}  # 거래일 기준


# ----------------------------------------------------------------------------
# 1. 평활화와 미분
# ----------------------------------------------------------------------------
def fit_window(n: int, window: int) -> int:
    """창 길이를 홀수로 맞추고, 데이터 길이를 넘지 않게 조정한다."""
    w = int(window)
    if w % 2 == 0:
        w += 1
    longest = n if n % 2 == 1 else n - 1
    w = min(w, longest)
    return max(w, 5)


def clean_close(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[close > 0].astype(float)
    if len(close) < 40:
        raise ValueError("분석하려면 종가 데이터가 최소 40일 이상 필요합니다.")
    return close


def analyze(close: pd.Series, window: int = 21) -> pd.DataFrame:
    """로그 주가를 평활화하고 1차·2차 미분을 구한다.

    반환 열
      close     : 실제 종가
      log       : ln(종가)
      trend     : 평활화된 추세선(가격 단위)
      trend_log : 평활화된 추세선(로그 단위)
      v         : 속도 = d(ln P)/dt  (1일당 로그수익률, 0.01 = 하루 약 1%)
      a         : 가속도 = d²(ln P)/dt²
    """
    close = clean_close(close)
    w = fit_window(len(close), window)
    x = np.log(close.to_numpy())
    return pd.DataFrame(
        {
            "close": close.to_numpy(),
            "log": x,
            "trend_log": savgol_filter(x, w, POLY),
            "v": savgol_filter(x, w, POLY, deriv=1),
            "a": savgol_filter(x, w, POLY, deriv=2),
        },
        index=close.index,
    ).assign(trend=lambda d: np.exp(d["trend_log"]))


def causal_derivatives(x: np.ndarray, window: int = 21):
    """'그 시점까지의 데이터만' 써서 구한 속도와 가속도.

    analyze() 는 그래프를 예쁘게 그리려고 각 점의 앞뒤 데이터를 모두 쓴다.
    하지만 예측을 검증할 때 미래 데이터를 쓰면 반칙(미래 참조)이다.
    그래서 여기서는 t 시점에서 [t-w+1, t] 구간만 보고 다항식을 맞춘 뒤
    맨 끝점에서의 미분값을 구한다. 앞쪽 w-1 개는 계산할 수 없어 NaN.
    """
    x = np.asarray(x, dtype=float)
    w = fit_window(len(x), window)
    c1 = savgol_coeffs(w, POLY, deriv=1, pos=w - 1, use="dot")
    c2 = savgol_coeffs(w, POLY, deriv=2, pos=w - 1, use="dot")
    windows = np.lib.stride_tricks.sliding_window_view(x, w)
    v = np.full(len(x), np.nan)
    a = np.full(len(x), np.nan)
    v[w - 1:] = windows @ c1
    a[w - 1:] = windows @ c2
    return v, a


# ----------------------------------------------------------------------------
# 2. 적분: 오른 정도와 내린 정도
# ----------------------------------------------------------------------------
def split_integral(v: np.ndarray):
    """속도 v 를 사다리꼴 공식으로 적분하되, 양수 부분과 음수 부분을 나눠서 구한다.

    구간마다 (상승 기여, 하락 기여)를 돌려준다. 속도의 부호가 구간 안에서
    바뀌면 0이 되는 지점을 찾아 두 개의 삼각형으로 정확히 나눈다.
    항상 sum(up) - sum(down) == 사다리꼴 적분값(v) 이 성립한다.
    """
    v = np.asarray(v, dtype=float)
    v0, v1 = v[:-1], v[1:]
    up = np.zeros(len(v0))
    down = np.zeros(len(v0))

    both_up = (v0 >= 0) & (v1 >= 0)
    both_down = (v0 <= 0) & (v1 <= 0) & ~both_up
    mixed = ~(both_up | both_down)

    up[both_up] = (v0[both_up] + v1[both_up]) / 2
    down[both_down] = -(v0[both_down] + v1[both_down]) / 2

    if mixed.any():
        m0, m1 = v0[mixed], v1[mixed]
        f = m0 / (m0 - m1)            # 구간 안에서 속도가 0이 되는 위치(0~1)
        first = m0 * f / 2            # 앞쪽 삼각형의 부호 있는 넓이
        second = m1 * (1 - f) / 2     # 뒤쪽 삼각형의 부호 있는 넓이
        up[mixed] = np.maximum(first, 0) + np.maximum(second, 0)
        down[mixed] = np.maximum(-first, 0) + np.maximum(-second, 0)
    return up, down


def period_summary(seg: pd.DataFrame) -> dict:
    """분석 구간에서 오른 정도·내린 정도·순변화를 적분으로 구한다."""
    v = seg["v"].to_numpy()
    up, down = split_integral(v)
    U, D = float(up.sum()), float(down.sum())
    trend_net = float(seg["trend_log"].iloc[-1] - seg["trend_log"].iloc[0])
    actual_net = float(seg["log"].iloc[-1] - seg["log"].iloc[0])
    return {
        "up_log": U,
        "down_log": D,
        "up_pct": math.expm1(U) * 100,            # 오른 구간만 이으면 +몇 %
        "down_pct": (1 - math.exp(-D)) * 100,     # 내린 구간만 이으면 -몇 %
        "net_pct_integral": math.expm1(U - D) * 100,   # 속도를 적분해 얻은 순변화
        "net_pct_trend": math.expm1(trend_net) * 100,  # 추세선의 실제 순변화(검산용)
        "net_pct_actual": math.expm1(actual_net) * 100,  # 실제 종가의 순변화
        "efficiency": abs(U - D) / (U + D) if (U + D) > 0 else 0.0,
        "up_days": int((v > 0).sum()),
        "down_days": int((v < 0).sum()),
        "cum_up": np.concatenate([[0.0], np.cumsum(up)]),
        "cum_down": np.concatenate([[0.0], np.cumsum(down)]),
    }


# ----------------------------------------------------------------------------
# 3. 극점·변곡점·파동
# ----------------------------------------------------------------------------
def _sign_changes(values: np.ndarray):
    """부호가 바뀌는 위치를 찾는다. (바뀐 위치, 바뀌기 전 부호) 목록을 돌려준다."""
    s = np.sign(values)
    for i in range(len(s)):           # 정확히 0인 값은 직전 부호를 따른다
        if s[i] == 0:
            s[i] = s[i - 1] if i > 0 else 1.0
    out = []
    for i in np.where(s[1:] != s[:-1])[0]:
        pos = i if abs(values[i]) <= abs(values[i + 1]) else i + 1
        out.append((int(pos), float(s[i])))
    return out


def turning_points(seg: pd.DataFrame) -> dict:
    """v = 0 인 곳이 극대·극소, a = 0 인 곳이 변곡점."""
    peaks, troughs = [], []
    for pos, before in _sign_changes(seg["v"].to_numpy()):
        (peaks if before > 0 else troughs).append(pos)
    inflections = [pos for pos, _ in _sign_changes(seg["a"].to_numpy())]
    return {"peaks": peaks, "troughs": troughs, "inflections": inflections}


def find_legs(seg: pd.DataFrame) -> pd.DataFrame:
    """극점과 극점 사이를 하나의 '파동'으로 보고, 각 파동의 크기를 구한다.

    파동의 크기는 그 구간에서 속도를 적분한 값과 같다.
    """
    v = seg["v"].to_numpy()
    tl = seg["trend_log"].to_numpy()
    cuts = sorted({0, len(v) - 1, *[pos for pos, _ in _sign_changes(v)]})
    rows = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        change = math.expm1(tl[e] - tl[s]) * 100
        rows.append(
            {
                "시작": seg.index[s],
                "끝": seg.index[e],
                "방향": "상승" if change >= 0 else "하락",
                "기간(거래일)": e - s,
                "변화율(%)": round(change, 2),
                "하루 평균(%)": round(change / max(e - s, 1), 3),
            }
        )
    return pd.DataFrame(rows)


def volatility(log_price: np.ndarray, lookback: int = 60) -> float:
    """최근 lookback 일 동안 하루 로그수익률의 표준편차(일일 변동성)."""
    r = np.diff(np.asarray(log_price, dtype=float))[-lookback:]
    return float(np.std(r, ddof=1)) if len(r) > 2 else 0.0


def phase(v0: float, a0: float, sigma: float) -> tuple[str, str]:
    """속도와 가속도의 부호로 현재 국면을 네 가지(+횡보)로 나눈다."""
    if abs(v0) < 0.05 * sigma:
        return "횡보", "속도가 0에 가까워서 뚜렷한 방향이 없습니다."
    if v0 > 0 and a0 > 0:
        return "상승 가속", "오르고 있고, 오르는 속도도 빨라지고 있습니다."
    if v0 > 0:
        return "상승 둔화", "아직 오르고 있지만 속도가 줄고 있습니다. 고점이 가까울 수 있습니다."
    if a0 < 0:
        return "하락 가속", "내리고 있고, 내리는 속도도 빨라지고 있습니다."
    return "하락 둔화", "아직 내리고 있지만 속도가 줄고 있습니다. 바닥을 다지는 중일 수 있습니다."


# ----------------------------------------------------------------------------
# 4. 예측: 미래의 속도를 적분한다
# ----------------------------------------------------------------------------
def trend_integral(v0, a0, h, tau: float, use_accel: bool = True):
    """앞으로 h 일 동안의 예상 로그 변화량.

    미래 속도를  v(s) = (v0 + a0·s)·e^(-s/tau)  로 가정한다.
    지금의 속도와 가속도로 출발하되, 시간이 지날수록 추세가 힘을 잃는다는 뜻이다
    (tau 가 클수록 추세가 오래 간다고 보는 것).  이것을 0부터 h 까지 적분하면

        ∫ v(s) ds = v0·tau·(1 - e^(-h/tau)) + a0·[tau² - (tau² + tau·h)·e^(-h/tau)]
    """
    h = np.asarray(h, dtype=float)
    decay = np.exp(-h / tau)
    out = v0 * tau * (1 - decay)
    if use_accel:
        out = out + a0 * (tau**2 - (tau**2 + tau * h) * decay)
    return out


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def backtest(
    close: pd.Series,
    window: int = 21,
    h: int = 10,
    tau: float = 10.0,
    use_accel: bool = True,
    min_train: int = 120,
    vol_lookback: int = 60,
) -> dict:
    """과거의 매일매일로 돌아가 '그날까지의 정보만으로' 예측하고, 실제와 비교한다.

    추세 신뢰 계수 k 도 이 과정에서 배운다.
      실제 h일 수익률 ≈ k × (추세를 적분해서 얻은 예상 수익률)
    k = 1 이면 추세를 그대로 믿는 것, 0 이면 추세가 쓸모없다는 것,
    음수면 추세가 이어지기보다 되돌려지는 경향이 있었다는 뜻이다.
    t 시점의 k 는 결과가 이미 확정된(t-h 이전) 예측들만으로 계산한다.
    """
    close = clean_close(close)
    x = np.log(close.to_numpy())
    n = len(x)
    v_c, a_c = causal_derivatives(x, window)
    m = trend_integral(v_c, a_c, h, tau, use_accel)      # 그날의 '날것' 추세 예측

    r = np.full(n, np.nan)                                # 그날 이후 h일의 실제 로그수익률
    r[:-h] = x[h:] - x[:-h]

    known = ~np.isnan(m) & ~np.isnan(r)
    c_rm = np.cumsum(np.where(known, r * m, 0.0))
    c_mm = np.cumsum(np.where(known, m * m, 0.0))
    c_n = np.cumsum(known.astype(float))

    # t 시점에 쓸 수 있는 것은 s <= t-h 인 쌍뿐이다(그래야 결과 r_s 를 이미 안다).
    k = np.full(n, np.nan)
    j = np.arange(n) - h
    ok = (j >= 0)
    jj = np.where(ok, j, 0)
    enough = ok & (c_n[jj] >= min_train) & (c_mm[jj] > 0)
    shrink = c_n[jj] / (c_n[jj] + 5.0 * h)               # 표본이 적을수록 0 쪽으로 당긴다
    with np.errstate(divide="ignore", invalid="ignore"):
        k_raw = c_rm[jj] / c_mm[jj] * shrink
    k[enough] = np.clip(k_raw[enough], -1.0, 1.5)

    daily = pd.Series(np.diff(x, prepend=np.nan))
    sigma = daily.rolling(vol_lookback, min_periods=20).std().to_numpy()

    pred = k * m
    ev = ~np.isnan(pred) & ~np.isnan(r) & ~np.isnan(sigma) & (sigma > 0)

    # 범위(밴드)도 스스로 보정한다. 기본 범위는 sigma·sqrt(h) 인데,
    # 과거 예측 오차가 그보다 컸다면 넓히고 작았다면 좁힌다(역시 t-h 이전 결과만 사용).
    raw_band = sigma * math.sqrt(h)
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.where(ev, ((r - pred) / raw_band) ** 2, 0.0)
    c_z2 = np.cumsum(z2)
    c_ev = np.cumsum(ev.astype(float))
    scale = np.ones(n)
    seen = ok & (c_ev[jj] >= 30)
    scale[seen] = np.clip(np.sqrt(c_z2[jj][seen] / c_ev[jj][seen]), 0.7, 2.5)

    table = pd.DataFrame(
        {
            "pred": pred[ev],
            "actual": r[ev],
            "k": k[ev],
            "band": (scale * raw_band)[ev],
        },
        index=close.index[ev],
    )

    k_now = float(k[-1]) if not np.isnan(k[-1]) else 0.0
    result = {
        "table": table,
        "k_now": k_now,
        "band_scale": float(scale[-1]),
        "h": h,
        "n_eval": int(ev.sum()),
    }
    if ev.sum() < 30:
        result["ok"] = False
        return result

    p, y = table["pred"].to_numpy(), table["actual"].to_numpy()
    moved = p != 0
    hit = float(np.mean(np.sign(p[moved]) == np.sign(y[moved]))) if moved.any() else float("nan")
    n_eff = max(ev.sum() / h, 1.0)        # 겹치는 구간을 감안한 '독립적인' 표본 수
    se = math.sqrt(0.25 / n_eff)          # 적중률 50% 가정에서의 표준오차
    mae_model = float(np.mean(np.abs(y - p)))
    mae_zero = float(np.mean(np.abs(y)))
    corr = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else 0.0
    result.update(
        {
            "ok": True,
            "hit_rate": hit,
            "base_up": float(np.mean(y > 0)),       # '무조건 오른다'고 했을 때의 적중률
            "n_eff": float(n_eff),
            "se": se,
            "significant": bool(abs(hit - 0.5) > 2 * se),
            "mae_model": mae_model,
            "mae_zero": mae_zero,
            "skill": 1 - mae_model / mae_zero if mae_zero > 0 else 0.0,
            "corr": corr,
            "coverage68": float(np.mean(np.abs(y - p) <= table["band"].to_numpy())),
        }
    )
    return result


def forecast(
    close: pd.Series,
    window: int = 21,
    h: int = 10,
    tau: float = 10.0,
    use_accel: bool = True,
    k: float = 0.0,
    news_score: float = 0.0,
    beta: float = 0.02,
    tau_news: float = 3.0,
    vol_lookback: int = 60,
    band_scale: float = 1.0,
) -> dict:
    """현재 시점의 예측 경로와 불확실성 범위.

    예상 로그 변화(s일 뒤) = k × 추세 적분(s) + 호재 보정(s)
    호재 보정(s) = beta × (호재점수/100) × 선반영 할인 × (1 - e^(-s/tau_news))
    범위는 과거 변동성으로 정한다: 표준편차 = band_scale × sigma × sqrt(s)
    (band_scale 은 백테스트에서 과거 오차로 보정한 값)
    """
    close = clean_close(close)
    x = np.log(close.to_numpy())
    v_c, a_c = causal_derivatives(x, window)
    v0, a0 = float(v_c[-1]), float(a_c[-1])
    sigma = volatility(x, vol_lookback)
    steps = np.arange(0, h + 1, dtype=float)

    trend_part = k * trend_integral(v0, a0, steps, tau, use_accel)

    # 선반영 할인: 호재 방향으로 최근 3일간 이미 크게 움직였다면 남은 효과를 줄인다.
    score = max(-100.0, min(100.0, float(news_score)))
    direction = np.sign(score)
    recent = float(x[-1] - x[-4]) if len(x) > 4 else 0.0
    z = direction * recent / (sigma * math.sqrt(3)) if sigma > 0 else 0.0
    discount = 1.0 / (1.0 + max(0.0, z))
    news_part = beta * (score / 100.0) * discount * (1 - np.exp(-steps / tau_news))

    center = trend_part + news_part
    sd = band_scale * sigma * np.sqrt(steps)
    p0 = float(close.iloc[-1])
    dates = [close.index[-1]] + list(
        pd.bdate_range(close.index[-1] + pd.offsets.BDay(1), periods=h)
    )
    p_up = _normal_cdf(center[-1] / sd[-1]) if sd[-1] > 0 else 0.5
    return {
        "dates": pd.DatetimeIndex(dates),
        "center": p0 * np.exp(center),
        "lo68": p0 * np.exp(center - sd),
        "hi68": p0 * np.exp(center + sd),
        "lo95": p0 * np.exp(center - 1.96 * sd),
        "hi95": p0 * np.exp(center + 1.96 * sd),
        "p0": p0,
        "v0": v0,
        "a0": a0,
        "sigma": sigma,
        "k": k,
        "trend_pct": math.expm1(trend_part[-1]) * 100,
        "news_pct": math.expm1(news_part[-1]) * 100,
        "total_pct": math.expm1(center[-1]) * 100,
        "raw_trend_pct": math.expm1(float(trend_integral(v0, a0, h, tau, use_accel))) * 100,
        "band68_pct": (math.expm1(center[-1] - sd[-1]) * 100, math.expm1(center[-1] + sd[-1]) * 100),
        "p_up": float(p_up),
        "priced_in_z": float(z),
        "discount": float(discount),
        "recent3_pct": math.expm1(recent) * 100,
    }
