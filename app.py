import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter
from pandas.tseries.offsets import BDay
import time
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 1. 페이지 설정 및 UI
# ---------------------------------------------------------
st.set_page_config(page_title="The Oracle: 맞춤형 타점 & 확률 분석기", page_icon="🔮", layout="wide")

st.title("🔮 The Oracle: 진입 시점 맞춤형 출구 전략기")
st.markdown("""
과거의 통계를 바탕으로 **내일 상승 확률 지도**를 그리고, 
내가 진입했던 날짜의 **'시그마(Sigma) 조건'**과 동일한 환경에서 과거에 어떻게 매도했을 때 가장 수익이 컸는지 2D 정밀 최적화로 분석합니다.
""")

with st.sidebar:
    st.header("⚙️ 내 진입 정보 입력")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    entry_date = st.date_input("진입 날짜 (매수일)")
    entry_price = st.number_input("매수 평단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003 # 고정 수수료 0.3%
    run_btn = st.button("🚀 확률 지도 & 맞춤형 전략 추출", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 확률 계산 및 맞춤형 2D 최적화 엔진
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_custom_oracle(ticker, ent_date, tax_rate, fee_rate):
    df = yf.download(ticker, start="2015-01-01", progress=False)
    if df.empty: return None, None, None, "데이터를 불러오지 못했습니다."
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Open', 'Close']].dropna()
    closes = df['Close'].values
    opens = df['Open'].values
    dates = df.index
    n_days = len(closes)
    
    win = 20
    sigmas = np.full(n_days, 999.0)
    slopes = np.full(n_days, -999.0)
    x = np.arange(win)
    
    # 지표 및 내일의 수익률 계산
    next_rets = np.zeros(n_days)
    for i in range(win, n_days):
        y = closes[i-win:i]
        s, inter, _, _, _ = linregress(x, y)
        std = np.std(y - (s*x + inter))
        if std > 0: sigmas[i] = (closes[i] - (s*(win-1)+inter)) / std
        if closes[i] > 0: slopes[i] = (s / closes[i]) * 100
        
        # 내일 상승/하락 여부 기록 (종가 기준)
        if i < n_days - 1:
            next_rets[i] = (closes[i+1] / closes[i]) - 1.0

    # ---------------------------------------------------------
    # 📊 Part 1: 내일 상승 확률 지도 (Probability Map)
    # ---------------------------------------------------------
    # 과거의 시그마 값을 0.1 단위로 그룹화하여 내일 오를 확률(승률) 계산
    prob_df = pd.DataFrame({'Sigma': np.round(sigmas[win:-1], 1), 'NextRet': next_rets[win:-1]})
    prob_df['IsUp'] = (prob_df['NextRet'] > 0).astype(int)
    
    # 시그마 구간별 상승 확률
    win_rates = prob_df.groupby('Sigma')['IsUp'].mean() * 100
    
    # 오늘자 가격 환산을 위한 파라미터
    y_last = closes[-win:]
    s_last, inter_last, _, _, _ = linregress(x, y_last)
    L_last = s_last*(win-1) + inter_last
    std_last = np.std(y_last - (s_last*x + inter_last))
    
    cur_price = closes[-1]
    cur_sigma = sigmas[-1]
    
    # 확률 구간별 밴드 생성 (90~99, 70~90, 50~70, 30~50, 10~30, 1~10)
    prob_bands = {
        '90% ~ 99%': [], '70% ~ 90%': [], '50% ~ 70%': [], 
        '30% ~ 50%': [], '10% ~ 30%': [], '1% ~ 10%': []
    }
    
    for sig_val, win_rate in win_rates.items():
        if sig_val == 999.0: continue
        price_at_sig = L_last + (sig_val * std_last)
        
        if 90 <= win_rate < 100: prob_bands['90% ~ 99%'].append(price_at_sig)
        elif 70 <= win_rate < 90: prob_bands['70% ~ 90%'].append(price_at_sig)
        elif 50 <= win_rate < 70: prob_bands['50% ~ 70%'].append(price_at_sig)
        elif 30 <= win_rate < 50: prob_bands['30% ~ 50%'].append(price_at_sig)
        elif 10 <= win_rate < 30: prob_bands['10% ~ 30%'].append(price_at_sig)
        elif 0 < win_rate < 10: prob_bands['1% ~ 10%'].append(price_at_sig)

    # ---------------------------------------------------------
    # 🎯 Part 2: 진입 시점 맞춤형 출구 최적화 (Custom Entry Optimization)
    # ---------------------------------------------------------
    ent_date_pd = pd.to_datetime(ent_date)
    # 입력한 날짜와 가장 가까운 거래일 찾기
    if ent_date_pd not in dates:
        closest_date_idx = np.argmin(np.abs(dates - ent_date_pd))
    else:
        closest_date_idx = dates.get_loc(ent_date_pd)
        
    my_ent_sigma = sigmas[closest_date_idx]
    
    if my_ent_sigma == 999.0:
        return None, None, None, "상장 초기이거나 데이터가 부족한 시점입니다. 다른 날짜를 선택하세요."

    # 사용자의 진입 조건을 시스템의 'Ent' 조건으로 고정 (음수 전환)
    # 사용자가 +1.5 시그마에 샀다면, 시스템은 Sigma <= 1.5 인 모든 과거를 백테스트
    custom_ent_param = np.round(-my_ent_sigma, 1)

    # 0.1 간격의 촘촘한 최적화 (단타 -1.0부터 장기 5.0까지)
    DROP_RANGE = np.round(np.arange(0.1, 5.1, 0.1), 1)
    EXT_RANGE = np.round(np.arange(-1.0, 5.1, 0.1), 1)
    
    shape = (len(DROP_RANGE), len(EXT_RANGE))
    ret_grid = np.full(shape, -100.0)
    adr_grid = np.full(shape, -100.0)
    all_res = []
    
    for i_drop, drop in enumerate(DROP_RANGE):
        for i_ext, ext in enumerate(EXT_RANGE):
            cap = 1.0; hold = False; buy_p = 0.0; ent_slope = 0.0
            trades = 0; hold_days = 0
            
            for k in range(win, n_days-1):
                if not hold:
                    # 사용자 진입 시그마와 같거나 더 유리한 조건일 때 진입
                    if sigmas[k] <= -custom_ent_param:
                        hold = True; buy_p = opens[k+1]; ent_slope = slopes[k]
                        trades += 1
                else:
                    hold_days += 1
                    if sigmas[k] >= ext or slopes[k] < (ent_slope - drop):
                        hold = False; sell_p = opens[k+1]
                        profit = sell_p - buy_p
                        tax_amt = profit * tax_rate if profit > 0 else 0
                        net_ret = ((sell_p - tax_amt) / buy_p) - 1.0 - fee_rate
                        cap *= (1.0 + net_ret)
                        
            if trades > 0:
                tot_ret = (cap - 1.0) * 100
                adr = (tot_ret / hold_days) if hold_days > 0 else 0
                ret_grid[i_drop, i_ext] = tot_ret
                adr_grid[i_drop, i_ext] = adr
                all_res.append({'Drop': drop, 'Ext': ext, 'TotRet': tot_ret, 'ADR': adr, 'Trades': trades})

    if not all_res: return None, None, None, "해당 진입 조건으로 백테스트 가능한 결과가 없습니다."
    
    # 이웃집 검증 (이웃 수익률, 이웃 괴리율 방어)
    smooth_ret = uniform_filter(ret_grid, size=3, mode='constant', cval=-100.0)
    df_res = pd.DataFrame(all_res)
    df_res['Nb_Ret'] = df_res.apply(lambda r: smooth_ret[np.where(DROP_RANGE==r['Drop'])[0][0], np.where(EXT_RANGE==r['Ext'])[0][0]], axis=1)
    
    # 최적의 전략 선정 (이웃 평균 수익률이 가장 높은 곳)
    best_strategy = df_res.sort_values('Nb_Ret', ascending=False).iloc[0]
    
    opt_ext = best_strategy['Ext']
    opt_drop = best_strategy['Drop']
    
    # ---------------------------------------------------------
    # 📈 Part 3: 현재 상태 및 스탑로스 계산
    # ---------------------------------------------------------
    recent_slopes = slopes[closest_date_idx:]
    peak_slope = np.max(recent_slopes[recent_slopes != -999.0]) if len(recent_slopes) > 0 else slopes[-1]
    
    target_price = L_last + (opt_ext * std_last)
    cut_slope = peak_slope - opt_drop
    
    status_data = {
        'CurPrice': cur_price, 'CurSigma': cur_sigma, 'CurSlope': slopes[-1],
        'EntSigma': my_ent_sigma, 'TargetPrice': target_price,
        'PeakSlope': peak_slope, 'CutSlope': cut_slope, 'OptDrop': opt_drop, 'OptExt': opt_ext
    }
    
    return prob_bands, status_data, best_strategy, None

# ---------------------------------------------------------
# ⚙️ 3. 결과 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner("✨ 10년 치 통계 데이터를 분석하여 확률 지도와 최적 출구 전략을 계산 중입니다..."):
        prob_bands, status, best_strat, err = run_custom_oracle(target_ticker, entry_date, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        # 내 수익률 계산
        my_rtn_str = ""
        if entry_price > 0:
            my_profit = (status['CurPrice'] / entry_price) - 1.0
            color = "#e74c3c" if my_profit > 0 else "#3498db"
            sign = "+" if my_profit > 0 else ""
            my_rtn_str = f" (현재 추정 수익률: <span style='color:{color}; font-weight:bold;'>{sign}{my_profit*100:.2f}%</span>)"

        # 1. 확률 지도 렌더링
        st.subheader("🗺️ 내일 가격 상승 확률 지도 (Probability Map)")
        st.markdown(f"**현재 주가:** ₩{status['CurPrice']:,.0f} (현재 시그마: {status['CurSigma']:.2f}){my_rtn_str}", unsafe_allow_html=True)
        st.caption("과거 10년간 시그마 레벨에 따른 다음 날 상승 확률을 현재 가격 대역으로 환산했습니다. (데이터가 없는 밴드는 주식 시장 특성상 해당 확률의 일일 변동이 발생하지 않음을 의미합니다.)")
        
        cols = st.columns(6)
        band_keys = list(prob_bands.keys())
        colors = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"]
        
        for i, col in enumerate(cols):
            band_name = band_keys[i]
            prices = prob_bands[band_name]
            
            with col:
                st.markdown(f"<h5 style='color:{colors[i]};'>{band_name}</h5>", unsafe_allow_html=True)
                if prices:
                    min_p, max_p = min(prices), max(prices)
                    st.write(f"₩{min_p:,.0f}\n~\n₩{max_p:,.0f}")
                else:
                    st.caption("데이터 없음")
                    
        st.markdown("---")

        # 2. 맞춤형 출구 전략 렌더링
        st.subheader(f"🎯 내 진입 시점 맞춤형 최적 출구 전략")
        st.markdown(f"> **나의 진입 환경:** {entry_date.strftime('%Y-%m-%d')} (당시 시그마: **{status['EntSigma']:.2f}**)")
        st.markdown(f"> **AI 백테스트 결과:** 나와 똑같은 조건(시그마)에서 진입했을 때, 과거 10년 동안 가장 안전하고 큰 수익을 낸 매도 공식은 **[익절 시그마 {status['OptExt']:.1f} / 손절 기울기 하락 {status['OptDrop']:.1f}%]** 입니다.")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.info("🔥 **맞춤형 목표 익절가**")
            st.metric(label=f"목표 시그마 ({status['OptExt']:.1f}) 도달 시", value=f"₩{status['TargetPrice']:,.0f}")
            st.caption("백테스트 결과 이 가격대에 도달했을 때 매도하는 것이 누적 수익금이 가장 컸습니다.")
            
        with c2:
            st.error("🚨 **맞춤형 손절/익절 마지노선 (Trailing Stop)**")
            st.metric(label=f"생명선 (기울기 {status['CutSlope']:.2f}%)", value=f"하락 시 즉시 매도")
            st.caption(f"진입 이후 달성한 최고 각도({status['PeakSlope']:.2f}%)에서 {status['OptDrop']:.1f}% 이상 꺾이면 미련 없이 매도하여 자산을 지키십시오.")
            
        is_danger = status['CurSlope'] < status['CutSlope']
        
        st.markdown("---")
        st.subheader("🤖 미스터 주의 최종 행동 지침")
        if is_danger:
            st.markdown(f"🚨 **[생명선 이탈]** 안타깝지만 진입 이후 유지되던 추세가 꺾였습니다. (현재 기울기: {status['CurSlope']:.2f}% < 마지노선: {status['CutSlope']:.2f}%). 더 큰 손실을 막거나 수익을 지키기 위해 **내일 시가에 매도**하시는 것을 통계적으로 강력히 권장합니다.")
        elif status['CurSigma'] >= status['OptExt']:
            st.markdown(f"💰 **[목표가 도달]** 축하합니다! 통계적 최적 매도 구간에 도달했습니다. **절반 이상 분할 익절**하여 수익을 확정 지으십시오.")
        else:
            st.markdown(f"🚀 **[순항 중 / 홀딩]** 아직 목표가에 도달하지 않았고, 추세(기울기)도 튼튼합니다. (현재 기울기: {status['CurSlope']:.2f}%). 안심하고 꽉 잡고 계십시오. 내일도 이 시스템이 생명선을 업데이트해 줄 것입니다.")
