import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter
import math
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 호가 절상 함수 (KRX 기준 반올림/올림)
# ---------------------------------------------------------
def round_to_tick(price, up=False):
    """주가를 KRX 호가 단위에 맞춰 교정합니다. (up=True면 절상)"""
    if price is None or np.isnan(price): return None
    
    if price < 2000: tick = 1
    elif price < 5000: tick = 5
    elif price < 20000: tick = 10
    elif price < 50000: tick = 50
    elif price < 200000: tick = 100
    elif price < 500000: tick = 500
    else: tick = 1000
        
    if up: return math.ceil(price / tick) * tick
    else: return round(price / tick) * tick

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V5 (T-Day Holding)", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V5: 보유 기간별(T일) 손익 예측기")
st.markdown("""
진입 지표 분석을 생략합니다. 과거 10년의 **장세(Regime)**를 5가지로 나누고,  
현재 장세에서 시그마 값에 따라 **1일부터 60일까지 보유했을 때의 90% 예상 수익률 구간**을 모두 도출합니다.
""")

with st.sidebar:
    st.header("⚙️ 내 진입 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    entry_date = st.date_input("진입 날짜 (매수일)")
    entry_price = st.number_input("매수 단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    run_btn = st.button("🚀 T일별 손익 분석 및 전략 추출", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_t_day_oracle(ticker, ent_date, ent_price, tax, fee_rate):
    try:
        raw = yf.download(ticker, start="2014-01-01", progress=False)
        if raw.empty: return None, "데이터를 불러오지 못했습니다."
            
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'Close']].dropna()
        closes = df['Close'].values
        opens = df['Open'].values
        dates = df.index
        n_days = len(closes)
        
        if n_days < 120: return None, "데이터가 부족합니다."

        win20 = 20
        win60 = 60
        sigmas = np.full(n_days, 999.0)
        slopes20 = np.full(n_days, -999.0)
        ann_slopes60 = np.full(n_days, -999.0) 
        
        x20 = np.arange(win20)
        x60 = np.arange(win60)
        
        for i in range(win60, n_days):
            y20 = closes[i-win20:i]
            s20, i20, _, _, _ = linregress(x20, y20)
            std20 = np.std(y20 - (s20*x20 + i20))
            if std20 > 0: sigmas[i] = (closes[i] - (s20*(win20-1)+i20)) / std20
            if closes[i] > 0: slopes20[i] = (s20 / closes[i]) * 100
            
            y60 = closes[i-win60:i]
            s60, _, _, _, _ = linregress(x60, y60)
            if closes[i] > 0: ann_slopes60[i] = (s60 / closes[i]) * 100 * 252

        # 🚦 장세(Regime) 분류
        regimes = np.full(n_days, 'Unknown', dtype=object)
        regimes[ann_slopes60 >= 40] = 'Strong Bull (🔥강한상승)'
        regimes[(ann_slopes60 >= 10) & (ann_slopes60 < 40)] = 'Bull (📈상승)'
        regimes[(ann_slopes60 > -10) & (ann_slopes60 < 10)] = 'Random (⚖️횡보)'
        regimes[(ann_slopes60 > -40) & (ann_slopes60 <= -10)] = 'Bear (📉하락)'
        regimes[ann_slopes60 <= -40] = 'Strong Bear (🧊강한하락)'

        ent_dt = pd.to_datetime(ent_date)
        closest_idx = np.argmin(np.abs(dates - ent_dt))
        my_ent_sig = sigmas[closest_idx]
        my_regime = regimes[closest_idx]
        
        if my_ent_sig == 999.0 or my_regime == 'Unknown':
            return None, "진입 날짜의 데이터가 부족합니다."

        regime_indices = np.where(regimes == my_regime)[0]
        if len(regime_indices) < 100:
            if my_regime == 'Strong Bull (🔥강한상승)': fallback = ['Strong Bull (🔥강한상승)', 'Bull (📈상승)']
            elif my_regime == 'Strong Bear (🧊강한하락)': fallback = ['Strong Bear (🧊강한하락)', 'Bear (📉하락)']
            else: fallback = [my_regime]
            regime_indices = np.where(np.isin(regimes, fallback))[0]

        # ---------------------------------------------------------
        # ⏱️ Part 1: T=1 ~ 60일 예상 손익률 (90% 신뢰구간) 도출
        # ---------------------------------------------------------
        t_results = []
        max_t = 60
        
        for t in range(1, max_t + 1):
            df_t = pd.DataFrame(columns=['Sigma', 'Return'])
            sig_list, ret_list = [], []
            
            # 종가 기준으로 T일 후 수익률 계산
            for i in regime_indices:
                if i + t < n_days:
                    sig_list.append(sigmas[i])
                    profit = closes[i+t] - closes[i]
                    tax_amt = profit * tax if profit > 0 else 0
                    ret = ((closes[i+t] - tax_amt) / closes[i]) - 1.0 - fee_rate
                    ret_list.append(ret * 100)
                    
            df_t = pd.DataFrame({'Sigma': sig_list, 'Return': ret_list})
            
            if df_t.empty: continue
                
            # 시그마 0.2 단위 그룹핑
            df_t['SigBin'] = np.round(df_t['Sigma'] / 0.2) * 0.2
            
            # 진입 시그마가 속한 그룹의 90% 신뢰구간 추출
            my_bin = np.round(my_ent_sig / 0.2) * 0.2
            bin_data = df_t[df_t['SigBin'] == my_bin]['Return']
            
            if len(bin_data) > 5:
                # 90% 구간 (하위 5% ~ 상위 95%)
                low_90 = np.percentile(bin_data, 5)
                high_90 = np.percentile(bin_data, 95)
                median_ret = np.median(bin_data)
            else:
                # 데이터가 부족하면 선형회귀식 사용
                if len(df_t) > 2:
                    slope, intercept, _, _, _ = linregress(df_t['Sigma'], df_t['Return'])
                    median_ret = slope * my_ent_sig + intercept
                    # 전체 데이터의 90% 잔차(Residual) 오차폭 적용
                    residuals = df_t['Return'] - (slope * df_t['Sigma'] + intercept)
                    err_margin = np.percentile(np.abs(residuals), 90)
                    low_90 = median_ret - err_margin
                    high_90 = median_ret + err_margin
                else:
                    low_90 = high_90 = median_ret = 0.0
                    
            t_results.append({
                'T': t, 'Median': median_ret, 'Low90': low_90, 'High90': high_90
            })

        # ---------------------------------------------------------
        # 🛡️ Part 2: 맞춤형 출구 최적화 (2D Grid)
        # ---------------------------------------------------------
        DROP_RANGE = np.round(np.arange(0.1, 5.1, 0.1), 1)
        EXT_RANGE = np.round(np.arange(-1.0, 5.1, 0.1), 1)
        shape = (len(DROP_RANGE), len(EXT_RANGE))
        ret_grid = np.full(shape, -100.0)
        c_ent_p = np.round(-my_ent_sig, 1)
        
        all_res = []
        for idp, dp in enumerate(DROP_RANGE):
            for iex, ex in enumerate(EXT_RANGE):
                cap, hold, bp, es, trades = 1.0, False, 0.0, 0.0, 0
                for k in range(win20, n_days-1):
                    if not hold:
                        if sigmas[k] <= -c_ent_p and regimes[k] == my_regime:
                            hold, bp, es, trades = True, opens[k+1], slopes20[k], trades + 1
                    else:
                        if sigmas[k] >= ex or slopes20[k] < (es - dp):
                            hold = False
                            profit = opens[k+1] - bp
                            tax_amt = profit * tax if profit > 0 else 0
                            net = ((opens[k+1] - tax_amt) / bp) - 1.0 - fee_rate
                            cap *= (1.0 + net)
                if trades > 0: 
                    tot_ret = (cap - 1.0) * 100
                    ret_grid[idp, iex] = tot_ret
                    all_res.append({'Drop': dp, 'Ext': ex, 'TotRet': tot_ret})

        smooth_ret = uniform_filter(ret_grid, size=3, mode='constant', cval=-100.0)
        if np.max(smooth_ret) == -100.0:
            return None, f"[{my_regime}] 장세에서 진입 시그마의 유효한 결과가 없습니다."
            
        df_res = pd.DataFrame(all_res)
        df_res['Nb_Ret'] = df_res.apply(lambda r: smooth_ret[np.where(DROP_RANGE==r['Drop'])[0][0], np.where(EXT_RANGE==r['Ext'])[0][0]], axis=1)
        
        best_strategy = df_res.sort_values('Nb_Ret', ascending=False).iloc[0]
        opt_ext = best_strategy['Ext']
        opt_drop = best_strategy['Drop']
        
        y_last = closes[-win20:]
        s_l, i_l, _, _, _ = linregress(x20, y_last)
        L_last = s_l*(win20-1) + i_l
        std_last = np.std(y_last - (s_l*x20 + i_l))
        
        # 목표가 절상 처리
        target_price = round_to_tick(L_last + (opt_ext * std_last), up=True)
        
        recent_slopes = slopes20[closest_idx:]
        peak_slope = np.max(recent_slopes[recent_slopes != -999.0]) if len(recent_slopes) > 0 else slopes20[-1]
        cut_slope = peak_slope - opt_drop

        res = {
            'regime': my_regime, 'ent_sigma': my_ent_sig,
            't_results': t_results,
            'opt_ext': opt_ext, 'opt_drop': opt_drop,
            'target_price': target_price, 'cut_slope': cut_slope,
            'cur_price': closes[-1], 'cur_sigma': sigmas[-1], 
            'cur_slope': slopes20[-1], 'peak_slope': peak_slope,
            'my_profit': ((closes[-1] / ent_price) - 1.0) * 100 if ent_price > 0 else 0.0
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner("📦 T=1~60일 손익 구간 탐색 및 최적 타점을 계산 중입니다... (1~2분 소요)"):
        res, err = run_t_day_oracle(target_ticker, entry_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 연산 완료! (해석된 장세: {res['regime']})")
        
        # --- Part 1: T일별 손익률 렌더링 ---
        st.subheader("🗓️ 1. 보유 기간(T일)별 예상 손익률 밴드 (90% 신뢰구간)")
        st.markdown(f"> 당신의 진입 조건(시그마 **{res['ent_sigma']:.2f}**)과 유사한 상황에서 매수 후 **T일 동안 보유**했을 때의 90% 확률 통계입니다.")
        
        # 결과를 15일 단위로 끊어서 컬럼으로 보여줌 (가독성 향상)
        t_df = pd.DataFrame(res['t_results'])
        c1, c2, c3, c4 = st.columns(4)
        
        for i, row in t_df.iterrows():
            t_val = int(row['T'])
            text = f"**T+{t_val:02d}일** : {row['Low90']:+5.1f}% ~ {row['High90']:+5.1f}% (평균 {row['Median']:+5.1f}%)"
            
            if t_val <= 15: c1.write(text)
            elif t_val <= 30: c2.write(text)
            elif t_val <= 45: c3.write(text)
            else: c4.write(text)
                
        st.markdown("---")
        
        # --- Part 2: 장세 맞춤형 출구 최적화 ---
        st.subheader("🎯 2. 장세 맞춤형 최적 출구 전략 (AI 최적화)")
        st.markdown(f"> T일 보유 기간과 무관하게, 추세(기울기)와 변동성(시그마)을 쫓아가며 가장 큰 누적 수익을 냈던 최적의 익절/손절 공식입니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🔥 **통계적 익절 목표가 (호가 절상 적용)**")
            st.metric(label=f"목표 시그마 ({res['opt_ext']:.1f}) 도달 시", value=f"₩{res['target_price']:,}")
            st.caption("AI가 찾아낸 수학적 최적 익절가입니다. 이 가격에 도달하면 전량 또는 분할 매도하십시오.")
            
        with col2:
            st.error(f"🚨 **생명선 (Trailing Stop)**")
            st.metric(label=f"기울기 {res['cut_slope']:.2f}% (현재 {res['cur_slope']:.2f}%)", value=f"하락 시 전량 매도")
            st.caption(f"최고 기울기({res['peak_slope']:.2f}%)에서 {res['opt_drop']:.1f}% 이상 꺾인 지점입니다.")
            
        is_danger = res['cur_slope'] < res['cut_slope']
        
        st.markdown("---")
        st.subheader("🤖 미스터 주의 최종 행동 지침")
        if is_danger:
            st.markdown(f"🚨 **[생명선 이탈]** 상승 추세가 꺾였습니다. (현재 기울기: {res['cur_slope']:.2f}% < 마지노선: {res['cut_slope']:.2f}%). T일 보유 확률과 무관하게 즉시 매도하여 자산을 보호하십시오.")
        elif res['cur_sigma'] >= res['opt_ext']:
            st.markdown(f"💰 **[목표가 도달]** 최적 익절 구간을 돌파했습니다. 미련 없이 익절하십시오.")
        else:
            rtn_text = f" (현재 수익률: {res['my_profit']:+.2f}%)" if entry_price > 0 else ""
            st.markdown(f"🚀 **[순항 중 / 홀딩]** 아직 목표가에 도달하지 않았습니다. 위의 **T일 예상 손익률 밴드**를 참고하여 나의 목표 기간까지 멘탈을 관리하며 홀딩하십시오.{rtn_text}")
