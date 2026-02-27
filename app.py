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
# ⚙️ 0. 호가 교정 함수 (KRX 기준)
# ---------------------------------------------------------
def round_to_tick(price, up=False):
    if price is None or np.isnan(price): return None
    
    if price < 2000: tick = 1
    elif price < 5000: tick = 5
    elif price < 20000: tick = 10
    elif price < 50000: tick = 50
    elif price < 200000: tick = 100
    elif price < 500000: tick = 500
    else: tick = 1000
        
    if up: return math.ceil(price / tick) * tick
    else: return math.floor(price / tick) * tick

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V7", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V7: 비대칭 편포 확률 밴드 적용")
st.markdown("""
장세(Regime)별로 데이터를 엄격하게 분리하고, 금융 데이터 특유의 **비대칭 편포(Skewness)**를 완벽하게 반영하여 
T일 후의 **90% 확률 가격 밴드**를 가장 정확하게 역산해 냅니다.
""")

with st.sidebar:
    st.header("⚙️ 내 진입 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    entry_date = st.date_input("진입 날짜 (매수일)")
    entry_price = st.number_input("매수 단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    run_btn = st.button("🚀 예측 가격 밴드 & 최적 타점 추출", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_t_day_oracle_v7(ticker, ent_date, ent_price, tax, fee_rate):
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
        cur_price = closes[-1]
        
        if my_ent_sig == 999.0 or my_regime == 'Unknown':
            return None, "진입 날짜의 데이터가 부족합니다."

        regime_indices = np.where(regimes == my_regime)[0]
        if len(regime_indices) < 50:
            return None, f"[{my_regime}] 과거 10년 중 이 장세 표본이 부족하여 신뢰도가 떨어집니다."

        # ---------------------------------------------------------
        # ⏱️ Part 1: 비대칭 편포를 반영한 T일 예상 가격 밴드
        # ---------------------------------------------------------
        t_results = []
        max_t = 60
        my_bin = np.round(my_ent_sig / 0.2) * 0.2
        
        for t in range(1, max_t + 1):
            sig_list, ret_list = [], []
            
            for i in regime_indices:
                if i + t < n_days:
                    sig_list.append(sigmas[i])
                    profit = closes[i+t] - closes[i]
                    tax_amt = profit * tax if profit > 0 else 0
                    ret = ((closes[i+t] - tax_amt) / closes[i]) - 1.0 - fee_rate
                    ret_list.append(ret)
                    
            df_t = pd.DataFrame({'Sigma': sig_list, 'Return': ret_list})
            if df_t.empty: continue
                
            df_t['SigBin'] = np.round(df_t['Sigma'] / 0.2) * 0.2
            bin_data = df_t[df_t['SigBin'] == my_bin]['Return']
            
            # 1순위: 실제 데이터가 충분할 때 (경험적 편포 그대로 반영)
            if len(bin_data) >= 5:
                low_90_ret = np.percentile(bin_data, 5)
                high_90_ret = np.percentile(bin_data, 95)
            # 2순위: 데이터 부족 시 선형회귀 + 비대칭 잔차 백분위수 적용
            else:
                if len(df_t) > 2:
                    slope, intercept, _, _, _ = linregress(df_t['Sigma'], df_t['Return'])
                    expected_ret = slope * my_ent_sig + intercept
                    
                    # 오차(Residual)의 절댓값이 아닌, 순수 오차의 비대칭 분포를 추출
                    residuals = df_t['Return'] - (slope * df_t['Sigma'] + intercept)
                    res_5 = np.percentile(residuals, 5)   # 하위 5% 오차 (보통 음수)
                    res_95 = np.percentile(residuals, 95) # 상위 95% 오차 (보통 양수)
                    
                    # 중심값에 비대칭 오차를 더해 완벽한 편포 밴드 구축
                    low_90_ret = expected_ret + res_5
                    high_90_ret = expected_ret + res_95
                else:
                    low_90_ret = high_90_ret = 0.0
            
            # 가격 역산 및 호가 교정
            low_price = round_to_tick(cur_price * (1 + low_90_ret), up=False)
            high_price = round_to_tick(cur_price * (1 + high_90_ret), up=True)
            
            t_results.append({
                'T': t, 'LowPrice': low_price, 'HighPrice': high_price
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
            return None, f"[{my_regime}] 장세에서 진입 시그마의 유효한 매도 전략 결과가 없습니다."
            
        df_res = pd.DataFrame(all_res)
        df_res['Nb_Ret'] = df_res.apply(lambda r: smooth_ret[np.where(DROP_RANGE==r['Drop'])[0][0], np.where(EXT_RANGE==r['Ext'])[0][0]], axis=1)
        
        top_5_strategies = df_res.sort_values('Nb_Ret', ascending=False).head(5)
        min_opt_ext = top_5_strategies['Ext'].min()
        max_opt_ext = top_5_strategies['Ext'].max()
        if max_opt_ext - min_opt_ext < 0.2: max_opt_ext += 0.3

        best_strategy = top_5_strategies.iloc[0]
        opt_drop = best_strategy['Drop']
        
        y_last = closes[-win20:]
        s_l, i_l, _, _, _ = linregress(x20, y_last)
        L_last = s_l*(win20-1) + i_l
        std_last = np.std(y_last - (s_l*x20 + i_l))
        
        target_price_min = round_to_tick(L_last + (min_opt_ext * std_last), up=True)
        target_price_max = round_to_tick(L_last + (max_opt_ext * std_last), up=True)
        
        closest_idx = np.argmin(np.abs(dates - ent_dt))
        recent_slopes = slopes20[closest_idx:]
        peak_slope = np.max(recent_slopes[recent_slopes != -999.0]) if len(recent_slopes) > 0 else slopes20[-1]
        cut_slope = peak_slope - opt_drop

        res = {
            'regime': my_regime, 'ent_sigma': my_ent_sig,
            't_results': t_results,
            'min_ext': min_opt_ext, 'max_ext': max_opt_ext, 'opt_drop': opt_drop,
            'target_min': target_price_min, 'target_max': target_price_max, 
            'cut_slope': cut_slope, 'cur_price': cur_price, 
            'cur_sigma': sigmas[-1], 'cur_slope': slopes20[-1], 'peak_slope': peak_slope,
            'my_profit': ((cur_price / ent_price) - 1.0) * 100 if ent_price > 0 else 0.0
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner("📦 비대칭 편포 확률을 적용하여 T=1~60일 가격 밴드를 도출 중입니다..."):
        res, err = run_t_day_oracle_v7(target_ticker, entry_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 연산 완료! (해석된 시장 장세: {res['regime']})")
        
        st.subheader("🗓️ 1. 보유 기간(T일)별 예상 가격 밴드 (90% 신뢰구간 / 편포 적용)")
        st.markdown(f"> **[{res['regime']}]** 장세의 비대칭적 수익 분포를 완벽하게 반영했습니다. (상/하위 5% 커트라인)")
        
        t_df = pd.DataFrame(res['t_results'])
        c1, c2, c3, c4 = st.columns(4)
        
        def format_band(row):
            return f"₩{row['LowPrice']:,.0f} ~ ₩{row['HighPrice']:,.0f}"

        with c1:
            st.markdown("**[1일 ~ 15일 뒤]**")
            for i in range(0, 15):
                if i < len(t_df): st.markdown(f"`T+{t_df.iloc[i]['T']:02d}` | {format_band(t_df.iloc[i])}")
        with c2:
            st.markdown("**[16일 ~ 30일 뒤]**")
            for i in range(15, 30):
                if i < len(t_df): st.markdown(f"`T+{t_df.iloc[i]['T']:02d}` | {format_band(t_df.iloc[i])}")
        with c3:
            st.markdown("**[31일 ~ 45일 뒤]**")
            for i in range(30, 45):
                if i < len(t_df): st.markdown(f"`T+{t_df.iloc[i]['T']:02d}` | {format_band(t_df.iloc[i])}")
        with c4:
            st.markdown("**[46일 ~ 60일 뒤]**")
            for i in range(45, 60):
                if i < len(t_df): st.markdown(f"`T+{t_df.iloc[i]['T']:02d}` | {format_band(t_df.iloc[i])}")
                
        st.markdown("---")
        
        st.subheader("🎯 2. 장세 맞춤형 실전 매도 구간 (Zone)")
        st.markdown(f"> 나의 진입 조건(Sigma **{res['ent_sigma']:.2f}**)에서 누적 수익금이 가장 컸던 밀집 타점입니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🔥 **통계적 분할 매도 구간 (Target Zone)**")
            st.metric(label=f"Sigma {res['min_ext']:.1f} ~ {res['max_ext']:.1f} 도달 시", 
                      value=f"₩{res['target_min']:,} ~ ₩{res['target_max']:,}")
            
        with col2:
            st.error(f"🚨 **생명선 (Trailing Stop)**")
            st.metric(label=f"기울기 {res['cut_slope']:.2f}% (현재 {res['cur_slope']:.2f}%)", value=f"하락 시 전량 매도")
            
        is_danger = res['cur_slope'] < res['cut_slope']
        
        st.markdown("---")
        st.subheader("🤖 미스터 주의 최종 행동 지침")
        if is_danger:
            st.markdown(f"🚨 **[생명선 이탈]** 상승 추세가 꺾였습니다. 즉시 매도하여 자산을 보호하십시오.")
        elif res['cur_sigma'] >= res['min_ext']:
            st.markdown(f"💰 **[매도 구간 진입]** 통계적 분할 매도 구간에 들어왔습니다. 정해진 호가에 맞춰 분할 익절하십시오.")
        else:
            rtn_text = f" (현재 수익률: {res['my_profit']:+.2f}%)" if entry_price > 0 else ""
            st.markdown(f"🚀 **[순항 중 / 홀딩]** 아직 매도 구간에 도달하지 않았습니다. T일 예상 가격을 참고하여 평온하게 들고 가십시오.{rtn_text}")
