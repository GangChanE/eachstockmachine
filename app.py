import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress, gaussian_kde
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

def get_kde_mode(data):
    """데이터에서 확률 밀도가 가장 높은 최빈점(Mode)을 찾습니다."""
    if len(data) < 2: return np.mean(data)
    kde = gaussian_kde(data)
    x_grid = np.linspace(min(data), max(data), 100)
    kde_vals = kde.evaluate(x_grid)
    return x_grid[np.argmax(kde_vals)]

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V8", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V8: 밀도 기반 정밀 밴드 & 타점 분석")
st.markdown("""
최소 표본수 20개를 엄격히 준수하여 상/하위 5%를 제거한 **순수 90% 확률 밴드**를 구축합니다.  
또한 '밀도가 가장 높은 통계적 중심가'를 제시하고, AI가 찾은 최적 타점이 평균 며칠 만에 도달하는지도 예측합니다.
""")

with st.sidebar:
    st.header("⚙️ 내 진입 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    entry_date = st.date_input("진입 날짜 (매수일)")
    entry_price = st.number_input("매수 단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    run_btn = st.button("🚀 정밀 타점 및 기간 예측", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_t_day_oracle_v8(ticker, ent_date, ent_price, tax, fee_rate):
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

        # 🚦 장세 분류
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
        # ⏱️ Part 1: 표본수 20개 이상 및 밀집 중심(KDE) 밴드 산출
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
            bin_data = df_t[df_t['SigBin'] == my_bin]['Return'].values
            
            # 표본수 20개 기준 적용
            if len(bin_data) >= 20:
                low_90_ret = np.percentile(bin_data, 5)
                high_90_ret = np.percentile(bin_data, 95)
                center_ret = get_kde_mode(bin_data) # 밀도가 가장 높은 최빈점
            else:
                if len(df_t) > 2:
                    slope, intercept, _, _, _ = linregress(df_t['Sigma'], df_t['Return'])
                    center_ret = slope * my_ent_sig + intercept
                    
                    residuals = df_t['Return'] - (slope * df_t['Sigma'] + intercept)
                    res_5 = np.percentile(residuals, 5)
                    res_95 = np.percentile(residuals, 95)
                    
                    low_90_ret = center_ret + res_5
                    high_90_ret = center_ret + res_95
                else:
                    low_90_ret = high_90_ret = center_ret = 0.0
            
            # 가격 역산 및 호가 교정
            low_price = round_to_tick(cur_price * (1 + low_90_ret), up=False)
            high_price = round_to_tick(cur_price * (1 + high_90_ret), up=True)
            center_price = round_to_tick(cur_price * (1 + center_ret), up=False)
            
            t_results.append({
                'T': t, 'LowPrice': low_price, 'CenterPrice': center_price, 'HighPrice': high_price
            })

        # ---------------------------------------------------------
        # 🛡️ Part 2: 맞춤형 출구 최적화 (보유 기간 분포 포함)
        # ---------------------------------------------------------
        DROP_RANGE = np.round(np.arange(0.1, 5.1, 0.1), 1)
        EXT_RANGE = np.round(np.arange(-1.0, 5.1, 0.1), 1)
        shape = (len(DROP_RANGE), len(EXT_RANGE))
        ret_grid = np.full(shape, -100.0)
        c_ent_p = np.round(-my_ent_sig, 1)
        
        all_res = []
        hold_days_dict = {} # 각 전략별 보유 기간 리스트 저장
        
        for idp, dp in enumerate(DROP_RANGE):
            for iex, ex in enumerate(EXT_RANGE):
                cap, hold, bp, es, trades = 1.0, False, 0.0, 0.0, 0
                h_days_list = []
                buy_idx = 0
                
                for k in range(win20, n_days-1):
                    if not hold:
                        if sigmas[k] <= -c_ent_p and regimes[k] == my_regime:
                            hold, bp, es, trades = True, opens[k+1], slopes20[k], trades + 1
                            buy_idx = k
                    else:
                        if sigmas[k] >= ex or slopes20[k] < (es - dp):
                            hold = False
                            profit = opens[k+1] - bp
                            tax_amt = profit * tax if profit > 0 else 0
                            net = ((opens[k+1] - tax_amt) / bp) - 1.0 - fee_rate
                            cap *= (1.0 + net)
                            h_days_list.append(k - buy_idx)
                            
                if trades > 0: 
                    tot_ret = (cap - 1.0) * 100
                    ret_grid[idp, iex] = tot_ret
                    all_res.append({'Drop': dp, 'Ext': ex, 'TotRet': tot_ret})
                    hold_days_dict[f"{dp}_{ex}"] = h_days_list

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
        opt_ext = best_strategy['Ext']
        
        # 보유 기간 90% 밴드 산출 (상/하위 5% 절사)
        best_h_days = hold_days_dict.get(f"{opt_drop}_{opt_ext}", [])
        if len(best_h_days) >= 20:
            h_low = int(np.percentile(best_h_days, 5))
            h_high = int(np.percentile(best_h_days, 95))
        elif len(best_h_days) > 0:
            h_low = min(best_h_days)
            h_high = max(best_h_days)
        else:
            h_low = h_high = 0
            
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
            'my_profit': ((cur_price / ent_price) - 1.0) * 100 if ent_price > 0 else 0.0,
            'h_low': h_low, 'h_high': h_high
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner("📦 표본 20개 이상 필터링 및 밀도 최빈값(KDE) 밴드를 연산 중입니다..."):
        res, err = run_t_day_oracle_v8(target_ticker, entry_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 연산 완료! (장세: {res['regime']})")
        
        # --- Part 1: T일 밴드 (밀집 중심가 포함) ---
        st.subheader("🗓️ 1. 보유 기간(T일)별 예상 가격 밴드 (통계적 중심가 포함)")
        st.markdown(f"> 표본수 20개 이상을 충족하는 90% 신뢰구간이며, **[  ]** 안의 금액은 가장 데이터가 조밀하게 뭉쳐있는 최빈(Mode) 가격입니다.")
        
        t_df = pd.DataFrame(res['t_results'])
        c1, c2, c3 = st.columns(3)
        
        def format_band(row):
            return f"₩{row['LowPrice']:,} ~ [**₩{row['CenterPrice']:,}**] ~ ₩{row['HighPrice']:,}"

        with c1:
            st.markdown("**[1일 ~ 20일 뒤]**")
            for i in range(0, 20):
                if i < len(t_df): st.markdown(f"`T+{t_df.iloc[i]['T']:02d}` | {format_band(t_df.iloc[i])}")
        with c2:
            st.markdown("**[21일 ~ 40일 뒤]**")
            for i in range(20, 40):
                if i < len(t_df): st.markdown(f"`T+{t_df.iloc[i]['T']:02d}` | {format_band(t_df.iloc[i])}")
        with c3:
            st.markdown("**[41일 ~ 60일 뒤]**")
            for i in range(40, 60):
                if i < len(t_df): st.markdown(f"`T+{t_df.iloc[i]['T']:02d}` | {format_band(t_df.iloc[i])}")
                
        st.markdown("---")
        
        # --- Part 2: 장세 맞춤형 출구 최적화 (보유 기간 밴드) ---
        st.subheader("🎯 2. 장세 맞춤형 분할 매도 타점 (AI 최적화)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🔥 **통계적 목표 익절가 (Target Zone)**")
            st.metric(label=f"Sigma {res['min_ext']:.1f} ~ {res['max_ext']:.1f} 도달 시", 
                      value=f"₩{res['target_min']:,} ~ ₩{res['target_max']:,}")
            # ★ 추가된 기능: 도달 예상 기간(T) 밴드 출력
            st.markdown(f"⏳ **과거 통계상 이 타점까지 도달하는 데 걸린 시간 (90% 확률):** \n👉 **`{res['h_low']}일 ~ {res['h_high']}일`** 내외")
            
        with col2:
            st.error(f"🚨 **생명선 (Trailing Stop)**")
            st.metric(label=f"기울기 {res['cut_slope']:.2f}% (현재 {res['cur_slope']:.2f}%)", value=f"하락 시 전량 매도")
            st.caption("기간에 상관없이 이 선이 깨지면 즉시 엑시트 하십시오.")
            
        is_danger = res['cur_slope'] < res['cut_slope']
        
        st.markdown("---")
        st.subheader("🤖 미스터 주의 최종 행동 지침")
        if is_danger:
            st.markdown(f"🚨 **[생명선 이탈]** 상승 추세가 꺾였습니다. 즉시 매도하여 자산을 보호하십시오.")
        elif res['cur_sigma'] >= res['min_ext']:
            st.markdown(f"💰 **[매도 구간 진입]** 통계적 분할 매도 구간에 들어왔습니다. 욕심을 버리고 익절하십시오.")
        else:
            rtn_text = f" (현재 수익률: {res['my_profit']:+.2f}%)" if entry_price > 0 else ""
            st.markdown(f"🚀 **[순항 중 / 홀딩]** 예상 도달 기간(`{res['h_low']}~{res['h_high']}일`)을 참고하여 멘탈을 관리하며 홀딩하십시오.{rtn_text}")
