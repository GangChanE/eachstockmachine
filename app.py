import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter
from pandas.tseries.offsets import BDay
import plotly.graph_objects as go
import math
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 호가 교정 함수
# ---------------------------------------------------------
def round_to_tick(price, up=False):
    if price is None or np.isnan(price) or price <= 0: return 0
    if price > 1e9: return int(price)
    
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
st.set_page_config(page_title="Quantum Oracle V13", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V13: 타임머신 & 실전 궤적 검증")
st.markdown("""
**1. 시계열 가중치(EWMA):** 최근 시장의 기세(모멘텀)에 높은 가중치를 주어 밋밋한 평균의 함정을 극복했습니다.  
**2. 장세 피로도(Hazard Rate):** 장세가 길어질수록 붕괴 확률을 높여, 비현실적인 영구 상승/하락 폭발을 방지합니다.
""")

with st.sidebar:
    st.header("⚙️ 타임머신 & 계좌 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    target_date = st.date_input("분석 기준일 (과거 타임머신 날짜)")
    entry_price = st.number_input("기준일 매수 단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    use_log_scale = st.checkbox("📈 Y축 로그 스케일 적용", value=False)
    run_btn = st.button("🚀 피로도 반영 몬테카를로 가동", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진 (EWMA + Hazard Markov + GBM)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_v13_oracle(ticker, target_date, ent_price, tax, fee_rate):
    try:
        raw = yf.download(ticker, start="2014-01-01", progress=False)
        if raw.empty: return None, "데이터 로드 실패."
            
        df_all = raw.copy()
        if isinstance(df_all.columns, pd.MultiIndex):
            df_all.columns = df_all.columns.get_level_values(0)
            
        df_all = df_all[['Open', 'Close']].dropna()
        target_dt = pd.to_datetime(target_date)
        
        # 🛡️ 데이터 분리
        df_past = df_all[df_all.index <= target_dt]
        df_future = df_all[df_all.index > target_dt]
        
        closes = df_past['Close'].values
        opens = df_past['Open'].values
        dates = df_past.index
        n_days = len(closes)
        
        if n_days < 120: return None, "과거 데이터 부족."

        win20, win60 = 20, 60
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

        REGIME_NAMES = ['Strong Bull', 'Bull', 'Random', 'Bear', 'Strong Bear']
        regimes = np.full(n_days, 'Random', dtype=object)
        regimes[ann_slopes60 >= 40] = 'Strong Bull'
        regimes[(ann_slopes60 >= 10) & (ann_slopes60 < 40)] = 'Bull'
        regimes[(ann_slopes60 > -10) & (ann_slopes60 < 10)] = 'Random'
        regimes[(ann_slopes60 > -40) & (ann_slopes60 <= -10)] = 'Bear'
        regimes[ann_slopes60 <= -40] = 'Strong Bear'

        current_regime = regimes[-1]
        cur_sigma = sigmas[-1]
        cur_slope = slopes20[-1]
        cur_price = closes[-1]
        
        # 🌟 1. 최신 시장 가중치 부여 (EWMA - Half life 252일)
        hl = 252.0
        decay = np.log(2) / hl
        weights = np.exp(-decay * np.arange(n_days-1, -1, -1)) # 최근일수록 가중치 1에 수렴

        # 📊 2. 가중치가 적용된 전이 행렬 (최근 흐름 반영)
        trans_matrix = {r1: {r2: 0.05 for r2 in REGIME_NAMES} for r1 in REGIME_NAMES} # Laplace Smoothing
        
        regime_blocks = []
        curr_r = regimes[win60]
        start_idx = win60
        
        for i in range(win60, n_days - 1):
            r_from = regimes[i]
            r_to = regimes[i+1]
            if r_from in trans_matrix and r_to in trans_matrix:
                trans_matrix[r_from][r_to] += weights[i] # 빈도 대신 가중치 합산
                
            if regimes[i+1] != curr_r:
                regime_blocks.append({'regime': curr_r, 'duration': i + 1 - start_idx})
                curr_r = regimes[i+1]
                start_idx = i + 1
        regime_blocks.append({'regime': curr_r, 'duration': n_days - start_idx})
                
        for r1 in REGIME_NAMES:
            total = sum(trans_matrix[r1].values())
            for r2 in REGIME_NAMES: trans_matrix[r1][r2] /= total

        # 🌟 3. 장세별 통계 계산 (가중 평균 및 피로도 한계치 산출)
        regime_stats = {}
        for r in REGIME_NAMES:
            r_indices = np.where(regimes == r)[0]
            valid_idx = [i for i in r_indices if i+1 < n_days and closes[i] > 0 and closes[i+1] > 0]
            
            if len(valid_idx) > 0:
                log_rets = np.log(closes[np.array(valid_idx)+1] / closes[np.array(valid_idx)])
                w = weights[valid_idx]
                mu = np.average(log_rets, weights=w)
                variance = np.average((log_rets - mu)**2, weights=w)
                sigma = np.sqrt(variance)
            else:
                mu, sigma = 0.0, 0.02
                
            r_blocks = [b['duration'] for b in regime_blocks if b['regime'] == r]
            # 피로도 한계선: 과거 해당 장세 수명의 95% 백분위수
            max_dur = np.percentile(r_blocks, 95) if len(r_blocks) > 2 else 20
            
            regime_stats[r] = {'mu': mu, 'sigma': sigma, 'max_dur': max(5, int(max_dur))}

        # 📈 4. 해저드율(Hazard Rate)이 결합된 몬테카를로 시뮬레이션
        n_sim = 1000
        days_ahead = 360
        sim_prices = np.zeros((n_sim, days_ahead))
        last_block = regime_blocks[-1]
        
        np.random.seed()
        
        for i in range(n_sim):
            c_r = current_regime if current_regime in REGIME_NAMES else 'Random'
            run_duration = last_block['duration']
            price = cur_price
            
            for t in range(days_ahead):
                # 🛡️ 피로도(Fatigue) 기반 확률 조정 로직
                base_probs = {nxt: trans_matrix[c_r][nxt] for nxt in REGIME_NAMES}
                max_d = regime_stats[c_r]['max_dur']
                
                # 수명이 한계치에 다가갈수록 피로도 지수 급증 (최대 0.95)
                fatigue = min(0.95, (run_duration / max_d) ** 2)
                
                stay_prob = base_probs[c_r]
                new_stay_prob = stay_prob * (1 - fatigue)
                diff = stay_prob - new_stay_prob
                
                base_probs[c_r] = new_stay_prob
                # 이탈한 확률을 시장을 진정시키는 방향으로 분배
                if c_r != 'Random':
                    base_probs['Random'] += diff # 추세가 길어지면 횡보장으로 회귀 강제
                else:
                    base_probs['Bear'] += diff / 2
                    base_probs['Bull'] += diff / 2
                    
                probs_arr = [base_probs[nxt] for nxt in REGIME_NAMES]
                probs_arr = np.array(probs_arr) / sum(probs_arr)
                
                next_r = np.random.choice(REGIME_NAMES, p=probs_arr)
                
                if next_r == c_r: run_duration += 1
                else: 
                    c_r = next_r
                    run_duration = 1
                    
                # 기하 브라운 운동 (GBM) 진행
                mu = regime_stats[c_r]['mu']
                sig = regime_stats[c_r]['sigma']
                price *= np.exp(np.random.normal(mu, sig))
                sim_prices[i, t] = price

        # 결과 백분위수 추출
        low_90_arr = np.percentile(sim_prices, 5, axis=0)
        high_90_arr = np.percentile(sim_prices, 95, axis=0)
        center_arr = np.percentile(sim_prices, 50, axis=0)
        
        trajectory = []
        base_date = dates[-1]
        max_pred_date = base_date
        
        for t in range(days_ahead):
            pred_date = base_date + BDay(t + 1)
            max_pred_date = pred_date
            trajectory.append({
                'Date': pred_date,
                'Center': round_to_tick(center_arr[t], up=False),
                'Low90': round_to_tick(low_90_arr[t], up=False),
                'High90': round_to_tick(high_90_arr[t], up=True)
            })

        # 검증용 실제 미래 데이터
        actual_future_dates = []
        actual_future_prices = []
        if not df_future.empty:
            df_future_cut = df_future[df_future.index <= max_pred_date]
            actual_future_dates = df_future_cut.index.tolist()
            actual_future_prices = df_future_cut['Close'].tolist()

        # 🎯 5. 듀얼 코어 백테스트 (현재 조건 유지)
        c_ent_p = np.round(-cur_sigma, 1) 
        DROP_RANGE = np.round(np.arange(0.1, 5.1, 0.1), 1)
        EXT_RANGE = np.round(np.arange(-1.0, 5.1, 0.1), 1)
        shape = (len(DROP_RANGE), len(EXT_RANGE))
        ret_grid = np.full(shape, -100.0)
        
        all_res = []
        for idp, dp in enumerate(DROP_RANGE):
            for iex, ex in enumerate(EXT_RANGE):
                cap, hold, bp, es, trades = 1.0, False, 0.0, 0.0, 0
                for k in range(win20, n_days-1):
                    if not hold:
                        if sigmas[k] <= -c_ent_p:
                            hold, bp, es, trades = True, opens[k+1], slopes20[k], trades + 1
                    else:
                        if sigmas[k] >= ex or slopes20[k] < (es - dp):
                            hold = False
                            net = ((opens[k+1] - max(0, opens[k+1]-bp)*tax) / bp) - 1.0 - fee_rate
                            cap *= (1.0 + net)
                if trades > 0: 
                    ret_grid[idp, iex] = (cap - 1.0) * 100
                    all_res.append({'Drop': dp, 'Ext': ex, 'TotRet': (cap-1)*100})

        smooth_ret = uniform_filter(ret_grid, size=3, mode='constant', cval=-100.0)
        y_last = closes[-win20:]
        s_l, i_l, _, _, _ = linregress(x20, y_last)
        L_last = s_l*(win20-1) + i_l
        std_last = np.std(y_last - (s_l*x20 + i_l))
        
        dual_results = {'short': None, 'long': None}

        if np.max(smooth_ret) != -100.0:
            df_res = pd.DataFrame(all_res)
            df_res['Nb_Ret'] = df_res.apply(lambda r: smooth_ret[np.where(DROP_RANGE==r['Drop'])[0][0], np.where(EXT_RANGE==r['Ext'])[0][0]], axis=1)
            
            short_df = df_res[(df_res['Ext'] <= 2.0) & (df_res['Drop'] <= 2.0)]
            if not short_df.empty:
                top_short = short_df.sort_values('Nb_Ret', ascending=False).head(5)
                dual_results['short'] = {
                    'opt_drop': top_short.iloc[0]['Drop'], 
                    'min_ext': top_short['Ext'].min(), 'max_ext': max(top_short['Ext'].max(), top_short['Ext'].min()+0.3),
                    'target_min': round_to_tick(L_last + (top_short['Ext'].min() * std_last), up=True),
                    'target_max': round_to_tick(L_last + (max(top_short['Ext'].max(), top_short['Ext'].min()+0.3) * std_last), up=True),
                    'cut_slope': cur_slope - top_short.iloc[0]['Drop']
                }

            long_df = df_res[(df_res['Ext'] >= 2.5) & (df_res['Drop'] >= 2.0)]
            if not long_df.empty:
                top_long = long_df.sort_values('Nb_Ret', ascending=False).head(5)
                dual_results['long'] = {
                    'opt_drop': top_long.iloc[0]['Drop'], 
                    'min_ext': top_long['Ext'].min(), 'max_ext': max(top_long['Ext'].max(), top_long['Ext'].min()+0.3),
                    'target_min': round_to_tick(L_last + (top_long['Ext'].min() * std_last), up=True),
                    'target_max': round_to_tick(L_last + (max(top_long['Ext'].max(), top_long['Ext'].min()+0.3) * std_last), up=True),
                    'cut_slope': cur_slope - top_long.iloc[0]['Drop']
                }

        display_curr = {'Strong Bull': '🔥강한상승', 'Bull': '📈상승', 'Random': '⚖️횡보', 'Bear': '📉하락', 'Strong Bear': '🧊강한하락'}.get(current_regime, current_regime)

        res = {
            'curr_regime': display_curr, 'cur_sigma': cur_sigma, 'cur_price': cur_price, 'cur_slope': cur_slope,
            'trajectory': trajectory, 'dual_results': dual_results,
            'actual_dates': actual_future_dates, 'actual_prices': actual_future_prices,
            'my_profit': ((cur_price / ent_price) - 1.0) * 100 if ent_price > 0 else 0.0
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 최근 장세에 가중치(EWMA)를 부여하고, 피로도(Hazard)가 반영된 1,000회 시뮬레이션을 가동 중입니다..."):
        res, err = run_v13_oracle(target_ticker, target_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ V13 정밀 분석 완료! (분석 기준일: {target_date})")
        
        st.subheader("📈 1. 1,000회 몬테카를로 360일 지수 궤적 vs 실제 주가")
        
        if use_log_scale:
            st.info("ℹ️ **로그 스케일(Log Scale):** 폭발적인 지수 곡선이 안정적인 비율로 교정되어 보입니다.")
            
        traj_df = pd.DataFrame(res['trajectory'])
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['High90'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Low90'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)', name='90% 확률 밴드'))
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Center'], mode='lines', line=dict(color='#e74c3c', width=2, dash='dot'), name='예상 통계적 중심', hovertemplate="<b>%{x|%Y-%m-%d}</b><br>예상가: ₩%{y:,.0f}<extra></extra>"))
        
        if res['actual_dates'] and len(res['actual_dates']) > 0:
            fig.add_trace(go.Scatter(x=res['actual_dates'], y=res['actual_prices'], mode='lines', line=dict(color='black', width=3), name='실제 시장 흐름 (Reality)'))
            
        if use_log_scale:
            fig.update_layout(yaxis_type="log")
            
        fig.update_layout(hovermode="x unified", height=500, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="미래 날짜", yaxis_title="주가 (원)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader(f"🎯 2. 기준일({target_date}) 당시 듀얼 매도 전략 (당시 Sigma: {res['cur_sigma']:.2f})")
        
        dual = res['dual_results']
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### ⚡ [단기 스윙형] 엑시트")
            if dual['short']:
                st.info(f"**목표 매도 구간**\n### ₩{dual['short']['target_min']:,} ~ ₩{dual['short']['target_max']:,}")
                st.error(f"**생명선 이탈 기준**\n### 기울기 {dual['short']['cut_slope']:.2f}% (당시 {res['cur_slope']:.2f}%)")
            else:
                st.write("단기 스윙 데이터 부족.")

        with c2:
            st.markdown("#### 📦 [장기 추세형] 엑시트")
            if dual['long']:
                st.success(f"**목표 매도 구간**\n### ₩{dual['long']['target_min']:,} ~ ₩{dual['long']['target_max']:,}")
                st.error(f"**생명선 이탈 기준**\n### 기울기 {dual['long']['cut_slope']:.2f}% (당시 {res['cur_slope']:.2f}%)")
            else:
                st.write("장기 추세 데이터 부족.")
