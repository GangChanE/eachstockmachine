import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d, uniform_filter
from pandas.tseries.offsets import BDay
import plotly.graph_objects as go
import math
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 고속 연산 및 호가 교정 함수
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

X_ARR = np.arange(20)
X_MEAN = 9.5
X_VAR_SUM = 665.0 

def calc_fast_sigma(prices_20):
    y_mean = np.mean(prices_20)
    slope = np.sum((X_ARR - X_MEAN) * (prices_20 - y_mean)) / X_VAR_SUM
    intercept = y_mean - slope * X_MEAN
    trend_line = slope * X_ARR + intercept
    std = np.std(prices_20 - trend_line)
    if std > 0: return (prices_20[-1] - trend_line[-1]) / std
    return 0.0

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V17", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V17: 야생의 변동성 복원 (Brownian Bridge)")
st.markdown("""
밋밋한 선형 보간을 폐기하고 **[브라운 브릿지(Brownian Bridge)]** 수학 모델을 도입했습니다.  
T일 뒤의 거시적 목표(도착점)를 향해 나아가면서도, 일일 변동성(Daily Volatility)이 살아 숨 쉬는 완벽히 현실적인 궤적을 그립니다.
""")

with st.sidebar:
    st.header("⚙️ 타임머신 & 계좌 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    target_date = st.date_input("분석 기준일 (과거 타임머신 날짜)")
    entry_price = st.number_input("기준일 매수 단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    use_log_scale = st.checkbox("📈 Y축 로그 스케일 적용", value=False)
    run_btn = st.button("🚀 브라운 브릿지 몬테카를로 가동", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진 (Brownian Bridge + T-Step)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_v17_oracle(ticker, target_date, ent_price, tax, fee_rate):
    try:
        raw = yf.download(ticker, start="2014-01-01", progress=False)
        if raw.empty: return None, "데이터 로드 실패."
            
        df_all = raw.copy()
        if isinstance(df_all.columns, pd.MultiIndex):
            df_all.columns = df_all.columns.get_level_values(0)
            
        df_all = df_all[['Open', 'Close']].dropna()
        target_dt = pd.to_datetime(target_date)
        
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
            sigmas[i] = calc_fast_sigma(closes[i-win20:i])
            s20, _, _, _, _ = linregress(x20, closes[i-win20:i])
            if closes[i] > 0: slopes20[i] = (s20 / closes[i]) * 100
            s60, _, _, _, _ = linregress(x60, closes[i-win60:i])
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
        
        # 🌟 1. 장세별 일일 변동성(Daily Volatility) 추출 (브릿지용)
        daily_vol_dict = {}
        regime_models = {}
        for r in REGIME_NAMES:
            r_indices = np.where(regimes == r)[0]
            if len(r_indices) < 50: 
                regime_models[r] = {'T': 5, 'slope': 0, 'inter': 0, 'res_std': 0.05}
                daily_vol_dict[r] = 0.02
                continue
            
            # 일일 변동성 계산
            log_rets = []
            for i in r_indices:
                if i+1 < n_days and closes[i] > 0: log_rets.append(np.log(closes[i+1]/closes[i]))
            daily_vol_dict[r] = np.std(log_rets) if log_rets else 0.02

            max_t = 30
            t_corrs = []
            for t in range(1, max_t + 1):
                x_sig, y_ret = [], []
                for i in r_indices:
                    if i + t < n_days:
                        x_sig.append(sigmas[i])
                        y_ret.append((closes[i+t] / closes[i]) - 1.0)
                if len(x_sig) > 30: t_corrs.append(np.corrcoef(x_sig, y_ret)[0, 1])
                else: t_corrs.append(0)
            
            best_t = np.argmin(uniform_filter1d(t_corrs, size=3)) + 1 
            
            x_sig, y_ret = [], []
            for i in r_indices:
                if i + best_t < n_days:
                    x_sig.append(sigmas[i])
                    y_ret.append((closes[i+best_t] / closes[i]) - 1.0)
                    
            if len(x_sig) > 2:
                s, inter, _, _, _ = linregress(x_sig, y_ret)
                res_std = np.std(np.array(y_ret) - (s * np.array(x_sig) + inter))
            else:
                s, inter, res_std = 0, 0, 0.05
                
            regime_models[r] = {'T': best_t, 'slope': s, 'inter': inter, 'res_std': res_std}

        trans_matrix = {r1: {r2: 0.1 for r2 in REGIME_NAMES} for r1 in REGIME_NAMES}
        for i in range(win60, n_days - 1): trans_matrix[regimes[i]][regimes[i+1]] += 1
        for r1 in REGIME_NAMES:
            tot = sum(trans_matrix[r1].values())
            for r2 in REGIME_NAMES: trans_matrix[r1][r2] /= tot

        # 📈 2. Brownian Bridge 결합 T-Step 몬테카를로
        n_sim = 1000
        days_ahead = 360
        sim_prices = np.zeros((n_sim, days_ahead))
        
        np.random.seed()
        
        for i in range(n_sim):
            c_r = current_regime
            hist = list(closes[-win20:]) # 초기 역사
            
            day_idx = 0
            while day_idx < days_ahead:
                probs = [trans_matrix[c_r][nxt] for nxt in REGIME_NAMES]
                c_r = np.random.choice(REGIME_NAMES, p=probs)
                
                model = regime_models[c_r]
                T = model['T']
                daily_vol = daily_vol_dict[c_r] # 해당 장세의 야생 변동성
                
                # 시그마 피드백 계산
                current_sim_sigma = calc_fast_sigma(np.array(hist[-20:]))
                
                # T일 뒤 목표가 설정
                expected_ret = model['slope'] * current_sim_sigma + model['inter']
                realized_ret = expected_ret + np.random.normal(0, model['res_std'])
                
                start_p = hist[-1]
                target_p = max(0.1, start_p * (1 + realized_ret))
                
                # 🌟 Brownian Bridge 알고리즘 (야생의 변동성을 살리며 도착점에 꽂힘)
                # 로그 스케일로 다리(Bridge) 건설
                log_start = np.log(start_p)
                log_target = np.log(target_p)
                
                bridge_prices = []
                for step in range(1, T + 1):
                    if step == T:
                        bridge_prices.append(target_p)
                    else:
                        # 브라운 브릿지 공식: 남은 거리의 선형 비율 + 무작위 진동(Volatility)
                        time_ratio = step / T
                        mean_log_p = log_start + time_ratio * (log_target - log_start)
                        
                        # 도착지점에 가까울수록 분산이 0으로 수렴하는 구조 (Tie-down)
                        bridge_var = (step * (T - step) / T) * (daily_vol ** 2)
                        bridge_std = np.sqrt(bridge_var) if bridge_var > 0 else 0
                        
                        sim_log_p = np.random.normal(mean_log_p, bridge_std)
                        bridge_prices.append(np.exp(sim_log_p))
                
                for bp in bridge_prices:
                    if day_idx < days_ahead:
                        sim_prices[i, day_idx] = bp
                        hist.append(bp)
                        day_idx += 1
                    else:
                        break

        # 📊 3. 다중 백분위수 (Fan Chart)
        low_90 = np.percentile(sim_prices, 5, axis=0)
        high_90 = np.percentile(sim_prices, 95, axis=0)
        low_80 = np.percentile(sim_prices, 10, axis=0)
        high_80 = np.percentile(sim_prices, 90, axis=0)
        low_70 = np.percentile(sim_prices, 15, axis=0)
        high_70 = np.percentile(sim_prices, 85, axis=0)
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
                'Low90': round_to_tick(low_90[t], up=False), 'High90': round_to_tick(high_90[t], up=True),
                'Low80': round_to_tick(low_80[t], up=False), 'High80': round_to_tick(high_80[t], up=True),
                'Low70': round_to_tick(low_70[t], up=False), 'High70': round_to_tick(high_70[t], up=True)
            })

        actual_future_dates = []
        actual_future_prices = []
        if not df_future.empty:
            df_future_cut = df_future[df_future.index <= max_pred_date]
            actual_future_dates = df_future_cut.index.tolist()
            actual_future_prices = df_future_cut['Close'].tolist()

        # 🎯 4. 듀얼 코어 백테스트 (현재 유지)
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
    with st.spinner(f"📦 변동성(Volatility)을 복원한 브라운 브릿지 시뮬레이션 가동 중..."):
        res, err = run_v17_oracle(target_ticker, target_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 야생의 변동성 복원 완료! (분석 기준일: {target_date})")
        
        st.subheader("📈 1. 다중 위상 궤적(Fan Chart) vs 실제 주가")
        st.markdown("> 인위적인 이동평균(Smoothing)을 제거하고, **브라운 브릿지(Brownian Bridge)** 수학 모델을 통해 일일 변동성이 펄떡이는 가장 현실적인 확률 밴드를 도출했습니다.")
        
        traj_df = pd.DataFrame(res['trajectory'])
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['High90'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Low90'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 152, 219, 0.1)', name='90% 확률 구간'))

        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['High80'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Low80'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 152, 219, 0.25)', name='80% 확률 구간'))

        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['High70'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Low70'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 152, 219, 0.4)', name='70% 확률 구간'))

        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Center'], mode='lines', line=dict(color='#e74c3c', width=2, dash='dot'), name='예상 중심가', hovertemplate="<b>%{x|%Y-%m-%d}</b><br>예상가: ₩%{y:,.0f}<extra></extra>"))
        
        if res['actual_dates'] and len(res['actual_dates']) > 0:
            fig.add_trace(go.Scatter(x=res['actual_dates'], y=res['actual_prices'], mode='lines', line=dict(color='black', width=3), name='실제 시장 흐름 (Reality)'))
            
        if use_log_scale: fig.update_layout(yaxis_type="log")
            
        fig.update_layout(hovermode="x unified", height=500, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="미래 날짜", yaxis_title="주가 (원)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader(f"🎯 2. 기준일({target_date}) 듀얼 매도 전략 (당시 Sigma: {res['cur_sigma']:.2f})")
        
        dual = res['dual_results']
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### ⚡ [단기 스윙형] 엑시트")
            if dual['short']:
                st.info(f"**목표 매도 구간**\n### ₩{dual['short']['target_min']:,} ~ ₩{dual['short']['target_max']:,}")
                st.error(f"**생명선 이탈 기준**\n### 기울기 {dual['short']['cut_slope']:.2f}%")
            else:
                st.write("단기 데이터 부족.")

        with c2:
            st.markdown("#### 📦 [장기 추세형] 엑시트")
            if dual['long']:
                st.success(f"**목표 매도 구간**\n### ₩{dual['long']['target_min']:,} ~ ₩{dual['long']['target_max']:,}")
                st.error(f"**생명선 이탈 기준**\n### 기울기 {dual['long']['cut_slope']:.2f}%")
            else:
                st.write("장기 데이터 부족.")
