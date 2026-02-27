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
st.set_page_config(page_title="Quantum Oracle V11", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V11: 타임머신 백테스트 & 궤적 검증")
st.markdown("""
과거의 특정 날짜를 선택하면, **그날까지의 데이터만**을 사용하여 향후 360일의 궤적과 타점을 예측합니다.  
예측된 밴드(90% 신뢰구간) 위에 **실제 주가 흐름을 오버레이(Overlay)**하여 모델의 정확도를 검증할 수 있습니다.
""")

with st.sidebar:
    st.header("⚙️ 타임머신 & 계좌 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    target_date = st.date_input("분석 기준일 (타임머신 날짜)", help="이 날짜를 기준으로 과거 데이터만 학습하여 미래를 예측합니다.")
    entry_price = st.number_input("기준일 매수 단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    run_btn = st.button("🚀 타임머신 가동 및 검증", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 타임머신 핵심 분석 엔진 (미래 데이터 차단)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_timemachine_oracle(ticker, target_date, ent_price, tax, fee_rate):
    try:
        # 데이터는 최신까지 모두 받되, 연산 시점에 분리(Slicing)합니다.
        raw = yf.download(ticker, start="2014-01-01", progress=False)
        if raw.empty: return None, "데이터 로드 실패."
            
        df_all = raw.copy()
        if isinstance(df_all.columns, pd.MultiIndex):
            df_all.columns = df_all.columns.get_level_values(0)
            
        df_all = df_all[['Open', 'Close']].dropna()
        
        target_dt = pd.to_datetime(target_date)
        
        # 🛡️ 데이터 1차 분리 (과거: 학습용 / 미래: 검증용)
        df_past = df_all[df_all.index <= target_dt]
        df_future = df_all[df_all.index > target_dt]
        
        closes = df_past['Close'].values
        opens = df_past['Open'].values
        dates = df_past.index
        n_days = len(closes)
        
        if n_days < 120: return None, f"선택하신 날짜({target_date}) 이전의 과거 데이터가 너무 적어 분석할 수 없습니다."

        # 지표 선계산 (오직 df_past 기준)
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

        current_regime = regimes[-1] # target_date 기준 장세
        cur_sigma = sigmas[-1]       # target_date 기준 시그마
        cur_slope = slopes20[-1]
        cur_price = closes[-1]
        
        # 📊 장세 마르코프 통계 (360일 궤적용)
        regime_blocks = []
        curr_r = regimes[win60]
        start_idx = win60
        for i in range(win60 + 1, n_days):
            if regimes[i] != curr_r:
                regime_blocks.append({'regime': curr_r, 'start': start_idx, 'end': i-1, 'duration': i - start_idx})
                curr_r = regimes[i]
                start_idx = i
        regime_blocks.append({'regime': curr_r, 'start': start_idx, 'end': n_days-1, 'duration': n_days - start_idx})
        
        regime_stats = {}
        for r in ['Strong Bull (🔥강한상승)', 'Bull (📈상승)', 'Random (⚖️횡보)', 'Bear (📉하락)', 'Strong Bear (🧊강한하락)']:
            r_blocks = [b for b in regime_blocks if b['regime'] == r]
            avg_dur = np.mean([b['duration'] for b in r_blocks]) if r_blocks else 20
            next_regimes = [regime_blocks[i+1]['regime'] for i, b in enumerate(regime_blocks[:-1]) if b['regime'] == r]
            most_likely_next = max(set(next_regimes), key=next_regimes.count) if next_regimes else 'Random (⚖️횡보)'
            
            r_indices = np.where(regimes == r)[0]
            daily_rets = []
            for idx in r_indices:
                if idx + 1 < n_days: daily_rets.append((closes[idx+1] - closes[idx])/closes[idx])
            mean_ret = np.mean(daily_rets) if daily_rets else 0.0
            std_ret = np.std(daily_rets) if daily_rets else 0.01
            regime_stats[r] = {'avg_dur': max(5, int(avg_dur)), 'next': most_likely_next, 'mean_ret': mean_ret, 'std_ret': std_ret}

        # 360일 궤적 생성 (미래 시뮬레이션)
        last_block = regime_blocks[-1]
        current_running_days = last_block['duration']
        avg_dur_current = regime_stats[current_regime]['avg_dur']
        remaining_days = max(1, avg_dur_current - current_running_days)
        
        path_regimes = []
        c_r, r_d = current_regime, remaining_days
        while len(path_regimes) < 360:
            take = min(r_d, 360 - len(path_regimes))
            path_regimes.extend([c_r] * take)
            c_r = regime_stats[c_r]['next']
            r_d = regime_stats[c_r]['avg_dur']
            
        trajectory = []
        sim_price, cum_var, base_date = cur_price, 0.0, dates[-1]
        max_pred_date = base_date
        
        for t, r in enumerate(path_regimes):
            mr, sr = regime_stats[r]['mean_ret'], regime_stats[r]['std_ret']
            sim_price *= (1 + mr)
            cum_var += (sr ** 2)
            std_cum = np.sqrt(cum_var)
            low_p = sim_price * (1 - 1.645 * std_cum)
            high_p = sim_price * (1 + 1.645 * std_cum)
            
            pred_date = base_date + BDay(t + 1)
            max_pred_date = pred_date
            
            trajectory.append({
                'T': t+1, 'Date': pred_date, 'Regime': r,
                'Center': round_to_tick(sim_price, up=False),
                'Low90': round_to_tick(low_p, up=False), 'High90': round_to_tick(high_p, up=True)
            })

        # 📈 실제 미래 데이터 추출 (검증용 Overlay)
        actual_future_dates = []
        actual_future_prices = []
        if not df_future.empty:
            # 예측 범위(max_pred_date)까지만 실제 데이터를 자름
            df_future_cut = df_future[df_future.index <= max_pred_date]
            actual_future_dates = df_future_cut.index.tolist()
            actual_future_prices = df_future_cut['Close'].tolist()

        # 🎯 분석일 시그마 기반: 듀얼 코어 백테스트
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
            
            # 단기 스윙형
            short_df = df_res[(df_res['Ext'] <= 2.0) & (df_res['Drop'] <= 2.0)]
            if not short_df.empty:
                top_short = short_df.sort_values('Nb_Ret', ascending=False).head(5)
                s_drop = top_short.iloc[0]['Drop']
                s_min_ext, s_max_ext = top_short['Ext'].min(), top_short['Ext'].max()
                if s_max_ext - s_min_ext < 0.2: s_max_ext += 0.3
                
                dual_results['short'] = {
                    'opt_drop': s_drop, 'min_ext': s_min_ext, 'max_ext': s_max_ext,
                    'target_min': round_to_tick(L_last + (s_min_ext * std_last), up=True),
                    'target_max': round_to_tick(L_last + (s_max_ext * std_last), up=True),
                    'cut_slope': cur_slope - s_drop
                }

            # 장기 보유형
            long_df = df_res[(df_res['Ext'] >= 2.5) & (df_res['Drop'] >= 2.0)]
            if not long_df.empty:
                top_long = long_df.sort_values('Nb_Ret', ascending=False).head(5)
                l_drop = top_long.iloc[0]['Drop']
                l_min_ext, l_max_ext = top_long['Ext'].min(), top_long['Ext'].max()
                if l_max_ext - l_min_ext < 0.2: l_max_ext += 0.3
                
                dual_results['long'] = {
                    'opt_drop': l_drop, 'min_ext': l_min_ext, 'max_ext': l_max_ext,
                    'target_min': round_to_tick(L_last + (l_min_ext * std_last), up=True),
                    'target_max': round_to_tick(L_last + (l_max_ext * std_last), up=True),
                    'cut_slope': cur_slope - l_drop
                }

        res = {
            'curr_regime': current_regime, 'cur_sigma': cur_sigma, 'cur_price': cur_price, 'cur_slope': cur_slope,
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
    with st.spinner(f"📦 {target_date} 시점으로 돌아가 미래 데이터를 가리고 연산 중입니다..."):
        res, err = run_timemachine_oracle(target_ticker, target_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 타임머신 분석 완료! (분석 기준일: {target_date})")
        
        # --- Part 1: 타임머신 궤적 및 실제 검증 그래프 ---
        st.subheader("📈 1. 360일 예상 궤적 vs 실제 주가 오버레이 (Walk-Forward Test)")
        st.markdown(f"> **분석 기준일({target_date})** 시점에 예측한 90% 확률 밴드 위에, **그 이후의 실제 시장 흐름(검은색 실선)**을 겹쳐 그렸습니다. 밴드를 이탈했는지 적중했는지 눈으로 확인하세요.")
        
        traj_df = pd.DataFrame(res['trajectory'])
        fig = go.Figure()
        
        # 상단 밴드
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['High90'], mode='lines', line=dict(width=0), showlegend=False))
        # 하단 밴드 (색칠)
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Low90'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)', name='90% 예측 밴드'))
        # 중심 가격
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Center'], mode='lines', line=dict(color='#e74c3c', width=2, dash='dot'), name='예상 중심 궤적'))
        
        # 🌟 실제 주가 오버레이 (검은색 실선)
        if res['actual_dates'] and len(res['actual_dates']) > 0:
            fig.add_trace(go.Scatter(
                x=res['actual_dates'], y=res['actual_prices'], mode='lines', 
                line=dict(color='black', width=3), name='실제 시장 흐름 (Reality)'
            ))
        else:
            st.info("💡 분석 기준일이 최근이어서 비교할 실제 미래 데이터가 없습니다. 순수 예측만 표시됩니다.")
            
        fig.update_layout(hovermode="x unified", height=500, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="미래 날짜", yaxis_title="주가 (원)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # --- Part 2: 기준일 당시의 듀얼 출구 전략 ---
        st.subheader(f"🎯 2. 기준일({target_date}) 당시 듀얼 매도 전략 (당시 Sigma: {res['cur_sigma']:.2f})")
        
        dual = res['dual_results']
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### ⚡ [단기 스윙형] 엑시트")
            if dual['short']:
                st.info(f"**목표 매도 구간 (Sigma {dual['short']['min_ext']:.1f} ~ {dual['short']['max_ext']:.1f})**\n### ₩{dual['short']['target_min']:,} ~ ₩{dual['short']['target_max']:,}")
                st.error(f"**생명선 이탈 기준**\n### 기울기 {dual['short']['cut_slope']:.2f}% (당시 {res['cur_slope']:.2f}%)")
            else:
                st.write("단기 스윙 데이터 부족.")

        with c2:
            st.markdown("#### 📦 [장기 추세형] 엑시트")
            if dual['long']:
                st.success(f"**목표 매도 구간 (Sigma {dual['long']['min_ext']:.1f} ~ {dual['long']['max_ext']:.1f})**\n### ₩{dual['long']['target_min']:,} ~ ₩{dual['long']['target_max']:,}")
                st.error(f"**생명선 이탈 기준**\n### 기울기 {dual['long']['cut_slope']:.2f}% (당시 {res['cur_slope']:.2f}%)")
            else:
                st.write("장기 추세 데이터 부족.")
