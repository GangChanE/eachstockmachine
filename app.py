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
import random

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 호가 교정 함수
# ---------------------------------------------------------
def round_to_tick(price, up=False):
    if price is None or np.isnan(price): return None
    if price <= 0: return 0 # 지수 하락 시 방어
    
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
st.set_page_config(page_title="Quantum Oracle V12", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V12: GBM & 풀 마르코프 체인")
st.markdown("""
**1. 기하 브라운 운동(GBM):** 주가의 지수적 특성과 복리 효과를 반영하여 로그 스케일의 완벽한 궤적을 그립니다.  
**2. 풀 마르코프 체인(Full Markov Chain):** 5대 장세가 고착되지 않고, 과거 통계 확률에 따라 다이내믹하게 전환됩니다.
""")

with st.sidebar:
    st.header("⚙️ 타임머신 & 계좌 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    target_date = st.date_input("분석 기준일 (타임머신 날짜)")
    entry_price = st.number_input("기준일 매수 단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    run_btn = st.button("🚀 지수적 궤적 및 타점 생성", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진 (미래 데이터 차단 + GBM + Markov)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_v12_oracle(ticker, target_date, ent_price, tax, fee_rate):
    try:
        raw = yf.download(ticker, start="2014-01-01", progress=False)
        if raw.empty: return None, "데이터 로드 실패."
            
        df_all = raw.copy()
        if isinstance(df_all.columns, pd.MultiIndex):
            df_all.columns = df_all.columns.get_level_values(0)
            
        df_all = df_all[['Open', 'Close']].dropna()
        target_dt = pd.to_datetime(target_date)
        
        # 🛡️ 데이터 분리 (과거 학습 / 미래 검증)
        df_past = df_all[df_all.index <= target_dt]
        df_future = df_all[df_all.index > target_dt]
        
        closes = df_past['Close'].values
        opens = df_past['Open'].values
        dates = df_past.index
        n_days = len(closes)
        
        if n_days < 120: return None, "과거 데이터가 부족하여 분석할 수 없습니다."

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
        REGIME_NAMES = ['Strong Bull', 'Bull', 'Random', 'Bear', 'Strong Bear']
        regimes = np.full(n_days, 'Unknown', dtype=object)
        regimes[ann_slopes60 >= 40] = 'Strong Bull'
        regimes[(ann_slopes60 >= 10) & (ann_slopes60 < 40)] = 'Bull'
        regimes[(ann_slopes60 > -10) & (ann_slopes60 < 10)] = 'Random'
        regimes[(ann_slopes60 > -40) & (ann_slopes60 <= -10)] = 'Bear'
        regimes[ann_slopes60 <= -40] = 'Strong Bear'

        current_regime = regimes[-1]
        cur_sigma = sigmas[-1]
        cur_slope = slopes20[-1]
        cur_price = closes[-1]
        
        # 📊 1. 마르코프 전이 행렬 (Transition Matrix) 및 상태 통계 구축
        regime_blocks = []
        curr_r = regimes[win60]
        start_idx = win60
        for i in range(win60 + 1, n_days):
            if regimes[i] != curr_r:
                regime_blocks.append({'regime': curr_r, 'duration': i - start_idx})
                curr_r = regimes[i]
                start_idx = i
        regime_blocks.append({'regime': curr_r, 'duration': n_days - start_idx})
        
        # 전이 행렬 초기화
        trans_matrix = {r1: {r2: 0 for r2 in REGIME_NAMES} for r1 in REGIME_NAMES}
        for i in range(len(regime_blocks) - 1):
            r_from = regime_blocks[i]['regime']
            r_to = regime_blocks[i+1]['regime']
            if r_from in trans_matrix and r_to in trans_matrix:
                trans_matrix[r_from][r_to] += 1
                
        # 확률로 변환
        for r1 in REGIME_NAMES:
            total = sum(trans_matrix[r1].values())
            if total > 0:
                for r2 in REGIME_NAMES: trans_matrix[r1][r2] /= total
            else:
                trans_matrix[r1]['Random'] = 1.0 # 데이터 없으면 횡보로

        # 장세별 통계 (로그 수익률 기반 GBM 파라미터 계산)
        regime_stats = {}
        for r in REGIME_NAMES:
            r_blocks = [b for b in regime_blocks if b['regime'] == r]
            avg_dur = np.mean([b['duration'] for b in r_blocks]) if r_blocks else 20
            
            r_indices = np.where(regimes == r)[0]
            log_rets = []
            for idx in r_indices:
                if idx + 1 < n_days and closes[idx] > 0 and closes[idx+1] > 0: 
                    log_rets.append(np.log(closes[idx+1] / closes[idx]))
                    
            mu = np.mean(log_rets) if log_rets else 0.0     # 일일 로그 수익률 평균
            sigma = np.std(log_rets) if log_rets else 0.02  # 일일 로그 변동성
            
            regime_stats[r] = {'avg_dur': max(5, int(avg_dur)), 'mu': mu, 'sigma': sigma}

        # 📈 2. 360일 궤적 생성 (Stochastic Transition + GBM)
        np.random.seed() # 랜덤 시드 초기화
        
        last_block = regime_blocks[-1]
        c_r = current_regime if current_regime in REGIME_NAMES else 'Random'
        r_d = max(1, regime_stats[c_r]['avg_dur'] - last_block['duration'])
        
        path_regimes = []
        while len(path_regimes) < 360:
            take = min(r_d, 360 - len(path_regimes))
            path_regimes.extend([c_r] * take)
            
            # 다음 장세를 '확률 행렬'에 따라 뽑기 (고착화 방지)
            probs = [trans_matrix[c_r][nxt] for nxt in REGIME_NAMES]
            c_r = np.random.choice(REGIME_NAMES, p=probs)
            
            # 수명은 정규분포를 섞어 약간의 랜덤성 부여
            mean_dur = regime_stats[c_r]['avg_dur']
            r_d = max(5, int(np.random.normal(mean_dur, mean_dur * 0.2))) 

        trajectory = []
        base_date = dates[-1]
        max_pred_date = base_date
        
        # 누적 파라미터 계산을 통한 GBM 90% 밴드
        cum_mu = 0.0
        cum_var = 0.0
        
        for t, r in enumerate(path_regimes):
            mu = regime_stats[r]['mu']
            sig = regime_stats[r]['sigma']
            
            cum_mu += (mu - 0.5 * (sig ** 2))
            cum_var += (sig ** 2)
            
            # 90% 신뢰구간 (정규분포 Z값 1.645)
            std_cum = np.sqrt(cum_var)
            
            center_price = cur_price * np.exp(cum_mu)
            low_p = cur_price * np.exp(cum_mu - 1.645 * std_cum)
            high_p = cur_price * np.exp(cum_mu + 1.645 * std_cum)
            
            pred_date = base_date + BDay(t + 1)
            max_pred_date = pred_date
            
            display_name = {'Strong Bull': '🔥강한상승', 'Bull': '📈상승', 'Random': '⚖️횡보', 'Bear': '📉하락', 'Strong Bear': '🧊강한하락'}.get(r, r)
            
            trajectory.append({
                'T': t+1, 'Date': pred_date, 'Regime': display_name,
                'Center': round_to_tick(center_price, up=False),
                'Low90': round_to_tick(low_p, up=False), 
                'High90': round_to_tick(high_p, up=True)
            })

        # 📈 3. 실제 미래 데이터 추출 (검증용 Overlay)
        actual_future_dates = []
        actual_future_prices = []
        if not df_future.empty:
            df_future_cut = df_future[df_future.index <= max_pred_date]
            actual_future_dates = df_future_cut.index.tolist()
            actual_future_prices = df_future_cut['Close'].tolist()

        # 🎯 4. 듀얼 코어 백테스트
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
    with st.spinner(f"📦 GBM (기하브라운운동) 및 마르코프 전이 연산 중..."):
        res, err = run_v12_oracle(target_ticker, target_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 타임머신 분석 완료! (분석 기준일: {target_date})")
        
        st.subheader("📈 1. GBM 360일 지수적 궤적 vs 실제 주가 검증")
        st.markdown(f"> **분석 기준일({target_date})** 시점의 **장세({res['curr_regime']})**를 시작으로, 확률적 전이(Markov)와 복리 효과(GBM)를 반영하여 그려낸 나팔꽃 형태의 로그 스케일 밴드입니다.")
        
        traj_df = pd.DataFrame(res['trajectory'])
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['High90'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Low90'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)', name='90% GBM 예측 밴드'))
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Center'], mode='lines', line=dict(color='#e74c3c', width=2, dash='dot'), name='예상 중심 궤적', customdata=traj_df['Regime'], hovertemplate="<b>%{x|%Y-%m-%d} (T+%{text})</b><br>예상 장세: %{customdata}<br>예상가: ₩%{y:,.0f}<extra></extra>", text=traj_df['T']))
        
        if res['actual_dates'] and len(res['actual_dates']) > 0:
            fig.add_trace(go.Scatter(x=res['actual_dates'], y=res['actual_prices'], mode='lines', line=dict(color='black', width=3), name='실제 시장 흐름 (Reality)'))
            
        # y축을 로그 스케일로 표시하려면 주석 해제 (단, 변동폭이 너무 크지 않으면 linear도 무방함)
        # fig.update_layout(yaxis_type="log")
            
        fig.update_layout(hovermode="x unified", height=500, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="미래 날짜", yaxis_title="주가 (원)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
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
