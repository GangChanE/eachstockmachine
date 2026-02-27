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
st.set_page_config(page_title="Quantum Oracle V10", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V10: 현재 시그마 기반 듀얼 타점 분석")
st.markdown("""
과거 진입일이 아닌 **'오늘(현재)'의 시그마**를 기준으로 백테스트를 수행합니다.  
현재 상태에서 **단기 스윙(단타)**으로 접근할 때와 **추세 추종(장투)**으로 접근할 때의 최적 익절/손절 구간을 각각 분리하여 제시합니다.
""")

with st.sidebar:
    st.header("⚙️ 내 계좌 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    entry_price = st.number_input("내 평균 단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    run_btn = st.button("🚀 장기 궤적 및 듀얼 타점 추출", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진 (마르코프 + 현재 시그마 듀얼 최적화)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_dual_oracle(ticker, ent_price, tax, fee_rate):
    try:
        raw = yf.download(ticker, start="2014-01-01", progress=False)
        if raw.empty: return None, "데이터 로드 실패."
            
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'Close']].dropna()
        closes = df['Close'].values
        opens = df['Open'].values
        dates = df.index
        n_days = len(closes)
        
        if n_days < 120: return None, "데이터 부족."

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

        current_regime = regimes[-1]
        
        # 📊 장세 마르코프 통계 (360일 궤적용 - 이전 버전 유지)
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

        # 360일 궤적 생성
        cur_price = closes[-1]
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
        for t, r in enumerate(path_regimes):
            mr, sr = regime_stats[r]['mean_ret'], regime_stats[r]['std_ret']
            sim_price *= (1 + mr)
            cum_var += (sr ** 2)
            std_cum = np.sqrt(cum_var)
            low_p = sim_price * (1 - 1.645 * std_cum)
            high_p = sim_price * (1 + 1.645 * std_cum)
            trajectory.append({
                'T': t+1, 'Date': base_date + BDay(t + 1), 'Regime': r,
                'Center': round_to_tick(sim_price, up=False),
                'Low90': round_to_tick(low_p, up=False), 'High90': round_to_tick(high_p, up=True)
            })

        # ---------------------------------------------------------
        # 🎯 현재 시그마 기반: 듀얼 코어 백테스트 (단기 vs 장기)
        # ---------------------------------------------------------
        cur_sigma = sigmas[-1]
        cur_slope = slopes20[-1]
        
        # '현재 시그마 이하'일 때 진입하는 조건 설정
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
                        # 현재 시그마와 비슷하거나 더 낮을 때 진입 (현재 상태를 모방)
                        if sigmas[k] <= -c_ent_p:
                            hold, bp, es, trades = True, opens[k+1], slopes20[k], trades + 1
                    else:
                        if sigmas[k] >= ex or slopes20[k] < (es - dp):
                            hold = False
                            net = ((opens[k+1] - max(0, opens[k+1]-bp)*tax) / bp) - 1.0 - fee_rate
                            cap *= (1.0 + net)
                if trades > 0: 
                    ret_grid[idp, iex] = (cap - 1.0) * 100
                    all_res.append({'Drop': dp, 'Ext': ex, 'TotRet': (cap-1)*100, 'Trades': trades})

        smooth_ret = uniform_filter(ret_grid, size=3, mode='constant', cval=-100.0)
        
        y_last = closes[-win20:]
        s_l, i_l, _, _, _ = linregress(x20, y_last)
        L_last = s_l*(win20-1) + i_l
        std_last = np.std(y_last - (s_l*x20 + i_l))
        
        # 기본 반환값 설정
        dual_results = {'short': None, 'long': None}

        if np.max(smooth_ret) != -100.0:
            df_res = pd.DataFrame(all_res)
            df_res['Nb_Ret'] = df_res.apply(lambda r: smooth_ret[np.where(DROP_RANGE==r['Drop'])[0][0], np.where(EXT_RANGE==r['Ext'])[0][0]], axis=1)
            
            # --- 단기 스윙형 (Short-term) 분리 ---
            # 짧게 먹고 나오는 전략 (Ext가 2.0 미만, Drop도 타이트하게)
            short_df = df_res[(df_res['Ext'] <= 2.0) & (df_res['Drop'] <= 2.0)]
            if not short_df.empty:
                top_short = short_df.sort_values('Nb_Ret', ascending=False).head(5)
                s_drop = top_short.iloc[0]['Drop']
                s_min_ext, s_max_ext = top_short['Ext'].min(), top_short['Ext'].max()
                if s_max_ext - s_min_ext < 0.2: s_max_ext += 0.3
                
                s_target_min = round_to_tick(L_last + (s_min_ext * std_last), up=True)
                s_target_max = round_to_tick(L_last + (s_max_ext * std_last), up=True)
                
                dual_results['short'] = {
                    'opt_drop': s_drop, 'min_ext': s_min_ext, 'max_ext': s_max_ext,
                    'target_min': s_target_min, 'target_max': s_target_max,
                    'cut_slope': cur_slope - s_drop
                }

            # --- 장기 보유형 (Long-term) 분리 ---
            # 길게 가져가는 추세 추종 (Ext가 2.5 이상, Drop을 넉넉하게 주어 잔파도 무시)
            long_df = df_res[(df_res['Ext'] >= 2.5) & (df_res['Drop'] >= 2.0)]
            if not long_df.empty:
                top_long = long_df.sort_values('Nb_Ret', ascending=False).head(5)
                l_drop = top_long.iloc[0]['Drop']
                l_min_ext, l_max_ext = top_long['Ext'].min(), top_long['Ext'].max()
                if l_max_ext - l_min_ext < 0.2: l_max_ext += 0.3
                
                l_target_min = round_to_tick(L_last + (l_min_ext * std_last), up=True)
                l_target_max = round_to_tick(L_last + (l_max_ext * std_last), up=True)
                
                dual_results['long'] = {
                    'opt_drop': l_drop, 'min_ext': l_min_ext, 'max_ext': l_max_ext,
                    'target_min': l_target_min, 'target_max': l_target_max,
                    'cut_slope': cur_slope - l_drop
                }

        res = {
            'curr_regime': current_regime, 'cur_sigma': cur_sigma, 'cur_price': cur_price, 'cur_slope': cur_slope,
            'trajectory': trajectory, 'dual_results': dual_results,
            'my_profit': ((cur_price / ent_price) - 1.0) * 100 if ent_price > 0 else 0.0
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner("📦 현재 상태 기준 듀얼 매도 전략을 연산 중입니다..."):
        res, err = run_dual_oracle(target_ticker, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success("✅ 연산 완료!")
        
        # --- Part 1: 360일 장기 궤적 ---
        st.subheader("📈 1. 향후 360일 예상 가격 궤적 (Interactive Chart)")
        traj_df = pd.DataFrame(res['trajectory'])
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['High90'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Low90'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)', name='90% 확률 밴드'))
        fig.add_trace(go.Scatter(x=traj_df['Date'], y=traj_df['Center'], mode='lines', line=dict(color='#e74c3c', width=2), name='예상 중심가', customdata=traj_df['Regime'], hovertemplate="<b>%{x|%Y-%m-%d} (T+%{text})</b><br>예상 장세: %{customdata}<br>예상가: ₩%{y:,.0f}<extra></extra>", text=traj_df['T']))
        
        fig.update_layout(hovermode="x unified", height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="미래 날짜", yaxis_title="예상 주가 (원)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # --- Part 2: 현재 기준 듀얼 출구 전략 ---
        st.subheader(f"🎯 2. 현재 상태 기준 최적 매도 구간 (현재 Sigma: {res['cur_sigma']:.2f})")
        st.markdown(f"> 당신이 지금 이 종목을 보유 중이라고 가정했을 때, **오늘의 시그마**를 바탕으로 '단기 스윙'과 '장기 보유' 전략을 각각 도출했습니다.")
        
        dual = res['dual_results']
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### ⚡ [단기 스윙형] 엑시트")
            st.caption("잔파도에서 짧게 수익을 챙기고 나오는 방어적 타점입니다.")
            if dual['short']:
                st.info(f"**목표 매도 구간 (Sigma {dual['short']['min_ext']:.1f} ~ {dual['short']['max_ext']:.1f})**\n### ₩{dual['short']['target_min']:,} ~ ₩{dual['short']['target_max']:,}")
                st.error(f"**손절/익절 마지노선 (Trailing Stop)**\n### 기울기 {dual['short']['cut_slope']:.2f}% 이탈 시")
                st.caption(f"※ 현재 기울기: {res['cur_slope']:.2f}%")
            else:
                st.write("단기 스윙에 적합한 통계적 데이터가 부족합니다.")

        with c2:
            st.markdown("#### 📦 [장기 추세형] 엑시트")
            st.caption("잔파도를 무시하고 굵은 추세를 끝까지 발라먹는 타점입니다.")
            if dual['long']:
                st.success(f"**목표 매도 구간 (Sigma {dual['long']['min_ext']:.1f} ~ {dual['long']['max_ext']:.1f})**\n### ₩{dual['long']['target_min']:,} ~ ₩{dual['long']['target_max']:,}")
                st.error(f"**손절/익절 마지노선 (Trailing Stop)**\n### 기울기 {dual['long']['cut_slope']:.2f}% 이탈 시")
                st.caption(f"※ 단기형보다 손절 각도가 넉넉하여(보유력 강화) 쉽게 털리지 않습니다.")
            else:
                st.write("장기 보유에 적합한 통계적 데이터가 부족합니다.")
                
        st.markdown("---")
        rtn_text = f"현재 추정 수익률: **{res['my_profit']:+.2f}%**" if entry_price > 0 else ""
        st.markdown(f"🤖 **미스터 주의 최종 지침:** 내 투자 성향이 단기인지 장기인지 선택하고, 정해진 타점의 밴드가 오면 덜어내십시오. {rtn_text}")
