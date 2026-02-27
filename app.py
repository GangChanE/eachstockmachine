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
st.set_page_config(page_title="Quantum Oracle V9 (360-Day Interactive)", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V9: 마르코프 장세 사이클 & 360일 예측")
st.markdown("""
현재 시장의 장세(Regime)가 과거 통계상 **며칠 동안 유지되었고, 언제 다음 장세로 전환될지**를 예측합니다.  
이를 바탕으로 T=1일부터 360일까지의 장기 가격 궤적(90% 신뢰구간)을 인터랙티브 그래프로 그려냅니다.
""")

with st.sidebar:
    st.header("⚙️ 내 진입 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    entry_date = st.date_input("진입 날짜 (매수일)")
    entry_price = st.number_input("매수 단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    run_btn = st.button("🚀 장기 궤적 생성 & 맞춤 타점 추출", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 마르코프 레짐 핵심 분석 엔진
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_markov_oracle(ticker, ent_date, ent_price, tax, fee_rate):
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

        # 🚦 1. 장세(Regime) 분류
        regimes = np.full(n_days, 'Unknown', dtype=object)
        regimes[ann_slopes60 >= 40] = 'Strong Bull (🔥강한상승)'
        regimes[(ann_slopes60 >= 10) & (ann_slopes60 < 40)] = 'Bull (📈상승)'
        regimes[(ann_slopes60 > -10) & (ann_slopes60 < 10)] = 'Random (⚖️횡보)'
        regimes[(ann_slopes60 > -40) & (ann_slopes60 <= -10)] = 'Bear (📉하락)'
        regimes[ann_slopes60 <= -40] = 'Strong Bear (🧊강한하락)'

        # ---------------------------------------------------------
        # 📊 2. 장세 수명(Duration) 및 전환 확률(Transition) 통계 추출
        # ---------------------------------------------------------
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
            
            # 다음 장세 예측 (가장 많이 전환된 장세)
            next_regimes = [regime_blocks[i+1]['regime'] for i, b in enumerate(regime_blocks[:-1]) if b['regime'] == r]
            most_likely_next = max(set(next_regimes), key=next_regimes.count) if next_regimes else 'Random (⚖️횡보)'
            
            # 해당 장세의 일일 평균 수익률 및 변동성
            r_indices = np.where(regimes == r)[0]
            daily_rets = []
            for idx in r_indices:
                if idx + 1 < n_days: daily_rets.append((closes[idx+1] - closes[idx])/closes[idx])
            mean_ret = np.mean(daily_rets) if daily_rets else 0.0
            std_ret = np.std(daily_rets) if daily_rets else 0.01
            
            regime_stats[r] = {'avg_dur': max(5, int(avg_dur)), 'next': most_likely_next, 'mean_ret': mean_ret, 'std_ret': std_ret}

        # ---------------------------------------------------------
        # 📈 3. 360일 장기 궤적 (Trajectory) 생성
        # ---------------------------------------------------------
        cur_price = closes[-1]
        last_block = regime_blocks[-1]
        current_regime = last_block['regime']
        current_running_days = last_block['duration']
        avg_dur_current = regime_stats[current_regime]['avg_dur']
        remaining_days = max(1, avg_dur_current - current_running_days)
        
        path_regimes = []
        c_r = current_regime
        r_d = remaining_days
        
        # 360일간의 장세 릴레이 시뮬레이션
        while len(path_regimes) < 360:
            take = min(r_d, 360 - len(path_regimes))
            path_regimes.extend([c_r] * take)
            c_r = regime_stats[c_r]['next']
            r_d = regime_stats[c_r]['avg_dur']
            
        trajectory = []
        sim_price = cur_price
        cum_var = 0.0
        base_date = dates[-1]
        
        for t, r in enumerate(path_regimes):
            mr = regime_stats[r]['mean_ret']
            sr = regime_stats[r]['std_ret']
            
            sim_price *= (1 + mr)
            cum_var += (sr ** 2)
            std_cum = np.sqrt(cum_var)
            
            # 90% 신뢰구간 (1.645 * 누적 표준편차)
            low_p = sim_price * (1 - 1.645 * std_cum)
            high_p = sim_price * (1 + 1.645 * std_cum)
            pred_date = base_date + BDay(t + 1)
            
            trajectory.append({
                'T': t+1, 'Date': pred_date, 'Regime': r,
                'Center': round_to_tick(sim_price, up=False),
                'Low90': round_to_tick(low_p, up=False),
                'High90': round_to_tick(high_p, up=True)
            })

        # ---------------------------------------------------------
        # 🛡️ 4. 맞춤형 출구 최적화 (기존 3x3x3 로직 유지)
        # ---------------------------------------------------------
        ent_dt = pd.to_datetime(ent_date)
        closest_idx = np.argmin(np.abs(dates - ent_dt))
        my_ent_sig = sigmas[closest_idx]
        my_regime = regimes[closest_idx]
        
        c_ent_p = np.round(-my_ent_sig, 1)
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
                        if sigmas[k] <= -c_ent_p and regimes[k] == my_regime:
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
        if np.max(smooth_ret) == -100.0:
            opt_ext, opt_drop, target_price_min, target_price_max = 0, 0, 0, 0
        else:
            df_res = pd.DataFrame(all_res)
            df_res['Nb_Ret'] = df_res.apply(lambda r: smooth_ret[np.where(DROP_RANGE==r['Drop'])[0][0], np.where(EXT_RANGE==r['Ext'])[0][0]], axis=1)
            
            top_5 = df_res.sort_values('Nb_Ret', ascending=False).head(5)
            opt_drop = top_5.iloc[0]['Drop']
            min_ext, max_ext = top_5['Ext'].min(), top_5['Ext'].max()
            if max_ext - min_ext < 0.2: max_ext += 0.3
            opt_ext = min_ext
            
            y_last = closes[-win20:]
            s_l, i_l, _, _, _ = linregress(x20, y_last)
            L_last = s_l*(win20-1) + i_l
            std_last = np.std(y_last - (s_l*x20 + i_l))
            
            target_price_min = round_to_tick(L_last + (min_ext * std_last), up=True)
            target_price_max = round_to_tick(L_last + (max_ext * std_last), up=True)

        recent_slopes = slopes20[closest_idx:]
        peak_slope = np.max(recent_slopes[recent_slopes != -999.0]) if len(recent_slopes) > 0 else slopes20[-1]
        cut_slope = peak_slope - opt_drop

        res = {
            'regime': my_regime, 'ent_sigma': my_ent_sig,
            'curr_regime': current_regime, 'curr_running_days': current_running_days,
            'avg_dur_curr': avg_dur_current, 'remaining_days': remaining_days,
            'next_regime_pred': regime_stats[current_regime]['next'],
            'trajectory': trajectory,
            'opt_ext': opt_ext, 'target_min': target_price_min, 'target_max': target_price_max, 
            'cut_slope': cut_slope, 'cur_price': cur_price, 'cur_slope': slopes20[-1],
            'my_profit': ((cur_price / ent_price) - 1.0) * 100 if ent_price > 0 else 0.0
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링 (Plotly 그래프 포함)
# ---------------------------------------------------------
if run_btn:
    with st.spinner("📦 마르코프 체인 알고리즘을 통한 360일 장기 궤적을 연산 중입니다..."):
        res, err = run_markov_oracle(target_ticker, entry_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 연산 완료!")
        
        # --- Part 1: 현재 장세 생명 주기 브리핑 ---
        st.subheader("⏳ 1. 현재 시장 장세 및 수명(Cycle) 예측")
        c1, c2, c3 = st.columns(3)
        c1.metric("현재 시장 장세", res['curr_regime'])
        c2.metric("현재 장세 진행 일수", f"{res['curr_running_days']}일째", f"과거 평균 수명: {res['avg_dur_curr']}일", delta_color="off")
        c3.metric("예상 전환 시점 (Next)", f"약 {res['remaining_days']}일 뒤", f"예상 다음 장세: {res['next_regime_pred']}", delta_color="normal")
        
        st.markdown("---")
        
        # --- Part 2: 360일 인터랙티브 궤적 그래프 ---
        st.subheader("📈 2. 향후 360일 예상 가격 궤적 (Interactive Chart)")
        st.markdown("> 차트 위에 마우스를 올리거나 터치하면 해당 지점의 **날짜, 예상 장세, 90% 범위 가격**을 볼 수 있습니다.")
        
        traj_df = pd.DataFrame(res['trajectory'])
        
        fig = go.Figure()
        
        # 상단 밴드
        fig.add_trace(go.Scatter(
            x=traj_df['Date'], y=traj_df['High90'], mode='lines',
            line=dict(width=0), name='상위 5% 한계', showlegend=False
        ))
        
        # 하단 밴드 (색칠)
        fig.add_trace(go.Scatter(
            x=traj_df['Date'], y=traj_df['Low90'], mode='lines',
            line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)',
            name='90% 확률 밴드'
        ))
        
        # 중심 가격 (통계적 밀집 구간)
        fig.add_trace(go.Scatter(
            x=traj_df['Date'], y=traj_df['Center'], mode='lines',
            line=dict(color='#e74c3c', width=2), name='예상 중심가',
            customdata=traj_df['Regime'],
            hovertemplate="<b>%{x|%Y-%m-%d} (T+%{text})</b><br>" +
                          "장세: %{customdata}<br>" +
                          "예상가: ₩%{y:,.0f}<extra></extra>",
            text=traj_df['T']
        ))
        
        fig.update_layout(
            hovermode="x unified", height=500, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="미래 날짜", yaxis_title="예상 주가 (원)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # --- Part 3: 최적 출구 전략 ---
        st.subheader("🎯 3. 진입 조건 맞춤형 최적 출구 전략")
        st.markdown(f"> 나의 진입 조건(**{res['regime']} / Sigma {res['ent_sigma']:.2f}**)에서 누적 수익을 가장 극대화했던 타점입니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🔥 **통계적 분할 매도 구간**")
            if res['target_min'] > 0:
                st.metric(label="목표 도달 시 밴드", value=f"₩{res['target_min']:,} ~ ₩{res['target_max']:,}")
            else:
                st.write("해당 조건의 유효한 익절 백테스트 데이터가 부족합니다.")
            
        with col2:
            st.error(f"🚨 **생명선 (Trailing Stop)**")
            st.metric(label=f"기울기 {res['cut_slope']:.2f}% (현재 {res['cur_slope']:.2f}%)", value=f"하락 시 전량 매도")
            
        st.markdown("---")
        if res['cur_slope'] < res['cut_slope']:
            st.error("🤖 **미스터 주의 지침:** 🚨 **[생명선 이탈]** 상승 추세가 꺾였습니다. 장기 예측과 무관하게 즉시 매도하여 자산을 보호하십시오.")
        else:
            rtn_text = f" (현재 수익률: {res['my_profit']:+.2f}%)" if entry_price > 0 else ""
            st.success(f"🤖 **미스터 주의 지침:** 🚀 **[순항 중 / 홀딩]** 위 그래프의 궤적을 그리며 우상향 중입니다. 평온하게 홀딩하십시오.{rtn_text}")
