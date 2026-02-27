import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 고속 연산 및 역산 엔진
# ---------------------------------------------------------
def get_linear_params(win):
    X = np.arange(win)
    X_mean = np.mean(X)
    X_var_sum = np.sum((X - X_mean)**2)
    return X, X_mean, X_var_sum

def calc_fast_slope(prices, X, X_mean, X_var_sum):
    y_mean = np.mean(prices)
    slope = np.sum((X - X_mean) * (prices - y_mean)) / X_var_sum
    current_price = prices[-1]
    return (slope / current_price) * 100 if current_price > 0 else 0.0

def calc_sigma(prices, X, X_mean, X_var_sum):
    y_mean = np.mean(prices)
    slope = np.sum((X - X_mean) * (prices - y_mean)) / X_var_sum
    intercept = y_mean - slope * X_mean
    trend_line = slope * X + intercept
    std = np.std(prices - trend_line)
    return (prices[-1] - trend_line[-1]) / std if std > 0 else 0.0

def vectorized_reverse_price(hist_19_matrix, target_slopes):
    weights = np.arange(19) - 9.5
    K = np.sum(weights * hist_19_matrix, axis=1)
    denom = 6.65 * target_slopes - 9.5
    denom[np.abs(denom) < 0.01] = np.sign(denom[np.abs(denom) < 0.01]) * 0.01 + 1e-9
    raw_prices = K / denom
    last_prices = hist_19_matrix[:, -1]
    return np.clip(raw_prices, last_prices * 0.7, last_prices * 1.3)

def reverse_calculate_price(prev_19_prices, target_slope_pct):
    K = np.sum((np.arange(19) - 9.5) * prev_19_prices)
    denom = 6.65 * target_slope_pct - 9.5
    if abs(denom) < 0.01:
        denom = -0.01 if denom < 0 else 0.01
    raw_price = K / denom
    last_price = prev_19_prices[-1]
    return np.clip(raw_price, last_price * 0.7, last_price * 1.3)

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V26.1", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V26.1: 방향 적중률 & 신뢰 구간")
st.markdown("""
단순 오차 크기가 아닌, **"오늘 사서 T일 뒤에 팔았을 때 오르고 내리는 '수익 방향(Direction)'을 얼마나 잘 맞췄는가(Hit Ratio %)"**를 기준으로 진짜 1위 T를 찾습니다.  
또한 미래 주가의 불확실성을 시각화한 **80% 신뢰 구간(Confidence Interval)** 밴드가 추가되었습니다.
""")

with st.sidebar:
    st.header("⚙️ 딥스캔 분석 설정")
    target_ticker = st.text_input("종목 코드 (우량주/ETF 권장)", value="069500.KS")
    run_btn = st.button("🚀 전체 역사 분석 및 오늘 전략 생성", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. AI 딥스캔 및 메타 모델 엔진
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def deep_scan_and_meta_predict(ticker):
    try:
        df_target = yf.download(ticker, start="2010-01-01", progress=False)
        df_vix = yf.download("^VIX", start="2010-01-01", progress=False)
        df_spx = yf.download("^GSPC", start="2010-01-01", progress=False)
        
        if df_target.empty: return None, "데이터 로드 실패."
        
        for d in [df_target, df_vix, df_spx]:
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)

        df_target.index = df_target.index.tz_localize(None)
        df_vix.index = df_vix.index.tz_localize(None)
        df_spx.index = df_spx.index.tz_localize(None)

        df = pd.DataFrame(index=df_target.index)
        df['Close'] = df_target['Close']
        df['Volume'] = df_target['Volume']
        
        df = df.join(df_vix[['Close']].rename(columns={'Close': 'VIX'}), how='left')
        df = df.join(df_spx[['Close']].rename(columns={'Close': 'SPX'}), how='left')
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        closes = df['Close'].values
        n_days = len(closes)
        if n_days < 500: return None, "과거 데이터 부족 (최소 500일 필요)."

        X_20, X_m_20, X_v_20 = get_linear_params(20)
        X_60, X_m_60, X_v_60 = get_linear_params(60)
        
        df['Slope_20'] = np.nan
        df['Sigma_20'] = np.nan
        df['Slope_60'] = np.nan
        
        for i in range(60, n_days):
            p20 = closes[i-20+1 : i+1]
            p60 = closes[i-60+1 : i+1]
            df.loc[df.index[i], 'Slope_20'] = calc_fast_slope(p20, X_20, X_m_20, X_v_20)
            df.loc[df.index[i], 'Sigma_20'] = calc_sigma(p20, X_20, X_m_20, X_v_20)
            df.loc[df.index[i], 'Slope_60'] = calc_fast_slope(p60, X_60, X_m_60, X_v_60)

        df['Slope_Accel'] = df['Slope_20'] - df['Slope_20'].shift(1)
        df['Slope_Divergence'] = df['Slope_20'] - df['Slope_60']
        df['VIX_Change'] = df['VIX'].pct_change(5).fillna(0)
        
        df['Target_Slope_Next'] = df['Slope_20'].shift(-1)
        
        features = ['Sigma_20', 'Slope_20', 'Slope_60', 'Slope_Accel', 'Slope_Divergence', 'VIX_Change']
        
        today_row = df.iloc[-1]
        ml_df = df.dropna(subset=features + ['Target_Slope_Next']).copy()
        
        X_all = ml_df[features].values
        Y_slope = ml_df['Target_Slope_Next'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        
        model_slope = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
        model_slope.fit(X_scaled, Y_slope)

        residuals = Y_slope - model_slope.predict(X_scaled)
        res_std = np.std(residuals)

        # ---------------------------------------------------------
        # 🌟 방향 적중률(Hit Ratio) 기반 초고속 벡터 백테스트
        # ---------------------------------------------------------
        eval_df = ml_df.iloc[-420:-20].copy()
        N_eval = len(eval_df)
        
        hit_matrix = np.zeros((N_eval, 20)) 
        curr_state_matrix = eval_df[features].values 
        
        hist_idx = [np.where(df.index == d)[0][0] for d in eval_df.index]
        hist_prices_matrix = np.array([closes[idx-19 : idx+1] for idx in hist_idx])
        base_prices = hist_prices_matrix[:, -1] 
        
        for step in range(20):
            x_in_scaled = scaler.transform(curr_state_matrix)
            next_slopes = model_slope.predict(x_in_scaled)
            
            prev_19 = hist_prices_matrix[:, -19:]
            next_prices = vectorized_reverse_price(prev_19, next_slopes)
            
            hist_prices_matrix = np.hstack((hist_prices_matrix[:, 1:], next_prices.reshape(-1, 1)))
            
            actual_future_prices = np.array([closes[idx + step + 1] for idx in hist_idx])
            pred_direction = np.sign(next_prices - base_prices)
            actual_direction = np.sign(actual_future_prices - base_prices)
            
            hits = (pred_direction == actual_direction).astype(int)
            hit_matrix[:, step] = hits
            
            curr_state_matrix[:, features.index('Slope_Accel')] = next_slopes - curr_state_matrix[:, features.index('Slope_20')]
            curr_state_matrix[:, features.index('Slope_20')] = next_slopes
            curr_state_matrix[:, features.index('Sigma_20')] *= 0.9 

        # ---------------------------------------------------------
        # 🧠 메타 모델 학습
        # ---------------------------------------------------------
        hit_rates_per_t = np.mean(hit_matrix, axis=0) * 100 
        passive_best_t = np.argmax(hit_rates_per_t) + 1 
        
        best_t_labels = []
        for i in range(N_eval):
            hits = hit_matrix[i]
            if np.sum(hits) == 0:
                best_t_labels.append(passive_best_t)
            else:
                valid_ts = np.where(hits == 1)[0]
                best_t_labels.append(valid_ts[-1] + 1)
        
        meta_features = eval_df[['Sigma_20', 'Slope_20']].values
        meta_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        meta_clf.fit(meta_features, best_t_labels)
        
        today_sigma = today_row['Sigma_20']
        today_slope = today_row['Slope_20']
        active_best_t = meta_clf.predict([[today_sigma, today_slope]])[0]
        meta_prob = np.max(meta_clf.predict_proba([[today_sigma, today_slope]])) * 100

        # ---------------------------------------------------------
        # 🔮 오늘 기준 미래 주가 역산 & 신뢰 구간 생성
        # ---------------------------------------------------------
        today_state = {f: today_row[f] for f in features}
        today_hist = list(closes[-20:])
        
        sim_dates = []
        sim_prices_mean = []
        sim_prices_upper = [] 
        sim_prices_lower = [] 
        
        c_date = df.index[-1]
        cumulative_std = 0
        
        for step in range(20):
            x_in = scaler.transform([[today_state[f] for f in features]])
            base_slope = model_slope.predict(x_in)[0]
            
            cumulative_std += (res_std * np.sqrt(step + 1)) * 0.1 
            upper_slope = base_slope + (1.28 * cumulative_std)
            lower_slope = base_slope - (1.28 * cumulative_std)
            
            mean_price = reverse_calculate_price(np.array(today_hist[-19:]), base_slope)
            upper_price = reverse_calculate_price(np.array(today_hist[-19:]), upper_slope)
            lower_price = reverse_calculate_price(np.array(today_hist[-19:]), lower_slope)
            
            u_p, l_p = max(upper_price, lower_price), min(upper_price, lower_price)
            
            c_date += BDay(1)
            sim_dates.append(c_date)
            sim_prices_mean.append(mean_price)
            sim_prices_upper.append(u_p)
            sim_prices_lower.append(l_p)
            
            today_hist.append(mean_price)
            
            today_state['Slope_Accel'] = base_slope - today_state['Slope_20']
            today_state['Slope_20'] = base_slope
            today_state['Sigma_20'] *= 0.9

        past_dates = df.index[-20:].tolist()
        past_prices = closes[-20:].tolist()

        res = {
            'hit_rates': hit_rates_per_t,
            'passive_t': passive_best_t,
            'active_t': int(active_best_t),
            'meta_prob': meta_prob,
            'today_sigma': today_sigma,
            'today_slope': today_slope,
            'past_dates': past_dates,
            'past_prices': past_prices,
            'sim_dates': sim_dates,
            'sim_prices_mean': sim_prices_mean,
            'sim_prices_upper': sim_prices_upper,
            'sim_prices_lower': sim_prices_lower
        }
        return res, None

    except Exception as e:
        return None, f"분석 중 오류 발생: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 400일치 방향 적중률(Hit Ratio) 분석 및 80% 신뢰 구간을 연산 중입니다..."):
        res, err = deep_scan_and_meta_predict(target_ticker)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 방향 적중률 최적화 및 신뢰 밴드 생성 완료!")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 🌐 1. 패시브 통계 (수익 방향 적중률 기준)")
            st.info(f"단순 오차가 아닌, 진입 후 주가의 '상승/하락 방향'을 가장 잘 맞춘 기간입니다.\n\n"
                    f"🏆 **역사적 고유 호흡(전체 1위): T = {res['passive_t']}일**\n\n"
                    f"(이 종목은 진입 후 평균적으로 {res['passive_t']}일 차에 가장 뚜렷한 추세를 보였습니다.)")
            
        with c2:
            st.markdown("#### 🧠 2. 메타 AI 액티브 추천 (오늘 장세 맞춤형)")
            st.success(f"오늘의 상태 (시그마: {res['today_sigma']:.2f}, 기울기: {res['today_slope']:.2f}%)\n\n"
                       f"🎯 **오늘 진입 시 최적 매도일: T = {res['active_t']}일**\n\n"
                       f"*(AI 패턴 확신도: {res['meta_prob']:.1f}% - 불확실한 장세면 패시브 T를 따르십시오)*")
        
        st.markdown("---")
        
        st.subheader("📊 3. 보유기간(T)별 '수익 방향 적중률' 랭킹 (높을수록 좋음)")
        rank_df = pd.DataFrame({
            'T (보유 일수)': [f"{i+1}일" for i in range(20)],
            '적중률 (%)': res['hit_rates']
        })
        fig_bar = go.Figure(go.Bar(
            x=rank_df['T (보유 일수)'], y=rank_df['적중률 (%)'],
            marker=dict(color=['#27ae60' if i+1 == res['active_t'] else '#95a5a6' for i in range(20)])
        ))
        fig_bar.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="방향 적중률 (%)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader(f"📈 4. 향후 20일 역산 궤적 및 80% 신뢰 구간 (Confidence Band)")
        st.markdown("> 시간이 지날수록 AI의 예측 오차가 누적되는 것을 반영하여, **주가가 흔들릴 수 있는 상하단 범위(회색 영역)**를 표시했습니다.")
        
        fig_proj = go.Figure()
        
        conn_dates = [res['past_dates'][-1]] + res['sim_dates']
        conn_upper = [res['past_prices'][-1]] + res['sim_prices_upper']
        conn_lower = [res['past_prices'][-1]] + res['sim_prices_lower']
        
        fig_proj.add_trace(go.Scatter(
            x=conn_dates + conn_dates[::-1], 
            y=conn_upper + conn_lower[::-1],
            fill='toself',
            fillcolor='rgba(149, 165, 166, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='80% 신뢰 구간'
        ))
        
        fig_proj.add_trace(go.Scatter(
            x=res['past_dates'], y=res['past_prices'], mode='lines+markers',
            line=dict(color='#2c3e50', width=4), name='과거 20일 실제 주가'
        ))
        
        conn_mean = [res['past_prices'][-1]] + res['sim_prices_mean']
        fig_proj.add_trace(go.Scatter(
            x=conn_dates, y=conn_mean, mode='lines+markers',
            line=dict(color='#e74c3c', width=3, dash='dot'), name='역산 기반 미래 궤적 (평균)'
        ))
        
        # 🌟 치명적인 버그 수정: Plotly의 add_vline annotation_text 에러 원천 차단
        rec_idx = res['active_t'] - 1
        rec_date = res['sim_dates'][rec_idx]
        
        # 선 긋기
        fig_proj.add_vline(x=rec_date, line_dash="dash", line_color="green")
        
        # 글자는 안전하게 독립적인 add_annotation으로 분리 배치
        fig_proj.add_annotation(
            x=rec_date, 
            y=1.05, # 그래프 살짝 위쪽에 위치
            yref="paper",
            text=f"🎯 AI 액티브 타겟 (T={res['active_t']})",
            showarrow=False,
            font=dict(color="green", size=13, weight="bold")
        )
        
        fig_proj.update_layout(hovermode="x unified", height=500, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="주가 (원)")
        st.plotly_chart(fig_proj, use_container_width=True)
