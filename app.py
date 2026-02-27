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
# ⚙️ 0. 고속 연산 및 대수학적 역산 엔진
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
    """
    [수학적 증명] 다차원 행렬(Matrix) 구조에서 내일의 주가를 한 번에 역산
    P_next = sum((i - 9.5) * P_i) / (6.65 * Slope_pct - 9.5)
    """
    weights = np.arange(19) - 9.5
    K = np.sum(weights * hist_19_matrix, axis=1)
    
    denom = 6.65 * target_slopes - 9.5
    # 분모 0 수렴 방지
    denom[np.abs(denom) < 0.01] = np.sign(denom[np.abs(denom) < 0.01]) * 0.01 + 1e-9
    
    raw_prices = K / denom
    last_prices = hist_19_matrix[:, -1]
    
    # 상하한가 30% 룰 적용 (기형적 역산값 방어)
    return np.clip(raw_prices, last_prices * 0.7, last_prices * 1.3)

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V25", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V25: AI 딥스캔 & 메타 추천기")
st.markdown("""
종목 코드만 입력하십시오. 기계가 과거 수백 일의 역사를 모조리 스캔하여 **가장 예측 적중률이 높았던 고유의 보유 기간(T)**을 발굴합니다.  
또한, 메타 AI가 '오늘 당장의 시그마와 슬로프'를 분석하여 오늘 장세에 딱 맞는 최적의 매도 타이밍을 추천합니다.
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
        # 1. 데이터 로드 및 타임존 제거
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
        df['High'] = df_target['High']
        df['Low'] = df_target['Low']
        
        df = df.join(df_vix[['Close']].rename(columns={'Close': 'VIX'}), how='left')
        df = df.join(df_spx[['Close']].rename(columns={'Close': 'SPX'}), how='left')
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        closes = df['Close'].values
        n_days = len(closes)
        if n_days < 500: return None, "과거 데이터 부족 (최소 500일 필요)."

        # 2. 피처 엔지니어링 (XGBoost 용)
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
        
        # 최근 20일은 검증 시 미래 가격이 없으므로 학습/백테스트에서 제외, 마지막 날은 '오늘'로 분리
        today_row = df.iloc[-1]
        ml_df = df.dropna(subset=features + ['Target_Slope_Next']).copy()
        
        X_all = ml_df[features].values
        Y_slope = ml_df['Target_Slope_Next'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        
        model_slope = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
        model_slope.fit(X_scaled, Y_slope)

        # ---------------------------------------------------------
        # 🌟 [초고속 텐서 백테스트] 과거 400일간 1~20일 역산 시뮬레이션
        # ---------------------------------------------------------
        # 최근 20일(미래 정답이 없는 구간)을 제외한 뒤에서 400일 추출
        eval_df = ml_df.iloc[-420:-20].copy()
        N_eval = len(eval_df)
        
        error_matrix = np.zeros((N_eval, 20)) # 400일 x 20일(T)
        
        # 초기 상태 행렬(Matrix) 구축
        curr_state_matrix = eval_df[features].values 
        
        # 각 날짜별 과거 20일 가격 히스토리 구축 (N_eval, 20)
        hist_idx = [np.where(df.index == d)[0][0] for d in eval_df.index]
        hist_prices_matrix = np.array([closes[idx-19 : idx+1] for idx in hist_idx])
        
        # 20 스텝 (T=1~20) 벡터화 연쇄 시뮬레이션
        for step in range(20):
            x_in_scaled = scaler.transform(curr_state_matrix)
            next_slopes = model_slope.predict(x_in_scaled)
            
            # 수학적 역산 (배열 연산으로 한 번에 계산)
            prev_19 = hist_prices_matrix[:, -19:]
            next_prices = vectorized_reverse_price(prev_19, next_slopes)
            
            # 히스토리에 새 가격 추가 (창 밀어내기)
            hist_prices_matrix = np.column_scaling = np.hstack((hist_prices_matrix[:, 1:], next_prices.reshape(-1, 1)))
            
            # 정답지 비교 (각 날짜별 step+1일 뒤 실제 주가)
            actual_future_prices = np.array([closes[idx + step + 1] for idx in hist_idx])
            errors = np.abs(next_prices - actual_future_prices) / actual_future_prices * 100
            error_matrix[:, step] = errors
            
            # 다음 턴을 위한 상태 업데이트
            curr_state_matrix[:, features.index('Slope_Accel')] = next_slopes - curr_state_matrix[:, features.index('Slope_20')]
            curr_state_matrix[:, features.index('Slope_20')] = next_slopes
            curr_state_matrix[:, features.index('Sigma_20')] *= 0.9 # 시그마 평균 회귀

        # ---------------------------------------------------------
        # 🧠 메타 모델 학습 (어떤 상황에서 어떤 T가 유리한가?)
        # ---------------------------------------------------------
        # 전체 400일 평균 오차를 바탕으로 패시브(전체 평균) 최적 T 추출
        mean_errors_per_t = np.mean(error_matrix, axis=0)
        passive_best_t = np.argmin(mean_errors_per_t) + 1
        
        # 각 날짜별 가장 오차가 적었던 T 추출
        best_t_labels = np.argmin(error_matrix, axis=1) + 1
        
        # 메타 모델: 입력값(진입일의 시그마, 슬로프) -> 출력값(최적 T)
        meta_features = eval_df[['Sigma_20', 'Slope_20']].values
        meta_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        meta_clf.fit(meta_features, best_t_labels)
        
        # 오늘(현재)의 상태 분석
        today_sigma = today_row['Sigma_20']
        today_slope = today_row['Slope_20']
        active_best_t = meta_clf.predict([[today_sigma, today_slope]])[0]
        
        # 메타 모델의 신뢰도(해당 클래스 확률) 측정
        meta_prob = np.max(meta_clf.predict_proba([[today_sigma, today_slope]])) * 100

        # ---------------------------------------------------------
        # 🔮 오늘 기준 미래 주가 역산 궤적 생성
        # ---------------------------------------------------------
        today_state = {f: today_row[f] for f in features}
        today_hist = list(closes[-20:])
        sim_future_prices = []
        sim_future_dates = []
        
        c_date = df.index[-1]
        for step in range(20):
            x_in = scaler.transform([[today_state[f] for f in features]])
            next_slope = model_slope.predict(x_in)[0]
            
            next_price = reverse_calculate_price(np.array(today_hist[-19:]), next_slope)
            
            c_date += BDay(1)
            sim_future_prices.append(next_price)
            sim_future_dates.append(c_date)
            today_hist.append(next_price)
            
            today_state['Slope_Accel'] = next_slope - today_state['Slope_20']
            today_state['Slope_20'] = next_slope
            today_state['Sigma_20'] *= 0.9

        # 최근 20일 과거 궤적 (그래프 연결용)
        past_dates = df.index[-20:].tolist()
        past_prices = closes[-20:].tolist()

        res = {
            'mean_errors': mean_errors_per_t,
            'passive_t': passive_best_t,
            'active_t': int(active_best_t),
            'meta_prob': meta_prob,
            'today_sigma': today_sigma,
            'today_slope': today_slope,
            'past_dates': past_dates,
            'past_prices': past_prices,
            'sim_dates': sim_future_dates,
            'sim_prices': sim_future_prices
        }
        return res, None

    except Exception as e:
        return None, f"분석 중 오류 발생: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 과거 400일의 평행우주를 동시 다발적으로 연산하여 메타 AI를 훈련 중입니다 (약 5초 소요)..."):
        res, err = deep_scan_and_meta_predict(target_ticker)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 종목 딥스캔 및 오늘자 메타 전략 생성 완료!")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 🌐 1. 패시브 통계 (전체 기간 평균)")
            st.info(f"이 종목이 지난 수년간 보여준 통계적 회귀의 평균치입니다.\n\n"
                    f"🏆 **역사적 고유 호흡(전체 1위): T = {res['passive_t']}일**\n\n"
                    f"가장 오차가 적고 예측이 확실하게 맞아떨어지는 평균적인 주기가 {res['passive_t']}일입니다.")
            
        with c2:
            st.markdown("#### 🧠 2. 메타 AI 액티브 추천 (오늘 장세 맞춤형)")
            st.success(f"오늘의 상태 (시그마: {res['today_sigma']:.2f}, 기울기: {res['today_slope']:.2f}%)\n\n"
                       f"🎯 **오늘 진입 시 최적 매도일: T = {res['active_t']}일**\n\n"
                       f"*(AI 확신도: {res['meta_prob']:.1f}% - 확신도가 낮으면 패시브 통계를 따르십시오)*")
        
        st.markdown("---")
        
        # --- 전체 T 랭킹 ---
        st.subheader("📊 3. 보유기간(T)별 예측 오차율 랭킹")
        rank_df = pd.DataFrame({
            'T (보유 일수)': [f"{i+1}일" for i in range(20)],
            '오차율 (%)': res['mean_errors']
        })
        fig_bar = go.Figure(go.Bar(
            x=rank_df['T (보유 일수)'], y=rank_df['오차율 (%)'],
            marker=dict(color=['#e74c3c' if i+1 == res['active_t'] else '#3498db' for i in range(20)])
        ))
        fig_bar.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="역산 주가 평균 오차율 (낮을수록 좋음)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # --- 미래 투영 궤적 ---
        st.subheader(f"📈 4. 오늘 기준: 20일 향후 역산 궤적 (Forward Projection)")
        st.markdown("> 과거 20일의 흐름을 이어받아, AI가 역산해 낸 **'순수 미래 20일'**의 점선 궤적입니다.")
        
        fig_proj = go.Figure()
        
        # 과거 20일 실제 데이터
        fig_proj.add_trace(go.Scatter(
            x=res['past_dates'], y=res['past_prices'], mode='lines+markers',
            line=dict(color='#2c3e50', width=4), name='과거 20일 실제 주가'
        ))
        
        # 미래 20일 예측 데이터 (과거 마지막 점과 연결)
        conn_dates = [res['past_dates'][-1]] + res['sim_dates']
        conn_prices = [res['past_prices'][-1]] + res['sim_prices']
        
        fig_proj.add_trace(go.Scatter(
            x=conn_dates, y=conn_prices, mode='lines+markers',
            line=dict(color='#e74c3c', width=3, dash='dot'), name='역산 기반 미래 궤적'
        ))
        
        # 추천 매도일 마킹
        rec_idx = res['active_t'] - 1
        fig_proj.add_vline(x=res['sim_dates'][rec_idx], line_dash="dash", line_color="green", 
                           annotation_text=f"AI 액티브 타겟 (T={res['active_t']})")
        
        fig_proj.update_layout(hovermode="x unified", height=450, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="주가 (원)")
        st.plotly_chart(fig_proj, use_container_width=True)
