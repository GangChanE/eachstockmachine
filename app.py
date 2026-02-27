import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier # 🌟 메타 모델용
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 고속 연산 및 역산(Reverse Engineering) 함수
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

def reverse_calculate_price(prev_19_prices, target_slope_pct):
    """
    [핵심 수학 로직] 과거 19일의 데이터와 목표 슬로프(%)를 통해 내일의 주가를 역산합니다.
    P_next = K / (6.65 * Slope_pct - 10.5)
    """
    K = np.sum((np.arange(19) - 9.5) * prev_19_prices)
    denom = 6.65 * target_slope_pct - 10.5
    
    # 분모가 0에 수렴하여 가격이 폭발하는 특이점(Singularity) 방지
    if abs(denom) < 0.01:
        denom = -0.01 if denom < 0 else 0.01
        
    raw_price = K / denom
    last_price = prev_19_prices[-1]
    
    # 한국 시장 상하한가 30% 룰 적용 (수학적 오류로 인한 음수 가격 완벽 차단)
    return np.clip(raw_price, last_price * 0.7, last_price * 1.3)

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V24", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V24: 메타 역산 & T일 최적화기")
st.markdown("""
AI가 예측한 슬로프/시그마를 통해 **미래 주가를 역산**합니다.  
과거 100일간의 시뮬레이션을 통해 1일~20일 중 **가장 예측 적중률이 높은 보유기간(T)**의 순위를 매기고, **메타 모델(Meta-AI)**이 오늘의 장세에 맞는 최적의 T를 추천합니다.
""")

with st.sidebar:
    st.header("⚙️ 하이퍼 타임머신 설정")
    target_ticker = st.text_input("종목 코드 (우량주/ETF 권장)", value="069500.KS")
    target_date = st.date_input("테스트 시작일 (타임머신 탑승일)")
    run_btn = st.button("🚀 전체 T 순위 분석 및 메타 추천", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. AI 학습, 역산 검증 및 메타 모델 훈련
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def analyze_optimal_horizon(ticker, target_date):
    try:
        # 데이터 로드 및 타임존 제거
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

        target_dt = pd.to_datetime(target_date).tz_localize(None)
        df_train = df[df.index <= target_dt].copy()
        df_future = df[df.index > target_dt].copy()
        
        closes = df_train['Close'].values
        n_days = len(closes)
        if n_days < 300: return None, "과거 데이터 부족."

        X_20, X_m_20, X_v_20 = get_linear_params(20)
        df_train['Slope_20'] = np.nan
        df_train['Sigma_20'] = np.nan
        
        for i in range(20, n_days):
            prices_20 = closes[i-20+1 : i+1]
            df_train.loc[df_train.index[i], 'Slope_20'] = calc_fast_slope(prices_20, X_20, X_m_20, X_v_20)
            df_train.loc[df_train.index[i], 'Sigma_20'] = calc_sigma(prices_20, X_20, X_m_20, X_v_20)

        # 피처 설계
        df_train['Slope_Accel'] = df_train['Slope_20'] - df_train['Slope_20'].shift(1)
        df_train['VIX_Change'] = df_train['VIX'].pct_change(5).fillna(0)
        
        df_train['Target_Slope_Next'] = df_train['Slope_20'].shift(-1)
        features = ['Sigma_20', 'Slope_20', 'Slope_Accel', 'VIX_Change']
        
        last_row = df_train.iloc[-1]
        ml_df = df_train.dropna(subset=features + ['Target_Slope_Next'])
        
        X_all = ml_df[features].values
        Y_slope = ml_df['Target_Slope_Next'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        
        model_slope = XGBRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
        model_slope.fit(X_scaled, Y_slope)

        # ---------------------------------------------------------
        # 🌟 [검증 페이즈] 과거 100일간 1~20일 주가 역산 시뮬레이션
        # ---------------------------------------------------------
        # 미래를 20일 이상 들여다봐야 하므로, 평가 기간은 -120일부터 -20일까지로 한정 (데이터 누수 방지)
        eval_indices = range(len(ml_df) - 120, len(ml_df) - 20)
        
        error_matrix = np.zeros((len(eval_indices), 20)) # (100일, T=20)
        best_t_labels = []
        meta_features = []
        
        for row_idx, i in enumerate(eval_indices):
            curr_state = {f: ml_df.iloc[i][f] for f in features}
            
            # 메타 모델 학습을 위해 해당 날짜의 시그마와 슬로프 저장
            meta_features.append([curr_state['Sigma_20'], curr_state['Slope_20']])
            
            # 주가 역산을 위한 과거 20일 기록 (리스트 복사)
            hist_prices = list(closes[i-19 : i+1])
            
            sim_prices = []
            for step in range(20):
                x_in = scaler.transform([[curr_state[f] for f in features]])
                next_slope = model_slope.predict(x_in)[0]
                
                # 🌟 수학적 주가 역산!
                prev_19 = hist_prices[-19:]
                next_price = reverse_calculate_price(prev_19, next_slope)
                sim_prices.append(next_price)
                hist_prices.append(next_price)
                
                # 가상의 내일을 위한 상태 업데이트
                curr_state['Slope_Accel'] = next_slope - curr_state['Slope_20']
                curr_state['Slope_20'] = next_slope
                # Sigma 등은 상수 또는 단순 감쇠로 처리하여 역산에 집중
                curr_state['Sigma_20'] = curr_state['Sigma_20'] * 0.9 

            # 정답지와 비교 (실제 주가)
            actual_prices = closes[i+1 : i+21]
            # MAPE (평균 절대 비율 오차) 계산
            errors = np.abs(np.array(sim_prices) - actual_prices) / actual_prices * 100
            error_matrix[row_idx, :] = errors
            
            # 이 날짜에 가장 에러가 적었던 최고의 T 찾기
            best_t_labels.append(np.argmin(errors) + 1)

        # ---------------------------------------------------------
        # 📈 전체 기간 T 순위 매기기
        # ---------------------------------------------------------
        mean_errors_per_t = np.mean(error_matrix, axis=0)
        ranking_indices = np.argsort(mean_errors_per_t) # 에러가 작은 순으로 정렬
        
        t_rankings = []
        for rank, t_idx in enumerate(ranking_indices):
            t_rankings.append({
                'Rank': rank + 1,
                'T_days': t_idx + 1,
                'Error_Pct': mean_errors_per_t[t_idx]
            })

        # ---------------------------------------------------------
        # 🧠 메타 모델(Meta-AI) 학습: 오늘의 상태에 맞는 T 추천
        # ---------------------------------------------------------
        meta_clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        meta_clf.fit(meta_features, best_t_labels)
        
        # 테스트 시작일의 상태로 최적의 T 예측
        today_meta_feature = [[last_row['Sigma_20'], last_row['Slope_20']]]
        recommended_t = meta_clf.predict(today_meta_feature)[0]

        # ---------------------------------------------------------
        # 🔮 타겟 날짜 실전 시뮬레이션 (추천된 T까지)
        # ---------------------------------------------------------
        curr_state = {f: last_row[f] for f in features}
        hist_prices = list(closes[-20:])
        sim_prices = []
        sim_dates = []
        
        c_date = target_dt
        for step in range(20):
            x_in = scaler.transform([[curr_state[f] for f in features]])
            next_slope = model_slope.predict(x_in)[0]
            
            prev_19 = hist_prices[-19:]
            next_price = reverse_calculate_price(prev_19, next_slope)
            
            c_date += BDay(1)
            sim_prices.append(next_price)
            sim_dates.append(c_date)
            hist_prices.append(next_price)
            
            curr_state['Slope_Accel'] = next_slope - curr_state['Slope_20']
            curr_state['Slope_20'] = next_slope
            curr_state['Sigma_20'] = curr_state['Sigma_20'] * 0.9

        # 실제 미래 데이터
        actual_dates = []
        actual_prices = []
        if not df_future.empty:
            df_eval = df.copy()
            future_indices = np.where(df.index > target_dt)[0]
            take_t = min(20, len(future_indices))
            for k in range(take_t):
                actual_dates.append(df.index[future_indices[k]])
                actual_prices.append(df['Close'].iloc[future_indices[k]])

        res = {
            't_rankings': t_rankings,
            'recommended_t': int(recommended_t),
            'target_date': target_dt,
            'curr_sigma': last_row['Sigma_20'],
            'curr_slope': last_row['Slope_20'],
            'sim_dates': sim_dates,
            'sim_prices': sim_prices,
            'actual_dates': actual_dates,
            'actual_prices': actual_prices
        }
        return res, None

    except Exception as e:
        return None, f"메타 시뮬레이션 중 오류 발생: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 과거 100일 치 가상 역산 백테스트와 메타 AI(Meta-AI)를 훈련 중입니다..."):
        res, err = analyze_optimal_horizon(target_ticker, target_date)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 주가 역산 및 최적 T 분석 완료!")
        
        # --- 메타 모델 추천 ---
        st.subheader("🧠 1. 메타 AI (Meta-Model)의 실시간 추천")
        st.info(f"**현재 상태:** 시그마 {res['curr_sigma']:.2f} / 기울기 {res['curr_slope']:.2f}%\n\n"
                f"메타 AI 분석 결과, 오늘 같은 장세에서는 **[ T = {res['recommended_t']}일 ]** 뒤에 매도하는 것이 역사적으로 가장 예측 정확도가 높았습니다.")
        
        st.markdown("---")
        
        # --- 전체 T 랭킹 (1~20위) ---
        st.subheader("📊 2. 모든 장세 포함: T일 보유기간 정확도 랭킹 (Top 10)")
        df_rank = pd.DataFrame(res['t_rankings'])
        
        fig_bar = go.Figure(go.Bar(
            x=df_rank['T_days'][:10].astype(str) + "일", 
            y=df_rank['Error_Pct'][:10],
            marker=dict(color='rgba(52, 152, 219, 0.8)')
        ))
        fig_bar.update_layout(height=350, yaxis_title="역산 주가 오차율 (%) - 낮을수록 좋음", xaxis_title="보유 기간 (T)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # --- 주가 궤적 오버레이 ---
        st.subheader(f"📈 3. 역산된 주가 궤적 vs 실제 주가 (최대 20일)")
        st.markdown("> AI가 슬로프를 예측하고, 대수학적 역산 공식을 통해 뽑아낸 **'가상의 내일 주가(점선)'**입니다.")
        
        fig_price = go.Figure()
        
        fig_price.add_trace(go.Scatter(
            x=res['sim_dates'], y=res['sim_prices'], mode='lines+markers',
            line=dict(color='#e74c3c', width=3, dash='dot'), name='역산 예측 주가'
        ))
        
        if len(res['actual_dates']) > 0:
            fig_price.add_trace(go.Scatter(
                x=res['actual_dates'], y=res['actual_prices'], mode='lines+markers',
                line=dict(color='#2c3e50', width=4), name='실제 시장 주가'
            ))
            
        # 메타 AI가 추천한 날짜에 세로선 긋기
        if res['recommended_t'] <= len(res['sim_dates']):
            rec_date = res['sim_dates'][res['recommended_t'] - 1]
            fig_price.add_vline(x=rec_date, line_dash="dash", line_color="green", annotation_text=f"추천 매도일 (T={res['recommended_t']})")

        fig_price.update_layout(hovermode="x unified", height=450, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="주가 (원)")
        st.plotly_chart(fig_price, use_container_width=True)
