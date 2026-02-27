import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 고속 연산 함수 (배열 길이 20으로 엄격히 통제)
# ---------------------------------------------------------
X_ARR_20 = np.arange(20)
X_MEAN_20 = 9.5
X_VAR_SUM_20 = 665.0 

X_ARR_60 = np.arange(60)
X_MEAN_60 = 29.5
X_VAR_SUM_60 = 17990.0

def calc_fast_slope(prices, X_ARR, X_MEAN, X_VAR_SUM):
    y_mean = np.mean(prices)
    slope = np.sum((X_ARR - X_MEAN) * (prices - y_mean)) / X_VAR_SUM
    current_price = prices[-1]
    return (slope / current_price) * 100 if current_price > 0 else 0.0

def calc_sigma(prices, X_ARR, X_MEAN, X_VAR_SUM):
    y_mean = np.mean(prices)
    slope = np.sum((X_ARR - X_MEAN) * (prices - y_mean)) / X_VAR_SUM
    intercept = y_mean - slope * X_MEAN
    trend_line = slope * X_ARR + intercept
    std = np.std(prices - trend_line)
    return (prices[-1] - trend_line[-1]) / std if std > 0 else 0.0

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V22.1", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V22.1: 무결점 AI 검증기")
st.markdown("""
과거 특정 날짜까지의 데이터만으로 AI를 학습시키고, 미래를 완벽히 가린 상태에서 T일 간의 '슬로프 & 시그마' 궤적을 예측합니다.  
그 후 **실제 시장에서 일어난 현실 궤적과 오버레이(Overlay)**하여 모델의 단기 스윙 정확도를 검증합니다.
""")

with st.sidebar:
    st.header("⚙️ 타임머신 검증 설정")
    target_ticker = st.text_input("종목 코드 (티커)", value="069500.KS")
    target_date = st.date_input("테스트 시작일 (타임머신 탑승일)")
    target_t = st.number_input("단기 예측 기간 (T일)", min_value=1, max_value=60, value=10, step=1)
    run_btn = st.button("🚀 예측 vs 현실 비교 가동", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. AI 학습 및 검증 엔진
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def backtest_ai_prediction(ticker, target_date, T):
    try:
        raw = yf.download(ticker, start="2012-01-01", progress=False)
        if raw.empty: return None, "데이터 로드 실패."
            
        df_all = raw.copy()
        if isinstance(df_all.columns, pd.MultiIndex):
            df_all.columns = df_all.columns.get_level_values(0)
            
        df_all = df_all[['Close', 'Volume']].dropna()
        target_dt = pd.to_datetime(target_date)
        
        # 🛡️ 미래 데이터 차단
        df_train = df_all[df_all.index <= target_dt].copy()
        df_future = df_all[df_all.index > target_dt].copy()
        
        closes = df_train['Close'].values
        n_days = len(closes)
        
        if n_days < 250: return None, "지정하신 날짜 이전의 과거 데이터가 부족합니다."

        win = 20
        df_train['Slope_20'] = np.nan
        df_train['Sigma_20'] = np.nan
        df_train['Slope_60'] = np.nan
        
        # 🌟 버그 수정: i-win+1 부터 i+1까지 정확히 20개 추출
        for i in range(60, n_days):
            prices_20 = closes[i-win+1 : i+1]
            prices_60 = closes[i-60+1 : i+1]
            
            df_train.loc[df_train.index[i], 'Slope_20'] = calc_fast_slope(prices_20, X_ARR_20, X_MEAN_20, X_VAR_SUM_20)
            df_train.loc[df_train.index[i], 'Sigma_20'] = calc_sigma(prices_20, X_ARR_20, X_MEAN_20, X_VAR_SUM_20)
            df_train.loc[df_train.index[i], 'Slope_60'] = calc_fast_slope(prices_60, X_ARR_60, X_MEAN_60, X_VAR_SUM_60)

        # 피처 엔지니어링
        df_train['Slope_Accel'] = df_train['Slope_20'] - df_train['Slope_20'].shift(1)
        df_train['Slope_Divergence'] = df_train['Slope_20'] - df_train['Slope_60']
        df_train['Drop_off_Shock'] = (df_train['Close'] / df_train['Close'].shift(win)) - 1.0
        df_train['Hist_Vol_20'] = df_train['Close'].pct_change().rolling(win).std() * np.sqrt(252)
        df_train['Skewness_20'] = df_train['Close'].pct_change().rolling(win).apply(skew, raw=True)
        df_train['Volume_Z'] = (df_train['Volume'] - df_train['Volume'].rolling(win).mean()) / df_train['Volume'].rolling(win).std()
        
        df_train['OBV'] = (np.sign(df_train['Close'].diff()) * df_train['Volume']).fillna(0).cumsum()
        df_train['OBV_Slope'] = df_train['OBV'].pct_change(win) * 100 
        
        ema_12 = df_train['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_train['Close'].ewm(span=26, adjust=False).mean()
        df_train['MACD'] = ema_12 - ema_26
        df_train['MACD_Slope'] = df_train['MACD'] - df_train['MACD'].shift(1)
        
        # 타겟 설정 (내일의 슬로프)
        df_train['Target_Slope_Next'] = df_train['Slope_20'].shift(-1)
        
        features = ['Sigma_20', 'Slope_20', 'Slope_60', 'Slope_Accel', 'Slope_Divergence', 
                    'Drop_off_Shock', 'Hist_Vol_20', 'Skewness_20', 'Volume_Z', 'OBV_Slope', 'MACD_Slope']
        
        # 🌟 버그 수정: 예측의 출발점이 될 '테스트 시작일 당일(last_row)' 추출
        # (Target_Slope_Next는 NaN이겠지만, Features는 모두 들어있음)
        last_row = df_train.iloc[-1]
        
        # AI 학습용 데이터 (내일의 정답이 없는 마지막 날은 제외하고 학습)
        ml_df = df_train.dropna(subset=features + ['Target_Slope_Next'])
        
        X_all = ml_df[features].values
        Y_all = ml_df['Target_Slope_Next'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        
        model = RandomForestRegressor(n_estimators=200, max_depth=7, min_samples_leaf=5, random_state=42)
        model.fit(X_scaled, Y_all)
        
        y_pred_train = model.predict(X_scaled)
        residuals_std = np.std(Y_all - y_pred_train)
        
        # ---------------------------------------------------------
        # 📈 3. AI 가상 예측 궤적 생성
        # ---------------------------------------------------------
        curr_state = {f: last_row[f] for f in features}
        
        pred_slopes = [curr_state['Slope_20']]
        pred_sigmas = [curr_state['Sigma_20']]
        pred_dates = [df_train.index[-1]] # 정확히 사용자가 선택한 날짜부터 시작
        
        np.random.seed(42) 
        
        current_date = df_train.index[-1]
        for step in range(T):
            x_input = np.array([[curr_state[f] for f in features]])
            x_input_scaled = scaler.transform(x_input)
            
            base_next_slope = model.predict(x_input_scaled)[0]
            stochastic_shock = np.random.normal(0, residuals_std * 0.8)
            next_slope = base_next_slope + stochastic_shock
            
            curr_state['Slope_Accel'] = next_slope - curr_state['Slope_20']
            curr_state['Slope_20'] = next_slope
            curr_state['Slope_60'] = curr_state['Slope_60'] * 0.95 + next_slope * 0.05
            curr_state['Slope_Divergence'] = curr_state['Slope_20'] - curr_state['Slope_60']
            
            next_sigma = (curr_state['Sigma_20'] * 0.8) + (curr_state['Slope_Accel'] * 2.0) + np.random.normal(0, 0.2)
            curr_state['Sigma_20'] = next_sigma
            
            current_date = current_date + BDay(1)
            pred_dates.append(current_date)
            pred_slopes.append(curr_state['Slope_20'])
            pred_sigmas.append(curr_state['Sigma_20'])

        # ---------------------------------------------------------
        # 🔍 4. 실제 현실 데이터 추출 (정답지 확인)
        # ---------------------------------------------------------
        actual_slopes = [last_row['Slope_20']]
        actual_sigmas = [last_row['Sigma_20']]
        actual_dates = [df_train.index[-1]]
        
        if not df_future.empty:
            df_eval = df_all.copy()
            eval_closes = df_eval['Close'].values
            
            future_indices = np.where(df_eval.index > target_dt)[0]
            take_t = min(T, len(future_indices))
            
            # 🌟 버그 수정: 인덱싱 에러(21개 추출) 방지를 위해 정확히 20개만 슬라이싱
            for k in range(take_t):
                idx = future_indices[k]
                prices_20 = eval_closes[idx-win+1 : idx+1] 
                
                real_slope = calc_fast_slope(prices_20, X_ARR_20, X_MEAN_20, X_VAR_SUM_20)
                real_sigma = calc_sigma(prices_20, X_ARR_20, X_MEAN_20, X_VAR_SUM_20)
                
                actual_dates.append(df_eval.index[idx])
                actual_slopes.append(real_slope)
                actual_sigmas.append(real_sigma)

        res = {
            'T': T,
            'pred_dates': pred_dates,
            'pred_slopes': pred_slopes,
            'pred_sigmas': pred_sigmas,
            'actual_dates': actual_dates,
            'actual_slopes': actual_slopes,
            'actual_sigmas': actual_sigmas
        }
        return res, None

    except Exception as e:
        return None, f"검증 중 오류 발생: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 5. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 {target_date} 시점으로 돌아가 11개 다변량 AI를 학습시키고, 현실 데이터와 비교 중입니다..."):
        res, err = backtest_ai_prediction(target_ticker, target_date, target_t)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ AI 예측 vs 현실 검증 완료! (테스트 시작일: {target_date})")
        
        # --- 슬로프(Slope) 비교 차트 ---
        st.subheader(f"📈 1. 추세 기울기(Slope) 검증")
        st.markdown("> AI가 예상한 기울기의 꺾임(파란 점선)과 실제 시장의 기울기 변화(파란 실선)를 비교합니다.")
        
        fig_slope = go.Figure()
        
        fig_slope.add_trace(go.Scatter(
            x=res['pred_dates'], y=res['pred_slopes'], mode='lines+markers',
            line=dict(color='#3498db', width=3, dash='dot'), name='AI 예상 슬로프'
        ))
        
        if len(res['actual_dates']) > 1:
            fig_slope.add_trace(go.Scatter(
                x=res['actual_dates'], y=res['actual_slopes'], mode='lines+markers',
                line=dict(color='#2c3e50', width=4), name='실제 현실 슬로프 (Reality)'
            ))
            
        fig_slope.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_slope.update_layout(hovermode="x unified", height=400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Slope (%)")
        st.plotly_chart(fig_slope, use_container_width=True)
        
        st.markdown("---")
        
        # --- 시그마(Sigma) 비교 차트 ---
        st.subheader(f"📉 2. 시그마(Sigma) 복원력 검증")
        st.markdown("> AI가 예상한 시그마의 평균 회귀(주황 점선)와 실제 고무줄의 튕김(주황 실선)을 비교합니다.")
        
        fig_sigma = go.Figure()
        
        fig_sigma.add_trace(go.Scatter(
            x=res['pred_dates'], y=res['pred_sigmas'], mode='lines+markers',
            line=dict(color='#e67e22', width=3, dash='dot'), name='AI 예상 시그마'
        ))
        
        if len(res['actual_dates']) > 1:
            fig_sigma.add_trace(go.Scatter(
                x=res['actual_dates'], y=res['actual_sigmas'], mode='lines+markers',
                line=dict(color='#d35400', width=4), name='실제 현실 시그마 (Reality)'
            ))
            
        fig_sigma.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_sigma.update_layout(hovermode="x unified", height=400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Sigma (이격도)")
        st.plotly_chart(fig_sigma, use_container_width=True)
