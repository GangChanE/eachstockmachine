import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 고속 연산 및 호가 교정
# ---------------------------------------------------------
X_ARR = np.arange(20)
X_MEAN = 9.5
X_VAR_SUM = 665.0 

def calc_fast_sigma_slope(prices_20):
    y_mean = np.mean(prices_20)
    slope = np.sum((X_ARR - X_MEAN) * (prices_20 - y_mean)) / X_VAR_SUM
    intercept = y_mean - slope * X_MEAN
    trend_line = slope * X_ARR + intercept
    std = np.std(prices_20 - trend_line)
    
    current_price = prices_20[-1]
    sigma = (current_price - trend_line[-1]) / std if std > 0 else 0.0
    slope_pct = (slope / current_price) * 100 if current_price > 0 else 0.0
    
    return sigma, slope_pct

def round_to_tick(price):
    if price is None or np.isnan(price) or price <= 0: return 0
    if price < 2000: tick = 1
    elif price < 5000: tick = 5
    elif price < 20000: tick = 10
    elif price < 50000: tick = 50
    elif price < 200000: tick = 100
    elif price < 500000: tick = 500
    else: tick = 1000
    return round(price / tick) * tick

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V20 (ML Slope)", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V20: 랜덤포레스트 슬로프 궤적망")
st.markdown("""
5가지 다차원 변수(시그마, 기울기 가속도, 이격도, 변동성, 롤링 효과 등)를 **랜덤포레스트(Random Forest) AI**가 학습합니다.  
인간이 알 수 없는 복잡한 가중치를 AI가 스스로 조율하여, T일 동안의 '추세 기울기(Slope)'가 꺾일지 솟구칠지를 연쇄적으로 예측합니다.
""")

with st.sidebar:
    st.header("⚙️ 딥러닝 예측 설정")
    target_ticker = st.text_input("종목 코드 (티커)", value="069500.KS")
    target_t = st.number_input("목표 릴레이 기간 (T일)", min_value=1, max_value=60, value=20, step=1)
    run_btn = st.button("🚀 AI 슬로프 예측망 가동", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 데이터 엔지니어링 & 피처(Feature) 생성
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def train_and_predict_slope(ticker, T):
    try:
        raw = yf.download(ticker, start="2014-01-01", progress=False)
        if raw.empty: return None, "데이터 로드 실패."
            
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Close', 'Volume']].dropna()
        closes = df['Close'].values
        volumes = df['Volume'].values
        n_days = len(closes)
        
        if n_days < 200: return None, "과거 데이터 부족."

        win = 20
        sigmas = np.full(n_days, np.nan)
        slopes = np.full(n_days, np.nan)
        
        for i in range(win, n_days):
            sig, slp = calc_fast_sigma_slope(closes[i-win:i])
            sigmas[i] = sig
            slopes[i] = slp
            
        df['Sigma'] = sigmas
        df['Slope'] = slopes
        
        # 🌟 다차원 피처(Features) 엔지니어링
        # 1. 자기 참조 (관성 & 가속도)
        df['Slope_1d_ago'] = df['Slope'].shift(1)
        df['Slope_Accel'] = df['Slope'] - df['Slope_1d_ago'] # 가속도 (2차 미분)
        
        # 2. 미시 구조 (Drop-off Effect: 20일 전 데이터가 빠져나가는 충격량)
        df['Drop_off_Shock'] = (df['Close'] / df['Close'].shift(win)) - 1.0
        df['Hist_Vol_20'] = df['Close'].pct_change().rolling(win).std() * np.sqrt(252) # 역사적 변동성
        
        # 3. 에너지 지표 (거래량 동반 여부)
        df['Volume_Z'] = (df['Volume'] - df['Volume'].rolling(win).mean()) / df['Volume'].rolling(win).std()
        
        # Target: 내일의 슬로프 (T+1)
        df['Target_Slope_Next'] = df['Slope'].shift(-1)
        
        # 학습용 데이터셋 정제
        ml_df = df.dropna()
        
        # 피처 리스트 (X)
        features = ['Sigma', 'Slope', 'Slope_Accel', 'Drop_off_Shock', 'Hist_Vol_20', 'Volume_Z']
        X_all = ml_df[features].values
        Y_all = ml_df['Target_Slope_Next'].values
        
        # 스케일링 (표준화)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        
        # 🌟 랜덤포레스트 모델 학습
        # 트리를 100개 만들어서 과적합을 방지하고 일반화된 패턴을 찾습니다.
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_scaled, Y_all)
        
        # 피처 중요도(Feature Importance) 추출
        importances = model.feature_importances_
        imp_dict = {f: imp for f, imp in zip(features, importances)}
        
        # ---------------------------------------------------------
        # 📈 3. 연쇄적 T일 미래 예측 (Auto-Regressive 릴레이)
        # ---------------------------------------------------------
        # 오늘(현재)의 가장 마지막 데이터 추출
        last_row = ml_df.iloc[-1]
        
        curr_sigma = last_row['Sigma']
        curr_slope = last_row['Slope']
        curr_accel = last_row['Slope_Accel']
        curr_drop = last_row['Drop_off_Shock']
        curr_vol = last_row['Hist_Vol_20']
        curr_vol_z = last_row['Volume_Z']
        
        pred_slopes = [curr_slope]
        pred_sigmas = [curr_sigma]
        
        # T일 동안 릴레이 시작
        for step in range(T):
            # 현재 상태를 모델의 입력 형태로 포장
            current_state = np.array([[curr_sigma, curr_slope, curr_accel, curr_drop, curr_vol, curr_vol_z]])
            current_state_scaled = scaler.transform(current_state)
            
            # AI의 예측: "내일의 슬로프는 이것이다!"
            next_slope = model.predict(current_state_scaled)[0]
            
            # 상태 업데이트 (가상의 내일로 이동)
            curr_accel = next_slope - curr_slope # 새로운 가속도 산출
            curr_slope = next_slope
            
            # 🌟 [핵심] 시그마(고무줄)의 평균 회귀 압력 반영
            # 슬로프가 눕거나 꺾이면, 시그마도 자연스럽게 0을 향해 수축합니다. (감쇠 계수 0.9 적용)
            curr_sigma = curr_sigma * 0.9 + (curr_slope * 0.1) 
            
            # Drop-off나 Volatility는 단기 고정 상수(보수적 추정)로 유지
            pred_slopes.append(curr_slope)
            pred_sigmas.append(curr_sigma)

        res = {
            'T': T,
            'importances': imp_dict,
            'pred_slopes': pred_slopes,
            'pred_sigmas': pred_sigmas
        }
        return res, None

    except Exception as e:
        return None, f"AI 학습 중 오류 발생: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 4. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 과거 10년 치 다변량 데이터를 랜덤포레스트 AI가 학습하여 T={target_t}일의 슬로프를 연쇄 예측 중입니다..."):
        res, err = train_and_predict_slope(target_ticker, target_t)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ AI 슬로프 예측 궤적 산출 완료!")
        
        # --- Part 1: AI가 판단한 중요 지표 순위 ---
        st.subheader("🧠 1. AI가 찾아낸 슬로프 결정 요인 (Feature Importance)")
        st.markdown("> 모델이 내일의 기울기를 예측할 때 **어떤 변수를 가장 신뢰하고 가중치를 많이 두었는지**를 보여줍니다.")
        
        imp_df = pd.DataFrame(list(res['importances'].items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
        
        fig_imp = go.Figure(go.Bar(
            x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
            marker=dict(color='rgba(46, 204, 113, 0.8)')
        ))
        fig_imp.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="AI 모델 반영 비중 (0~1)")
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("---")
        
        # --- Part 2: 연쇄 예측 궤적 (Slope & Sigma) ---
        st.subheader(f"📈 2. 향후 {target_t}일 간의 '추세 기울기(Slope)' 예상 궤적")
        st.markdown("> AI 모델이 오늘부터 T일 뒤까지, 매일매일의 슬로프 변화와 시그마 복원력을 연쇄적으로 추적한 결과입니다.")
        
        x_days = np.arange(target_t + 1)
        y_slopes = res['pred_slopes']
        y_sigmas = res['pred_sigmas']
        
        from plotly.subplots import make_subplots
        fig_traj = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 1. 슬로프 궤적 (파란 실선)
        fig_traj.add_trace(go.Scatter(
            x=x_days, y=y_slopes, mode='lines+markers',
            line=dict(color='#3498db', width=3), name='예상 슬로프 (%)'
        ), secondary_y=False)
        
        # 2. 시그마 궤적 (주황 점선)
        fig_traj.add_trace(go.Scatter(
            x=x_days, y=y_sigmas, mode='lines+markers',
            line=dict(color='#e67e22', width=2, dash='dot'), name='예상 시그마 (이격도)'
        ), secondary_y=True)
        
        # 기준선 0 추가
        fig_traj.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, secondary_y=False)
        
        fig_traj.update_layout(
            hovermode="x unified", height=400, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="미래 경과 일수 (T+n)"
        )
        fig_traj.update_yaxes(title_text="<b>추세 기울기 (%)</b>", secondary_y=False)
        fig_traj.update_yaxes(title_text="<b>시그마 (고무줄)</b>", secondary_y=True)
        
        st.plotly_chart(fig_traj, use_container_width=True)
        
        st.info("💡 **미스터 주의 해석:** 파란색 선(슬로프)이 0을 향해 꺾인다면 상승 추세의 수명이 다해가고 있다는 뜻입니다. 동시에 주황색 선(시그마)이 0을 향해 수축하는 평균 회귀(Mean Reversion)를 함께 확인하십시오.")
