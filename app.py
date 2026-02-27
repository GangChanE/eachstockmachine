import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 고속 연산 엔진
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

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V23.1", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V23.1: 듀얼 XGBoost (Timezone 교정)")
st.markdown("""
다중 타임프레임, 이탈 충격량, VIX 등 **극한의 피처**를 학습하는 듀얼 XGBoost 검증기입니다.  
국가 간 타임존(Timezone) 충돌 문제를 완벽히 해결하여 글로벌 매크로 데이터를 오류 없이 융합합니다.
""")

with st.sidebar:
    st.header("⚙️ 하이퍼 타임머신 설정")
    target_ticker = st.text_input("종목 코드 (우량주/ETF 권장)", value="069500.KS")
    target_date = st.date_input("테스트 시작일 (타임머신 탑승일)")
    target_t = st.number_input("단기 예측 기간 (T일)", min_value=1, max_value=60, value=10, step=1)
    run_btn = st.button("🚀 듀얼 XGBoost 가동 및 검증", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 극한 피처 엔지니어링 및 듀얼 모델 학습
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def backtest_xgboost_extreme(ticker, target_date, T):
    try:
        # 1. 다중 자산 데이터 로드
        df_target = yf.download(ticker, start="2010-01-01", progress=False)
        df_vix = yf.download("^VIX", start="2010-01-01", progress=False)
        df_spx = yf.download("^GSPC", start="2010-01-01", progress=False)
        
        if df_target.empty: return None, "타겟 종목 데이터 로드 실패."
        
        # 야후 파이낸스 MultiIndex 컬럼 평탄화
        for d in [df_target, df_vix, df_spx]:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

        # 🌟 핵심 버그 수정: 타임존 꼬리표 강제 제거 (Timezone Naive)
        # 한국 날짜와 미국 날짜가 충돌하지 않도록 순수 문자열 날짜로 통일합니다.
        df_target.index = df_target.index.tz_localize(None)
        df_vix.index = df_vix.index.tz_localize(None)
        df_spx.index = df_spx.index.tz_localize(None)

        df = pd.DataFrame(index=df_target.index)
        df['Close'] = df_target['Close']
        df['Volume'] = df_target['Volume']
        df['High'] = df_target['High']
        df['Low'] = df_target['Low']
        
        # 매크로 지표 조인 (날짜 1:1 매칭)
        df = df.join(df_vix[['Close']].rename(columns={'Close': 'VIX'}), how='left')
        df = df.join(df_spx[['Close']].rename(columns={'Close': 'SPX'}), how='left')
        
        # 휴장일 차이로 인한 NaN은 직전 영업일 데이터로 꼼꼼히 채움 (ffill 후 bfill)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        target_dt = pd.to_datetime(target_date).tz_localize(None)
        df_train = df[df.index <= target_dt].copy()
        df_future = df[df.index > target_dt].copy()
        
        closes = df_train['Close'].values
        n_days = len(closes)
        if n_days < 300: return None, f"과거 데이터 부족 (현재 {n_days}일. 최소 300일 필요)."

        # 🌟 2. 다중 타임프레임 슬로프/시그마 추출
        windows = [5, 10, 20, 60]
        params = {w: get_linear_params(w) for w in windows}
        
        for w in windows:
            df_train[f'Slope_{w}'] = np.nan
            df_train[f'Sigma_{w}'] = np.nan
            
        for i in range(max(windows), n_days):
            for w in windows:
                prices = closes[i-w+1 : i+1]
                X, X_m, X_v = params[w]
                df_train.loc[df_train.index[i], f'Slope_{w}'] = calc_fast_slope(prices, X, X_m, X_v)
                df_train.loc[df_train.index[i], f'Sigma_{w}'] = calc_sigma(prices, X, X_m, X_v)

        # 🌟 3. 극한 피처 엔지니어링
        df_train['Drop_Price_Ratio'] = df_train['Close'] / df_train['Close'].shift(19)
        df_train['Drop_Sigma_20'] = df_train['Sigma_20'].shift(19)
        
        df_train['Slope_20_Accel'] = df_train['Slope_20'] - df_train['Slope_20'].shift(1)
        df_train['Sigma_20_Accel'] = df_train['Sigma_20'] - df_train['Sigma_20'].shift(1)
        df_train['Slope_Divergence'] = df_train['Slope_20'] - df_train['Slope_60']
        
        rets = df_train['Close'].pct_change().fillna(0)
        df_train['Vol_20'] = rets.rolling(20).std() * np.sqrt(252)
        
        # 왜도/첨도 계산 시 에러(NaN) 방지를 위한 안전 코드
        def safe_skew(x):
            v = skew(x)
            return v if not np.isnan(v) else 0.0
        def safe_kurt(x):
            v = kurtosis(x)
            return v if not np.isnan(v) else 0.0

        df_train['Skew_20'] = rets.rolling(20).apply(safe_skew, raw=True)
        df_train['Kurt_20'] = rets.rolling(20).apply(safe_kurt, raw=True)
        
        # RSI, MACD, ATR
        up = np.where(rets > 0, rets, 0)
        down = np.where(rets < 0, -rets, 0)
        rs = pd.Series(up).rolling(14).mean() / (pd.Series(down).rolling(14).mean() + 1e-9)
        df_train['RSI_14'] = 100 - (100 / (1 + rs.values))
        
        ema_12 = df_train['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_train['Close'].ewm(span=26, adjust=False).mean()
        df_train['MACD'] = ema_12 - ema_26
        df_train['ATR_14'] = (df_train['High'] - df_train['Low']).rolling(14).mean() / df_train['Close']
        
        # 거시 지표 변화량
        df_train['VIX_Change'] = df_train['VIX'].pct_change(5).fillna(0)
        df_train['SPX_Ret_20'] = df_train['SPX'].pct_change(20).fillna(0)
        
        # 🌟 4. 타겟 및 학습
        df_train['Target_Slope_Next'] = df_train['Slope_20'].shift(-1)
        df_train['Target_Sigma_Next'] = df_train['Sigma_20'].shift(-1)
        
        features = [
            'Slope_5', 'Slope_10', 'Slope_20', 'Slope_60', 
            'Sigma_5', 'Sigma_10', 'Sigma_20', 'Sigma_60',
            'Drop_Price_Ratio', 'Drop_Sigma_20',
            'Slope_20_Accel', 'Sigma_20_Accel', 'Slope_Divergence',
            'Vol_20', 'Skew_20', 'Kurt_20',
            'RSI_14', 'MACD', 'ATR_14',
            'VIX_Change', 'SPX_Ret_20'
        ]
        
        last_row = df_train.iloc[-1]
        ml_df = df_train.dropna(subset=features + ['Target_Slope_Next', 'Target_Sigma_Next'])
        
        if ml_df.empty: return None, "유효한 학습 데이터가 0개입니다. 시작일을 더 과거로 설정해 주세요."
        
        X_all = ml_df[features].values
        Y_slope = ml_df['Target_Slope_Next'].values
        Y_sigma = ml_df['Target_Sigma_Next'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        
        model_slope = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
        model_slope.fit(X_scaled, Y_slope)
        
        model_sigma = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
        model_sigma.fit(X_scaled, Y_sigma)
        
        res_std_slope = np.std(Y_slope - model_slope.predict(X_scaled))
        res_std_sigma = np.std(Y_sigma - model_sigma.predict(X_scaled))
        
        imp_dict = {f: imp for f, imp in zip(features, model_slope.feature_importances_)}

        # ---------------------------------------------------------
        # 📈 5. AI 가상 예측 궤적 생성
        # ---------------------------------------------------------
        curr_state = {f: last_row[f] for f in features}
        
        pred_slopes = [curr_state['Slope_20']]
        pred_sigmas = [curr_state['Sigma_20']]
        pred_dates = [df_train.index[-1]]
        
        np.random.seed(42)
        current_date = df_train.index[-1]
        
        for step in range(T):
            x_input = np.array([[curr_state[f] for f in features]])
            x_input_scaled = scaler.transform(x_input)
            
            base_next_slope = model_slope.predict(x_input_scaled)[0]
            base_next_sigma = model_sigma.predict(x_input_scaled)[0]
            
            # 노이즈 주입
            next_slope = base_next_slope + np.random.normal(0, res_std_slope * 0.7)
            next_sigma = base_next_sigma + np.random.normal(0, res_std_sigma * 0.7)
            
            curr_state['Slope_20_Accel'] = next_slope - curr_state['Slope_20']
            curr_state['Sigma_20_Accel'] = next_sigma - curr_state['Sigma_20']
            curr_state['Slope_20'] = next_slope
            curr_state['Sigma_20'] = next_sigma
            
            curr_state['Slope_60'] = curr_state['Slope_60'] * 0.95 + next_slope * 0.05
            curr_state['Slope_Divergence'] = next_slope - curr_state['Slope_60']
            
            current_date = current_date + BDay(1)
            pred_dates.append(current_date)
            pred_slopes.append(next_slope)
            pred_sigmas.append(next_sigma)

        # ---------------------------------------------------------
        # 🔍 6. 실제 현실 데이터 추출
        # ---------------------------------------------------------
        actual_slopes = [last_row['Slope_20']]
        actual_sigmas = [last_row['Sigma_20']]
        actual_dates = [df_train.index[-1]]
        
        if not df_future.empty:
            eval_closes = df['Close'].values
            future_indices = np.where(df.index > target_dt)[0]
            take_t = min(T, len(future_indices))
            
            X_20, X_m_20, X_v_20 = get_linear_params(20)
            
            for k in range(take_t):
                idx = future_indices[k]
                prices_20 = eval_closes[idx-20+1 : idx+1] 
                
                real_slope = calc_fast_slope(prices_20, X_20, X_m_20, X_v_20)
                real_sigma = calc_sigma(prices_20, X_20, X_m_20, X_v_20)
                
                actual_dates.append(df.index[idx])
                actual_slopes.append(real_slope)
                actual_sigmas.append(real_sigma)

        res = {
            'T': T,
            'importances': imp_dict,
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
# ⚙️ 7. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 타임존 꼬리표를 제거하고, 21개 다차원 변수를 듀얼 XGBoost로 학습 중입니다..."):
        res, err = backtest_xgboost_extreme(target_ticker, target_date, target_t)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 무결점 듀얼 XGBoost 검증 완료! (시작일: {target_date})")
        
        st.subheader("🧠 1. XGBoost 피처 중요도 (Slope 예측 기준)")
        imp_df = pd.DataFrame(list(res['importances'].items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
        fig_imp = go.Figure(go.Bar(x=imp_df['Importance'], y=imp_df['Feature'], orientation='h', marker=dict(color='rgba(231, 76, 60, 0.8)')))
        fig_imp.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="AI 모델 가중치 (0~1)")
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader(f"📈 2. 추세 기울기(Slope) AI 예측 vs 현실 검증")
        fig_slope = go.Figure()
        fig_slope.add_trace(go.Scatter(x=res['pred_dates'], y=res['pred_slopes'], mode='lines+markers', line=dict(color='#3498db', width=3, dash='dot'), name='AI 예상 슬로프'))
        if len(res['actual_dates']) > 1:
            fig_slope.add_trace(go.Scatter(x=res['actual_dates'], y=res['actual_slopes'], mode='lines+markers', line=dict(color='#2c3e50', width=4), name='실제 현실 슬로프'))
        fig_slope.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_slope.update_layout(hovermode="x unified", height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_slope, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader(f"📉 3. 시그마(Sigma) 복원력 AI 예측 vs 현실 검증")
        fig_sigma = go.Figure()
        fig_sigma.add_trace(go.Scatter(x=res['pred_dates'], y=res['pred_sigmas'], mode='lines+markers', line=dict(color='#e67e22', width=3, dash='dot'), name='AI 예상 시그마'))
        if len(res['actual_dates']) > 1:
            fig_sigma.add_trace(go.Scatter(x=res['actual_dates'], y=res['actual_sigmas'], mode='lines+markers', line=dict(color='#d35400', width=4), name='실제 현실 시그마'))
        fig_sigma.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_sigma.update_layout(hovermode="x unified", height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_sigma, use_container_width=True)
