import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 고속 연산용 상수 및 함수
# ---------------------------------------------------------
X_ARR = np.arange(20)
X_MEAN = 9.5
X_VAR_SUM = 665.0 

def calc_fast_sigmas(closes, win=20):
    """전체 기간의 20일 시그마를 고속으로 추출합니다."""
    n_days = len(closes)
    sigmas = np.full(n_days, np.nan)
    for i in range(win, n_days):
        prices_20 = closes[i-win:i]
        y_mean = np.mean(prices_20)
        slope = np.sum((X_ARR - X_MEAN) * (prices_20 - y_mean)) / X_VAR_SUM
        intercept = y_mean - slope * X_MEAN
        trend_line = slope * X_ARR + intercept
        std = np.std(prices_20 - trend_line)
        if std > 0:
            sigmas[i] = (closes[i] - trend_line[-1]) / std
    return sigmas

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V19", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V19: 평균 회귀 반감기 분석기")
st.markdown("""
진입 시점의 **시그마(x)**와 T일 후의 **시그마 변화량(y)**을 비교하여, 해당 종목이 가장 빠르고 정확하게 본래의 추세(평균)로 돌아오는 최적의 보유 기간(T)을 찾아냅니다.  
선형성(결정계수 $R^2$)이 가장 높은 Top 10 그래프를 출력합니다.
""")

with st.sidebar:
    st.header("⚙️ 백테스트 설정")
    target_ticker = st.text_input("종목 코드 (티커)", value="069500.KS") # KODEX 200 등 점잖은 종목 추천
    max_t = st.number_input("최대 탐색 기간 (Max T)", min_value=10, max_value=120, value=60, step=10)
    run_btn = st.button("🚀 최적의 회귀 주기(T) 탐색", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진 (Sigma vs Delta Sigma)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def analyze_mean_reversion(ticker, max_t):
    try:
        raw = yf.download(ticker, start="2014-01-01", progress=False)
        if raw.empty: return None, "데이터 로드 실패."
            
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Close']].dropna()
        closes = df['Close'].values
        n_days = len(closes)
        
        if n_days < 120: return None, "과거 데이터 부족."

        # 모든 날짜의 시그마 계산
        sigmas = calc_fast_sigmas(closes)
        df['Sigma'] = sigmas
        
        results = []
        
        # T=1 부터 max_t 까지 반복
        for t in range(1, max_t + 1):
            # T일 후의 시그마 값을 당겨옴 (Shift)
            df[f'Sigma_T{t}'] = df['Sigma'].shift(-t)
            # 시그마 변화량 (y축) = T일 후 시그마 - 오늘 시그마
            df[f'Delta_Sigma_T{t}'] = df[f'Sigma_T{t}'] - df['Sigma']
            
            # NaN 제거
            valid_df = df.dropna(subset=['Sigma', f'Delta_Sigma_T{t}'])
            x = valid_df['Sigma'].values
            y = valid_df[f'Delta_Sigma_T{t}'].values
            
            if len(x) > 50:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                r_squared = r_value ** 2 # 결정계수 (선형성의 뚜렷함)
                
                # 잔차(Residual)의 표준편차 계산
                expected_y = slope * x + intercept
                residuals = y - expected_y
                res_std = np.std(residuals)
                
                results.append({
                    'T': t,
                    'R2': r_squared,
                    'Correlation': r_value,
                    'Slope': slope,
                    'Intercept': intercept,
                    'Residual_Std': res_std,
                    'x_data': x,
                    'y_data': y
                })
                
        # R-squared(결정계수)가 가장 높은 순으로 정렬 (가장 선형성이 두드러지는 T)
        results_sorted = sorted(results, key=lambda k: k['R2'], reverse=True)
        return results_sorted[:10], None # Top 10 반환

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링 (Plotly 2D Scatter)
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 T=1~{max_t}일 간의 시그마 복원력을 테스트 중입니다..."):
        top_results, err = analyze_mean_reversion(target_ticker, max_t)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 분석 완료! 가장 뚜렷한 평균 회귀를 보여주는 Top 10 주기(T)입니다.")
        st.markdown("> **해석 방법:** 그래프의 점들이 우하향 대각선에 예쁘게 모여있을수록($R^2$가 높을수록) 훌륭한 주기입니다. 진입 시그마가 +3일 때 시그마 변화량이 -3 근처라면 완벽히 제자리로 돌아왔다는 뜻입니다.")
        
        # 2개씩 짝지어서 5줄로 출력 (총 10개)
        for i in range(0, 10, 2):
            cols = st.columns(2)
            
            for j in range(2):
                if i + j < len(top_results):
                    res = top_results[i + j]
                    t_val = res['T']
                    r2_val = res['R2']
                    res_std = res['Residual_Std']
                    
                    fig = go.Figure()
                    
                    # 산점도 (실제 데이터 점들)
                    fig.add_trace(go.Scatter(
                        x=res['x_data'], y=res['y_data'],
                        mode='markers',
                        marker=dict(size=3, color='rgba(52, 152, 219, 0.4)'),
                        name='실제 변화량'
                    ))
                    
                    # 선형 추세선
                    x_line = np.array([np.min(res['x_data']), np.max(res['x_data'])])
                    y_line = res['Slope'] * x_line + res['Intercept']
                    
                    fig.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode='lines',
                        line=dict(color='red', width=3),
                        name=f'추세선 (Slope: {res["Slope"]:.2f})'
                    ))
                    
                    # 기준선 (0선)
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    fig.update_layout(
                        title=f"🥇 Rank {i+j+1} | T = {t_val}일 뒤<br><sup>선형성(R²): {r2_val:.3f} | 오차(Std): {res_std:.2f}</sup>",
                        xaxis_title="진입 당일 시그마 (x)",
                        yaxis_title=f"T+{t_val}일 후 시그마 변화량 (y)",
                        height=400,
                        margin=dict(l=0, r=0, t=50, b=0),
                        showlegend=False
                    )
                    
                    with cols[j]:
                        st.plotly_chart(fig, use_container_width=True)
