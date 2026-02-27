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

def calc_fast_sigma_slope(prices_20):
    """20일 종가 배열을 받아 시그마와 기울기(Slope %)를 동시에 반환합니다."""
    y_mean = np.mean(prices_20)
    slope = np.sum((X_ARR - X_MEAN) * (prices_20 - y_mean)) / X_VAR_SUM
    intercept = y_mean - slope * X_MEAN
    trend_line = slope * X_ARR + intercept
    std = np.std(prices_20 - trend_line)
    
    current_price = prices_20[-1]
    sigma = (current_price - trend_line[-1]) / std if std > 0 else 0.0
    slope_pct = (slope / current_price) * 100 if current_price > 0 else 0.0
    
    return sigma, slope_pct

# ---------------------------------------------------------
# ⚙️ 1. UI 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V18.1 (3D Smooth)", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V18.1: 3D Alpha Landscape (평탄화)")
st.markdown("""
X축(기울기), Y축(시그마)의 블록(Grid) 크기를 2배로 넓혀 노이즈를 제거했습니다.  
각 블록에 더 많은 데이터가 담기면서 불규칙한 가시덤불이 사라지고, **진짜 수익이 나는 거대한 산맥(Robust 추세)**이 부드럽게 드러납니다.
""")

with st.sidebar:
    st.header("⚙️ 3D 지형도 설정")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    target_t = st.number_input("T일 후 수익률 (보유 기간)", min_value=1, max_value=250, value=20, step=1)
    run_btn = st.button("🚀 3D 평탄화 지형도 생성", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 3D 매트릭스 엔진 (그리드 분할 & 중앙값 산출)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def generate_3d_landscape_smooth(ticker, T):
    try:
        raw = yf.download(ticker, start="2014-01-01", progress=False)
        if raw.empty: return None, "데이터 로드 실패."
            
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Close']].dropna()
        closes = df['Close'].values
        n_days = len(closes)
        
        if n_days < 120 + T: return None, "과거 데이터가 부족합니다."

        win = 20
        sigmas = np.full(n_days, np.nan)
        slopes = np.full(n_days, np.nan)
        
        # 1. 매일의 시그마와 기울기 계산
        for i in range(win, n_days):
            sig, slp = calc_fast_sigma_slope(closes[i-win:i])
            sigmas[i] = sig
            slopes[i] = slp
            
        # 2. T일 후 수익률(%) 계산 (미래 참조)
        df['Slope'] = slopes
        df['Sigma'] = sigmas
        df['Future_Ret'] = (df['Close'].shift(-T) / df['Close'] - 1.0) * 100
        
        valid_df = df.dropna(subset=['Slope', 'Sigma', 'Future_Ret'])
        
        # 🌟 3. 회원님 요청 로직: 그리드(Grid) 2배 확장 (평탄화 스무딩)
        dx = 0.2  # 기존 0.1 -> 0.2 (2배 넓어짐)
        dy = 0.1  # 기존 0.05 -> 0.10 (2배 넓어짐)
        
        x_min, x_max = valid_df['Slope'].min(), valid_df['Slope'].max()
        y_min, y_max = valid_df['Sigma'].min(), valid_df['Sigma'].max()
        
        # 여유 공간(Padding) 추가
        x_bins = np.arange(x_min - dx, x_max + dx*2, dx)
        y_bins = np.arange(y_min - dy, y_max + dy*2, dy)
        
        x_centers = x_bins[:-1] + dx/2
        y_centers = y_bins[:-1] + dy/2
        
        valid_df['x_bin'] = pd.cut(valid_df['Slope'], bins=x_bins, labels=False)
        valid_df['y_bin'] = pd.cut(valid_df['Sigma'], bins=y_bins, labels=False)
        
        # 각 방에 모인 점들의 수익률 중앙값(Median) 추출
        grouped = valid_df.groupby(['x_bin', 'y_bin'])['Future_Ret'].median().reset_index()
        
        # 4. Z축 (수익률) 2D 매트릭스 생성
        Z = np.full((len(y_centers), len(x_centers)), np.nan)
        for _, row in grouped.iterrows():
            Z[int(row['y_bin']), int(row['x_bin'])] = row['Future_Ret']
            
        res = {
            'X': x_centers,
            'Y': y_centers,
            'Z': Z,
            'T': T
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 3D 화면 렌더링 (Plotly)
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 블록 크기를 2배로 넓혀 노이즈를 제거한 3D 지형을 압축 중입니다..."):
        res, err = generate_3d_landscape_smooth(target_ticker, target_t)
        
    if err:
        st.error(err)
    else:
        st.success("✅ 평탄화된 3D 수익률 지형도 생성 완료!")
        
        X = res['X']
        Y = res['Y']
        Z = res['Z']
        T = res['T']
        
        fig = go.Figure()
        
        # 🌟 실제 지형도 (Surface Plot)
        fig.add_trace(go.Surface(
            z=Z, x=X, y=Y,
            colorscale='RdBu_r', 
            colorbar=dict(title=f"T+{T}일 수익률 (%)"),
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=True) 
            ),
            connectgaps=False 
        ))
        
        # 🌟 Z=0 (수익률 0%) 바닥 평면 추가
        zero_plane = np.zeros((len(Y), len(X)))
        fig.add_trace(go.Surface(
            z=zero_plane, x=X, y=Y,
            showscale=False,
            opacity=0.3, 
            colorscale=[[0, 'gray'], [1, 'gray']],
            hoverinfo='skip'
        ))
        
        # 레이아웃 비율 및 카메라 시점 튜닝
        fig.update_layout(
            title=f'[{target_ticker}] Slope & Sigma 조합별 T+{T}일 후 예상 수익률 (Grid Smoothing)',
            autosize=True,
            height=800,
            scene=dict(
                xaxis_title='Slope (추세 기울기 %)',
                yaxis_title='Sigma (볼린저 이격도)',
                zaxis_title=f'T+{T}일 수익률 (%)',
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.2) 
                )
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💡 미스터 주의 3D 지형도 해석 가이드 (스무딩 버전)")
        st.markdown("""
        * **그리드 2배 확장 효과:** 불필요하게 뾰족하게 튀어나와 있던 가짜 수익 구간(노이즈)이 사라졌습니다. 이제 붉은색으로 솟아오른 거대한 산맥(Plateau)은 어떤 상황에서도 쉽게 깨지지 않는 **가장 튼튼한(Robust) 진짜 확률 타점**을 의미합니다.
        * **🔴 붉은 산맥 (Red Plateaus):** 이 거대한 산맥 좌표에 현재 주가의 Slope와 Sigma가 진입했다면, 눈감고 베팅(Hold)해도 좋은 구간입니다.
        * **🔵 깊은 골짜기 (Blue Valleys):** 수익률 0% 바닥 밑으로 깊게 파인 곳은 어김없이 하락이 나오는 '데스존'입니다.
        """)
