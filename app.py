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
st.set_page_config(page_title="Quantum Oracle V18 (3D)", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V18: 3D Alpha Landscape")
st.markdown("""
X축(기울기), Y축(시그마)을 바닥 평면(Grid)으로 삼고, **T일 후의 수익률 중앙값**을 Z축으로 솟아오르게 만든 3D 지형도입니다.  
붉은 산봉우리가 형성된 좌표가 가장 확률 높은 매수 타점이며, 푸른 계곡은 강력한 하락(손절)을 의미합니다. 마우스로 회전시키며 추세를 분석해 보세요.
""")

with st.sidebar:
    st.header("⚙️ 3D 지형도 설정")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    target_t = st.number_input("T일 후 수익률 (보유 기간)", min_value=1, max_value=250, value=20, step=1)
    run_btn = st.button("🚀 3D 수익률 지형도 생성", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 3D 매트릭스 엔진 (그리드 분할 & 중앙값 산출)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def generate_3d_landscape(ticker, T):
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
        # Shift 연산: 오늘 산 주식이 T일 뒤에 몇 % 올랐는가?
        df['Slope'] = slopes
        df['Sigma'] = sigmas
        df['Future_Ret'] = (df['Close'].shift(-T) / df['Close'] - 1.0) * 100
        
        # NaN 데이터 제거 (맨 앞 20일, 맨 뒤 T일 날아감)
        valid_df = df.dropna(subset=['Slope', 'Sigma', 'Future_Ret'])
        
        # 🌟 3. 회원님 요청 로직: X(Slope), Y(Sigma) 그리드화
        dx = 0.1  # Slope 그리드 간격
        dy = 0.05 # Sigma 그리드 간격
        
        x_min, x_max = valid_df['Slope'].min(), valid_df['Slope'].max()
        y_min, y_max = valid_df['Sigma'].min(), valid_df['Sigma'].max()
        
        # 여유 공간(Padding) 추가
        x_bins = np.arange(x_min - dx, x_max + dx*2, dx)
        y_bins = np.arange(y_min - dy, y_max + dy*2, dy)
        
        # 그리드의 중앙값 좌표 (X, Y축 틱)
        x_centers = x_bins[:-1] + dx/2
        y_centers = y_bins[:-1] + dy/2
        
        # 데이터를 그리드 방(Bin)에 배정
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
    with st.spinner(f"📦 수만 개의 데이터를 {target_t}일 수익률 기준 3D 그리드로 압축 중입니다..."):
        res, err = generate_3d_landscape(target_ticker, target_t)
        
    if err:
        st.error(err)
    else:
        st.success("✅ 3D 수익률 지형도 매트릭스 생성 완료!")
        
        X = res['X']
        Y = res['Y']
        Z = res['Z']
        T = res['T']
        
        fig = go.Figure()
        
        # 🌟 실제 지형도 (Surface Plot) 추가
        # 비어있는 그리드(NaN)를 무시하고 이어진 지형도를 그림
        fig.add_trace(go.Surface(
            z=Z, x=X, y=Y,
            colorscale='RdBu_r', # 붉은색(수익), 푸른색(손실) 계열
            colorbar=dict(title=f"T+{T}일 수익률 (%)"),
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=True) # 바닥에 그림자(등고선) 투영
            ),
            connectgaps=False # 데이터가 아예 없는 구간은 끊어버림(절벽 표현)
        ))
        
        # 🌟 Z=0 (수익률 0%) 바닥 평면 추가 (기준선 역할)
        zero_plane = np.zeros((len(Y), len(X)))
        fig.add_trace(go.Surface(
            z=zero_plane, x=X, y=Y,
            showscale=False,
            opacity=0.3, # 반투명한 유리 바닥
            colorscale=[[0, 'gray'], [1, 'gray']],
            hoverinfo='skip'
        ))
        
        # 레이아웃 비율 및 카메라 시점 튜닝
        fig.update_layout(
            title=f'[{target_ticker}] Slope & Sigma 조합별 T+{T}일 후 예상 수익률',
            autosize=True,
            height=800,
            scene=dict(
                xaxis_title='Slope (추세 기울기 %)',
                yaxis_title='Sigma (볼린저 이격도)',
                zaxis_title=f'T+{T}일 수익률 (%)',
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.2) # 기본 3D 회전 각도
                )
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💡 미스터 주의 3D 지형도 해석 가이드")
        st.markdown("""
        * **🔴 붉은 산봉우리 (Red Peaks):** 해당 `기울기(X)`와 `시그마(Y)` 조합이 만들어질 때 매수하면, T일 뒤에 가장 통계적으로 높은 수익을 안겨주었던 **알파(Alpha) 타점**입니다. 봉우리가 넓고 평평할수록 실전에 강한 튼튼한 전략입니다.
        * **🔵 푸른 심해 (Blue Valleys):** 수익률이 0% 이하(유리 바닥 밑)로 추락한 **절대 진입 금지 구역(손절 구간)**입니다.
        * **🌐 반투명 회색 유리판:** 수익률이 0%인 **본전(Break-Even) 커트라인**입니다. 지형이 이 유리판 위에 떠 있어야만 수익이 난다는 뜻입니다.
        * 마우스 좌클릭으로 지형을 **회전**시키고, 우클릭이나 휠로 **줌인/줌아웃**하여 특정 좌표의 데이터를 정밀 타겟팅 하십시오.
        """)
