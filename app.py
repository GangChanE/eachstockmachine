import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter # 🌟 공간 스무딩을 위한 핵심 모듈
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
st.set_page_config(page_title="Quantum Oracle V18.2 (High-Res Smooth)", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V18.2: 3D 고해상도 공간 스무딩")
st.markdown("""
그리드 간격은 촘촘하게(고해상도) 유지하되, **주변 3x3 블록의 데이터를 함께 흡수하여 중앙값을 내는 공간 스무딩(Spatial Smoothing)** 필터를 적용했습니다.  
해상도를 잃지 않으면서도 노이즈가 제거된, 가장 완벽하고 부드러운 수익률 산맥이 드러납니다.
""")

with st.sidebar:
    st.header("⚙️ 3D 지형도 설정")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    target_t = st.number_input("T일 후 수익률 (보유 기간)", min_value=1, max_value=250, value=20, step=1)
    
    # 스무딩 강도 조절 옵션 추가
    smooth_size = st.slider("데이터 수집 반경 (Smoothing Size)", min_value=1, max_value=7, value=3, step=2, help="1이면 원본, 3이면 주변 3x3, 5면 5x5 블록을 뭉쳐서 계산합니다.")
    run_btn = st.button("🚀 고해상도 평탄화 지형도 생성", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 3D 매트릭스 엔진 (고해상도 그리드 & 2D 필터링)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def generate_3d_landscape_spatial(ticker, T, filter_size):
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
        
        # 🌟 3. 고해상도 그리드 복구 (dx=0.1, dy=0.05)
        dx = 0.1  
        dy = 0.05 
        
        x_min, x_max = valid_df['Slope'].min(), valid_df['Slope'].max()
        y_min, y_max = valid_df['Sigma'].min(), valid_df['Sigma'].max()
        
        # 여유 공간(Padding) 추가
        x_bins = np.arange(x_min - dx, x_max + dx*2, dx)
        y_bins = np.arange(y_min - dy, y_max + dy*2, dy)
        
        x_centers = x_bins[:-1] + dx/2
        y_centers = y_bins[:-1] + dy/2
        
        valid_df['x_bin'] = pd.cut(valid_df['Slope'], bins=x_bins, labels=False)
        valid_df['y_bin'] = pd.cut(valid_df['Sigma'], bins=y_bins, labels=False)
        
        # 각 방에 모인 점들의 1차 중앙값(Median) 추출
        grouped = valid_df.groupby(['x_bin', 'y_bin'])['Future_Ret'].median().reset_index()
        
        # 4. Z축 (수익률) 2D 매트릭스 생성
        Z_raw = np.full((len(y_centers), len(x_centers)), np.nan)
        for _, row in grouped.iterrows():
            Z_raw[int(row['y_bin']), int(row['x_bin'])] = row['Future_Ret']
            
        # 🌟 5. 핵심 로직: 2D 공간 스무딩 필터 적용 (수집 반경 확장)
        # NaN 값을 무시하고 필터링하기 위해 numpy 연산 트릭 사용
        # (NaN을 0으로 만들고, 가중치 행렬로 나누어 실제 데이터가 있는 곳만 평균냄)
        
        mask = ~np.isnan(Z_raw)
        Z_filled = np.nan_to_num(Z_raw, nan=0.0)
        
        if filter_size > 1:
            # 반경 내 데이터의 합
            Z_sum = uniform_filter(Z_filled, size=filter_size, mode='constant', cval=0.0) * (filter_size**2)
            # 반경 내 유효 데이터 개수 카운트
            valid_count = uniform_filter(mask.astype(float), size=filter_size, mode='constant', cval=0.0) * (filter_size**2)
            
            # 유효 데이터가 있는 곳만 나누어 평균(스무딩) 도출
            Z_smooth = np.full_like(Z_raw, np.nan)
            valid_mask = valid_count > 0
            Z_smooth[valid_mask] = Z_sum[valid_mask] / valid_count[valid_mask]
        else:
            Z_smooth = Z_raw # 필터 사이즈가 1이면 원본 유지
            
        res = {
            'X': x_centers,
            'Y': y_centers,
            'Z': Z_smooth, # 스무딩된 매트릭스 반환
            'T': T,
            'FilterSize': filter_size
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 3D 화면 렌더링 (Plotly)
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 고해상도 그리드 생성 후 {smooth_size}x{smooth_size} 공간 스무딩을 적용 중입니다..."):
        res, err = generate_3d_landscape_spatial(target_ticker, target_t, smooth_size)
        
    if err:
        st.error(err)
    else:
        st.success("✅ 고해상도 & 평탄화 3D 지형도 생성 완료!")
        
        X = res['X']
        Y = res['Y']
        Z = res['Z']
        T = res['T']
        
        fig = go.Figure()
        
        # 🌟 스무딩된 실제 지형도 (Surface Plot)
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
            title=f'[{target_ticker}] T+{T}일 수익률 (해상도 유지 & {smooth_size}x{smooth_size} 공간 스무딩)',
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
        st.subheader("💡 미스터 주의 3D 공간 필터링 해석")
        st.markdown("""
        * **해상도와 평탄화의 결합:** 그리드 자체는 아주 잘게 쪼개어(Slope 0.1, Sigma 0.05) **정밀한 굴곡(해상도)**을 살려내면서도, 각 지점의 높이를 정할 때 **주변 N칸의 데이터를 긁어모아 평균**을 내었습니다. 
        * **오버피팅 방어:** 단 하나의 튀는 데이터 때문에 뾰족하게 솟아오른 노이즈가 주변의 정상 데이터들과 섞이면서 부드럽게 깎여나갔습니다.
        * 왼쪽 사이드바의 **[데이터 수집 반경] 슬라이더**를 조절해 보세요. 숫자를 높일수록 지형이 더 넓고 둥글둥글한 거대 산맥으로 변하고, 1로 내리면 날카로운 원본 노이즈를 볼 수 있습니다.
        """)
