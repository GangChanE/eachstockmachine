import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter
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
st.set_page_config(page_title="Quantum Oracle V18.3", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V18.3: 3D 지형도 (절대 색상 고정)")
st.markdown("""
**절대 색상 스케일(Absolute Color Scale)**이 적용되었습니다.  
이제 종목이 바뀌거나 필터 강도를 조절해도 **수익률 0%는 항상 중간색(흰색), +30% 이상은 진한 붉은색, -30% 이하는 진한 푸른색**으로 고정되어 다른 종목들과 직관적인 비교가 가능합니다.
""")

with st.sidebar:
    st.header("⚙️ 3D 지형도 설정")
    target_ticker = st.text_input("종목 코드 (티커)", value="069500.KS") # KODEX 200 기본값
    target_t = st.number_input("T일 후 수익률 (보유 기간)", min_value=1, max_value=250, value=20, step=1)
    smooth_size = st.slider("데이터 수집 반경 (Smoothing Size)", min_value=1, max_value=7, value=3, step=2)
    
    st.markdown("---")
    # 🌟 사용자 맞춤형 색상 고정 범위 설정 추가
    color_limit = st.number_input("컬러 기준선 (± %)", min_value=10, max_value=100, value=30, step=5, help="이 수치 이상/이하의 수익률은 가장 진한 색으로 칠해집니다.")
    
    run_btn = st.button("🚀 절대 색상 지형도 생성", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 3D 매트릭스 엔진
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
        
        for i in range(win, n_days):
            sig, slp = calc_fast_sigma_slope(closes[i-win:i])
            sigmas[i] = sig
            slopes[i] = slp
            
        df['Slope'] = slopes
        df['Sigma'] = sigmas
        df['Future_Ret'] = (df['Close'].shift(-T) / df['Close'] - 1.0) * 100
        
        valid_df = df.dropna(subset=['Slope', 'Sigma', 'Future_Ret'])
        
        dx = 0.1  
        dy = 0.05 
        
        x_min, x_max = valid_df['Slope'].min(), valid_df['Slope'].max()
        y_min, y_max = valid_df['Sigma'].min(), valid_df['Sigma'].max()
        
        x_bins = np.arange(x_min - dx, x_max + dx*2, dx)
        y_bins = np.arange(y_min - dy, y_max + dy*2, dy)
        
        x_centers = x_bins[:-1] + dx/2
        y_centers = y_bins[:-1] + dy/2
        
        valid_df['x_bin'] = pd.cut(valid_df['Slope'], bins=x_bins, labels=False)
        valid_df['y_bin'] = pd.cut(valid_df['Sigma'], bins=y_bins, labels=False)
        
        grouped = valid_df.groupby(['x_bin', 'y_bin'])['Future_Ret'].median().reset_index()
        
        Z_raw = np.full((len(y_centers), len(x_centers)), np.nan)
        for _, row in grouped.iterrows():
            Z_raw[int(row['y_bin']), int(row['x_bin'])] = row['Future_Ret']
            
        mask = ~np.isnan(Z_raw)
        Z_filled = np.nan_to_num(Z_raw, nan=0.0)
        
        if filter_size > 1:
            Z_sum = uniform_filter(Z_filled, size=filter_size, mode='constant', cval=0.0) * (filter_size**2)
            valid_count = uniform_filter(mask.astype(float), size=filter_size, mode='constant', cval=0.0) * (filter_size**2)
            
            Z_smooth = np.full_like(Z_raw, np.nan)
            valid_mask = valid_count > 0
            Z_smooth[valid_mask] = Z_sum[valid_mask] / valid_count[valid_mask]
        else:
            Z_smooth = Z_raw
            
        res = {
            'X': x_centers,
            'Y': y_centers,
            'Z': Z_smooth,
            'T': T
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 3D 화면 렌더링 (Plotly)
# ---------------------------------------------------------
if run_btn:
    with st.spinner(f"📦 고해상도 그리드 생성 후 공간 스무딩 및 절대 색상 스케일을 적용 중입니다..."):
        res, err = generate_3d_landscape_spatial(target_ticker, target_t, smooth_size)
        
    if err:
        st.error(err)
    else:
        st.success("✅ 지형도 생성 완료!")
        
        X = res['X']
        Y = res['Y']
        Z = res['Z']
        T = res['T']
        
        fig = go.Figure()
        
        # 🌟 핵심: cmin과 cmax를 강제로 고정하여 색상 기준을 절대화
        fig.add_trace(go.Surface(
            z=Z, x=X, y=Y,
            colorscale='RdBu_r', 
            cmin=-color_limit, # 진한 파랑색의 기준 (예: -30%)
            cmax=color_limit,  # 진한 빨강색의 기준 (예: +30%)
            colorbar=dict(title=f"수익률 (%)<br>고정 기준: ±{color_limit}%"),
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=True) 
            ),
            connectgaps=False 
        ))
        
        # Z=0 (수익률 0%) 바닥 평면 추가
        zero_plane = np.zeros((len(Y), len(X)))
        fig.add_trace(go.Surface(
            z=zero_plane, x=X, y=Y,
            showscale=False,
            opacity=0.3, 
            colorscale=[[0, 'gray'], [1, 'gray']],
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f'[{target_ticker}] T+{T}일 절대 색상 수익률 지형도',
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
        st.info(f"💡 **절대 색상 고정 안내:** 좌측 사이드바에서 설정한 `±{color_limit}%`를 기준으로 색상이 칠해집니다. 따라서 종목을 변경하며 여러 번 돌려도 붉은색의 짙음만으로 어떤 종목이 더 폭발적인 알파(Alpha)를 가졌는지 공정하게 1:1 비교가 가능합니다.")
