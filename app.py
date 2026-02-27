import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d, uniform_filter
import time
import warnings

# scikit-learn 설치 여부 체크 및 처리
try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 1. 페이지 설정 및 UI
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V2", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle: 데이터 로딩 최적화 버전")
st.markdown("""
최근 `yfinance` 업데이트 대응 및 데이터 로딩 안정성을 강화한 버전입니다.  
**고무줄 복원 주기(T일)** 탐색과 **맞춤형 출구 전략**을 동시에 수행합니다.
""")

with st.sidebar:
    st.header("⚙️ 분석 정보 입력")
    target_ticker = st.text_input("종목 코드 (티커)", value="005930.KS")
    entry_date = st.date_input("진입 날짜 (매수일)")
    entry_price = st.number_input("매수 평단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    run_btn = st.button("🚀 데이터 분석 및 진단 실행", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진 (안전한 데이터 로드)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_quantum_oracle_v2(ticker, ent_date, ent_price, tax, fee_rate):
    try:
        # 데이터 로드 (안전한 방식)
        raw = yf.download(ticker, start="2015-01-01", progress=False)
        
        if raw.empty:
            return None, "데이터가 비어있습니다. 티커를 확인해주세요."
            
        # yfinance 신버전 멀티인덱스 대응
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            # 'Close', 'Open' 등의 레벨만 남기기
            df.columns = df.columns.get_level_values(0)
            
        # 필수 컬럼 존재 확인
        if 'Close' not in df.columns or 'Open' not in df.columns:
            return None, "필수 데이터(Open/Close)가 부족합니다."
            
        df = df[['Open', 'Close']].dropna()
        closes = df['Close'].values
        opens = df['Open'].values
        dates = df.index
        n_days = len(closes)
        
        if n_days < 50:
            return None, "데이터가 너무 적습니다. (최소 50거래일 필요)"

        # 지표 선계산
        win = 20
        sigmas = np.full(n_days, 999.0)
        slopes = np.full(n_days, -999.0)
        x = np.arange(win)
        
        for i in range(win, n_days):
            y_seg = closes[i-win:i]
            s, inter, _, _, _ = linregress(x, y_seg)
            std = np.std(y_seg - (s*x + inter))
            if std > 0: sigmas[i] = (closes[i] - (s*(win-1)+inter)) / std
            if closes[i] > 0: slopes[i] = (s / closes[i]) * 100

        valid_idx = np.where(sigmas != 999.0)[0]

        # --- Part 1: T일 탐색 ---
        max_t = 30
        t_correlations = []
        for t in range(1, max_t + 1):
            x_sig, y_ret = [], []
            for i in valid_idx:
                if i + t < n_days:
                    x_sig.append(sigmas[i])
                    y_ret.append((closes[i+t] / closes[i]) - 1.0)
            if len(x_sig) > 10:
                t_correlations.append(np.corrcoef(x_sig, y_ret)[0, 1])
            else:
                t_correlations.append(0)
        
        smooth_corrs = uniform_filter1d(t_correlations, size=5)
        best_t = np.argmin(smooth_corrs) + 1 
        best_corr = smooth_corrs[best_t - 1]

        # --- Part 2: 확률 밴드 (로지스틱 회귀) ---
        bands = {}
        cur_prob = 50.0
        if HAS_SKLEARN:
            X_train, Y_train = [], []
            for i in valid_idx:
                if i + best_t < n_days:
                    X_train.append([sigmas[i]])
                    Y_train.append(1 if closes[i+best_t] > closes[i] else 0)
            
            if len(X_train) > 50:
                model = LogisticRegression().fit(X_train, Y_train)
                coef_a, intercept_b = model.coef_[0][0], model.intercept_[0]
                
                y_last = closes[-win:]
                s_l, i_l, _, _, _ = linregress(x, y_last)
                L_last = s_l*(win-1) + i_l
                std_last = np.std(y_last - (s_l*x + i_l))
                
                def get_p_for_prob(p):
                    if coef_a == 0: return None
                    sig = - (np.log(1/p - 1) + intercept_b) / coef_a
                    return L_last + (sig * std_last)

                bands = {
                    "90% ~ 99%": (get_p_for_prob(0.99), get_p_for_prob(0.90)),
                    "70% ~ 90%": (get_p_for_prob(0.90), get_p_for_prob(0.70)),
                    "50% ~ 70%": (get_p_for_prob(0.70), get_p_for_prob(0.50)),
                    "30% ~ 50%": (get_p_for_prob(0.50), get_p_for_prob(0.30)),
                    "10% ~ 30%": (get_p_for_prob(0.30), get_p_for_prob(0.10)),
                    " 1% ~ 10%": (get_p_for_prob(0.10), get_p_for_prob(0.01))
                }
                cur_prob = model.predict_proba([[sigmas[-1]]])[0][1] * 100

        # --- Part 3: 출구 최적화 ---
        ent_dt = pd.to_datetime(ent_date)
        closest_idx = np.argmin(np.abs(dates - ent_dt))
        my_ent_sig = sigmas[closest_idx]
        
        DROP_RANGE = np.round(np.arange(0.1, 5.1, 0.1), 1)
        EXT_RANGE = np.round(np.arange(-1.0, 5.1, 0.1), 1)
        shape = (len(DROP_RANGE), len(EXT_RANGE))
        ret_grid = np.full(shape, -100.0)
        
        c_ent_p = np.round(-my_ent_sig, 1)
        for idp, dp in enumerate(DROP_RANGE):
            for iex, ex in enumerate(EXT_RANGE):
                cap, hold, bp, es, trades = 1.0, False, 0.0, 0.0, 0
                for k in range(win, n_days-1):
                    if not hold:
                        if sigmas[k] <= -c_ent_p:
                            hold, bp, es, trades = True, opens[k+1], slopes[k], trades + 1
                    else:
                        if sigmas[k] >= ex or slopes[k] < (es - dp):
                            hold = False
                            net = ((opens[k+1] - (max(0, opens[k+1]-bp)*tax)) / bp) - 1.0 - fee_rate
                            cap *= (1.0 + net)
                if trades > 0: ret_grid[idp, iex] = (cap - 1.0) * 100

        smooth_ret = uniform_filter(ret_grid, size=3, mode='constant', cval=-100.0)
        best_idx = np.unravel_index(np.argmax(smooth_ret), smooth_ret.shape)
        o_dp, o_ex = DROP_RANGE[best_idx[0]], EXT_RANGE[best_idx[1]]
        
        # 최종 리턴
        res = {
            'best_t': best_t, 'best_corr': best_corr, 'bands': bands, 
            'cur_price': closes[-1], 'cur_sigma': sigmas[-1], 'cur_prob': cur_prob,
            'ent_sigma': my_ent_sig, 'opt_ext': o_ex, 'opt_drop': o_dp,
            'target_price': L_last + (o_ex * std_last), 
            'cut_slope': np.max(slopes[closest_idx:]) - o_dp,
            'cur_slope': slopes[-1], 'peak_slope': np.max(slopes[closest_idx:])
        }
        return res, None

    except Exception as e:
        return None, f"오류 발생: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner("📦 데이터를 안전하게 불러와 분석 중입니다..."):
        res, err = run_quantum_oracle_v2(target_ticker, entry_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 분석 완료! (최적 주기 T={res['best_t']}일)")
        
        # 1. 확률 지도
        st.subheader("🗺️ Probability Map (T일 후 상승 확률)")
        cols = st.columns(6)
        colors = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"]
        for i, (name, val) in enumerate(res['bands'].items()):
            with cols[i]:
                st.markdown(f"<h5 style='color:{colors[i]};'>{name}</h5>", unsafe_allow_html=True)
                if val[0] and val[1]:
                    st.write(f"₩{min(val):,.0f} ~ ₩{max(val):,.0f}")
                    if min(val) <= res['cur_price'] <= max(val): st.write("👈 **현재 위치**")
                else: st.caption("도달 불가")
        
        # 2. 맞춤 전략
        st.markdown("---")
        st.subheader("🎯 맞춤형 출구 전략")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("목표 익절가", f"₩{res['target_price']:,.0f}", f"Sigma {res['opt_ext']:.1f}")
        with c2:
            st.metric("생명선 (기울기)", f"{res['cut_slope']:.2f}%", f"현재 {res['cur_slope']:.2f}%")
        
        if res['cur_slope'] < res['cut_slope']:
            st.error("🚨 **[추세 이탈]** 생명선이 깨졌습니다. 즉시 매도를 검토하세요.")
        else:
            st.info("🚀 **[추세 유지]** 목표가 도달 혹은 생명선 이탈 전까지 홀딩하세요.")
