import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d, uniform_filter
import time
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 1. 페이지 설정 및 UI
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V3 (Regime-Switching)", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V3: 장세 맞춤형 예언자")
st.markdown("""
시장을 5가지 장세(Regime)로 정밀 타격하여 분리합니다.  
**진입 시점의 장세**를 분석하고, 해당 장세 전용 **고무줄 복원 주기(T일)**와 **예상 등락률 함수**를 도출한 뒤, 가장 완벽한 2D 익절/손절 구간을 계산합니다.
""")

with st.sidebar:
    st.header("⚙️ 내 진입 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    entry_date = st.date_input("진입 날짜 (매수일)")
    entry_price = st.number_input("매수 평단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    run_btn = st.button("🚀 장세 진단 및 전략 최적화", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진 (Regime-Switching)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_regime_oracle(ticker, ent_date, ent_price, tax, fee_rate):
    try:
        raw = yf.download(ticker, start="2014-01-01", progress=False)
        if raw.empty: return None, "데이터를 불러오지 못했습니다."
            
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'Close']].dropna()
        closes = df['Close'].values
        opens = df['Open'].values
        dates = df.index
        n_days = len(closes)
        
        if n_days < 120: return None, "데이터가 부족합니다."

        # 지표 선계산
        win20 = 20
        win60 = 60
        sigmas = np.full(n_days, 999.0)
        slopes20 = np.full(n_days, -999.0)
        ann_slopes60 = np.full(n_days, -999.0) # 장세 진단용
        
        x20 = np.arange(win20)
        x60 = np.arange(win60)
        
        for i in range(win60, n_days):
            # 20일 시그마 및 단기 기울기
            y20 = closes[i-win20:i]
            s20, i20, _, _, _ = linregress(x20, y20)
            std20 = np.std(y20 - (s20*x20 + i20))
            if std20 > 0: sigmas[i] = (closes[i] - (s20*(win20-1)+i20)) / std20
            if closes[i] > 0: slopes20[i] = (s20 / closes[i]) * 100
            
            # 60일 장기 기울기 (연환산 %로 변환)
            y60 = closes[i-win60:i]
            s60, _, _, _, _ = linregress(x60, y60)
            if closes[i] > 0:
                ann_slopes60[i] = (s60 / closes[i]) * 100 * 252

        # ---------------------------------------------------------
        # 🚦 장세(Regime) 분류
        # ---------------------------------------------------------
        regimes = np.full(n_days, 'Unknown', dtype=object)
        regimes[ann_slopes60 >= 40] = 'Strong Bull (🔥강한상승)'
        regimes[(ann_slopes60 >= 10) & (ann_slopes60 < 40)] = 'Bull (📈상승)'
        regimes[(ann_slopes60 > -10) & (ann_slopes60 < 10)] = 'Random (⚖️횡보)'
        regimes[(ann_slopes60 > -40) & (ann_slopes60 <= -10)] = 'Bear (📉하락)'
        regimes[ann_slopes60 <= -40] = 'Strong Bear (🧊강한하락)'

        # ---------------------------------------------------------
        # 🎯 진입 시점 환경 분석
        # ---------------------------------------------------------
        ent_dt = pd.to_datetime(ent_date)
        closest_idx = np.argmin(np.abs(dates - ent_dt))
        
        my_ent_sig = sigmas[closest_idx]
        my_regime = regimes[closest_idx]
        
        if my_ent_sig == 999.0 or my_regime == 'Unknown':
            return None, "진입 날짜의 데이터가 부족하여 장세 진단이 불가합니다."

        # 해당 장세에 속하는 인덱스만 추출
        regime_indices = np.where(regimes == my_regime)[0]
        
        # 샘플 수가 너무 적으면 (100일 미만) 인접 장세로 병합 (Fallback)
        if len(regime_indices) < 100:
            if my_regime == 'Strong Bull (🔥강한상승)': fallback = ['Strong Bull (🔥강한상승)', 'Bull (📈상승)']
            elif my_regime == 'Strong Bear (🧊강한하락)': fallback = ['Strong Bear (🧊강한하락)', 'Bear (📉하락)']
            else: fallback = [my_regime]
            regime_indices = np.where(np.isin(regimes, fallback))[0]

        # ---------------------------------------------------------
        # ⏱️ 장세 맞춤형 T일 최적화 (해당 장세 데이터만 사용)
        # ---------------------------------------------------------
        max_t = 30
        t_corrs = []
        
        for t in range(1, max_t + 1):
            x_sig, y_ret = [], []
            for i in regime_indices:
                if i + t < n_days:
                    x_sig.append(sigmas[i])
                    y_ret.append((closes[i+t] / closes[i]) - 1.0)
            
            if len(x_sig) > 30:
                t_corrs.append(np.corrcoef(x_sig, y_ret)[0, 1])
            else:
                t_corrs.append(0)
                
        smooth_corrs = uniform_filter1d(t_corrs, size=5)
        # 음의 상관관계가 가장 강한 T 찾기
        best_t = np.argmin(smooth_corrs) + 1
        best_corr = smooth_corrs[best_t - 1]

        # 📈 장세 맞춤형 선형 함수 생성 (Sigma -> Expected Return)
        final_x_sig, final_y_ret = [], []
        for i in regime_indices:
            if i + best_t < n_days:
                final_x_sig.append(sigmas[i])
                final_y_ret.append((closes[i+best_t] / closes[i]) - 1.0)
                
        # 1차 선형 회귀 (y = ax + b)
        poly_coeffs = np.polyfit(final_x_sig, final_y_ret, 1)
        poly_func = np.poly1d(poly_coeffs)
        
        # 내 시그마를 대입한 T일 후 예상 등락률
        expected_ret = poly_func(my_ent_sig) * 100 

        # ---------------------------------------------------------
        # 🛡️ 장세 맞춤형 출구 최적화 (2D Grid)
        # ---------------------------------------------------------
        # *진입 시그마(Ent)를 내 조건으로 고정하고, 내 장세에서만 백테스트*
        DROP_RANGE = np.round(np.arange(0.1, 5.1, 0.1), 1)
        EXT_RANGE = np.round(np.arange(-1.0, 5.1, 0.1), 1)
        shape = (len(DROP_RANGE), len(EXT_RANGE))
        ret_grid = np.full(shape, -100.0)
        
        c_ent_p = np.round(-my_ent_sig, 1)
        
        for idp, dp in enumerate(DROP_RANGE):
            for iex, ex in enumerate(EXT_RANGE):
                cap, hold, bp, es, trades = 1.0, False, 0.0, 0.0, 0
                for k in range(win20, n_days-1):
                    # 중요: '내 장세'와 동일한 날짜에만 진입 허용
                    if not hold:
                        if sigmas[k] <= -c_ent_p and regimes[k] == my_regime:
                            hold, bp, es, trades = True, opens[k+1], slopes20[k], trades + 1
                    else:
                        if sigmas[k] >= ex or slopes20[k] < (es - dp):
                            hold = False
                            profit = opens[k+1] - bp
                            tax_amt = profit * tax if profit > 0 else 0
                            net = ((opens[k+1] - tax_amt) / bp) - 1.0 - fee_rate
                            cap *= (1.0 + net)
                if trades > 0: ret_grid[idp, iex] = (cap - 1.0) * 100

        # 이웃집 검증
        smooth_ret = uniform_filter(ret_grid, size=3, mode='constant', cval=-100.0)
        best_idx = np.unravel_index(np.argmax(smooth_ret), smooth_ret.shape)
        
        if np.max(smooth_ret) == -100.0:
            return None, f"[{my_regime}] 장세에서 진입 시그마({my_ent_sig:.2f})에 해당하는 유효한 백테스트 결과가 없습니다."
            
        opt_drop, opt_ext = DROP_RANGE[best_idx[0]], EXT_RANGE[best_idx[1]]
        
        # 오늘자 최종 데이터 산출
        y_last = closes[-win20:]
        s_l, i_l, _, _, _ = linregress(x20, y_last)
        L_last = s_l*(win20-1) + i_l
        std_last = np.std(y_last - (s_l*x20 + i_l))
        
        target_price = L_last + (opt_ext * std_last)
        recent_slopes = slopes20[closest_idx:]
        peak_slope = np.max(recent_slopes[recent_slopes != -999.0]) if len(recent_slopes) > 0 else slopes20[-1]
        cut_slope = peak_slope - opt_drop

        res = {
            'regime': my_regime, 'ent_sigma': my_ent_sig,
            'best_t': best_t, 'best_corr': best_corr, 'expected_ret': expected_ret,
            'poly_coeffs': poly_coeffs, # [a, b]
            'opt_ext': opt_ext, 'opt_drop': opt_drop,
            'target_price': target_price, 'cut_slope': cut_slope,
            'cur_price': closes[-1], 'cur_sigma': sigmas[-1], 'cur_slope': slopes20[-1], 'peak_slope': peak_slope,
            'my_profit': ((closes[-1] / ent_price) - 1.0) * 100 if ent_price > 0 else 0.0
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner("📦 10년 치 주가 궤적을 5대 장세로 분리하여 양자 궤적(T)을 연산 중입니다... (1~2분 소요)"):
        res, err = run_regime_oracle(target_ticker, entry_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 연산 완료! (해석된 장세: {res['regime']})")
        
        # --- Part 1: 장세 및 함수 분석 ---
        st.subheader("🚦 1. 진입 시점 장세 진단 및 예상 등락률 (T일 함수)")
        st.markdown(f"> 당신이 진입한 날짜({entry_date.strftime('%Y-%m-%d')})의 시장 60일 연환산 추세는 **{res['regime']}** 상태였습니다.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(label="당시 나의 진입 시그마", value=f"{res['ent_sigma']:.2f}")
        c2.metric(label=f"해당 장세 최적 주기 (T)", value=f"{res['best_t']}일 뒤", delta=f"상관도 {res['best_corr']:.3f}", delta_color="inverse")
        
        # 시그마 대입 함수 결과
        color = "#e74c3c" if res['expected_ret'] > 0 else "#3498db"
        c3.markdown(f"**T일 뒤 예상 통계 등락률:**")
        c3.markdown(f"<h3 style='color:{color};'>{res['expected_ret']:+.2f}%</h3>", unsafe_allow_html=True)
        
        st.caption(f"* 함수식(y=ax+b): y = {res['poly_coeffs'][0]:.4f} * Sigma + {res['poly_coeffs'][1]:.4f} (이 장세에서는 시그마가 낮을수록 T일 후 수익률이 상승합니다.)")
        st.markdown("---")
        
        # --- Part 2: 장세 맞춤형 출구 최적화 ---
        st.subheader("🎯 2. 장세 독립형 맞춤 출구 전략 (AI 최적화)")
        st.markdown(f"> 오직 **[{res['regime']}]** 이었던 과거의 날들 중에서, 나의 조건(Sigma **{res['ent_sigma']:.2f}**)으로 샀을 때 가장 수익이 컸던 3x3x3 이웃 검증 익절/손절 타점입니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🔥 **{res['regime']} 전용 익절 목표가**")
            st.metric(label=f"목표 시그마 ({res['opt_ext']:.1f}) 도달 시", value=f"₩{res['target_price']:,.0f}")
            st.caption("상승장이라면 이 수치가 높을 것이고, 하락장이라면 이 수치가 매우 보수적으로 잡혔을 것입니다.")
            
        with col2:
            st.error(f"🚨 **{res['regime']} 전용 생명선 (Trailing Stop)**")
            st.metric(label=f"기울기 {res['cut_slope']:.2f}% (현재 {res['cur_slope']:.2f}%)", value=f"하락 시 즉시 매도")
            st.caption(f"최고 기울기({res['peak_slope']:.2f}%)에서 {res['opt_drop']:.1f}% 이상 꺾인 지점입니다.")
            
        is_danger = res['cur_slope'] < res['cut_slope']
        
        st.markdown("---")
        st.subheader("🤖 미스터 주의 최종 행동 지침")
        if is_danger:
            st.markdown(f"🚨 **[생명선 이탈]** 안타깝지만 진입 이후 유지되던 추세가 꺾였습니다. (현재 기울기: {res['cur_slope']:.2f}% < 마지노선: {res['cut_slope']:.2f}%). 즉시 전량 매도하여 손실을 끊거나 수익을 지키십시오.")
        elif res['cur_sigma'] >= res['opt_ext']:
            st.markdown(f"💰 **[목표가 도달]** 축하합니다! 이 장세에서 먹을 수 있는 최적 매도 구간에 도달했습니다. 미련 없이 분할 익절하십시오.")
        else:
            rtn_text = f" (현재 추정 수익률: {res['my_profit']:+.2f}%)" if entry_price > 0 else ""
            st.markdown(f"🚀 **[순항 중 / 홀딩]** 아직 이 장세의 목표가에 도달하지 않았고, 추세(현재 기울기 {res['cur_slope']:.2f}%)도 견고합니다. 평온하게 보유하십시오.{rtn_text}")
