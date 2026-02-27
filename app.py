import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d, uniform_filter
from sklearn.linear_model import LogisticRegression
import time
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 1. 페이지 설정 및 UI
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle: 3x3x3 출구 전략기", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle: 양자 궤적 & 출구 전략기")
st.markdown("""
종목의 **고무줄 복원 주기(T일)**를 찾아내어 내일의 상승 확률 지도를 그립니다.  
또한, 나의 **'진입 시그마'**와 동일한 과거 환경을 2D(Drop x Ext)로 촘촘하게 역추적하여 **나만의 최적 익절/손절 타점**을 도출합니다.
""")

with st.sidebar:
    st.header("⚙️ 내 진입 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="005930.KS", help="한국 주식은 .KS 또는 .KQ")
    entry_date = st.date_input("진입 날짜 (매수일)")
    entry_price = st.number_input("매수 평단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003 # 고정 수수료 0.3%
    run_btn = st.button("🚀 양자 궤적 분석 실행", type="primary")
    
    st.markdown("---")
    st.caption("※ 3x3x3 초정밀 그리드 연산(약 9만 개 조합)과 T일 시계열 탐색을 수행하므로 약 1~2분이 소요됩니다.")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진 (캐싱 적용)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_quantum_oracle(ticker, ent_date, ent_price, tax, fee_rate):
    # 데이터 로드
    df = yf.download(ticker, start="2015-01-01", progress=False)
    if df.empty: return None, "데이터를 불러오지 못했습니다. 티커를 확인해주세요."
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Open', 'Close']].dropna()
    closes = df['Close'].values
    opens = df['Open'].values
    dates = df.index
    n_days = len(closes)
    
    win = 20
    sigmas = np.full(n_days, 999.0)
    slopes = np.full(n_days, -999.0)
    x = np.arange(win)
    
    for i in range(win, n_days):
        y = closes[i-win:i]
        s, inter, _, _, _ = linregress(x, y)
        std = np.std(y - (s*x + inter))
        if std > 0: sigmas[i] = (closes[i] - (s*(win-1)+inter)) / std
        if closes[i] > 0: slopes[i] = (s / closes[i]) * 100

    valid_idx = np.where(sigmas != 999.0)[0]

    # --- Part 1: 최적의 복원 주기(T) 탐색 ---
    max_t = 30
    t_correlations = []
    
    for t in range(1, max_t + 1):
        x_sig, y_ret = [], []
        for i in valid_idx:
            if i + t < n_days:
                x_sig.append(sigmas[i])
                y_ret.append((closes[i+t] / closes[i]) - 1.0)
        if len(x_sig) > 0:
            corr = np.corrcoef(x_sig, y_ret)[0, 1]
            t_correlations.append(corr)
        else:
            t_correlations.append(0)
            
    smooth_corrs = uniform_filter1d(t_correlations, size=5)
    best_t = np.argmin(smooth_corrs) + 1 
    best_corr = smooth_corrs[best_t - 1]

    # --- Part 2: 로지스틱 회귀 확률 밴드 생성 ---
    X_train, Y_train = [], []
    for i in valid_idx:
        if i + best_t < n_days:
            X_train.append([sigmas[i]])
            Y_train.append(1 if closes[i+best_t] > closes[i] else 0)
            
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    coef_a = model.coef_[0][0]
    intercept_b = model.intercept_[0]
    
    def get_sigma_for_prob(p):
        if p >= 1.0 or p <= 0.0: return None
        return - (np.log(1/p - 1) + intercept_b) / coef_a

    y_last = closes[-win:]
    s_last, inter_last, _, _, _ = linregress(x, y_last)
    L_last = s_last*(win-1) + inter_last
    std_last = np.std(y_last - (s_last*x + inter_last))
    
    def get_price_for_prob(p):
        sig = get_sigma_for_prob(p)
        if sig is None: return None
        return L_last + (sig * std_last)

    bands = {
        "90% ~ 99%": (get_price_for_prob(0.99), get_price_for_prob(0.90)),
        "70% ~ 90%": (get_price_for_prob(0.90), get_price_for_prob(0.70)),
        "50% ~ 70%": (get_price_for_prob(0.70), get_price_for_prob(0.50)),
        "30% ~ 50%": (get_price_for_prob(0.50), get_price_for_prob(0.30)),
        "10% ~ 30%": (get_price_for_prob(0.30), get_price_for_prob(0.10)),
        " 1% ~ 10%": (get_price_for_prob(0.10), get_price_for_prob(0.01))
    }
    
    cur_sigma = sigmas[-1]
    cur_price = closes[-1]
    cur_prob = model.predict_proba([[cur_sigma]])[0][1] * 100

    # --- Part 3: 진입 시점 맞춤형 출구 최적화 ---
    ent_date_pd = pd.to_datetime(ent_date)
    if ent_date_pd not in dates:
        closest_date_idx = np.argmin(np.abs(dates - ent_date_pd))
    else:
        closest_date_idx = dates.get_loc(ent_date_pd)
        
    my_ent_sigma = sigmas[closest_date_idx]
    if my_ent_sigma == 999.0:
        return None, "진입 날짜의 데이터가 부족합니다. (상장 초기)"
        
    custom_ent_param = np.round(-my_ent_sigma, 1)
    
    DROP_RANGE = np.round(np.arange(0.1, 5.1, 0.1), 1)
    EXT_RANGE = np.round(np.arange(-1.0, 5.1, 0.1), 1)
    shape = (len(DROP_RANGE), len(EXT_RANGE))
    ret_grid = np.full(shape, -100.0)
    adr_grid = np.full(shape, -100.0)
    all_res = []
    
    for i_drop, drop in enumerate(DROP_RANGE):
        for i_ext, ext in enumerate(EXT_RANGE):
            cap = 1.0; hold = False; buy_p = 0.0; ent_slope = 0.0
            trades = 0; hold_days = 0
            
            for k in range(win, n_days-1):
                if not hold:
                    if sigmas[k] <= -custom_ent_param:
                        hold = True; buy_p = opens[k+1]; ent_slope = slopes[k]
                else:
                    hold_days += 1
                    if sigmas[k] >= ext or slopes[k] < (ent_slope - drop):
                        hold = False; sell_p = opens[k+1]
                        tax_amt = (sell_p - buy_p) * tax if (sell_p - buy_p) > 0 else 0
                        net_ret = ((sell_p - tax_amt) / buy_p) - 1.0 - fee_rate
                        cap *= (1.0 + net_ret)
                        trades += 1
                        
            if trades > 0:
                tot_ret = (cap - 1.0) * 100
                adr = (tot_ret / hold_days) if hold_days > 0 else 0
                ret_grid[i_drop, i_ext] = tot_ret
                adr_grid[i_drop, i_ext] = adr
                all_res.append({'Drop': drop, 'Ext': ext, 'TotRet': tot_ret, 'ADR': adr})

    if not all_res:
        return None, "해당 진입 조건으로 백테스트 가능한 결과가 없습니다."

    smooth_ret = uniform_filter(ret_grid, size=3, mode='constant', cval=-100.0)
    df_res = pd.DataFrame(all_res)
    df_res['Nb_Ret'] = df_res.apply(lambda r: smooth_ret[np.where(DROP_RANGE==r['Drop'])[0][0], np.where(EXT_RANGE==r['Ext'])[0][0]], axis=1)
    
    best_strategy = df_res.sort_values('Nb_Ret', ascending=False).iloc[0]
    opt_ext = best_strategy['Ext']
    opt_drop = best_strategy['Drop']
    
    target_price = L_last + (opt_ext * std_last)
    recent_slopes = slopes[closest_date_idx:]
    peak_slope = np.max(recent_slopes[recent_slopes != -999.0]) if len(recent_slopes) > 0 else slopes[-1]
    cut_slope = peak_slope - opt_drop
    
    my_profit = ((cur_price / ent_price) - 1.0) * 100 if ent_price > 0 else 0.0

    result_data = {
        'best_t': best_t, 'best_corr': best_corr,
        'bands': bands, 'cur_price': cur_price, 'cur_sigma': cur_sigma, 'cur_prob': cur_prob,
        'ent_sigma': my_ent_sigma, 'my_profit': my_profit,
        'opt_ext': opt_ext, 'opt_drop': opt_drop,
        'target_price': target_price, 'cut_slope': cut_slope,
        'cur_slope': slopes[-1], 'peak_slope': peak_slope
    }
    return result_data, None

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    if not target_ticker:
        st.warning("티커를 입력해주세요.")
    else:
        with st.spinner("✨ 10년 치 양자 궤적 분석 및 시그마 평면을 탐색 중입니다. (1~2분 소요)"):
            res, err = run_quantum_oracle(target_ticker, entry_date, entry_price, tax_rate, fee)
            
        if err:
            st.error(err)
        else:
            # --- Part 1: 확률 밴드 렌더링 ---
            st.subheader(f"⏱️ 1. 고무줄 복원 주기 탐색")
            st.info(f"데이터 분석 결과, 이 종목은 고무줄(시그마)이 당겨졌을 때 **[{res['best_t']}일 뒤]**에 반대 방향으로 튕겨 나가는 경향(상관계수: {res['best_corr']:.3f})이 가장 강합니다.")
            
            st.subheader(f"🗺️ 2. {res['best_t']}일 후 상승 확률 밴드 (Probability Map)")
            st.markdown(f"**현재 주가:** ₩{res['cur_price']:,.0f} (현재 시그마: {res['cur_sigma']:.2f}) $\\rightarrow$ **{res['best_t']}일 뒤 상승 확률: <span style='color:#e74c3c; font-size:1.2em;'>{res['cur_prob']:.1f}%</span>**", unsafe_allow_html=True)
            
            # 밴드 출력
            cols = st.columns(6)
            band_keys = list(res['bands'].keys())
            colors = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"]
            
            for i, col in enumerate(cols):
                band_name = band_keys[i]
                p_high, p_low = res['bands'][band_name]
                with col:
                    st.markdown(f"<h5 style='color:{colors[i]};'>{band_name}</h5>", unsafe_allow_html=True)
                    if p_high is None or p_low is None:
                        st.caption("도달 불가 영역")
                    else:
                        low_p = min(p_high, p_low)
                        high_p = max(p_high, p_low)
                        mark = " 👈 현재 주가" if low_p <= res['cur_price'] <= high_p else ""
                        
                        # 음수 가격도 통계적 지표로서 그대로 노출 (사용자 요청 반영)
                        st.write(f"₩{low_p:,.0f}\n~\n₩{high_p:,.0f}{mark}")
            
            st.markdown("---")
            
            # --- Part 2: 맞춤형 출구 전략 렌더링 ---
            st.subheader("🎯 3. 진입 시점 맞춤형 출구 전략 (AI 최적화)")
            st.markdown(f"> **나의 진입 환경:** {entry_date.strftime('%Y-%m-%d')} (당시 시그마: **{res['ent_sigma']:.2f}**) / 현재 수익률: **{res['my_profit']:+.2f}%**")
            st.markdown(f"> **AI 분석 결과:** 나와 완벽히 동일한 시그마 조건에서 진입했을 때, 과거 10년 동안 누적 수익금이 가장 컸던 매도 공식은 **[익절 시그마 {res['opt_ext']:.1f} / 손절 기울기 하락 {res['opt_drop']:.1f}%]** 입니다.")
            
            c1, c2 = st.columns(2)
            with c1:
                st.success("💰 **EOD 목표 익절가**")
                st.metric(label=f"목표 시그마 ({res['opt_ext']:.1f}) 도달 시", value=f"₩{res['target_price']:,.0f}")
                st.caption("AI가 찾아낸 수학적 최적의 이익 실현 구간입니다.")
                
            with c2:
                st.error("🚨 **생명선 마지노선 (Trailing Stop)**")
                st.metric(label=f"기울기 {res['cut_slope']:.2f}% (현재 {res['cur_slope']:.2f}%)", value=f"하락 시 즉시 매도")
                st.caption(f"진입 후 달성한 최고 각도({res['peak_slope']:.2f}%)에서 {res['opt_drop']:.1f}% 이상 꺾인 선입니다.")
                
            st.markdown("---")
            st.subheader("🤖 미스터 주의 최종 행동 지침")
            if res['cur_slope'] < res['cut_slope']:
                st.markdown(f"🚨 **[생명선 이탈 비상]** 최근의 상승 추세가 통계적 임계점 미만으로 꺾였습니다. (현재 기울기: {res['cur_slope']:.2f}% < 마지노선: {res['cut_slope']:.2f}%). 즉시 매도하여 자산을 보호하십시오.")
            elif res['cur_sigma'] >= res['opt_ext']:
                st.markdown(f"🎉 **[목표가 도달]** 축하합니다! 통계적 최적 매도 구간을 돌파했습니다. 탐욕을 버리고 분할 매도로 수익을 확정 지으십시오.")
            else:
                st.markdown(f"🚀 **[순항 중 / 홀딩]** 아직 목표가에 도달하지 않았고, 생명선도 튼튼합니다. 평온한 마음으로 보유하십시오.")
