import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d, uniform_filter
import time
import warnings

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 0. 한국거래소(KRX) 호가 단위 맞춤형 함수
# ---------------------------------------------------------
def round_to_tick(price):
    """주가를 KRX 호가 단위에 맞춰 반올림합니다."""
    if price is None or np.isnan(price): return None
    
    if price < 2000:
        tick = 1
    elif price < 5000:
        tick = 5
    elif price < 20000:
        tick = 10
    elif price < 50000:
        tick = 50
    elif price < 200000:
        tick = 100
    elif price < 500000:
        tick = 500
    else:
        tick = 1000
        
    return round(price / tick) * tick

# ---------------------------------------------------------
# ⚙️ 1. 페이지 설정 및 UI
# ---------------------------------------------------------
st.set_page_config(page_title="Quantum Oracle V4 (호가 최적화)", page_icon="🔮", layout="wide")

st.title("🔮 The Quantum Oracle V4: 확률 구간 & 실전 호가 에디션")
st.markdown("""
하나의 가격이 아닌 **[분할 매도 확률 구간(Zone)]**을 제시하며, 
모든 출력 가격은 한국거래소(KRX)의 **실제 호가 단위(Tick Size)**에 맞춰 자동 교정됩니다.
""")

with st.sidebar:
    st.header("⚙️ 내 진입 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS")
    entry_date = st.date_input("진입 날짜 (매수일)")
    entry_price = st.number_input("매수 평단가 (원)", value=0.0, step=1000.0)
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0) / 100.0
    fee = 0.003
    run_btn = st.button("🚀 실전 매도 구간 분석", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 분석 엔진
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_quantum_oracle_v4(ticker, ent_date, ent_price, tax, fee_rate):
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

        win20 = 20
        win60 = 60
        sigmas = np.full(n_days, 999.0)
        slopes20 = np.full(n_days, -999.0)
        ann_slopes60 = np.full(n_days, -999.0) 
        
        x20 = np.arange(win20)
        x60 = np.arange(win60)
        
        for i in range(win60, n_days):
            y20 = closes[i-win20:i]
            s20, i20, _, _, _ = linregress(x20, y20)
            std20 = np.std(y20 - (s20*x20 + i20))
            if std20 > 0: sigmas[i] = (closes[i] - (s20*(win20-1)+i20)) / std20
            if closes[i] > 0: slopes20[i] = (s20 / closes[i]) * 100
            
            y60 = closes[i-win60:i]
            s60, _, _, _, _ = linregress(x60, y60)
            if closes[i] > 0: ann_slopes60[i] = (s60 / closes[i]) * 100 * 252

        # 🚦 장세(Regime) 분류
        regimes = np.full(n_days, 'Unknown', dtype=object)
        regimes[ann_slopes60 >= 40] = 'Strong Bull (🔥강한상승)'
        regimes[(ann_slopes60 >= 10) & (ann_slopes60 < 40)] = 'Bull (📈상승)'
        regimes[(ann_slopes60 > -10) & (ann_slopes60 < 10)] = 'Random (⚖️횡보)'
        regimes[(ann_slopes60 > -40) & (ann_slopes60 <= -10)] = 'Bear (📉하락)'
        regimes[ann_slopes60 <= -40] = 'Strong Bear (🧊강한하락)'

        ent_dt = pd.to_datetime(ent_date)
        closest_idx = np.argmin(np.abs(dates - ent_dt))
        my_ent_sig = sigmas[closest_idx]
        my_regime = regimes[closest_idx]
        
        if my_ent_sig == 999.0 or my_regime == 'Unknown':
            return None, "진입 날짜의 데이터가 부족합니다."

        regime_indices = np.where(regimes == my_regime)[0]
        if len(regime_indices) < 100:
            if my_regime == 'Strong Bull (🔥강한상승)': fallback = ['Strong Bull (🔥강한상승)', 'Bull (📈상승)']
            elif my_regime == 'Strong Bear (🧊강한하락)': fallback = ['Strong Bear (🧊강한하락)', 'Bear (📉하락)']
            else: fallback = [my_regime]
            regime_indices = np.where(np.isin(regimes, fallback))[0]

        # ⏱️ 장세 맞춤형 T일 최적화
        max_t = 30
        t_corrs = []
        for t in range(1, max_t + 1):
            x_sig, y_ret = [], []
            for i in regime_indices:
                if i + t < n_days:
                    x_sig.append(sigmas[i])
                    y_ret.append((closes[i+t] / closes[i]) - 1.0)
            if len(x_sig) > 30: t_corrs.append(np.corrcoef(x_sig, y_ret)[0, 1])
            else: t_corrs.append(0)
                
        smooth_corrs = uniform_filter1d(t_corrs, size=5)
        best_t = np.argmin(smooth_corrs) + 1
        best_corr = smooth_corrs[best_t - 1]

        # 🛡️ 출구 최적화 (2D Grid)
        DROP_RANGE = np.round(np.arange(0.1, 5.1, 0.1), 1)
        EXT_RANGE = np.round(np.arange(-1.0, 5.1, 0.1), 1)
        shape = (len(DROP_RANGE), len(EXT_RANGE))
        ret_grid = np.full(shape, -100.0)
        c_ent_p = np.round(-my_ent_sig, 1)
        
        all_res = []
        for idp, dp in enumerate(DROP_RANGE):
            for iex, ex in enumerate(EXT_RANGE):
                cap, hold, bp, es, trades = 1.0, False, 0.0, 0.0, 0
                for k in range(win20, n_days-1):
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
                if trades > 0: 
                    tot_ret = (cap - 1.0) * 100
                    ret_grid[idp, iex] = tot_ret
                    all_res.append({'Drop': dp, 'Ext': ex, 'TotRet': tot_ret})

        smooth_ret = uniform_filter(ret_grid, size=3, mode='constant', cval=-100.0)
        if np.max(smooth_ret) == -100.0:
            return None, f"[{my_regime}] 장세에서 진입 시그마({my_ent_sig:.2f})의 유효한 결과가 없습니다."
            
        df_res = pd.DataFrame(all_res)
        df_res['Nb_Ret'] = df_res.apply(lambda r: smooth_ret[np.where(DROP_RANGE==r['Drop'])[0][0], np.where(EXT_RANGE==r['Ext'])[0][0]], axis=1)
        
        # 🌟 상위 5개 전략을 모아 '확률적 매도 구간(Zone)' 도출
        top_5_strategies = df_res.sort_values('Nb_Ret', ascending=False).head(5)
        min_opt_ext = top_5_strategies['Ext'].min()
        max_opt_ext = top_5_strategies['Ext'].max()
        
        # 만약 구간이 너무 좁다면(예: 0.1 이하 차이) 인위적으로 최소 0.3 시그마 밴드를 열어줌
        if max_opt_ext - min_opt_ext < 0.2:
            max_opt_ext += 0.3

        best_strategy = top_5_strategies.iloc[0]
        opt_drop = best_strategy['Drop']
        
        y_last = closes[-win20:]
        s_l, i_l, _, _, _ = linregress(x20, y_last)
        L_last = s_l*(win20-1) + i_l
        std_last = np.std(y_last - (s_l*x20 + i_l))
        
        # 가격 역산 및 **호가 단위 반올림 적용**
        target_price_min = round_to_tick(L_last + (min_opt_ext * std_last))
        target_price_max = round_to_tick(L_last + (max_opt_ext * std_last))
        
        recent_slopes = slopes20[closest_idx:]
        peak_slope = np.max(recent_slopes[recent_slopes != -999.0]) if len(recent_slopes) > 0 else slopes20[-1]
        cut_slope = peak_slope - opt_drop

        # 로지스틱 확률 밴드 (호가 단위 적용)
        bands = {}
        cur_prob = 50.0
        if HAS_SKLEARN:
            X_train, Y_train = [], []
            for i in regime_indices:
                if i + best_t < n_days:
                    X_train.append([sigmas[i]])
                    Y_train.append(1 if closes[i+best_t] > closes[i] else 0)
            
            if len(X_train) > 50:
                model = LogisticRegression().fit(X_train, Y_train)
                coef_a, intercept_b = model.coef_[0][0], model.intercept_[0]
                
                def get_p_for_prob(p):
                    if coef_a == 0: return None
                    sig = - (np.log(1/p - 1) + intercept_b) / coef_a
                    raw_price = L_last + (sig * std_last)
                    return round_to_tick(raw_price) # 호가 단위 적용

                bands = {
                    "90% ~ 99%": (get_p_for_prob(0.99), get_p_for_prob(0.90)),
                    "70% ~ 90%": (get_p_for_prob(0.90), get_p_for_prob(0.70)),
                    "50% ~ 70%": (get_p_for_prob(0.70), get_p_for_prob(0.50)),
                    "30% ~ 50%": (get_p_for_prob(0.50), get_p_for_prob(0.30)),
                    "10% ~ 30%": (get_p_for_prob(0.30), get_p_for_prob(0.10)),
                    " 1% ~ 10%": (get_p_for_prob(0.10), get_p_for_prob(0.01))
                }
                cur_prob = model.predict_proba([[sigmas[-1]]])[0][1] * 100

        res = {
            'regime': my_regime, 'ent_sigma': my_ent_sig,
            'best_t': best_t, 'best_corr': best_corr,
            'min_ext': min_opt_ext, 'max_ext': max_opt_ext, 'opt_drop': opt_drop,
            'target_min': target_price_min, 'target_max': target_price_max, 
            'cut_slope': cut_slope, 'cur_price': closes[-1], 
            'cur_sigma': sigmas[-1], 'cur_slope': slopes20[-1], 'peak_slope': peak_slope,
            'my_profit': ((closes[-1] / ent_price) - 1.0) * 100 if ent_price > 0 else 0.0,
            'bands': bands, 'cur_prob': cur_prob
        }
        return res, None

    except Exception as e:
        return None, f"시스템 오류: {str(e)}"

# ---------------------------------------------------------
# ⚙️ 3. 화면 렌더링
# ---------------------------------------------------------
if run_btn:
    with st.spinner("📦 실전 호가 단위 최적화 및 매도 확률 구간을 연산 중입니다..."):
        res, err = run_quantum_oracle_v4(target_ticker, entry_date, entry_price, tax_rate, fee)
        
    if err:
        st.error(err)
    else:
        st.success(f"✅ 연산 완료! (해석된 장세: {res['regime']})")
        
        # --- Part 1: 확률 밴드 렌더링 ---
        if res['bands']:
            st.subheader(f"🗺️ {res['best_t']}일 후 상승 확률 밴드 (호가 적용)")
            cols = st.columns(6)
            colors = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"]
            for i, (name, val) in enumerate(res['bands'].items()):
                with cols[i]:
                    st.markdown(f"<h5 style='color:{colors[i]};'>{name}</h5>", unsafe_allow_html=True)
                    if val[0] and val[1]:
                        low_p, high_p = min(val), max(val)
                        mark = " 👈 현재" if low_p <= res['cur_price'] <= high_p else ""
                        st.write(f"₩{low_p:,}\n~\n₩{high_p:,}{mark}")
                    else: st.caption("도달 불가")
        
        st.markdown("---")
        
        # --- Part 2: 장세 맞춤형 출구 최적화 (Zone) ---
        st.subheader("🎯 장세 맞춤형 실전 매도 구간 (Zone)")
        st.markdown(f"> 나의 진입 조건(Sigma **{res['ent_sigma']:.2f}**)에서 누적 수익금이 가장 컸던 **상위 5개 전략의 밀집 구간**입니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🔥 **통계적 분할 매도 구간 (Target Zone)**")
            st.metric(label=f"Sigma {res['min_ext']:.1f} ~ {res['max_ext']:.1f} 도달 시", 
                      value=f"₩{res['target_min']:,} ~ ₩{res['target_max']:,}")
            st.caption("단일 가격이 아닙니다. 주가가 이 박스권(Zone)에 진입하면 보유 물량을 분할하여 익절하십시오. 가격은 실제 호가 단위에 맞춰 교정되었습니다.")
            
        with col2:
            st.error(f"🚨 **생명선 (Trailing Stop)**")
            st.metric(label=f"기울기 {res['cut_slope']:.2f}% (현재 {res['cur_slope']:.2f}%)", value=f"하락 시 전량 매도")
            st.caption(f"최고 기울기({res['peak_slope']:.2f}%)에서 {res['opt_drop']:.1f}% 이상 꺾인 지점입니다. 목표 구간에 도달하지 않았어도 이 선이 깨지면 미련 없이 매도하십시오.")
            
        is_danger = res['cur_slope'] < res['cut_slope']
        
        st.markdown("---")
        st.subheader("🤖 미스터 주의 최종 행동 지침")
        if is_danger:
            st.markdown(f"🚨 **[생명선 이탈]** 상승 추세가 꺾였습니다. (현재 기울기: {res['cur_slope']:.2f}% < 마지노선: {res['cut_slope']:.2f}%). 더 늦기 전에 즉시 전량 매도하십시오.")
        elif res['cur_sigma'] >= res['min_ext']:
            st.markdown(f"💰 **[매도 구간 진입]** 통계적 분할 매도 구간(Zone)에 들어왔습니다. 욕심을 버리고 정해진 호가에 맞춰 분할 익절을 시작하십시오.")
        else:
            rtn_text = f" (현재 수익률: {res['my_profit']:+.2f}%)" if entry_price > 0 else ""
            st.markdown(f"🚀 **[순항 중 / 홀딩]** 아직 매도 구간에 도달하지 않았습니다. 생명선도 튼튼하게 주가를 받치고 있으니 평온하게 들고 가십시오.{rtn_text}")
