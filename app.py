import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter
import time
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 1. 페이지 설정 및 UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="Diamond Exit: 보유자 전용 타점 분석기",
    page_icon="💎",
    layout="wide"
)

st.title("💎 Diamond Exit: \"내가 가진 종목, 어디까지 오를까?\"")
st.markdown("""
이미 보유 중인 주식의 **'최적 매도 타점(익절/손절)'**만 집중적으로 분석합니다.  
과거 10년 치 상승 파동 데이터를 분석하여, **어깨와 머리의 가격**을 역산하고 **추세가 꺾이는 생명선**을 제시합니다.
""")

with st.sidebar:
    st.header("⚙️ 내 계좌 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS", help="한국 주식은 .KS 또는 .KQ")
    avg_price = st.number_input("내 평균 단가 (원/달러)", value=0.0, step=1000.0, help="현재 평단가를 입력하면 수익률을 계산해 줍니다. (선택사항)")
    run_btn = st.button("🚀 최적 매도 타점 분석", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. Exit 전용 2D 최적화 엔진
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_exit_optimization(ticker):
    # 출구 전략(Exit) 전용 그리드
    DROP_RANGE = np.round(np.arange(0.5, 6.1, 0.2), 1)  # 28 steps
    EXT_RANGE = np.round(np.arange(1.0, 6.1, 0.2), 1)   # 26 steps
    
    df = yf.download(ticker, start="2015-01-01", progress=False)
    if df.empty: return None, None, "데이터를 불러오지 못했습니다."
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Open', 'Close']].dropna()
    closes = df['Close'].values
    opens = df['Open'].values
    n_days = len(closes)
    
    # 지표 계산
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

    # 보유자 마인드 백테스트: 추세가 시작될 때(Sigma가 0을 돌파할 때) 샀다고 가정하고, 
    # 어떤 Ext와 Drop에서 팔아야 가장 수익금이 큰지(Avg Profit) 2D로 찾습니다.
    shape = (len(DROP_RANGE), len(EXT_RANGE))
    profit_grid = np.full(shape, -100.0)
    all_res = []
    
    for i_drop, drop in enumerate(DROP_RANGE):
        for i_ext, ext in enumerate(EXT_RANGE):
            hold = False; buy_p = 0.0; ent_slope = 0.0
            trades = 0; total_profit_pct = 0.0
            
            for k in range(win, n_days-1):
                if not hold:
                    # 가상의 추세 진입점 (모멘텀 양전환)
                    if sigmas[k-1] < 0 and sigmas[k] >= 0:
                        hold = True; buy_p = opens[k+1]; ent_slope = slopes[k]
                else:
                    # 목표가 도달 OR 생명선 이탈 시 매도
                    if sigmas[k] >= ext or slopes[k] < (ent_slope - drop):
                        hold = False; sell_p = opens[k+1]
                        ret = (sell_p / buy_p) - 1.0
                        total_profit_pct += ret
                        trades += 1
                        
            avg_profit = (total_profit_pct / trades * 100) if trades > 0 else 0
            profit_grid[i_drop, i_ext] = avg_profit
            all_res.append({'Drop': drop, 'Ext': ext, 'AvgProfit': avg_profit, 'Trades': trades})

    if not all_res: return None, None, "분석 불가 종목입니다."
    
    df_res = pd.DataFrame(all_res)
    # 최소 매매 횟수 필터 후 가장 평균 수익이 높은 조합(최적의 매도 타점) 추출
    valid = df_res[df_res['Trades'] > (n_days/252)] 
    best_exit = valid.sort_values('AvgProfit', ascending=False).iloc[0]
    
    # 역사적 광기 구간 (상위 5% 시그마) 추출
    extreme_sigma = np.percentile(sigmas[sigmas != 999.0], 95)
    
    # 오늘자 지표 계산
    y_last = closes[-win:]
    s_last, inter_last, _, _, _ = linregress(x, y_last)
    L_last = s_last*(win-1) + inter_last
    std_last = np.std(y_last - (s_last*x + inter_last))
    
    cur_price = closes[-1]
    cur_sigma = sigmas[-1]
    cur_slope = slopes[-1]
    
    # 20일 내 최고 기울기 (Trailing Stop 기준점)
    recent_slopes = slopes[-win:]
    peak_slope = np.max(recent_slopes[recent_slopes != -999.0])
    
    # 가격 역산
    opt_ext = best_exit['Ext']
    opt_drop = best_exit['Drop']
    
    target_1_price = L_last + (opt_ext * std_last)
    target_2_price = L_last + (extreme_sigma * std_last)
    
    status_data = {
        'CurPrice': cur_price, 'CurSigma': cur_sigma, 'CurSlope': cur_slope,
        'PeakSlope': peak_slope, 'Target1': target_1_price, 'Target2': target_2_price,
        'OptExt': opt_ext, 'OptDrop': opt_drop, 'ExtSigma': extreme_sigma
    }
    
    return status_data, df_res, None

# ---------------------------------------------------------
# ⚙️ 3. 결과 렌더링
# ---------------------------------------------------------
if run_btn:
    if not target_ticker:
        st.warning("티커를 입력해주세요.")
    else:
        with st.spinner("✨ 차트에 올라탄 당신을 위해 최적의 하차 지점을 계산 중입니다..."):
            status, _, err = run_exit_optimization(target_ticker)
            
        if err:
            st.error(err)
        else:
            # 내 수익률 계산
            my_rtn_str = "평단가 미입력"
            my_profit = 0
            if avg_price > 0:
                my_profit = (status['CurPrice'] / avg_price) - 1.0
                color = "red" if my_profit > 0 else "blue"
                sign = "+" if my_profit > 0 else ""
                my_rtn_str = f"<span style='color:{color}; font-weight:bold;'>{sign}{my_profit*100:.2f}%</span>"

            # 현재 상태 판단
            is_danger = status['CurSlope'] < (status['PeakSlope'] - status['OptDrop'])
            is_overbought = status['CurSigma'] >= status['OptExt']
            
            st.markdown(f"### 📊 현재 내 계좌 상태 (현재가: ₩{status['CurPrice']:,.0f})")
            st.markdown(f"**현재 추정 수익률:** {my_rtn_str}", unsafe_allow_html=True)
            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            
            # 1차 목표가 (통계적 어깨)
            with c1:
                st.info("🎯 **1차 목표가 (통계적 어깨)**")
                st.metric(label=f"목표 시그마: {status['OptExt']:.2f}", value=f"₩{status['Target1']:,.0f}")
                st.caption("과거 통계상 이 지점에서 팔았을 때 누적 수익금이 가장 컸습니다. 분할 매도 시작점입니다.")
                
            # 2차 목표가 (역사적 광기)
            with c2:
                st.success("🔥 **2차 목표가 (역사적 광기)**")
                st.metric(label=f"상위 5% 시그마: {status['ExtSigma']:.2f}", value=f"₩{status['Target2']:,.0f}")
                st.caption("과거 10년간 이 주식이 이성을 잃고 올랐던 최고점 라인입니다. 전량 익절을 고려하세요.")
                
            # 트레일링 스탑 (생명선)
            with c3:
                st.error("🚨 **생명선 (Trailing Stop)**")
                cut_slope = status['PeakSlope'] - status['OptDrop']
                st.metric(label="손절/익절 각도 (Slope)", value=f"{cut_slope:.2f}%")
                st.caption(f"최근 최고 각도({status['PeakSlope']:.2f}%)에서 {status['OptDrop']:.1f}% 꺾이면 추세가 죽은 것입니다. 이 각도가 깨지면 미련 없이 던지세요.")
                
            st.markdown("---")
            
            # AI 행동 지침
            st.subheader("🤖 미스터 주의 행동 지침")
            if is_danger:
                st.markdown(f"> 🚨 **[추세 이탈 경보]** 최근의 상승 추세(기울기)가 통계적 임계점({status['OptDrop']}%) 이상 꺾였습니다. 수익 중이라면 **즉시 익절**, 손실 중이라면 **칼손절**을 권장합니다. 미련을 가지면 지하실을 봅니다.")
            elif is_overbought:
                st.markdown(f"> 💰 **[수익 실현 구간]** 주가가 1차 목표가를 돌파했습니다! 보유 물량의 절반 이상을 매도하여 **수익을 확정** 지으세요. 나머지는 2차 목표가(₩{status['Target2']:,.0f}) 또는 생명선이 깨질 때까지 들고 가보십시오.")
            else:
                st.markdown(f"> 🚀 **[쾌속 질주 중]** 아직 목표가에 도달하지 않았고, 추세(기울기)도 꺾이지 않았습니다. **안심하고 계속 홀딩(Hold) 하십시오.** 주가가 오를수록 생명선(각도)도 따라 올라가며 여러분의 수익을 지켜줄 것입니다.")
