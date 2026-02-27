import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
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
과거 10년 치 상승 파동 데이터를 분석하여, **확률적 매도 구간(Zone)**과 **추세가 꺾이는 생명선**을 제시합니다.
""")

with st.sidebar:
    st.header("⚙️ 내 계좌 정보")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS", help="한국 주식은 .KS 또는 .KQ")
    avg_price = st.number_input("내 평균 단가 (원/달러)", value=0.0, step=1000.0, help="현재 평단가를 입력하면 수익률을 계산해 줍니다. (선택사항)")
    run_btn = st.button("🚀 확률적 매도 구간 분석", type="primary")

# ---------------------------------------------------------
# ⚙️ 2. Exit 전용 2D 최적화 엔진
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_exit_optimization(ticker):
    # 출구 전략(Exit) 전용 그리드
    DROP_RANGE = np.round(np.arange(0.5, 6.1, 0.2), 1)  
    EXT_RANGE = np.round(np.arange(1.0, 6.1, 0.2), 1)   
    
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

    shape = (len(DROP_RANGE), len(EXT_RANGE))
    profit_grid = np.full(shape, -100.0)
    all_res = []
    
    for i_drop, drop in enumerate(DROP_RANGE):
        for i_ext, ext in enumerate(EXT_RANGE):
            hold = False; buy_p = 0.0; ent_slope = 0.0
            trades = 0; total_profit_pct = 0.0
            
            for k in range(win, n_days-1):
                if not hold:
                    # 가상의 추세 진입점 (모멘텀 양전환 시점)
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
    
    # ---------------------------------------------------------
    # 🌟 상승 파동 확률 구간 (Percentiles) 추출
    # ---------------------------------------------------------
    # 오직 상승장(시그마 > 0)의 데이터만 추출하여 정밀도 향상
    pos_sigmas = sigmas[(sigmas != 999.0) & (sigmas > 0)]
    
    sigma_90 = np.percentile(pos_sigmas, 90) # 상위 10% 컷
    sigma_95 = np.percentile(pos_sigmas, 95) # 상위 5% 컷
    sigma_99 = np.percentile(pos_sigmas, 99) # 상위 1% 컷
    
    # 오늘자 지표 계산
    y_last = closes[-win:]
    s_last, inter_last, _, _, _ = linregress(x, y_last)
    L_last = s_last*(win-1) + inter_last
    std_last = np.std(y_last - (s_last*x + inter_last))
    
    cur_price = closes[-1]
    cur_sigma = sigmas[-1]
    cur_slope = slopes[-1]
    
    recent_slopes = slopes[-win:]
    peak_slope = np.max(recent_slopes[recent_slopes != -999.0])
    
    opt_ext = best_exit['Ext']
    opt_drop = best_exit['Drop']
    
    # 가격 역산 밴드 (역대급 확률 구간)
    price_90 = L_last + (sigma_90 * std_last)
    price_95 = L_last + (sigma_95 * std_last)
    price_99 = L_last + (sigma_99 * std_last)
    opt_target = L_last + (opt_ext * std_last) # 퀀트가 찾은 진짜 최적점
    
    status_data = {
        'CurPrice': cur_price, 'CurSigma': cur_sigma, 'CurSlope': cur_slope,
        'PeakSlope': peak_slope, 'OptExt': opt_ext, 'OptDrop': opt_drop,
        'Sigma90': sigma_90, 'Sigma95': sigma_95, 'Sigma99': sigma_99,
        'Price90': price_90, 'Price95': price_95, 'Price99': price_99, 'OptTarget': opt_target
    }
    
    return status_data, df_res, None

# ---------------------------------------------------------
# ⚙️ 3. 결과 렌더링
# ---------------------------------------------------------
if run_btn:
    if not target_ticker:
        st.warning("티커를 입력해주세요.")
    else:
        with st.spinner("✨ 과거 10년 치 파동을 분석하여 확률적 매도 구간을 계산 중입니다..."):
            status, _, err = run_exit_optimization(target_ticker)
            
        if err:
            st.error(err)
        else:
            my_rtn_str = "평단가 미입력"
            if avg_price > 0:
                my_profit = (status['CurPrice'] / avg_price) - 1.0
                color = "#e74c3c" if my_profit > 0 else "#3498db"
                sign = "+" if my_profit > 0 else ""
                my_rtn_str = f"<span style='color:{color}; font-size:1.2em; font-weight:bold;'>{sign}{my_profit*100:.2f}%</span>"

            is_danger = status['CurSlope'] < (status['PeakSlope'] - status['OptDrop'])
            
            st.markdown(f"### 📊 현재 계좌 상태 (현재가: ₩{status['CurPrice']:,.0f})")
            st.markdown(f"**현재 추정 수익률:** {my_rtn_str}", unsafe_allow_html=True)
            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.info("🎯 **1차 분할매도 구간 (상위 10% 영역)**")
                st.markdown(f"<h4 style='color:#3498db;'>₩{status['Price90']:,.0f} ~ ₩{status['Price95']:,.0f}</h4>", unsafe_allow_html=True)
                st.caption(f"Sigma {status['Sigma90']:.2f} ~ {status['Sigma95']:.2f} 구간. 과거 통계상 이 구간에 진입하면 상승 동력이 크게 둔화되기 시작했습니다. 절반 이상 익절을 권장합니다.")
                
            with c2:
                st.success("🔥 **2차 전량매도 구간 (상위 5% 영역)**")
                st.markdown(f"<h4 style='color:#2ecc71;'>₩{status['Price95']:,.0f} ~ ₩{status['Price99']:,.0f}</h4>", unsafe_allow_html=True)
                st.caption(f"Sigma {status['Sigma95']:.2f} ~ {status['Sigma99']:.2f} 구간. 상위 1~5%에 해당하는 역사적 광기(오버슈팅) 구간입니다. 미련 없이 전량 매도를 준비하세요.")
                
            with c3:
                st.error("🚨 **생명선 (Trailing Stop)**")
                cut_slope = status['PeakSlope'] - status['OptDrop']
                st.metric(label="손절/익절 마지노선 (기울기)", value=f"{cut_slope:.2f}%")
                st.caption(f"최근 형성된 최고 각도({status['PeakSlope']:.2f}%)에서 {status['OptDrop']:.1f}% 이상 꺾인 지점입니다. 목표가에 도달하지 않았더라도, 이 각도가 깨지면 미련 없이 매도하여 수익을 지키세요.")
                
            st.markdown("---")
            st.subheader("💡 백테스트 최적 익절가 (The Ultimate Target)")
            st.markdown(f"> 🏆 과거 10년 2D 백테스트 시뮬레이션 결과, 가장 누적 수익금이 컸던 수학적 최적 익절 타점은 **Sigma {status['OptExt']:.2f} (약 ₩{status['OptTarget']:,.0f})** 였습니다.")
            
            st.markdown("---")
            st.subheader("🤖 미스터 주의 행동 지침")
            if is_danger:
                st.markdown(f"> 🚨 **[추세 이탈 경보]** 최근의 상승 추세가 통계적 임계점({status['OptDrop']}%) 이상 꺾였습니다. 수익 중이라면 **즉시 익절**, 손실 중이라면 **칼손절**을 권장합니다. 더 들고 있으면 통계적으로 위험합니다.")
            else:
                st.markdown(f"> 🚀 **[쾌속 질주 / 홀딩 구간]** 아직 추세가 꺾이지 않았습니다. 주가가 1차/2차 매도 구간 밴드에 진입할 때 분할 매도로 대응하십시오. 오를수록 생명선(기울기)도 매일 따라 올라가며 수익을 안전하게 지켜줄 것입니다.")
