import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter
from pandas.tseries.offsets import BDay
import time
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 1. 페이지 설정 및 UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="The Oracle: 3x3x3 퀀트 예언자",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 The Oracle: 우량주 3x3x3 퀀트 예언자")
st.markdown("""
개별 우량주의 티커를 입력하면 **94,550개의 3변수 하이퍼-그리드**를 탐색하여 최적의 파라미터를 찾고, 
과거의 통계를 바탕으로 **다음 매수/매도 시점을 역산(Forecasting)**합니다.
""")

with st.sidebar:
    st.header("⚙️ 분석 설정")
    target_ticker = st.text_input("종목 코드 (티커)", value="000660.KS", help="한국 주식은 .KS(코스피) 또는 .KQ(코스닥)를 붙여주세요. 미국 주식은 AAPL 등 그대로 입력합니다.")
    tax_rate = st.number_input("세율 적용 (%)", value=0.0, step=1.0, help="국내 주식은 0, 해외 주식 및 배당 ETF는 22 또는 15.4를 입력하세요.") / 100.0
    fee = st.number_input("수수료/슬리피지 (%)", value=0.3, step=0.1) / 100.0
    run_btn = st.button("🚀 전략 최적화 및 진단 실행", type="primary")
    
    st.markdown("---")
    st.caption("※ 0.1 간격의 초정밀 그리드 탐색을 수행하므로 연산에 1~3분 정도 소요될 수 있습니다.")

# ---------------------------------------------------------
# ⚙️ 2. 핵심 최적화 엔진 (캐싱 적용)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_oracle_optimization(ticker, tax, fee_rate):
    # 그리드 설정
    DROP_RANGE = np.round(np.arange(0.1, 5.1, 0.1), 1)  # 50 steps
    ENT_RANGE = np.round(np.arange(1.0, 4.1, 0.1), 1)   # 31 steps
    EXT_RANGE = np.round(np.arange(-1.0, 5.1, 0.1), 1)  # 61 steps
    
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
    years = n_days / 252.0
    min_trades = max(1, int(1.5 * years))
    
    # 지표 선계산
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
        
    shape = (len(DROP_RANGE), len(ENT_RANGE), len(EXT_RANGE))
    ret_grid = np.full(shape, -100.0)
    adr_grid = np.full(shape, -100.0)
    all_res = []
    
    # 1차 시뮬레이션
    for i_drop, drop in enumerate(DROP_RANGE):
        for i_ent, ent in enumerate(ENT_RANGE):
            neg_ent = -ent
            for i_ext, ext in enumerate(EXT_RANGE):
                cap = 1.0
                hold = False
                buy_p = 0.0
                ent_slope = 0.0
                trades = 0
                wins = 0
                hold_days = 0
                
                for k in range(win, n_days-1):
                    if not hold:
                        if sigmas[k] <= neg_ent:
                            hold = True; buy_p = opens[k+1]; ent_slope = slopes[k]; trades += 1
                    else:
                        hold_days += 1
                        if sigmas[k] >= ext or slopes[k] < (ent_slope - drop):
                            hold = False
                            sell_p = opens[k+1]
                            profit = sell_p - buy_p
                            tax_amt = profit * tax if profit > 0 else 0
                            net_ret = ((sell_p - tax_amt) / buy_p) - 1.0 - fee_rate
                            cap *= (1.0 + net_ret)
                            if net_ret > 0: wins += 1
                            
                if trades >= min_trades:
                    tot_ret = (cap - 1.0) * 100
                    adr = (tot_ret / hold_days) if hold_days > 0 else 0
                    ret_grid[i_drop, i_ent, i_ext] = tot_ret
                    adr_grid[i_drop, i_ent, i_ext] = adr
                    
                    all_res.append({
                        'Drop': drop, 'Ent': ent, 'Ext': ext,
                        'TotRet': tot_ret, 'ADR': adr, 'WinRate': (wins/trades)*100,
                        'Trades': trades, 'Idx': (i_drop, i_ent, i_ext)
                    })

    if not all_res: return None, "조건을 만족하는 전략이 없습니다."
        
    # Smoothing
    smooth_ret = uniform_filter(ret_grid, size=3, mode='constant', cval=-100.0)
    smooth_adr = uniform_filter(adr_grid, size=3, mode='constant', cval=-100.0)
    
    df_res = pd.DataFrame(all_res)
    df_res['Nb_Ret'] = df_res['Idx'].apply(lambda x: smooth_ret[x])
    df_res['Nb_ADR'] = df_res['Idx'].apply(lambda x: smooth_adr[x])
    
    # 필터링
    valid = df_res[df_res['WinRate'] >= 65.0]
    if valid.empty: valid = df_res[df_res['WinRate'] >= 60.0]

    # TOP 3 선별
    top_bal = valid.sort_values('Nb_Ret', ascending=False).head(1)
    top_bal['Type'] = '⚖️ [종합 밸런스형]'
    
    long_term = valid[(valid['Ent'] >= 2.5) & (valid['Ext'] >= 2.5)]
    top_lt = long_term.sort_values('Nb_ADR', ascending=False).head(1) if not long_term.empty else valid.sort_values('TotRet', ascending=False).head(1)
    top_lt['Type'] = '📦 [장기 보유형]'
    
    short_term = valid[(valid['Ent'] < 2.5) & (valid['Ext'] < 2.5)]
    top_st = short_term.sort_values('Nb_ADR', ascending=False).head(1) if not short_term.empty else valid.sort_values('WinRate', ascending=False).head(1)
    top_st['Type'] = '⚡ [단기 스윙형]'

    final = pd.concat([top_bal, top_lt, top_st]).drop_duplicates(subset=['Drop', 'Ent', 'Ext'])
    
    # ---------------------------------------------------------
    # 정밀 재시뮬레이션 (Forecasting & History)
    # ---------------------------------------------------------
    y_last = closes[-win:]
    s_last, inter_last, _, _, _ = linregress(x, y_last)
    L_last = s_last*(win-1) + inter_last
    std_last = np.std(y_last - (s_last*x + inter_last))
    
    results_data = []
    
    for _, r in final.iterrows():
        drop, ent, ext = r['Drop'], r['Ent'], r['Ext']
        
        hold = False; buy_p = 0.0; ent_slope = 0.0
        last_buy_date = None; last_sell_date = None; last_sell_idx = None
        last_net_ret = 0.0
        
        trade_rets, hold_days_list, wait_days_list = [], [], []
        
        for k in range(win, n_days-1):
            if not hold:
                if sigmas[k] <= -ent:
                    hold = True; buy_p = opens[k+1]; ent_slope = slopes[k]
                    last_buy_date = dates[k+1]
                    if last_sell_idx is not None: wait_days_list.append(k - last_sell_idx)
            else:
                if sigmas[k] >= ext or slopes[k] < (ent_slope - drop):
                    hold = False; sell_p = opens[k+1]
                    last_sell_date = dates[k+1]; last_sell_idx = k
                    
                    profit = sell_p - buy_p
                    tax_amt = profit * tax if profit > 0 else 0
                    ret = ((sell_p - tax_amt) / buy_p) - 1.0 - fee_rate
                    last_net_ret = ret * 100
                    trade_rets.append(last_net_ret)
                    
                    dur = dates.get_loc(last_sell_date) - dates.get_loc(last_buy_date)
                    hold_days_list.append(dur)

        avg_ret = np.mean(trade_rets) if trade_rets else 0
        std_ret = np.std(trade_rets) if trade_rets else 0
        avg_hold = np.mean(hold_days_list) if hold_days_list else 0
        std_hold = np.std(hold_days_list) if hold_days_list else 0
        avg_wait = np.mean(wait_days_list) if wait_days_list else 0
        std_wait = np.std(wait_days_list) if wait_days_list else 0
        
        target_buy_p = L_last + (-ent * std_last)
        target_sell_p = L_last + (ext * std_last)
        
        results_data.append({
            'Type': r['Type'], 'Drop': drop, 'Ent': ent, 'Ext': ext,
            'TotRet': r['TotRet'], 'WinRate': r['WinRate'], 'Trades': r['Trades'],
            'AvgRet': avg_ret, 'StdRet': std_ret,
            'AvgHold': avg_hold, 'StdHold': std_hold,
            'AvgWait': avg_wait, 'StdWait': std_wait,
            'Hold': hold, 'CurPrice': closes[-1], 'CurSigma': sigmas[-1], 'CurSlope': slopes[-1],
            'TargetBuy': target_buy_p, 'TargetSell': target_sell_p,
            'LastBuyDate': last_buy_date, 'LastSellDate': last_sell_date, 
            'LastNetRet': last_net_ret, 'EntSlope': ent_slope
        })
        
    return results_data, None

# ---------------------------------------------------------
# ⚙️ 3. 결과 렌더링
# ---------------------------------------------------------
if run_btn:
    if not target_ticker:
        st.warning("티커를 입력해주세요.")
    else:
        with st.spinner("✨ 시스템이 9만 개의 우주를 탐색 중입니다. 잠시만 기다려주세요..."):
            start_t = time.time()
            results, err = run_oracle_optimization(target_ticker, tax_rate, fee)
            elapsed = time.time() - start_t
            
        if err:
            st.error(err)
        else:
            st.success(f"✅ 최적화 완료! (탐색 소요 시간: {elapsed:.1f}초)")
            
            for res in results:
                with st.container():
                    st.markdown(f"### {res['Type']} `(Drop: {res['Drop']} / Ent: {res['Ent']} / Ext: {res['Ext']})`")
                    
                    # 통계 요약 컬럼
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("10년 누적 수익률", f"{res['TotRet']:,.0f}%")
                    c2.metric("매매 승률", f"{res['WinRate']:.1f}%", f"총 {int(res['Trades'])}회 매매")
                    c3.metric("평균 매매 수익", f"{res['AvgRet']:.2f}%", f"편차 ±{res['StdRet']:.2f}%", delta_color="off")
                    c4.metric("평균 보유/대기", f"{res['AvgHold']:.1f}일", f"대기 {res['AvgWait']:.1f}일", delta_color="off")
                    
                    # 상태 분석 창
                    if res['Hold']:
                        st.info("🟢 **[현재 상태] : 보유 중 (Holding)**")
                        
                        min_hold = max(0, int(res['AvgHold'] - res['StdHold']))
                        max_hold = int(res['AvgHold'] + res['StdHold'])
                        est_sell_start = res['LastBuyDate'] + BDay(min_hold)
                        est_sell_end = res['LastBuyDate'] + BDay(max_hold)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"- **최근 매수일:** {res['LastBuyDate'].strftime('%Y-%m-%d')} (진입 기울기: {res['EntSlope']:.2f}%)")
                            st.write(f"- **현재 가격:** ₩{res['CurPrice']:,.0f} (시그마: {res['CurSigma']:.2f} / 기울기: {res['CurSlope']:.2f}%)")
                        with col2:
                            st.write(f"🎯 **목표 익절가:** 약 ₩{res['TargetSell']:,.0f} (Sigma {res['Ext']} 도달 시)")
                            st.write(f"🚨 **손절 기준선:** 기울기 {res['EntSlope'] - res['Drop']:.2f}% 이탈 시 시가 매도")
                        
                        st.warning(f"⏳ **예상 매도권:** {est_sell_start.strftime('%Y-%m-%d')} ~ {est_sell_end.strftime('%Y-%m-%d')} 내외")
                        
                    else:
                        st.error("🔵 **[현재 상태] : 대기 중 (Waiting / 현금 보유)**")
                        
                        if res['LastSellDate'] is not None:
                            # FOMO 방지 멘탈 케어 로직 추가
                            st.markdown(f"""
                            > 🎯 **직전 매매 성과:** {res['LastBuyDate'].strftime('%Y-%m-%d')} 매수 $\\rightarrow$ {res['LastSellDate'].strftime('%Y-%m-%d')} 매도  
                            > 💰 **최종 확정 수익률: <span style='color:#e74c3c; font-size:1.1em; font-weight:bold;'>+{res['LastNetRet']:.2f}%</span>**
                            
                            *💡 **Mental Care:** 이미 이 전략으로 성공적인 수익을 거두었습니다. 최근 매도 이후 주가가 올랐더라도 아쉬워하지 마세요. 그것은 내 몫이 아닙니다. 다음 '과매도' 구간까지 현금을 쥐고 기다리는 자만이 복리를 누릴 수 있습니다.*
                            """, unsafe_allow_html=True)
                            
                            min_wait = max(0, int(res['AvgWait'] - res['StdWait']))
                            max_wait = int(res['AvgWait'] + res['StdWait'])
                            est_buy_start = res['LastSellDate'] + BDay(min_wait)
                            est_buy_end = res['LastSellDate'] + BDay(max_wait)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"- **현재 가격:** ₩{res['CurPrice']:,.0f} (시그마: {res['CurSigma']:.2f})")
                            with col2:
                                st.write(f"🎯 **다음 목표 매수가:** 약 ₩{res['TargetBuy']:,.0f} (Sigma -{res['Ent']} 터치)")
                                
                            st.warning(f"⏳ **예상 매수권:** {est_buy_start.strftime('%Y-%m-%d')} ~ {est_buy_end.strftime('%Y-%m-%d')} 내외")
                        else:
                            st.write(f"- **현재 가격:** ₩{res['CurPrice']:,.0f} (시그마: {res['CurSigma']:.2f})")
                            st.write(f"🎯 **목표 매수가:** 약 ₩{res['TargetBuy']:,.0f} (Sigma -{res['Ent']} 터치)")
                            
                    st.markdown("---")
