# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import os
import time

app = Flask(__name__)
CORS(app)

# ==========================================
# 📊 量化特徵工程模組 (保留你所有的核心邏輯)
# ==========================================
def calculate_technical_indicators(df):
    if df.empty or len(df) < 30: return df
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['MACD_Signal']
    
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['K'], df['D'] = 50.0, 50.0
    for i in range(1, len(df)):
        if pd.isna(df['RSV'].iloc[i]): continue
        df.loc[df.index[i], 'K'] = df['K'].iloc[i-1] * (2/3) + df['RSV'].iloc[i] * (1/3)
        df.loc[df.index[i], 'D'] = df['D'].iloc[i-1] * (2/3) + df['K'].iloc[i] * (1/3)
    return df

def get_latest_news(ticker_symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker_symbol}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        return [item.get_text() for item in soup.find_all('h3', limit=3)]
    except:
        return ["無法獲取即時新聞"]

# --- 基礎資料清洗器 ---
def is_valid(val) -> bool:
    if val is None: return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (TypeError, ValueError): return False

def is_positive(val) -> bool: return is_valid(val) and float(val) > 0
def is_non_negative(val) -> bool: return is_valid(val) and float(val) >= 0
def safe_float(val, fallback=None): return float(val) if is_valid(val) else fallback
def fmt_pct(val, decimals=2) -> str: return f"{float(val) * 100:.{decimals}f}%" if is_valid(val) else "N/A"
def fmt_ratio(val, decimals=2) -> str: return f"{float(val):.{decimals}f}x" if is_valid(val) else "N/A"
def fmt_num(val, decimals=2) -> str: return f"{float(val):.{decimals}f}" if is_valid(val) else "N/A"
def clean(val): return "N/A" if val is None or (isinstance(val, float) and math.isnan(val)) else round(val, 2) if isinstance(val, float) else val
def clean_number(value, default=0.0, min_val=None, max_val=None):
    if value is None: return default
    try:
        num = float(value)
        if math.isnan(num) or math.isinf(num): return default
        if min_val is not None and num < min_val: return default
        if max_val is not None and num > max_val: return default
        return num
    except (ValueError, TypeError): return default

# --- 高階指標計算 ---
def calc_div_yield(info: dict) -> str:
    try:
        raw = info.get('dividendYield') or info.get('trailingAnnualDividendYield')
        if not is_valid(raw): return "N/A"
        val = float(raw)
        if val < 0 or val > 0.30: return "N/A"
        if val == 0.0: return "0.00%"
        return f"{val * 100:.2f}%"
    except: return "N/A"

def calc_ev_fcf(info: dict) -> str:
    try:
        ev = safe_float(info.get('enterpriseValue'))
        fcf = safe_float(info.get('freeCashflow'))
        if not is_positive(ev) or not is_positive(fcf): return "N/A"
        ratio = ev / fcf
        if ratio <= 0 or ratio > 500: return "N/A"
        return fmt_ratio(ratio)
    except: return "N/A"

def calc_fcf_yield(info: dict) -> str:
    try:
        fcf = safe_float(info.get('freeCashflow'))
        market_cap = safe_float(info.get('marketCap'))
        if not is_valid(fcf) or not is_positive(market_cap): return "N/A"
        yield_val = fcf / market_cap
        if abs(yield_val) > 0.5: return "N/A"
        return fmt_pct(yield_val)
    except: return "N/A"

def calc_roic(info: dict) -> str:
    try:
        revenue = safe_float(info.get('totalRevenue'))
        margin = safe_float(info.get('operatingMargins'))
        if not is_positive(revenue) or not is_valid(margin): return "N/A"
        raw_tax = safe_float(info.get('effectiveTaxRate'))
        tax_rate = raw_tax if is_valid(raw_tax) and 0.0 <= raw_tax <= 0.60 else 0.21
        nopat = revenue * margin * (1 - tax_rate)
        total_assets = safe_float(info.get('totalAssets'))
        current_liabilities = safe_float(info.get('currentLiabilities'))
        if not is_positive(total_assets) or not is_non_negative(current_liabilities): return "N/A"
        invested_capital = total_assets - current_liabilities
        if invested_capital <= 0: return "N/A"
        roic = nopat / invested_capital
        if roic < -0.5 or roic > 2.0: return "N/A"
        return fmt_pct(roic)
    except: return "N/A"

# ⚡ 带重试的 Yahoo 数据获取
def fetch_stock_data(symbol, retries=3, delay=1):
    for i in range(retries):
        try:
            tk = yf.Ticker(symbol)
            info = tk.info
            hist = tk.history(period="6mo")
            return tk, info, hist
        except Exception as e:
            if "Rate limited" in str(e) and i < retries - 1:
                time.sleep(delay * (2 ** i))
                continue
            raise

# ==========================================
# ⚡ 股票深度雷達掃描路由器 (純數據，無 AI)
# ==========================================
@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    try:
        req = request.json
        symbol = req.get('symbol', '').upper()

        if symbol.isdigit() and len(symbol) == 4:
            symbol += ".TW"

        try:
            tk, info, hist = fetch_stock_data(symbol)
        except Exception as yf_err:
            return jsonify({"error": f"Yahoo Finance 暫時無法存取: {str(yf_err)}"}), 503
        
        if hist.empty: 
            return jsonify({"error": f"找不到股票數據: {symbol}"}), 404
        
        hist = calculate_technical_indicators(hist)
        latest = hist.iloc[-1]

        # 髒數區優化與生肉加工
        yoy_raw = info.get('earningsGrowth')
        margin_raw = info.get('profitMargins')
        pb_raw = info.get('priceToBook')
        debteq_raw = info.get('debtToEquity')

        yoy_clean = clean_number(yoy_raw, default=None)
        yoy_str = f"{yoy_clean * 100:.2f}%" if yoy_clean is not None else "N/A"

        margin_clean = clean_number(margin_raw, default=None)
        margin_str = f"{margin_clean * 100:.2f}%" if margin_clean is not None else "N/A"

        pb_val = clean_number(pb_raw, default=None, min_val=0, max_val=30)
        pb_str = f"{pb_val:.2f}" if pb_val is not None else "N/A"

        debteq_val = clean_number(debteq_raw, default=None, min_val=0, max_val=1000)
        debteq_str = f"{debteq_val:.2f}%" if debteq_val is not None else "N/A"

        # 完美打包所有數據
        stock_pack = {
            "symbol": symbol,
            "price": clean(info.get('currentPrice', info.get('regularMarketPrice', latest['Close']))),
            "change_52w": clean(info.get('52WeekChange')),
            "beta": clean(info.get('beta')),
            "pe": clean(info.get('trailingPE')),
            "peg": clean(info.get('pegRatio')),
            "div": calc_div_yield(info),
            "roe": clean(info.get('returnOnEquity')),
            "de": clean(info.get('debtToEquity')),
            "rsi": clean(latest.get('RSI', None)),
            "k": clean(latest.get('K', None)),
            "d": clean(latest.get('D', None)),
            "macd_hist": clean(latest.get('MACD_Hist', None)),
            "ev_fcf": calc_ev_fcf(info),
            "roic": calc_roic(info),
            "news": get_latest_news(symbol),
            "yoy": yoy_str,
            "pb": pb_str,
            "net_margin": margin_str,
            "debt_eq": debteq_str
        }

        # 直接回傳給前端，不再呼叫 AI
        return jsonify({"stock_data": stock_pack})

    except Exception as e:
        return jsonify({"error": f"後端系統錯誤: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
