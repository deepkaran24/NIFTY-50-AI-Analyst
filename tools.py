import os
import yfinance as yf
from yahooquery import Ticker
from textblob import TextBlob
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from difflib import get_close_matches
from newsapi import NewsApiClient
from gnews import GNews
import requests

# Import helpers from utils
from utils import clean_symbol, format_currency

@tool
def get_stock_quote(symbol: str) -> dict:
    """
    Get live stock price. 
    Tries YahooQuery first (faster), then yfinance (reliable backup).
    """
    clean_sym = clean_symbol(symbol)
    yahoo_sym = f"{clean_sym}.NS"
    
    # Source 1: YahooQuery
    try:
        stock = Ticker(yahoo_sym)
        price_info = stock.price.get(yahoo_sym, {})
        price = (price_info.get("regularMarketPrice") or 
                 price_info.get("currentPrice") or 
                 price_info.get("previousClose"))
        if price: return {"price": round(price, 2), "source": "YahooQuery", "success": True}
    except: pass

    # Source 2: yfinance
    try:
        stock = yf.Ticker(yahoo_sym)
        data = stock.history(period="5d")
        if not data.empty:
            return {"price": round(data['Close'].iloc[-1], 2), "source": "yfinance", "success": True}
    except: pass

    return {"error": "Failed to fetch price.", "success": False}

@tool
def get_historical_average(symbol: str) -> dict:
    """
    Get 30-day historical average using the 'yahooquery' library.
    """
    clean_sym = symbol.upper().replace(".NS", "").replace(".BO", "").strip()
    # default to NSE
    yahoo_sym = f"{clean_sym}.NS"
    
    try:
        t = Ticker(yahoo_sym)
     
        df = t.history(period='30d', interval='1d')
        
        if isinstance(df, dict) or df.empty:
             return {"error": "No data found", "success": False}

        avg_price = df['close'].mean()
        
        return {
            "symbol": yahoo_sym,
            "average_30d": round(avg_price, 2),
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}

@tool
def get_company_fundamentals(symbol: str, data_point: str = "all") -> dict:
    """
    Fetch comprehensive fundamentals (Valuation, Profitability, Health).
    Uses Yahoo Finance (Primary) -> Screener.in (Fallback).
    """
    clean_sym = symbol.upper().replace(".NS", "").replace(".BO", "").strip()
    data = {}
    
    # ---------------------------------------------------------
    # STRATEGY 1: Yahoo Finance 
    # ---------------------------------------------------------
    suffixes = [".NS", ".BO"]
    for suffix in suffixes:
        try:
            ticker = f"{clean_sym}{suffix}"
            yf_stock = yf.Ticker(ticker)
            info = yf_stock.info
            
            if info and info.get("currentPrice"):
                def safe(key): return info.get(key)
                
                # Inline Helper for Percentages
                def pct(key):
                    val = info.get(key)
                    return f"{round(val * 100, 2)}%" if val is not None else "N/A"

                data = {
                    "name": info.get("longName", clean_sym),
                    
                    # Currency Fields
                    "current_price": format_currency(safe("currentPrice")),
                    "market_cap": format_currency(safe("marketCap")),
                    "book_value": format_currency(safe("bookValue")),
                    "total_debt": format_currency(safe("totalDebt")),
                    "free_cash_flow": format_currency(safe("freeCashflow")),
                    
                    # Valuation
                    "pe_ratio": round(safe("trailingPE"), 2) if safe("trailingPE") else "N/A",
                    "peg_ratio": safe("pegRatio") or "N/A",
                    "dividend_yield": pct("dividendYield"),
                    
                    # Profitability
                    "roe": pct("returnOnEquity"),
                    "roa": pct("returnOnAssets"),
                    "net_profit_margin": pct("profitMargins"),
                    "operating_margin": pct("operatingMargins"),
                    
                    # Financial Health
                    "debt_to_equity": round(safe("debtToEquity")/100, 2) if safe("debtToEquity") else "N/A",
                    "current_ratio": safe("currentRatio") or "N/A",
                    
                    # Growth
                    "revenue_growth": pct("revenueGrowth"),
                    "earnings_growth": pct("earningsGrowth"),
                    
                    "source": f"yfinance ({suffix})",
                    "success": True
                }
                break 
        except: continue

    # ---------------------------------------------------------
    # STRATEGY 2: Screener.in
    # ---------------------------------------------------------
    if not data:
        try:
            url = f"https://www.screener.in/company/{clean_sym}/"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                def get_sc(label):
                    tags = soup.find_all('span', class_='name')
                    for tag in tags:
                        if label.lower() in tag.text.strip().lower():
                            val = tag.find_next('span', class_='number')
                            if val: return val.text.strip().replace(",", "")
                    return None

                # Safe float conversion
                try: mcap = float(get_sc('Market Cap')) 
                except: mcap = None
                try: price = float(get_sc('Current Price'))
                except: price = None
                try: bv = float(get_sc('Book Value'))
                except: bv = None

                data = {
                    "name": clean_sym,
                    "source": "Screener.in (Fallback)",
                    "current_price": f"₹{price}" if price else "N/A",
                    # Note: Screener MCAP is usually in Cr already, we just append 'Cr'
                    "market_cap": f"₹{mcap} Cr" if mcap else "N/A",
                    
                    "pe_ratio": get_sc('Stock P/E') or "N/A",
                    "roce": f"{get_sc('ROCE')}%",
                    "roe": f"{get_sc('ROE')}%",
                    "dividend_yield": f"{get_sc('Dividend Yield')}%",
                    "book_value": f"₹{bv}" if bv else "N/A",
                    "success": True
                }
        except Exception as e:
            print(f"Screener Error: {e}")

    if not data: 
        return {"error": f"No data found for {clean_sym}", "success": False}

    if data_point.lower() == "all": return data

    match = get_close_matches(data_point, data.keys(), n=1, cutoff=0.5)
    if match: return {match[0]: data[match[0]], "success": True}
    
    return {"error": f"Metric '{data_point}' not found.", "success": False}


@tool
def get_news_headlines(company_name: str) -> dict:
    """
    Fetch top financial news, prioritizing Moneycontrol , Economic Times .
    """
    headlines = []
    seen = set()

    def add(new_titles):
        for h in new_titles:
            h = h.strip()
            if h not in seen:
                headlines.append(h)
                seen.add(h)

    # 1. Trusted Sources (Priority)
    try:
        gnews = GNews(language='en', country='IN', period='7d', max_results=5)
        # Force results from trusted domains
        query = f"{company_name} (site:moneycontrol.com OR site:economictimes.indiatimes.com)"
        add([n['title'] for n in gnews.get_news(query)])
    except: pass
    
    # 2. NewsAPI 
    api_key = os.getenv("NEWS_API_KEY")
    if api_key and len(headlines) < 10:
        try:
            n_client = NewsApiClient(api_key=api_key)
            resp = n_client.get_everything(q=f"{company_name} stock", language='en', sort_by='relevancy', page_size=5)
            if resp.get('status') == 'ok':
                add([a['title'] for a in resp['articles']])
        except: pass

    return {"headlines": headlines[:10], "success": bool(headlines)}
@tool
def analyze_sentiment(headlines: list) -> dict:
    """
    Analyze sentiment of headlines (Positive/Negative/Neutral).
    """
    if not headlines: return {"sentiment_label": "Neutral", "score": 0}
    try:
        polarities = [TextBlob(h).sentiment.polarity for h in headlines]
        avg = sum(polarities) / len(polarities)
        label = "Positive" if avg > 0.05 else "Negative" if avg < -0.05 else "Neutral"
        return {"sentiment_label": label, "score": round(avg, 2)}
    except: return {"sentiment_label": "Neutral", "score": 0}

@tool
def suggest_trade(current_price: float, average_price: float, sentiment_label: str) -> dict:
    """
    Generate trade suggestion (Buy/Sell/Hold) based on Price Trend vs 30-day Avg + Sentiment.
    """
    if average_price == 0: return {"recommendation": "Hold", "reason": "No Data"}
    
    diff = ((current_price - average_price) / average_price) * 100
    rec = "HOLD ⚖️"
    
    # Logic: >3% Deviation + Matching Sentiment = Signal
    if diff > 3 and sentiment_label == "Positive": rec = "BUY 🚀"
    elif diff < -3 and sentiment_label == "Negative": rec = "SELL ⬇️"
    
    return {"recommendation": rec, "trend": f"{diff:.1f}%"}

@tool
def search_web(query: str):
    """
    Searches the web for reasons,demerger,aquistions, future plans, or qualitative analysis etc. 
    Returns text snippets.
    """
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    try:
        return search.invoke(query)
    except Exception as e:
        return f"Search failed: {e}"
