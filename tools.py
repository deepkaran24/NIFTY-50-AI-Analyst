import os
import requests
import yfinance as yf
from yahooquery import Ticker
from textblob import TextBlob
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from difflib import get_close_matches
from newsapi import NewsApiClient
from gnews import GNews
from bs4 import BeautifulSoup

# Import helpers from utils
from utils import clean_symbol, format_currency

@tool
def get_stock_quote(symbol: str) -> dict:
    """
    Get live stock price. 
    Priority: YahooQuery -> yfinance 
    """
    clean_sym = clean_symbol(symbol)
    yahoo_sym = f"{clean_sym}.NS"
    
    # Source 1: YahooQuery (Fastest)
    try:
        stock = Ticker(yahoo_sym)
        price_info = stock.price.get(yahoo_sym, {})
        # Check various keys for price
        price = (price_info.get("regularMarketPrice") or 
                 price_info.get("currentPrice") or 
                 price_info.get("previousClose"))
        
        if price: 
            return {"price": round(price, 2), "source": "YahooQuery", "success": True}
    except Exception:
        pass

    # Source 2: yfinance
    try:
        stock = yf.Ticker(yahoo_sym)
        if hasattr(stock, 'fast_info'):
            price = stock.fast_info.last_price
            if price:
                return {"price": round(price, 2), "source": "yfinance ", "success": True}
    except Exception:
        pass

    return {"error": "Failed to fetch price from all sources.", "success": False}

@tool
def get_historical_average(symbol: str) -> dict:
    """
    Get 30-day historical average.
    Priority: YahooQuery -> yfinance.
    """
    clean_sym = clean_symbol(symbol).upper().replace(".NS", "").strip()
    yahoo_sym = f"{clean_sym}.NS"
    
    # Source 1: YahooQuery
    try:
        t = Ticker(yahoo_sym)
        df = t.history(period='30d', interval='1d')
        
        if not isinstance(df, dict) and not df.empty:
            avg_price = df['close'].mean()
            return {
                "symbol": yahoo_sym,
                "average_30d": round(avg_price, 2),
                "source": "YahooQuery",
                "success": True
            }
    except Exception:
        pass

    # Source 2: yfinance 
    try:
        stock = yf.Ticker(yahoo_sym)
        hist = stock.history(period="30d")
        
        if not hist.empty:
            avg_price = hist['Close'].mean()
            return {
                "symbol": yahoo_sym,
                "average_30d": round(avg_price, 2),
                "source": "yfinance",
                "success": True
            }
    except Exception as e:
        return {"error": str(e), "success": False}
    
    return {"error": "No historical data found.", "success": False}

@tool
def get_company_fundamentals(symbol: str, data_point: str = "all") -> dict:
    """
    Fetch comprehensive fundamentals (Valuation, Profitability, Health).
    Priority: yfinance -> Screener.in
    """
    clean_sym = clean_symbol(symbol).upper().replace(".NS", "").replace(".BO", "").strip()
    data = {}
    
    # ---------------------------------------------------------
    # STRATEGY 1: Yahoo Finance 
    # ---------------------------------------------------------
    suffixes = [".NS"]
    for suffix in suffixes:
        try:
            ticker = f"{clean_sym}{suffix}"
            yf_stock = yf.Ticker(ticker)
            info = yf_stock.info
            
            # Check if we actually got valid data 
            if info and info.get("marketCap"):
                def safe(key): return info.get(key)
                def pct(key):
                    val = info.get(key)
                    return f"{round(val * 100, 2)}%" if val is not None else "N/A"

                data = {
                    "name": info.get("longName", clean_sym),
                    "current_price": format_currency(safe("currentPrice")),
                    "market_cap": format_currency(safe("marketCap")),
                    "book_value": format_currency(safe("bookValue")),
                    "total_debt": format_currency(safe("totalDebt")),
                    "free_cash_flow": format_currency(safe("freeCashflow")),
                    "pe_ratio": round(safe("trailingPE"), 2) if safe("trailingPE") else "N/A",
                    "peg_ratio": safe("pegRatio") or "N/A",
                    "dividend_yield": pct("dividendYield"),
                    "roe": pct("returnOnEquity"),
                    "roa": pct("returnOnAssets"),
                    "net_profit_margin": pct("profitMargins"),
                    "operating_margin": pct("operatingMargins"),
                    "debt_to_equity": round(safe("debtToEquity")/100, 2) if safe("debtToEquity") else "N/A",
                    "current_ratio": safe("currentRatio") or "N/A",
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
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            
            # Added timeout to prevent hanging
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                def get_sc(label):
                    tags = soup.find_all('span', class_='name')
                    for tag in tags:
                        if label.lower() == tag.text.strip().lower():
                            val = tag.find_next('span', class_='number')
                            if val: return val.text.strip().replace(",", "")
                    return None

                # Extract basic metrics
                mcap = get_sc('Market Cap')
                price = get_sc('Current Price')
                bv = get_sc('Book Value')
                pe = get_sc('Stock P/E')
                roce = get_sc('ROCE')
                roe = get_sc('ROE')
                div_yield = get_sc('Dividend Yield')

                data = {
                    "name": clean_sym,
                    "source": "Screener.in (Fallback)",
                    "current_price": f"₹{price}" if price else "N/A",
                    "market_cap": f"₹{mcap} Cr" if mcap else "N/A",
                    "pe_ratio": pe or "N/A",
                    "roce": f"{roce}%" if roce else "N/A",
                    "roe": f"{roe}%" if roe else "N/A",
                    "dividend_yield": f"{div_yield}%" if div_yield else "N/A",
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
    Fetch top financial news. 
    Priority: GNews (Trusted) -> GNews (Broad) -> NewsAPI
    """
    headlines = []
    seen = set()

    def add(new_titles):
        for h in new_titles:
            h = h.strip()
            # Basic deduplication and cleanup
            if h and h not in seen:
                headlines.append(h)
                seen.add(h)

    # 1. Trusted Sources 
    try:
        gnews = GNews(language='en', country='IN', period='7d', max_results=5)
        query = f'"{company_name}" site:moneycontrol.com OR site:economictimes.indiatimes.com'
        add([n['title'] for n in gnews.get_news(query)])
    except: pass

    # 2. Broader GNews Search 
    if len(headlines) < 5:
        try:
            gnews_t2 = GNews(language='en', country='IN', period='5d', max_results=5)
            query_t2 = f'"{company_name}" site:livemint.com OR site:business-standard.com OR site:cnbctv18.com'
            add([n['title'] for n in gnews_t2.get_news(query_t2)])
        except: pass

    # 3. NewsAPI
    api_key = os.getenv("NEWS_API_KEY")
    if api_key and len(headlines) < 5:
        try:
            n_client = NewsApiClient(api_key=api_key)
            resp = n_client.get_everything(q=f"{company_name}", language='en', sort_by='publishedAt', page_size=5)
            if resp.get('status') == 'ok':
                add([a['title'] for a in resp['articles']])
        except: pass

    return {"headlines": headlines[:10], "success": bool(headlines)}

@tool
def analyze_sentiment(headlines: list) -> dict:
    """
    Analyze sentiment of headlines.
    """
    if not headlines: 
        return {"sentiment_label": "Neutral", "score": 0}
    
    try:
        polarities = [TextBlob(h).sentiment.polarity for h in headlines]
        if not polarities:
            return {"sentiment_label": "Neutral", "score": 0}
            
        avg = sum(polarities) / len(polarities)
        label = "Positive" if avg > 0.05 else "Negative" if avg < -0.05 else "Neutral"
        return {"sentiment_label": label, "score": round(avg, 2)}
    except: 
        return {"sentiment_label": "Neutral", "score": 0}

@tool
def suggest_trade(current_price: float, average_price: float, sentiment_label: str) -> dict:
    """
    Generate trade suggestion based on Price vs 30d Avg + Sentiment.
    """
    # Ensure inputs are numbers
    try:
        current_price = float(current_price)
        average_price = float(average_price)
    except (ValueError, TypeError):
        return {"recommendation": "Hold", "reason": "Invalid Price Data"}

    if average_price == 0: 
        return {"recommendation": "Hold", "reason": "No Historical Data"}
    
    diff = ((current_price - average_price) / average_price) * 100
    rec = "HOLD ⚖️"
    
    # Logic: >3% Deviation + Matching Sentiment = Signal
    if diff > 3 and sentiment_label == "Positive": rec = "BUY 🚀"
    elif diff < -3 and sentiment_label == "Negative": rec = "SELL ⬇️"
    elif diff > 10: rec = "WATCH (Overbought?) ⚠️" # Added safety check
    elif diff < -10: rec = "WATCH (Oversold?) ⚠️" # Added safety check
    
    return {"recommendation": rec, "trend": f"{diff:.1f}%"}

@tool
def search_web(query: str):
    """
    Searches the web for qualitative analysis.
    """
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        search = DuckDuckGoSearchResults(api_wrapper=wrapper)
        return search.invoke(query)
    except Exception as e:
        return f"Search failed: {e}"