import os
import yfinance as yf
from yahooquery import Ticker
from textblob import TextBlob
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from difflib import get_close_matches

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

    # Source 2: yfinance (Fallback with 5-day history for weekends)
    try:
        stock = yf.Ticker(yahoo_sym)
        data = stock.history(period="5d")
        if not data.empty:
            return {"price": round(data['Close'].iloc[-1], 2), "source": "yfinance", "success": True}
    except: pass

    return {"error": "Failed to fetch price.", "success": False}

@tool
def get_historical_average(symbol: str) -> dict:
    """Get 30-day historical average price."""
    clean_sym = clean_symbol(symbol)
    yahoo_sym = f"{clean_sym}.NS"
    try:
        stock = yf.Ticker(yahoo_sym)
        hist = stock.history(period="1mo")
        if not hist.empty:
            return {"average_price": round(hist['Close'].mean(), 2), "success": True}
    except: pass
    return {"average_price": 0, "error": "No history found", "success": False}

@tool
def get_company_fundamentals(symbol: str, data_point: str = "all") -> dict:
    """
    Fetch factual Indian stock fundamentals (PE, Market Cap, ROE, etc.).
    Returns verified INR data only.
    """
    clean_sym = clean_symbol(symbol)
    yahoo_sym = f"{clean_sym}.NS"
    
    data = {}
    try:
        yf_stock = yf.Ticker(yahoo_sym)
        info = yf_stock.info
        
        def safe(key): return info.get(key)

        if info.get("currency") == "INR":
            data = {
                "name": info.get("longName", clean_sym),
                "market_cap": format_currency(safe("marketCap")),
                "current_price": f"₹{safe('currentPrice')}",
                "pe_ratio": round(safe("trailingPE"), 2) if safe("trailingPE") else "N/A",
                "roe": f"{round(safe('returnOnEquity')*100, 2)}%" if safe("returnOnEquity") else "N/A",
                "dividend_yield": f"{round(safe('dividendYield')*100, 2)}%" if safe("dividendYield") else "N/A",
                "book_value": format_currency(safe("bookValue")),
                "source": "yfinance",
                "success": True
            }
    except Exception as e:
        return {"error": str(e), "success": False}

    if not data: return {"error": "No data found", "success": False}

    # Return specific metric if requested
    if data_point.lower() == "all": return data

    match = get_close_matches(data_point, data.keys(), n=1, cutoff=0.6)
    if match: return {match[0]: data[match[0]], "success": True}
    
    return data

@tool
def get_news_headlines(company_name: str) -> dict:
    """Fetch top 5 financial news headlines using GNews."""
    from gnews import GNews
    try:
        google_news = GNews(language='en', country='IN', period='3d', max_results=5)
        # Search query like "RELIANCE stock"
        news = google_news.get_news(f"{company_name} stock")
        headlines = [n['title'] for n in news]
        if headlines: return {"headlines": headlines[:5], "success": True}
    except: pass
    return {"headlines": [], "message": "No news found.", "success": True}

@tool
def analyze_sentiment(headlines: list) -> dict:
    """
    Analyze sentiment of headlines (Positive/Negative/Neutral).
    Uses a low threshold (0.05) to catch financial nuances.
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
    Searches the web for reasons, future plans, or qualitative analysis. 
    Returns text snippets and links.
    """
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    try:
        return search.invoke(query)
    except Exception as e:
        return f"Search failed: {e}"