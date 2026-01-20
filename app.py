import os
import sys
import datetime,time
import random
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from langchain_core.pydantic_v1 import BaseModel, Field

from newsapi import NewsApiClient
from gnews import GNews

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts.chat import ChatPromptTemplate
from langchain.tools import tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# --- Input Schemas ---
# These help the LLM understand exactly what to provide
class SearchInput(BaseModel):
    query: str = Field(description="The search query string to find information.")

class ReadBlogInput(BaseModel):
    url: str = Field(description="The valid URL of the article or blog post to read.")

# --- Helper for Anti-Bot ---
def get_random_header():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    ]
    return {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/"
    }


# ================= 0. SETUP =================
# Load environment variables
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found in .env file.")
    sys.exit(1)

app = Flask(__name__)

# ================= 1. THINKING LOG HANDLER =================
class ThinkingLogHandler(BaseCallbackHandler):
    """Captures tool usage to show 'Thinking' on the UI."""
    def __init__(self):
        self.logs = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name")
        self.logs.append(f"🛠️ Using tool: {tool_name}...")

NIFTY_50_MAPPING = {
    "RELIANCE": "RELIANCE", "TCS": "TCS", "HDFC BANK": "HDFCBANK", "ICICI BANK": "ICICIBANK",
    "INFOSYS": "INFY", "SBI": "SBIN", "BHARTI AIRTEL": "BHARTIARTL", "ITC": "ITC",
    "LARSEN": "LT", "L&T": "LT", "HINDUSTAN UNILEVER": "HINDUNILVR", "AXIS BANK": "AXISBANK",
    "KOTAK": "KOTAKBANK", "TATA MOTORS PV": "TMPV", "TITAN": "TITAN", "BAJAJ FINANCE": "BAJFINANCE",
    "SUN PHARMA": "SUNPHARMA", "HCL TECH": "HCLTECH", "MARUTI": "MARUTI", "ASIAN PAINTS": "ASIANPAINT",
    "ADANI ENTERPRISES": "ADANIENT", "ADANI PORTS": "ADANIPORTS", "ULTRATECH": "ULTRACEMCO",
    "POWER GRID": "POWERGRID", "WIPRO": "WIPRO", "NTPC": "NTPC", "ONGC": "ONGC",
    "JSW STEEL": "JSWSTEEL", "TATA STEEL": "TATASTEEL", "M&M": "M&M", "MAHINDRA": "M&M",
    "COAL INDIA": "COALINDIA", "BAJAJ FINSERV": "BAJAJFINSV", "EICHER MOTORS": "EICHERMOT",
    "NESTLE": "NESTLEIND", "BPCL": "BPCL", "GRASIM": "GRASIM", "BRITANNIA": "BRITANNIA",
    "TECH MAHINDRA": "TECHM", "HINDALCO": "HINDALCO", "CIPLA": "CIPLA", "DR REDDY": "DRREDDY",
    "APOLLO HOSPITALS": "APOLLOHOSP", "TATA CONSUMER": "TATACONSUM",
    "DIVIS LAB": "DIVISLAB", "HERO MOTOCORP": "HEROMOTOCO", "BAJAJ AUTO": "BAJAJ-AUTO",
    "SHRIRAM FINANCE": "SHRIRAMFIN", "TRENT": "TRENT", "BEL": "BEL"
}

# ================= 2. SYMBOL CLEANING =================
def clean_symbol(symbol: str) -> str:
    """Ensures we use the correct NIFTY 50 ticker."""
    q = symbol.upper().strip()
    
    # Check the NIFTY_50_MAPPING first
    for name, sym in NIFTY_50_MAPPING.items():
        if name == q or name in q: # Exact or partial match in map keys
            return sym
            
    # Fallback: Strip existing suffixes if user typed them
    return q.replace(".NS", "").replace(".BO", "").strip()
def format_currency(value):
    if not value or isinstance(value, str): return "N/A"
    if value >= 10000000: return f"₹{value / 10000000:,.2f} Cr"
    return f"₹{value:,.2f}"

# ================= 3. MULTI-SOURCE  TOOLS =================

@tool
def get_stock_quote(symbol: str) -> dict:
    """
    Get live stock price.
    Strategy: nsetools (Direct NSE) -> YahooQuery -> yfinance -> investpy -> Error.
    """
    clean_sym = clean_symbol(symbol)
    yahoo_sym = f"{clean_sym}.NS"
    
    # Source 1: nsetools (Official NSE Scraping)
    try:
        from nsetools import Nse
        nse = Nse()
        quote = nse.get_quote(clean_sym)
        if quote and quote.get("lastPrice"):
            return {"price": float(quote["lastPrice"]), "source": "nsetools", "success": True}
    except: pass

    # Source 2: YahooQuery (Fast)
    try:
        from yahooquery import Ticker
        stock = Ticker(yahoo_sym)
        price_info = stock.price.get(yahoo_sym, {})
        price = (price_info.get("regularMarketPrice") or 
                 price_info.get("currentPrice") or 
                 price_info.get("previousClose"))
        if price: return {"price": round(price, 2), "source": "YahooQuery", "success": True}
    except: pass

    # Source 3: yfinance (Reliable)
    try:
        import yfinance as yf
        stock = yf.Ticker(yahoo_sym)
        data = stock.history(period="1d")
        if not data.empty:
            return {"price": round(data['Close'].iloc[-1], 2), "source": "yfinance", "success": True}
    except: pass

    # Source 4: investpy (Investing.com)
    try:
        import investpy
        data = investpy.get_stock_recent_data(stock=clean_sym, country='india', order='descending')
        if not data.empty:
            return {"price": round(data['Close'].iloc[0], 2), "source": "investpy", "success": True}
    except: pass

    return {"error": "Failed to fetch price from all sources.", "success": False}

@tool
def get_historical_average(symbol: str) -> dict:
    """
    Get 30-day historical average price.
    Strategy: YahooQuery -> yfinance -> investpy.
    """
    clean_sym = clean_symbol(symbol)
    yahoo_sym = f"{clean_sym}.NS"

    # Source 1: YahooQuery
    try:
        from yahooquery import Ticker
        stock = Ticker(yahoo_sym)
        hist = stock.history(period="1mo")
        if not hist.empty and 'close' in hist.columns:
            return {"average_price": round(hist['close'].mean(), 2), "success": True}
    except: pass

    # Source 2: yfinance
    try:
        import yfinance as yf
        stock = yf.Ticker(yahoo_sym)
        hist = stock.history(period="1mo")
        if not hist.empty:
            return {"average_price": round(hist['Close'].mean(), 2), "success": True}
    except: pass

    # Source 3: investpy
    try:
        import investpy
        df = investpy.get_stock_recent_data(stock=clean_sym, country='india')
        df = df.tail(30) # Get last 30 days
        if not df.empty:
            return {"average_price": round(df['Close'].mean(), 2), "success": True}
    except: pass

    return {"average_price": 0, "error": "No history found", "success": False}



@tool
def get_company_fundamentals(symbol: str, data_point: str = "all") -> dict:
    """
    Fetch factual Indian stock fundamentals.
    Priority: YahooQuery (Annual) → yfinance fallback.
    Currency strictly validated to INR.
    """

    from difflib import get_close_matches
    from yahooquery import Ticker
    import yfinance as yf

    clean_sym = clean_symbol(symbol)
    yahoo_sym = f"{clean_sym}.NS"

    # ----------------- HELPERS -----------------
    def safe(val, default=0):
        return val if val not in (None, "", "NaN") else default

    def pct(val):
        return f"{round(val * 100, 2)}%" if val else "N/A"

    def valid_inr(info):
        return info.get("currency") == "INR"

    # ----------------- METRIC MAP -----------------
    metric_map = {
        "market cap": "market_cap",
        "debt": "total_debt",
        "pe": "pe_ratio",
        "profit": "net_income",
        "revenue": "total_revenue",
        "cash": "operating_cash_flow",
        "free cash": "free_cash_flow",
        "roe": "roe",
        "price": "current_price"
    }

    data = {}

    # ================== STEP 1: YAHOOQUERY ==================
    try:
        yq = Ticker(yahoo_sym)
        modules = yq.get_modules(
            "financialData defaultKeyStatistics summaryDetail price"
        ).get(yahoo_sym, {})

        price = modules.get("price", {})
        if valid_inr(price):

            fin = modules.get("financialData", {})
            stats = modules.get("defaultKeyStatistics", {})
            summary = modules.get("summaryDetail", {})

            revenue = safe(fin.get("totalRevenue"))
            fcf = safe(fin.get("freeCashflow"))

            # Sanity check
            if revenue and fcf > revenue:
                fcf = 0

            data = {
                "name": price.get("longName", clean_sym),
                "currency": "INR",
                "market_cap": format_currency(summary.get("marketCap")),
                "current_price": f"₹{safe(fin.get('currentPrice'))}",
                "pe_ratio": round(summary.get("trailingPE"), 2) if summary.get("trailingPE") else "N/A",
                "total_debt": format_currency(fin.get("totalDebt")),
                "total_revenue": format_currency(revenue),
                "net_income": format_currency(
                    stats.get("netIncomeToCommon") or fin.get("netIncomeToCommon")
                ),
                "operating_cash_flow": format_currency(fin.get("operatingCashflow")),
                "free_cash_flow": format_currency(fcf) if fcf else "N/A",
                "roe": pct(fin.get("returnOnEquity")),
                "source": "YahooQuery (TTM)",
                "success": True
            }

    except Exception as e:
        print(f"[YahooQuery Error] {e}")

    # ================== STEP 2: YFINANCE FALLBACK ==================
    if not data:
        try:
            yf_stock = yf.Ticker(yahoo_sym)
            info = yf_stock.info

            if valid_inr(info) and info.get("marketCap"):
                data = {
                    "name": info.get("longName", clean_sym),
                    "currency": "INR",
                    "market_cap": format_currency(info.get("marketCap")),
                    "current_price": f"₹{safe(info.get('currentPrice'))}",
                    "pe_ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else "N/A",
                    "total_debt": format_currency(info.get("totalDebt")),
                    "total_revenue": format_currency(info.get("totalRevenue")),
                    "net_income": format_currency(info.get("netIncomeToCommon")),
                    "operating_cash_flow": format_currency(info.get("operatingCashflow")),
                    "free_cash_flow": format_currency(info.get("freeCashflow")),
                    "roe": pct(info.get("returnOnEquity")),
                    "source": "yfinance",
                    "success": True
                }
        except Exception as e:
            print(f"[yfinance Error] {e}")

    if not data:
        return {"error": f"No verified INR data found for {clean_sym}", "success": False}

    # ================== FILTERING ==================
    if data_point.lower() == "all":
        return data

    query = data_point.lower().replace("_", " ")

    for key, field in metric_map.items():
        if key in query:
            return {
                field: data.get(field, "N/A"),
                "metric": key,
                "currency": "INR",
                "source": data["source"],
                "success": True
            }

    match = get_close_matches(query, metric_map.keys(), n=1, cutoff=0.6)
    if match:
        field = metric_map[match[0]]
        return {
            field: data.get(field, "N/A"),
            "metric": match[0],
            "note": f"Did you mean '{match[0]}'?",
            "source": data["source"],
            "success": True
        }

    return data


@tool
def get_news_headlines(company_name: str) -> dict:
    """
    Fetch RELEVANT financial news headlines with strict filtering.
    Strategy: NewsAPI (Trusted Domains) -> Google News (Financial Query) -> investpy (Fallback).
    """
    headlines = []
    
    # 1. Strict Query Construction
    # Using strict keywords ensures we get stock market news, not just product announcements.
    search_query = f'"{company_name}" AND (stock OR share OR earnings OR profit OR quarterly OR dividend)'
    
    # 2. Trusted Financial Domains (for NewsAPI)
    trusted_domains = (
        "moneycontrol.com,livemint.com,economictimes.indiatimes.com,"
        "business-standard.com,financialexpress.com,bloombergquint.com,"
        "cnbctv18.com,ndtvprofit.com"
    )

    # --- SOURCE 1: NewsAPI ---
    try:
        from newsapi import NewsApiClient
        import datetime
        
        api_key = os.getenv("NEWSAPI_KEY")
        if api_key:
            api = NewsApiClient(api_key=api_key)
            
            # Limit to last 3 days to keep news fresh
            week_ago = (datetime.date.today() - datetime.timedelta(days=3)).isoformat()
            
            data = api.get_everything(
                q=search_query,
                domains=trusted_domains, # Strict domain filtering
                language="en",
                sort_by="relevancy",
                page_size=5,
                from_param=week_ago
            )
            
            # Format: "Headline (Source)"
            headlines = [f"{a['title']} ({a['source']['name']})" for a in data.get("articles", [])]
            
    except Exception as e:
        print(f"NewsAPI Error: {e}")

    # --- SOURCE 2: Google News (Best Coverage, No API Key needed) ---
    if not headlines:
        try:
            from gnews import GNews
            # '7d' = last 7 days. country='IN' ensures Indian context.
            google_news = GNews(language='en', country='IN', period='7d', max_results=5)
            
            # Google News sometimes ignores AND operators, so we make the string descriptive
            gnews_query = f"{company_name} stock price financial news India"
            news = google_news.get_news(gnews_query)
            
            headlines = [n['title'] for n in news]
            
        except Exception as e:
            print(f"GNews Error: {e}")

    # --- SOURCE 3: investpy---
    if not headlines:
        try:
            import investpy
            clean_sym = clean_symbol(company_name)
            # investpy requires the exact symbol or name
            inv_news = investpy.get_stock_news(stock=clean_sym, country='india')
            headlines = inv_news['title'].head(5).tolist()
        except Exception:
            # investpy often errors if the stock isn't found exactly or if IP is blocked.
            pass

    # --- FINAL CHECK ---
    if not headlines:
        return {
            "headlines": [], 
            "message": "No relevant financial news found in the last 7 days.", 
            "success": True
        }
    
    # Limit to top 5 headlines
    return {"headlines": headlines[:5], "success": True}

@tool
def analyze_sentiment(headlines: list) -> dict:
    """Analyze sentiment locally."""
    if not headlines: return {"sentiment_label": "Neutral", "score": 0}
    try:
        polarities = [TextBlob(h).sentiment.polarity for h in headlines]
        avg = sum(polarities) / len(polarities)
        label = "Positive" if avg > 0.15 else "Negative" if avg < -0.15 else "Neutral"
        return {"sentiment_label": label, "score": round(avg, 2)}
    except: return {"sentiment_label": "Neutral", "score": 0}

@tool
def suggest_trade(current_price: float, average_price: float, sentiment_label: str) -> dict:
    """Generate trade suggestion based on average price of last 30 days."""
    if average_price == 0: return {"recommendation": "Hold", "reason": "No Data"}
    diff = ((current_price - average_price) / average_price) * 100
    rec = "HOLD ⚖️"
    if diff > 3 and sentiment_label == "Positive": rec = "BUY 🚀"
    elif diff < -3 and sentiment_label == "Negative": rec = "SELL ⬇️"
    return {"recommendation": rec, "trend": f"{diff:.1f}%"}

# ================= WEB SEARCH & SCRAPING TOOLS =================


@tool("search_web", args_schema=SearchInput)
def search_web(query: str):
    """
    Searches the internet for analysis, blogs, or news articles.
    Use this when you need to find information about demergers, 
    future plans, or qualitative analysis.
    Returns a summary and URLs.
    """
    search = DuckDuckGoSearchRun()
    max_retries = 3
    
    # DuckDuckGo often throws rate limit errors. We add a simple retry mechanism.
    for attempt in range(max_retries):
        try:
            # invoke returns the string result directly
            return search.invoke(query)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Search failed after {max_retries} attempts. Error: {e}"
            time.sleep(1.5) # Wait briefly before retrying
    return "Search failed."


@tool("read_blog_content", args_schema=ReadBlogInput)
def read_blog_content(url: str):
    """
    Reads the full text content of a specific URL.
    Use this AFTER using 'search_web' to get the details of a specific article.
    Smartly removes navigation bars, footers, and ads.
    """
    try:
        # 1. Network Request with Anti-Bot headers
        response = requests.get(url, headers=get_random_header(), timeout=15)
        response.raise_for_status()
        
        # 2. Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 3. Clean "Noise" Tags (This is critical for quality data)
        # We remove scripts, styles, navbars, footers, and headers
        tags_to_remove = [
            "script", "style", "nav", "footer", "header", "aside", 
            "form", "iframe", "noscript", "svg"
        ]
        for tag in soup(tags_to_remove):
            tag.decompose() # Completely remove the tag from the tree
            
        # 4. Extract Text Smartly
        # get_text with separator handles paragraphs better than raw text
        text = soup.get_text(separator="\n", strip=True)
        
        # 5. Clean up excessive newlines
        import re
        clean_text = re.sub(r'\n+', '\n', text)
        
        # 6. Length Check
        if len(clean_text) < 200:
             return f"Error: Content too short. The site ({url}) might be blocking bots or requires JavaScript."
             
        # 7. Truncate for LLM Context Window
        return clean_text[:8000] + "..." if len(clean_text) > 8000 else clean_text

    except requests.exceptions.Timeout:
        return "Error: The request timed out. The server took too long to respond."
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the server. Check the URL."
    except Exception as e:
        return f"Error reading article: {str(e)}"

# ================= 4. AGENT SETUP =================

tools = [get_stock_quote, get_historical_average, get_company_fundamentals, get_news_headlines, analyze_sentiment, suggest_trade,read_blog_content,search_web]

company_list_str = ", ".join(NIFTY_50_MAPPING.keys())
system_message = f"""
You are a **NIFTY 50 Stock Analysis Assistant**

### SCOPE
Analyze **ONLY** these NIFTY 50 stocks:
{company_list_str}
Refuse all others or give similar available stocks.

### TOOLS
- **Analysis / Buy–Sell** → use ALL tools 
- **Simple data** (e.g., price) → use ONLY the required tool
- **Sentiment-only** → analyze_sentiment only

### OUTPUT
- Use Markdown tables
- Cite tool sources
- Label sections clearly
- tell if company fundamentals are good or bad based on standard financial ratios.
- Summarize news sentiment briefly
- Provide clear Buy/Sell/Hold recommendation with reasoning
- If using web search, summarize key points from articles
- If no data found, say "N/A" or "No Data"
- if multiple tools used, summarize findings coherently.


### RULES
- Educational insights only
- If any tool fails → explain briefly and **HOLD**
- Neutral, professional tone
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True, 

)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,memory=memory)

# ================= 5. FLASK ROUTES =================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input: return jsonify({"error": "Empty message"}), 400

    log_handler = ThinkingLogHandler()
    
    try:
        response = executor.invoke(
            {"input": user_input},
            config={"callbacks": [log_handler]}
        )
        return jsonify({
            "response": response["output"],
            "logs": log_handler.logs
        })
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}", "logs": []})

if __name__ == '__main__':
    app.run(debug=True, port=5000)