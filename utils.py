import random

# Constants: Full NIFTY 50 Mapping
NIFTY_50_MAPPING = {
    "RELIANCE": "RELIANCE", "TCS": "TCS", "HDFC BANK": "HDFCBANK", "ICICI BANK": "ICICIBANK",
    "INFOSYS": "INFY", "SBI": "SBIN", "BHARTI AIRTEL": "BHARTIARTL", "ITC": "ITC",
    "LARSEN": "LT", "L&T": "LT", "HINDUSTAN UNILEVER": "HINDUNILVR", "AXIS BANK": "AXISBANK",
    "KOTAK": "KOTAKBANK", "TATA MOTORS": "TMPV", "TITAN": "TITAN", "BAJAJ FINANCE": "BAJFINANCE",
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


def clean_symbol(symbol: str) -> str:
    """Ensures we use the correct NIFTY 50 ticker."""
    q = symbol.upper().strip()
    
    # 1. Exact Match
    if q in NIFTY_50_MAPPING:
        return NIFTY_50_MAPPING[q]
        
    # 2. Partial Match 
    for name, sym in NIFTY_50_MAPPING.items():
        if name in q: 
            return sym
            
    # 3. Fallback
    return q.replace(".NS", "").replace(".BO", "").strip()

def format_currency(value):
    """Formats large numbers into Crores or readable strings."""
    if not value or isinstance(value, str): return "N/A"
    if value >= 10000000: return f"₹{value / 10000000:,.2f} Cr"
    return f"₹{value:,.2f}"
