# 📈 NIFTY 50 AI Analyst

A robust, AI-powered stock analysis agent designed exclusively for the **NIFTY 50** index. It leverages **LangChain** and **OpenAI** to act as an intelligent financial assistant, capable of fetching live prices, analyzing fundamentals, reading news sentiment, and generating trade suggestions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.1-green) ![Flask](https://img.shields.io/badge/Flask-3.0-orange) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-purple)

## 🚀 Features

* **NIFTY 50 Exclusive:** Strictly enforces analysis only for NIFTY 50 companies (rejects US stocks or non-index Indian stocks).
* **Multi-Source Data Fetching:** Robust fallback logic to ensure 99.9% uptime.
    * *Price:* `nsetools` → `YahooQuery` → `yfinance` → `investpy`
    * *News:* `NewsAPI` (Trusted Domains) → `Google News`
    * *Fundamentals:* `YahooQuery` → `yfinance`
* **Smart Fundamentals:** Fetches accurate  data for Market Cap, PE, Debt, Revenue, and Cash Flow. Corrects anomalies (e.g., Free Cash Flow > Revenue).
* **Sentiment Analysis:** Uses `TextBlob` to analyze news headlines and gauge market sentiment (Positive/Negative/Neutral).
* **Thinking Process UI:** The Web Interface shows exactly what the AI is doing (e.g., "🛠️ Fetching Price...", "🛠️ Reading News...") before giving an answer.
* **Fuzzy Matching:** Understands typos like "profit" or synonyms like "sales" (for Revenue).
