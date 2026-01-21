import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Import tools (read_blog_content is removed)
from tools import (
    get_stock_quote, get_historical_average, get_company_fundamentals, 
    get_news_headlines, analyze_sentiment, suggest_trade, search_web
)
from utils import NIFTY_50_MAPPING

def initialize_agent():
    # 1. Define Tools List
    tools = [
        get_stock_quote, get_historical_average, get_company_fundamentals,
        get_news_headlines, analyze_sentiment, suggest_trade, search_web
    ]

    # 2. Define System Prompt
    company_list_str = ", ".join(NIFTY_50_MAPPING.keys())
    
    system_message = f"""
    You are a **NIFTY 50 Stock Analysis Assistant**.
    
    ### SCOPE
    Analyze **ONLY** these NIFTY 50 stocks:
    {company_list_str}
    
    ### WORKFLOW
    - **Quantitative Data:** Use `get_stock_quote` and `get_company_fundamentals` for numbers.
    - **Sentiment:** Use `get_news_headlines` then `analyze_sentiment`.
    - **Reasoning/Deep Dive:** Use `search_web`. The tool provides text snippets. **Analyze these snippets directly** to find reasons for price movement (e.g., "Earnings report", "New contract", "Global market fall").
    - **Trade Call:** Use `suggest_trade` combining Price Trend + Sentiment.


    ### OUTPUT GUIDELINES
    - Use Markdown tables for data.
    - Cite sources (e.g., "Source: YahooFinance", "Source: Web Search Snippets").
    - Be concise. Do not make up facts.
    - 
    - Give answer mostly in bullet points and clean format so that it doesn't look cluttered.
    - also provide a summary analysis at the end in a separate "Analysis" section along with trade recommendation.
    - if no data is found from tools, respond with "Data Not Available" message and tell to hold the stock.
    - If `search_web` provides a reason, mention it clearly in the "Analysis" section.

    ##
    -it is for educational purposes only and should not be considered as financial advice.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # 3. Setup Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 4. Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 5. Create Agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        memory=memory
    )