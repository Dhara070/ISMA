from __future__ import annotations

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from agents.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are an expert Indian stock market technical analyst AI assistant.

Your role:
- Analyse NSE/BSE stocks using technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands, etc.)
- Provide clear, structured, and actionable observations based on the data
- Explain what the indicators mean in simple language when the user is learning
- When asked about a stock, ALWAYS use the available tools to fetch real data first — never guess prices

Rules:
- All prices are in Indian Rupees (₹)
- Use NSE symbols (e.g. RELIANCE, TCS, INFY, HDFCBANK)
- When comparing stocks, use the compare_stocks tool
- Always include the overall signal (Bullish / Bearish / Neutral) in your response
- If the user asks for a recommendation, frame it as "signals suggest" not "you should buy/sell"

IMPORTANT DISCLAIMER: Always remind users that this analysis is for educational purposes only
and does not constitute financial advice. Past performance does not guarantee future results."""


def create_agent():
    """Create and return the LangChain ReAct agent with Ollama."""
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    return agent


def chat(agent, user_message: str) -> str:
    """Send a message to the agent and return the final text response."""
    result = agent.invoke(
        {"messages": [("human", user_message)]},
    )

    ai_messages = [
        m for m in result["messages"]
        if hasattr(m, "type") and m.type == "ai" and m.content
    ]
    if ai_messages:
        return ai_messages[-1].content
    return "I wasn't able to generate a response. Please try rephrasing your question."


def chat_stream(agent, user_message: str):
    """Stream the agent response token-by-token. Yields content strings."""
    for chunk in agent.stream(
        {"messages": [("human", user_message)]},
        stream_mode="messages",
    ):
        msg, metadata = chunk
        if hasattr(msg, "content") and msg.content and metadata.get("langgraph_node") == "agent":
            yield msg.content
