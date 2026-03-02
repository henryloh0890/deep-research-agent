# Deep Research Agent 🔍

An autonomous AI research agent that searches the web, synthesizes information, and generates structured reports.

## Features (in progress)
- 🔎 Autonomous web search with multi-query planning
- 📝 Structured report generation
- 💾 Research history logged to SQLite
- 🤖 Multi-LLM support (Claude + Groq/Llama)

## Tech Stack
- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM tooling
- [Claude](https://anthropic.com) — primary LLM
- [Groq](https://groq.com) — fast free LLM
- [DuckDuckGo Search](https://pypi.org/project/ddgs/) — web search

## Setup
```bash
conda create -n research-agent python=3.11 -y
conda activate research-agent
pip install -r requirements.txt
```

Add your API keys to `.env`:
```
ANTHROPIC_API_KEY=your_key
GROQ_API_KEY=your_key
```

## Progress
- [x] Stage 1: Working search agent prototype
- [x] Stage 2: Multi-step planning + web scraper
- [x] Stage 3: Data analysis integration
- [x] Stage 4: Polished report generation
- [ ] Stage 5: Gradio UI
