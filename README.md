# 🔍 Deep Research Agent

An autonomous AI research agent that searches the web, scrapes pages, and synthesizes findings into structured markdown reports — powered by LangGraph and your choice of LLM.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **Autonomous research loop** — searches the web 3–4 times, scrapes the most relevant pages, and synthesizes everything into a structured report
- **Multi-LLM support** — choose from 7 models across Anthropic, Groq, and Google
- **Live progress streaming** — watch every tool call in real time via the Gradio UI or CLI
- **Token tracking** — every run logs input/output token counts to SQLite
- **Research history** — all runs stored in a local database, queryable with pandas
- **Dual entry points** — run via web UI (`app.py`) or terminal (`main.py`)
- **Clean module structure** — production-ready `src/` layout

---

## 🤖 Available Models

| Model | Provider | Cost |
|-------|----------|------|
| Claude Sonnet 4.6 *(recommended)* | Anthropic | Paid |
| Claude Haiku 4.5 | Anthropic | Paid (cheaper) |
| Llama 3.3 70B | Groq | Free |
| Llama 3.1 8B | Groq | Free (fastest) |
| Gemini 2.5 Flash | Google | Free tier |
| Gemini 2.5 Flash Lite | Google | Free tier |
| Gemini 2.0 Flash | Google | Free tier |

---

## 🏗️ Project Structure

```
deep-research-agent/
│
├── notebooks/
│   ├── 01_search_agent_prototype.ipynb     # Stage 1: basic search agent
│   ├── 02_planning_and_scraping.ipynb      # Stage 2: scraper + report generation
│   ├── 03_sqlite_logging_and_analysis.ipynb # Stage 3: logging + token tracking
│   └── 04_using_src_modules.ipynb          # Stage 4: clean module usage
│
├── src/
│   ├── agent.py       # Core agent logic, model factory, run_research()
│   ├── tools.py       # Search, scraper, and report-saving tools
│   └── database.py    # SQLite logging and pandas analysis
│
├── reports/           # Generated markdown reports
├── app.py             # Gradio web UI
├── main.py            # CLI entry point
├── requirements.txt
└── .env               # API keys (never committed)
```

---

## ⚡ Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/henryloh0890/deep-research-agent.git
cd deep-research-agent

conda create -n research-agent python=3.11 -y
conda activate research-agent

pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
GROQ_API_KEY=your_key_here        # free at console.groq.com
GOOGLE_API_KEY=your_key_here      # free at aistudio.google.com
```

### 3. Run

**Web UI:**
```bash
python app.py
```
Opens at `http://localhost:7860`

**CLI:**
```bash
python main.py "impact of AI on healthcare 2025"
```

**View research history:**
```bash
python main.py --history
```

**Quiet mode (no live output):**
```bash
python main.py "quantum computing trends" --quiet
```

---

## 🔄 How It Works

```
User provides topic
       ↓
Agent searches the web (3–4 queries)
       ↓
Agent selects and scrapes 2–3 most relevant pages
       ↓
LLM synthesizes findings into structured report
       ↓
Report saved to /reports as markdown
       ↓
Run logged to SQLite (topic, duration, tokens, status)
```

The agent runs a **ReAct loop** (Reason + Act) — it decides what to search, reads the results, decides what to scrape, reads the pages, then synthesizes everything into a final report.

---

## 📊 Example Output

Running `python main.py "future of renewable energy 2025"` produces:

```
🚀 Starting research on: 'future of renewable energy 2025'
--------------------------------------------------
[2.1s] 🤖 Calling tool → search_web
         Input: {'query': 'renewable energy trends 2025'}
[4.8s] 🔧 [search_web] returned results
[5.1s] 🤖 Calling tool → search_web
         Input: {'query': 'solar wind power developments 2025'}
[7.3s] 🔧 [search_web] returned results
...
[28.4s] 🤖 Claude writing response...
--------------------------------------------------
✅ Research complete!
   Searches made: 4
   Pages scraped: 3
   Duration: 31.2s
   Tokens — Input: 18,432 | Output: 2,841 | Total: 21,273
   Run ID: 12
```

Generated reports include:
- Executive Summary
- Key Findings with cited source URLs
- Detailed Analysis
- Sources (only URLs actually visited)
- Conclusion

---

## 🗄️ Research History

All runs are logged to `research_history.db`. Analyse with pandas in the notebook or load directly:

```python
from src.database import load_history, print_stats

# Load as DataFrame
df = load_history()

# Print full summary
print_stats()
```

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Agent orchestration | LangGraph 1.0 |
| LLM integrations | LangChain, langchain-anthropic, langchain-openai, langchain-google-genai |
| Web search | ddgs (DuckDuckGo) |
| Web scraping | httpx + BeautifulSoup4 |
| Data analysis | pandas |
| Database | SQLite3 |
| Web UI | Gradio |
| Environment | Anaconda + Python 3.11 |

---

## 📈 Roadmap

- [x] Stage 1 — Working search agent
- [x] Stage 2 — Web scraper + report generation
- [x] Stage 3 — SQLite logging + token tracking
- [x] Stage 4 — Clean modules + CLI
- [x] Stage 5 — Gradio UI with model cascading
- [ ] Stage 6 — Export reports to PDF
- [ ] Stage 7 — Deploy to Hugging Face Spaces

---

## 📝 Key Lessons Learned

A few prompt engineering insights discovered while building this:

- **Vague goals cause infinite loops** — `"cast a wide net"` causes agents to search forever. `"search exactly 3 times"` works.
- **Temperature matters** — `temperature=0` for reasoning/planning steps, higher for writing steps
- **Multi-LLM routing** — different models need different prompting strategies; Gemini responds better to explicit ordered steps
- **Lost in the middle** — LLMs perform worse with too much context; 3–4 quality sources beats 10 unfiltered ones

---

## 🔑 Getting API Keys

| Provider | URL | Free Tier |
|----------|-----|-----------|
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | Free credits on signup |
| Groq | [console.groq.com](https://console.groq.com) | Fully free |
| Google | [aistudio.google.com](https://aistudio.google.com) | Generous free tier |

---

## ⚠️ Notes

- Never commit your `.env` file — it's in `.gitignore` by default
- DuckDuckGo search occasionally rate limits; the agent retries up to 3 times automatically
- Token costs apply for Anthropic models; Groq and Gemini free tiers are sufficient for development

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
