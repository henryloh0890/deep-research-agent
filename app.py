import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import gradio as gr
import glob as glob_module
import os
import re
import time as t
from langchain_core.messages import HumanMessage
from src.agent import build_agent, build_prompt, AVAILABLE_MODELS
from src.tools import run_metrics, reset_metrics
from src.database import load_history, log_research_run, init_database

init_database()


def run_research_streaming(topic: str, model_name: str):
    """Run research with selected model and yield live progress."""

    if not topic.strip():
        yield "Please enter a research topic.", "", ""
        return

    reset_metrics()
    agent = build_agent(model_name)

    start_time = t.time()
    status = "success"
    report_filename = ""
    input_tokens = 0
    output_tokens = 0
    last_chunk = None
    progress_log = []

    def log(msg):
        progress_log.append(msg)
        return "\n".join(progress_log)

    yield log(f"🚀 Starting research on: '{topic}'"), "", ""
    yield log(f"🤖 Model: {model_name}"), "", ""
    yield log("-" * 50), "", ""

    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=build_prompt(topic))]},
        ):
            last_chunk = chunk
            elapsed = round(t.time() - start_time, 1)

            if "agent" in chunk:
                for message in chunk["agent"].get("messages", []):
                    content = message.content

                    # Replace the entire token extraction block with this:
                    usage = message.response_metadata.get("usage", {})
                    usage_meta = getattr(message, "usage_metadata", {}) or {}

                    # Anthropic
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)
                    # Groq/OpenAI
                    input_tokens += usage.get("prompt_tokens", 0)
                    output_tokens += usage.get("completion_tokens", 0)
                    # Gemini — uses usage_metadata attribute directly
                    if isinstance(usage_meta, dict):
                        input_tokens += usage_meta.get("input_tokens", 0)
                        output_tokens += usage_meta.get("output_tokens", 0)

                    if isinstance(content, list):
                        for block in content:
                            if hasattr(block, "name"):
                                yield log(f"[{elapsed}s] 🤖 Calling tool → {block.name}"), "", ""
                                if hasattr(block, "input"):
                                    yield log(f"         Input: {str(block.input)[:120]}"), "", ""
                    elif isinstance(content, str) and content.strip():
                        yield log(f"[{elapsed}s] 🤖 Writing report..."), "", ""

            elif "tools" in chunk:
                for message in chunk["tools"].get("messages", []):
                    tool_name = getattr(message, "name", "unknown")
                    yield log(f"[{elapsed}s] 🔧 [{tool_name}] returned results"), "", ""

        # Extract report filename — use most recently created file
        import glob as glob_module
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        md_files = glob_module.glob(os.path.join(reports_dir, "*.md"))
        if md_files:
            report_filename = os.path.basename(max(md_files, key=os.path.getctime))

    except Exception as e:
        status = f"error: {str(e)}"
        yield log(f"\n❌ Error: {e}"), "", ""

    duration = round(t.time() - start_time, 2)

    run_id = log_research_run(
        topic=topic,
        num_searches=run_metrics["num_searches"],
        num_pages_scraped=run_metrics["num_pages_scraped"],
        report_filename=report_filename,
        duration_seconds=duration,
        llm_used=model_name,
        status=status,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )

    # Load report content
    report_content = ""
    if report_filename:
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        report_path = os.path.join(reports_dir, report_filename)
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()

    summary = f"""## Run Complete ✅
- **Topic:** {topic}
- **Model:** {model_name}
- **Searches made:** {run_metrics['num_searches']}
- **Pages scraped:** {run_metrics['num_pages_scraped']}
- **Duration:** {duration}s
- **Tokens:** {input_tokens + output_tokens:,} (Input: {input_tokens:,} | Output: {output_tokens:,})
- **Run ID:** {run_id}
- **Report file:** {report_filename}
"""

    yield log(f"\n[DONE] Research complete in {duration}s | Run ID: {run_id}"), summary, report_content


def get_history():
    df = load_history()
    if df.empty:
        return None
    return df[["id", "timestamp", "topic", "llm_used", "num_searches",
               "duration_seconds", "total_tokens", "status"]]


# UI
with gr.Blocks(title="Deep Research Agent", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # 🔍 Deep Research Agent
    *Autonomous web research powered by LangGraph*
    """)

    with gr.Tab("Research"):
        with gr.Row():
            with gr.Column(scale=2):
                topic_input = gr.Textbox(
                    label="Research Topic",
                    placeholder="e.g. 'Impact of AI on healthcare 2025'",
                    lines=2
                )
                model_selector = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="Claude Sonnet 4.6 (Recommended)",
                    label="Model",
                    info="Select the LLM to use for research"
                )
                # Show model description dynamically
                model_info = gr.Markdown(
                    value=f"ℹ️ {AVAILABLE_MODELS['Claude Sonnet 4.6 (Recommended)']['description']}"
                )
                run_btn = gr.Button("🚀 Start Research", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                progress_output = gr.Textbox(
                    label="Live Progress",
                    lines=15,
                    max_lines=30,
                    interactive=False
                )
            with gr.Column(scale=1):
                summary_output = gr.Markdown(label="Run Summary")

        gr.Markdown("### 📄 Generated Report")
        report_output = gr.Markdown()

        # Update model info when selection changes
        def update_model_info(model_name):
            desc = AVAILABLE_MODELS[model_name]["description"]
            return f"ℹ️ {desc}"

        model_selector.change(
            fn=update_model_info,
            inputs=model_selector,
            outputs=model_info
        )

        run_btn.click(
            fn=run_research_streaming,
            inputs=[topic_input, model_selector],
            outputs=[progress_output, summary_output, report_output]
        )

    with gr.Tab("History"):
        refresh_btn = gr.Button("🔄 Refresh History")
        history_table = gr.Dataframe(
            value=get_history(),
            label="Research Runs",
            interactive=False
        )
        refresh_btn.click(fn=get_history, outputs=history_table)

    with gr.Tab("About"):
        gr.Markdown("""
        ## About This Project

        An autonomous AI research agent that searches the web, scrapes pages,
        and synthesizes findings into structured reports.

        ### Available Models
        | Model | Provider | Cost |
        |-------|----------|------|
        | Claude Sonnet 4.6 | Anthropic | Paid |
        | Claude Haiku 4.5 | Anthropic | Paid (cheaper) |
        | Llama 3.3 70B | Groq | Free |
        | Llama 3.1 8B | Groq | Free |
        | Gemini 2.5 Flash | Google | Free tier |
        | Gemini 2.5 Flash Lite | Google | Free tier |
        | Gemini 2.0 Flash | Google | Free tier |

        ### Tech Stack
        - **LangGraph** — agent orchestration
        - **LangChain** — LLM tooling
        - **DuckDuckGo** — web search
        - **BeautifulSoup** — web scraping
        - **SQLite + Pandas** — history logging
        - **Gradio** — this UI
        """)

if __name__ == "__main__":
    app.launch(inbrowser=True)