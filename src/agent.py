import os
import re
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.tools import tools, run_metrics, reset_metrics
from src.database import init_database, log_research_run

import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Initialize database on import
init_database()

# LLM
claude = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

# Agent
agent = create_react_agent(claude, tools=tools)

def build_prompt(topic: str) -> str:
    return f"""
    You are a thorough research agent. Follow these steps carefully:
        
        1. Search for '{topic}' — do this 3 to 4 times with different queries
        
        2. From all the URLs you find, select the 3 to 4 MOST RELEVANT and CREDIBLE ones:
           - Prefer: official sources, academic sites, reputable news outlets, industry blogs
           - Avoid: forums, social media, low quality blogs, paywalled content
        
        3. Scrape each selected page and evaluate whether it contains genuinely useful 
           information. Skip any page that is too thin or irrelevant.
        
        4. Synthesize your findings into a structured report with these sections:
           - Executive Summary
           - Key Findings (cite the source URL next to each finding)
           - Detailed Analysis
           - Sources (only URLs you actually visited and found useful)
           - Conclusion
           You can add additional section if it is deemed informative and important.
        
        5. Save the report with a descriptive filename related to '{topic}'
        
        Important: Never include sources you did not actually visit and scrape.
        You must complete all steps in under 20 tool calls total.
    """

def run_research(topic: str, verbose: bool = True) -> dict:
    """
    Run a full research session on a given topic.
    
    Args:
        topic: The research topic
        verbose: Whether to print live progress
    
    Returns:
        dict with keys: topic, report, filename, metrics, run_id
    """
    reset_metrics()
    
    start_time = time.time()
    status = "success"
    report_filename = ""
    input_tokens = 0
    output_tokens = 0
    last_chunk = None

    if verbose:
        print(f"🚀 Starting research on: '{topic}'", flush=True)
        print("-" * 50, flush=True)

    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=build_prompt(topic))]},
            config={"recursion_limit": 40}
        ):
            last_chunk = chunk
            elapsed = round(time.time() - start_time, 1)

            if "agent" in chunk:
                for message in chunk["agent"].get("messages", []):
                    content = message.content

                    # Accumulate token usage
                    usage = message.response_metadata.get("usage", {})
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)

                    if verbose:
                        if isinstance(content, list):
                            for block in content:
                                if hasattr(block, "name"):
                                    print(f"[{elapsed}s] 🤖 Calling tool → {block.name}", flush=True)
                                    if hasattr(block, "input"):
                                        print(f"         Input: {str(block.input)[:120]}", flush=True)
                        elif isinstance(content, str) and content.strip():
                            print(f"[{elapsed}s] 🤖 Claude writing response...", flush=True)
                            print(f"         Preview: {content[:150]}...", flush=True)

            elif "tools" in chunk:
                if verbose:
                    for message in chunk["tools"].get("messages", []):
                        tool_name = getattr(message, "name", "unknown")
                        print(f"[{elapsed}s] 🔧 [{tool_name}] returned results", flush=True)

        # Extract report filename
        if last_chunk and "agent" in last_chunk:
            messages = last_chunk["agent"].get("messages", [])
            if messages:
                final_content = messages[-1].content
                if isinstance(final_content, str):
                    match = re.search(r'[\w_-]+\.md', final_content)
                    if match:
                        report_filename = match.group(0)

    except Exception as e:
        status = f"error: {str(e)}"
        if verbose:
            print(f"\n❌ Error: {e}", flush=True)

    duration = round(time.time() - start_time, 2)

    run_id = log_research_run(
        topic=topic,
        num_searches=run_metrics["num_searches"],
        num_pages_scraped=run_metrics["num_pages_scraped"],
        report_filename=report_filename,
        duration_seconds=duration,
        llm_used="claude-sonnet-4-5",
        status=status,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )

    if verbose:
        print("\n" + "-" * 50, flush=True)
        print(f"✅ Research complete!", flush=True)
        print(f"   Topic: {topic}", flush=True)
        print(f"   Searches made: {run_metrics['num_searches']}", flush=True)
        print(f"   Pages scraped: {run_metrics['num_pages_scraped']}", flush=True)
        print(f"   Duration: {duration}s", flush=True)
        print(f"   Tokens — Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {input_tokens + output_tokens:,}", flush=True)
        print(f"   Run ID: {run_id}", flush=True)

    return {
        "topic": topic,
        "report_filename": report_filename,
        "run_id": run_id,
        "duration": duration,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status": status
    }