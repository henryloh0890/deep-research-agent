from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Available models config
AVAILABLE_MODELS = {
    "Claude Sonnet 4.5 (Recommended)": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "description": "Best quality, paid"
    },
    "Claude Haiku 4.5 (Fast & Cheap)": {
        "provider": "anthropic", 
        "model": "claude-haiku-4-5-20251001",
        "description": "Faster, lower cost"
    },
    "Groq Llama 3.3 70B (Free)": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "description": "Free, fast"
    },
    "Groq Llama 3.1 8B (Free, Fastest)": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "description": "Free, fastest, lower quality"
    },
    "Gemini 2.0 Flash (Free Tier)": {
        "provider": "gemini",
        "model": "gemini-2.0-flash",
        "description": "Google, generous free tier"
    },
}

def build_llm(model_name: str):
    """Build an LLM instance from a model display name."""
    config = AVAILABLE_MODELS[model_name]
    provider = config["provider"]
    model = config["model"]

    if provider == "anthropic":
        return ChatAnthropic(model=model, temperature=0)
    
    elif provider == "groq":
        return ChatOpenAI(
            model=model,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
    
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def build_agent(model_name: str):
    """Build a fresh agent with the selected model."""
    llm = build_llm(model_name)
    return create_react_agent(llm, tools=tools)


def run_research(topic: str, model_name: str = "Claude Sonnet 4.6 (Recommended)", verbose: bool = True) -> dict:
    """
    Run a full research session on a given topic.
    
    Args:
        topic: The research topic
        model_name: Display name from AVAILABLE_MODELS
        verbose: Whether to print live progress
    
    Returns:
        dict with keys: topic, report_filename, run_id, duration, tokens, status
    """
    reset_metrics()

    agent = build_agent(model_name)
    provider = AVAILABLE_MODELS[model_name]["provider"]

    start_time = time.time()
    status = "success"
    report_filename = ""
    input_tokens = 0
    output_tokens = 0
    last_chunk = None

    if verbose:
        print(f"🚀 Starting research on: '{topic}'", flush=True)
        print(f"🤖 Model: {model_name}", flush=True)
        print("-" * 50, flush=True)

    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=build_prompt(topic))]},
            config={"recursion_limit": 25}
        ):
            last_chunk = chunk
            elapsed = round(time.time() - start_time, 1)

            if "agent" in chunk:
                for message in chunk["agent"].get("messages", []):
                    content = message.content

                    # Token tracking — works for Anthropic and OpenAI-compatible
                    usage = message.response_metadata.get("usage", {})
                    input_tokens += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

                    if verbose:
                        if isinstance(content, list):
                            for block in content:
                                if hasattr(block, "name"):
                                    print(f"[{elapsed}s] 🤖 Calling tool → {block.name}", flush=True)
                                    if hasattr(block, "input"):
                                        print(f"         Input: {str(block.input)[:120]}", flush=True)
                        elif isinstance(content, str) and content.strip():
                            print(f"[{elapsed}s] 🤖 Writing response...", flush=True)
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
        llm_used=model_name,
        status=status,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )

    if verbose:
        print("\n" + "-" * 50, flush=True)
        print(f"✅ Research complete!", flush=True)
        print(f"   Topic: {topic}", flush=True)
        print(f"   Model: {model_name}", flush=True)
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