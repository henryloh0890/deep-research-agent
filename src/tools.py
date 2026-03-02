import os
import datetime
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import tool

# Shared metrics tracker
run_metrics = {
    "num_searches": 0,
    "num_pages_scraped": 0,
    "scraped_urls": []
}

def reset_metrics():
    run_metrics["num_searches"] = 0
    run_metrics["num_pages_scraped"] = 0
    run_metrics["scraped_urls"] = []    

@tool
def search_web(query: str) -> str:
    """Search the web for information on a given query."""
    run_metrics["num_searches"] += 1
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5, timeout=10))
            if not results:
                return "No results found."
            formatted = []
            for r in results:
                formatted.append(f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}\n")
            return "\n---\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def scrape_webpage(url: str) -> str:
    """Scrape the full text content of a webpage given its URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        soup = BeautifulSoup(response.text, "html.parser")
        
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        text = soup.get_text(separator="\n", strip=True)
        
        run_metrics["num_pages_scraped"] += 1
        run_metrics["scraped_urls"].append(url)
        
        return text[:3000] if len(text) > 3000 else text
    
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

@tool
def save_report(content: str, filename: str = "") -> str:
    """Save a research report as a markdown file."""
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.md"
    
    os.makedirs(reports_dir, exist_ok=True)
    filepath = os.path.join(reports_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return f"Report saved to {filepath} with filename {filename}"

# Export tools list
tools = [search_web, scrape_webpage, save_report]