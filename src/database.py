import os
import sqlite3
import datetime
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "research_history.db")

def init_database():
    """Create the database and tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            topic TEXT NOT NULL,
            num_searches INTEGER,
            num_pages_scraped INTEGER,
            report_filename TEXT,
            duration_seconds REAL,
            llm_used TEXT,
            status TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0
        )
    """)
    
    # Migrate existing database if needed
    existing_columns = [
        row[1] for row in cursor.execute("PRAGMA table_info(research_runs)")
    ]
    for col in ["input_tokens", "output_tokens", "total_tokens"]:
        if col not in existing_columns:
            cursor.execute(f"ALTER TABLE research_runs ADD COLUMN {col} INTEGER DEFAULT 0")
    
    conn.commit()
    conn.close()

def log_research_run(topic, num_searches, num_pages_scraped,
                     report_filename, duration_seconds, llm_used,
                     status, input_tokens=0, output_tokens=0):
    """Log a completed research run to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO research_runs 
        (timestamp, topic, num_searches, num_pages_scraped,
         report_filename, duration_seconds, llm_used, status,
         input_tokens, output_tokens, total_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().isoformat(),
        topic,
        num_searches,
        num_pages_scraped,
        report_filename,
        duration_seconds,
        llm_used,
        status,
        input_tokens,
        output_tokens,
        input_tokens + output_tokens
    ))
    
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id

def load_history() -> pd.DataFrame:
    """Load all research runs into a pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM research_runs", conn)
    conn.close()
    return df

def print_stats():
    """Print a summary of all research runs."""
    df = load_history()
    if df.empty:
        print("No research runs yet.")
        return
    
    print("\n=== Research History ===")
    print(df[["id", "timestamp", "topic", "num_searches", "duration_seconds", "status"]].to_string())
    
    print("\n=== Token Usage ===")
    print(df[["id", "topic", "input_tokens", "output_tokens", "total_tokens"]].to_string())
    
    print("\n=== Summary Stats ===")
    print(f"Total runs:               {len(df)}")
    print(f"Successful runs:          {len(df[df['status'] == 'success'])}")
    print(f"Avg duration:             {df['duration_seconds'].mean():.1f}s")
    print(f"Avg searches per run:     {df['num_searches'].mean():.1f}")
    print(f"Total Claude tokens used: {df['total_tokens'].sum():,}")
    print(f"Avg tokens per run:       {df['total_tokens'].mean():.0f}")