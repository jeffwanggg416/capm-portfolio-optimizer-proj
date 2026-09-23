import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

DB_PATH = Path(__file__).parent.parent / "database" / "market_data.db"


def get_connection() -> sqlite3.Connection:
    """Open a connection to the SQLite database, creating the file if needed."""
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database() -> None:
    """Create tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS securities (
            ticker TEXT PRIMARY KEY,
            first_available_date TEXT,
            last_updated TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            adjusted_close REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)

    conn.commit()
    conn.close()


def get_last_updated(ticker: str) -> str | None:
    """Return the most recent date stored for a ticker, or None if not tracked yet."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT MAX(date) FROM price_history WHERE ticker = ?", (ticker,)
    )
    result = cursor.fetchone()[0]
    conn.close()
    return result


def save_price_history(ticker: str, df: pd.DataFrame) -> None:
    """Insert or update price rows for a ticker. df must have an 'Adj Close' column and a date index."""
    conn = get_connection()
    cursor = conn.cursor()

    # Drop any rows where Adj Close is missing before inserting
    clean_df = df.dropna(subset=["Adj Close"])

    if clean_df.empty:
        conn.close()
        raise ValueError(f"No valid adjusted close data available for '{ticker}' after removing missing values.")

    rows = [
        (ticker, date.strftime("%Y-%m-%d"), float(row["Adj Close"]))
        for date, row in clean_df.iterrows()
    ]

    cursor.executemany(
        """
        INSERT OR REPLACE INTO price_history (ticker, date, adjusted_close)
        VALUES (?, ?, ?)
        """,
        rows,
    )

    now = datetime.now().isoformat()
    cursor.execute(
        """
        INSERT INTO securities (ticker, first_available_date, last_updated)
        VALUES (?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET last_updated = excluded.last_updated
        """,
        (ticker, clean_df.index.min().strftime("%Y-%m-%d"), now),
    )

    conn.commit()
    conn.close()


def load_price_history(ticker: str) -> pd.DataFrame:
    """Load all cached price history for a ticker as a DataFrame indexed by date."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT date, adjusted_close FROM price_history WHERE ticker = ? ORDER BY date",
        conn,
        params=(ticker,),
        parse_dates=["date"],
        index_col="date",
    )
    conn.close()
    return df


def needs_update(ticker: str, max_staleness_days: int = 1) -> bool:
    """Check whether a ticker's cached data is stale and needs a refresh."""
    last_date_str = get_last_updated(ticker)
    if last_date_str is None:
        return True

    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
    return (datetime.now() - last_date) > timedelta(days=max_staleness_days)