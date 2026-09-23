import yfinance as yf
import pandas as pd
from src import database


def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    security = yf.Ticker(ticker)
    history = security.history(period=period, auto_adjust=False)

    if history.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. It may be invalid or delisted.")

    return history


def get_cached_or_fresh_prices(ticker: str) -> pd.DataFrame:
    """Return price history for a ticker, using the SQLite cache when possible."""
    database.initialize_database()

    if database.needs_update(ticker):
        print(f"Fetching fresh data for {ticker}...")
        fresh_data = fetch_price_history(ticker)
        database.save_price_history(ticker, fresh_data)
    else:
        print(f"Using cached data for {ticker}.")

    return database.load_price_history(ticker)


if __name__ == "__main__":
    df = get_cached_or_fresh_prices("AAPL")
    print(df.head())
    print(df.tail())
    print(f"\nTotal rows: {len(df)}")

def fetch_risk_free_rate() -> float:
    """
    Fetch the current 3-month Treasury bill yield from Yahoo Finance
    (^IRX) and return it as a decimal (e.g. 0.0403 for 4.03%),
    rounded to 5 decimal places.

    Falls back to config.RISK_FREE_RATE_FALLBACK if the fetch fails.
    """
    from config import RISK_FREE_RATE_FALLBACK

    try:
        irx = yf.Ticker("^IRX")
        recent = irx.history(period="5d")

        if recent.empty:
            raise ValueError("No data returned for ^IRX")

        latest_yield_pct = recent["Close"].iloc[-1]
        return round(latest_yield_pct / 100, 5)

    except Exception as e:
        print(f"Warning: could not fetch live risk-free rate ({e}). Using fallback: {RISK_FREE_RATE_FALLBACK}")
        return RISK_FREE_RATE_FALLBACK