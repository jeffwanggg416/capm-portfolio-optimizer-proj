import yfinance as yf

DEFENSIVE_ASSET = "VTIP"


def parse_ticker_input(raw_input: str) -> list[str]:
    """
    Convert a raw comma-separated string into a clean list of tickers:
    uppercase, whitespace-stripped, duplicates removed, order preserved.
    """
    if not raw_input or not raw_input.strip():
        raise ValueError("Ticker input cannot be empty.")

    raw_tickers = raw_input.split(",")
    cleaned = [t.strip().upper() for t in raw_tickers if t.strip()]

    seen = set()
    deduped = []
    for ticker in cleaned:
        if ticker not in seen:
            seen.add(ticker)
            deduped.append(ticker)

    return deduped


def add_defensive_asset(tickers: list[str]) -> list[str]:
    """Ensure VTIP is present in the universe, without duplicating it."""
    if DEFENSIVE_ASSET not in tickers:
        tickers = tickers + [DEFENSIVE_ASSET]
    return tickers


def is_valid_ticker(ticker: str) -> bool:
    """
    Check whether a ticker exists and has retrievable price data.
    Uses a very short history request rather than the full 5 years,
    since we only need to confirm the ticker is real.
    """
    try:
        test_data = yf.Ticker(ticker).history(period="5d")
        return not test_data.empty
    except Exception:
        return False


def validate_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    """
    Split a list of tickers into (valid, invalid) based on whether
    Yahoo Finance actually has data for each one.
    """
    valid = []
    invalid = []

    for ticker in tickers:
        if is_valid_ticker(ticker):
            valid.append(ticker)
        else:
            invalid.append(ticker)

    return valid, invalid


def build_universe(raw_input: str) -> list[str]:
    """
    Full pipeline: parse user input, add VTIP, validate everything,
    and raise a clear error if any tickers are invalid.
    """
    tickers = parse_ticker_input(raw_input)
    tickers = add_defensive_asset(tickers)

    valid, invalid = validate_tickers(tickers)

    if invalid:
        raise ValueError(
            f"The following tickers could not be found: {', '.join(invalid)}. "
            f"Please check the symbols and try again."
        )

    return valid

if __name__ == "__main__":
    test_input = "aapl, msft, AAPL, nvda"

    try:
        universe = build_universe(test_input)
        print("Valid universe:", universe)
    except ValueError as e:
        print("Validation error:", e)