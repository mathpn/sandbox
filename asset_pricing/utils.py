import pandas as pd
from datetime import timedelta
from bcb import sgs
import yfinance as yf


def fetch_yf_close_prices(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data["Close"]


def fetch_br_risk_free_rate(start_date, end_date):
    print("Fetching SELIC rate from BCB...")

    # BCB API only allows requests up to 10 years
    # Split the request into chunks if needed
    max_days_per_request = 365 * 10

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    total_days = (end_date - start_date).days

    if total_days <= max_days_per_request:
        selic_data = sgs.get({"selic": 11}, start=start_date, end=end_date)
        return selic_data

    print(f"Request spans {total_days / 365:.1f} years. Splitting into chunks...")

    all_data = []
    current_start = start_date

    while current_start < end_date:
        chunk_end = min(current_start + timedelta(days=max_days_per_request), end_date)

        print(f"  Fetching from {current_start.date()} to {chunk_end.date()}...")
        chunk_data = sgs.get({"selic": 11}, start=current_start, end=chunk_end)
        all_data.append(chunk_data)

        current_start = chunk_end + timedelta(days=1)

    selic_data = pd.concat(all_data)
    print(f"Successfully fetched {len(selic_data)} data points")

    return selic_data
