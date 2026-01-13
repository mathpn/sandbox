# %%

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from bcb import sgs

# %%


def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data["Close"]


def fetch_risk_free_rate(start_date, end_date):
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


# %%


def calculate_returns(prices):
    returns = prices.pct_change().dropna()
    return returns


def calculate_selic_daily_returns(selic_daily_pct):
    """
    Convert SELIC daily rates (in % per day) to daily returns.
    BCB provides SELIC as % per day, so we divide by 100.
    """
    # SELIC comes as % per day, convert to decimal
    selic_returns = selic_daily_pct["selic"] / 100
    return selic_returns


# %%


def plot_cumulative_returns(returns_dict):
    """Plot total cumulative returns using compound returns formula."""
    plt.figure(figsize=(12, 6))

    for label, returns in returns_dict.items():
        # Calculate cumulative returns: (1 + r1) * (1 + r2) * ... - 1
        cumulative_returns = (1 + returns).cumprod() - 1
        # Convert to percentage for display
        plt.plot(cumulative_returns.index, cumulative_returns * 100, label=label)

    plt.title("Total Cumulative Returns Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True)
    plt.show()


# %%

end_date = datetime.now()
start_date = end_date - timedelta(days=25 * 365)
bvsp_prices = fetch_data("^BVSP", start_date, end_date)

# %%

selic_data = fetch_risk_free_rate(start_date, end_date)

# %%

bvsp_returns = calculate_returns(bvsp_prices)
selic_returns = calculate_selic_daily_returns(selic_data)

# %%

# Align the two return series
common_index = bvsp_returns.index.intersection(selic_returns.index)
bvsp_returns_aligned = bvsp_returns.loc[common_index]
selic_returns_aligned = selic_returns.loc[common_index]

# %%

# Print total returns for verification
total_selic = (1 + selic_returns_aligned).prod() - 1
total_bvsp = ((1 + bvsp_returns_aligned).prod() - 1).iloc[0]
print(f"Total SELIC return: {total_selic * 100:.2f}%")
print(f"Total BVSP return: {total_bvsp * 100:.2f}%")
print(f"Period: {common_index[0].date()} to {common_index[-1].date()}")

# %%

returns_dict = {"^BVSP": bvsp_returns_aligned, "SELIC": selic_returns_aligned}
plot_cumulative_returns(returns_dict)

# %%


def plot_rolling_returns(returns_dict, window_days=21):
    plt.figure(figsize=(12, 6))

    for label, returns in returns_dict.items():
        # Calculate rolling monthly returns using compound formula
        rolling_returns = (1 + returns).rolling(window=window_days).apply(
            lambda x: x.prod(), raw=True
        ) - 1
        # Convert to percentage for display
        plt.plot(rolling_returns.index, rolling_returns * 100, label=label)

    plt.title(f"Rolling {window_days}-Day Returns Comparison")
    plt.xlabel("Date")
    plt.ylabel("Rolling Monthly Return (%)")
    plt.legend()
    plt.grid(True)
    plt.show()


# %%

plot_rolling_returns(returns_dict, window_days=365)

# %%
