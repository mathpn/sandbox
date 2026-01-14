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
    # SELIC comes as % per day, convert to decimal
    selic_returns = selic_daily_pct["selic"] / 100
    return selic_returns


# %%


def returns_to_price_index(returns, initial_value=100):
    """Convert a return series to a synthetic price index."""
    return initial_value * (1 + returns).cumprod()


def calculate_rolling_returns_calendar_days(prices, window_days):
    """
    Calculate rolling returns based on calendar days.

    For each date t, calculates: (price[t] / price[t - window_days]) - 1
    where window_days is in calendar time. Uses the closest available trading
    day at or before the lookback date.
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()

    df = pd.DataFrame({"price": prices})

    df["lookback_date"] = pd.to_datetime(df.index) - pd.Timedelta(days=window_days)

    df_hist = pd.DataFrame({"price_past": prices})
    df_hist["date"] = df_hist.index

    # For each lookback date, find the price at the closest available trading day
    # direction='backward' means we take the last available price at or before the lookback date
    merged = pd.merge_asof(
        df.reset_index(),
        df_hist,
        left_on="lookback_date",
        right_on="date",
        direction="backward",
    )

    rolling_returns = (merged["price"] / merged["price_past"]) - 1
    rolling_returns.index = prices.index

    return rolling_returns


# %%


def plot_cumulative_returns(prices_dict):
    plt.figure(figsize=(12, 6))

    for label, prices in prices_dict.items():
        cumulative_returns = (prices / prices.iloc[0]) - 1
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

# Convert SELIC returns to a synthetic price index
selic_returns = calculate_selic_daily_returns(selic_data)
selic_prices = returns_to_price_index(selic_returns, initial_value=100)

# %%

common_index = bvsp_prices.index.intersection(selic_prices.index)
bvsp_prices_aligned = bvsp_prices.loc[common_index]
selic_prices_aligned = selic_prices.loc[common_index]

# %%

total_selic = (selic_prices_aligned.iloc[-1] / selic_prices_aligned.iloc[0]) - 1
total_bvsp = ((bvsp_prices_aligned.iloc[-1] / bvsp_prices_aligned.iloc[0]) - 1).iloc[0]
print(f"Total SELIC return: {total_selic * 100:.2f}%")
print(f"Total BVSP return: {total_bvsp * 100:.2f}%")
print(f"Period: {common_index[0].date()} to {common_index[-1].date()}")

# %%

prices_dict = {"^BVSP": bvsp_prices_aligned, "SELIC": selic_prices_aligned}
plot_cumulative_returns(prices_dict)

# %%


def plot_rolling_returns(prices_dict, window_days=21):
    plt.figure(figsize=(12, 6))

    for label, prices in prices_dict.items():
        rolling_returns = calculate_rolling_returns_calendar_days(prices, window_days)
        plt.plot(rolling_returns.index, rolling_returns * 100, label=label)

    plt.title(f"Rolling {window_days}-Day Returns Comparison")
    plt.xlabel("Date")
    plt.ylabel("Rolling Monthly Return (%)")
    plt.legend()
    plt.grid(True)
    plt.show()


# %%

plot_rolling_returns(prices_dict, window_days=365)

# %%
