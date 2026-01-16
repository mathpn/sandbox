# %%

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd

from utils import fetch_br_risk_free_rate, fetch_yf_close_prices


def calculate_returns(prices):
    returns = prices.pct_change().dropna()
    return returns


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
bvsp_prices = fetch_yf_close_prices("^BVSP", start_date, end_date)

# %%

selic_data = fetch_br_risk_free_rate(start_date, end_date)

# %%

selic_prices = returns_to_price_index(selic_data, initial_value=100)

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
