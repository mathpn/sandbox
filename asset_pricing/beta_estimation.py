import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class BetaEstimator:
    """Estimate market-risk beta for Brazilian ETFs using CAPM model"""

    def __init__(self, market_index="^BVSP"):
        """Initialize with market index (default: Ibovespa)"""
        self.market_index = market_index
        self.market_data = None
        self.results = {}

    def fetch_data(self, tickers, start_date, end_date):
        """Fetch historical price data from Yahoo Finance"""
        data = {}

        print(f"Fetching {self.market_index}...")
        self.market_data = yf.download(
            self.market_index, start=start_date, end=end_date, progress=False
        )["Close"]  # XXX Adj Close

        for ticker in tickers:
            print(f"Fetching {ticker}...")
            data[ticker] = yf.download(
                ticker, start=start_date, end=end_date, progress=False
            )["Close"]  # XXX Adj Close

        return data

    def calculate_returns(self, prices, frequency="daily"):
        """Calculate returns from price series (daily or weekly)"""
        if frequency == "weekly":
            prices = prices.resample("W").last()

        returns = prices.pct_change().dropna()
        return returns

    def estimate_beta(self, etf_ticker, etf_data, frequency="daily"):
        """Estimate beta using OLS regression"""
        market_returns = self.calculate_returns(self.market_data, frequency)
        etf_returns = self.calculate_returns(etf_data, frequency)

        # TODO inner join fix
        df = pd.merge(
            market_returns.rename(columns={self.market_index: "market"}),
            etf_returns.rename(columns={etf_ticker: "etf"}),
            on="Date",
            how="inner",
        ).dropna()
        print(df.head())

        if len(df) < 30:
            print(f"Warning: Only {len(df)} observations for {etf_ticker}")

        # OLS regression: etf_returns = alpha + beta * market_returns
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df["market"], df["etf"]
        )

        result = {
            "ticker": etf_ticker,
            "beta": slope,
            "alpha": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_error": std_err,
            "observations": len(df),
            "correlation": df["market"].corr(df["etf"]),
            "etf_volatility": df["etf"].std()
            * np.sqrt(252 if frequency == "daily" else 52),
            "market_volatility": df["market"].std()
            * np.sqrt(252 if frequency == "daily" else 52),
            "returns_data": df,
        }

        self.results[etf_ticker] = result
        return result

    def plot_regression(self, etf_ticker):
        """Plot regression line and scatter plot"""
        if etf_ticker not in self.results:
            print(f"No results for {etf_ticker}")
            return

        result = self.results[etf_ticker]
        df = result["returns_data"]

        plt.figure(figsize=(10, 6))
        plt.scatter(df["market"], df["etf"], alpha=0.5, s=20)

        x_line = np.linspace(df["market"].min(), df["market"].max(), 100)
        y_line = result["alpha"] + result["beta"] * x_line
        plt.plot(x_line, y_line, "r-", linewidth=2, label=f"β = {result['beta']:.3f}")

        plt.xlabel("Market Returns (Ibovespa)")
        plt.ylabel(f"ETF Returns ({etf_ticker})")
        plt.title(f"CAPM Beta Estimation: {etf_ticker}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        plt.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

        stats_text = f"R² = {result['r_squared']:.3f}\n"
        stats_text += f"α = {result['alpha'] * 100:.3f}%\n"
        stats_text += f"n = {result['observations']}"
        plt.text(
            0.05,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.show()

    def summary_table(self):
        """Create summary table of all estimated betas"""
        if not self.results:
            print("No results to display")
            return None

        summary = pd.DataFrame(
            [
                {
                    "Ticker": r["ticker"],
                    "Beta": r["beta"],
                    "Alpha (%)": r["alpha"] * 100,
                    "R²": r["r_squared"],
                    "Correlation": r["correlation"],
                    "ETF Vol (%)": r["etf_volatility"] * 100,
                    "Observations": r["observations"],
                }
                for r in self.results.values()
            ]
        )

        return summary


if __name__ == "__main__":
    # TODO excess returns
    estimator = BetaEstimator(market_index="^BVSP")

    etfs = [
        "^BVSP",  # sanity check
        "BOVA11.SA",
        "SMAL11.SA",
        "IVVB11.SA",
        "DIVO11.SA",
        "GOLD11.SA",
        "HASH11.SA",
        "AUVP11.SA",
    ]

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    print("Fetching data...")
    etf_data = estimator.fetch_data(etfs, start_date, end_date)

    print("\nEstimating betas...")
    for ticker, data in etf_data.items():
        result = estimator.estimate_beta(ticker, data, frequency="daily")
        print(f"\n{ticker}:")
        print(f"  Beta: {result['beta']:.3f}")
        print(f"  R²: {result['r_squared']:.3f}")
        print(f"  Observations: {result['observations']}")

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    summary = estimator.summary_table()
    print(summary.to_string(index=False))

    for etf in etfs:
        estimator.plot_regression(etf)
