from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


class BetaEstimator:
    """Estimate market-risk beta for Brazilian assets using CAPM model"""

    def __init__(self, market_index: str):
        """Initialize with market index ticker"""
        self.market_index = market_index
        self.market_data = None
        self.risk_free_rate = None
        self.results = {}

    def fetch_data(self, tickers, start_date, end_date):
        """Fetch historical price data from Yahoo Finance"""
        data = {}

        print(f"Fetching {self.market_index}...")
        market_df = yf.download(
            self.market_index, start=start_date, end=end_date, progress=False
        )
        self.market_data = market_df["Close"].squeeze()

        for ticker in tickers:
            print(f"Fetching {ticker}...")
            ticker_df = yf.download(
                ticker, start=start_date, end=end_date, progress=False
            )
            data[ticker] = ticker_df["Close"].squeeze()

        return data

    def fetch_risk_free_rate(self, start_date, end_date):
        """
        Fetch SELIC rate from Brazilian Central Bank and convert to daily returns.

        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format

        Returns:
        --------
        pd.Series : Daily risk-free rate returns with DatetimeIndex
        """
        from bcb import sgs

        print(f"Fetching SELIC rate from BCB...")

        # BCB Series 11 (daily over-SELIC rate)
        # Returns annual percentage rate
        selic_data = sgs.get({"selic": 11}, start=start_date, end=end_date)

        # Convert annual percentage to daily decimal returns
        # Formula: (1 + r_annual/100)^(1/252) - 1
        # 252 = typical trading days per year in Brazil
        selic_daily = (1 + selic_data["selic"] / 100) ** (1 / 252) - 1

        # Forward-fill for weekends/holidays (SELIC doesn't change)
        selic_daily = selic_daily.ffill()

        return selic_daily

    def calculate_returns(self, prices, frequency="daily", log_returns=True):
        """
        Calculate returns from price series.

        Parameters:
        -----------
        prices : pd.Series
            Price time series with DatetimeIndex
        frequency : str
            'daily' or 'weekly'
        log_returns : bool
            If True, calculate log returns (recommended).
            If False, simple returns.

        Returns:
        --------
        pd.Series : Returns time series

        Notes:
        ------
        Log returns are preferred because:
        - Time-additive: log(P_t/P_0) = sum of log returns
        - More symmetric (up/down movements)
        - Better statistical properties (closer to normal distribution)
        """
        if frequency == "weekly":
            prices = prices.resample("W").last()

        if log_returns:
            # Log returns: ln(P_t / P_{t-1})
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            # Simple returns: (P_t - P_{t-1}) / P_{t-1}
            returns = prices.pct_change().dropna()

        return returns

    def estimate_beta(
        self,
        etf_ticker,
        etf_data,
        frequency="daily",
        use_robust_se=True,
        outlier_method="winsorize",
    ):
        """
        Estimate CAPM beta using OLS regression on excess returns.

        The CAPM model: E[R_i - R_f] = alpha + beta * E[R_m - R_f]

        Parameters:
        -----------
        etf_ticker : str
            ETF ticker symbol
        etf_data : pd.Series
            ETF price data
        frequency : str
            'daily' or 'weekly'
        use_robust_se : bool
            If True, use heteroskedasticity-robust standard errors (HC1)
        outlier_method : str or None
            'winsorize', 'zscore', 'iqr', or None

        Returns:
        --------
        dict : Beta estimation results with confidence intervals
        """
        import statsmodels.api as sm

        market_returns = self.calculate_returns(
            self.market_data, frequency, log_returns=True
        )
        etf_returns = self.calculate_returns(etf_data, frequency, log_returns=True)

        # SELIC is already a daily rate, not prices - just resample if needed
        if frequency == "weekly":
            rf_rate = self.risk_free_rate.resample("W").last()
        else:
            rf_rate = self.risk_free_rate

        market_df = market_returns.to_frame(name="market")
        etf_df = etf_returns.to_frame(name="etf")
        rf_df = rf_rate.to_frame(name="rf")

        df = market_df.join(etf_df, how="inner").join(rf_df, how="inner").dropna()

        df["market_excess"] = df["market"] - df["rf"]
        df["etf_excess"] = df["etf"] - df["rf"]

        if len(df) < 30:
            print(f"Warning: Only {len(df)} observations for {etf_ticker}")
            return None

        if outlier_method:
            df = self.detect_outliers(df, method=outlier_method)
            if len(df) < 30:
                print(
                    f"Warning: After outlier removal, only {len(df)} observations for {etf_ticker}"
                )
                return None

        X = sm.add_constant(df["market_excess"])
        y = df["etf_excess"]

        if use_robust_se:
            model = sm.OLS(y, X).fit(cov_type="HC1")
        else:
            model = sm.OLS(y, X).fit()

        beta = model.params["market_excess"]
        alpha = model.params["const"]

        conf_int = model.conf_int(alpha=0.05)
        beta_ci_lower = conf_int.loc["market_excess", 0]
        beta_ci_upper = conf_int.loc["market_excess", 1]
        alpha_ci_lower = conf_int.loc["const", 0]
        alpha_ci_upper = conf_int.loc["const", 1]

        annualization_factor = 252 if frequency == "daily" else 52

        result = {
            "ticker": etf_ticker,
            "beta": beta,
            "beta_ci_lower": beta_ci_lower,
            "beta_ci_upper": beta_ci_upper,
            "alpha": alpha,
            "alpha_ci_lower": alpha_ci_lower,
            "alpha_ci_upper": alpha_ci_upper,
            "r_squared": model.rsquared,
            "p_value": model.pvalues["market_excess"],
            "std_error": model.bse["market_excess"],
            "observations": len(df),
            "correlation": df["market_excess"].corr(df["etf_excess"]),
            "etf_volatility": df["etf_excess"].std() * np.sqrt(annualization_factor),
            "market_volatility": df["market_excess"].std()
            * np.sqrt(annualization_factor),
            "returns_data": df,
            "model": model,
        }

        self.results[etf_ticker] = result
        return result

    def detect_outliers(self, returns_data, method="winsorize", limits=(0.01, 0.01)):
        """
        Detect and handle outliers in returns data.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            DataFrame with 'market_excess' and 'etf_excess' columns
        method : str
            'winsorize' - Cap extreme values at percentiles (recommended)
            'zscore' - Remove observations with |z| > 3
            'iqr' - Remove observations outside 1.5*IQR
        limits : tuple
            For winsorize: (lower_percentile, upper_percentile)
            Default (0.01, 0.01) = cap at 1st and 99th percentiles

        Returns:
        --------
        pd.DataFrame : Cleaned returns data

        Notes:
        ------
        Winsorization is recommended because it handles outliers without
        losing observations, which is important for beta stability.
        """
        from scipy import stats
        from scipy.stats.mstats import winsorize

        df = returns_data.copy()

        if method == "winsorize":
            # Cap extreme values at specified percentiles
            df["market_excess"] = winsorize(df["market_excess"], limits=limits)
            df["etf_excess"] = winsorize(df["etf_excess"], limits=limits)

        elif method == "zscore":
            # Remove observations with |z-score| > 3
            z_market = np.abs(stats.zscore(df["market_excess"]))
            z_etf = np.abs(stats.zscore(df["etf_excess"]))
            df = df[(z_market < 3) & (z_etf < 3)]

        elif method == "iqr":
            # Remove observations outside 1.5*IQR (Tukey's fences)
            for col in ["market_excess", "etf_excess"]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        return df

    def estimate_rolling_beta(self, etf_ticker, window=252, min_periods=None):
        """
        Estimate rolling beta over time using a sliding window.

        Useful for detecting changes in systematic risk over time.

        Parameters:
        -----------
        etf_ticker : str
            ETF ticker symbol
        window : int
            Rolling window size in days (default 252 = 1 trading year)
        min_periods : int
            Minimum observations required (default = window)

        Returns:
        --------
        pd.DataFrame : Rolling betas with date index

        Notes:
        ------
        Uses the covariance/variance formula: β = Cov(R_etf, R_market) / Var(R_market)
        This is faster than running OLS in each window.
        """
        if etf_ticker not in self.results:
            print(f"No results for {etf_ticker}. Run estimate_beta first.")
            return None

        # Get the excess returns data from previous estimation
        df = self.results[etf_ticker]["returns_data"]

        if min_periods is None:
            min_periods = window

        # Calculate rolling beta using covariance/variance formula
        # β = Cov(R_etf - Rf, R_market - Rf) / Var(R_market - Rf)
        rolling_cov = (
            df["etf_excess"]
            .rolling(window=window, min_periods=min_periods)
            .cov(df["market_excess"])
        )

        rolling_var = (
            df["market_excess"].rolling(window=window, min_periods=min_periods).var()
        )

        rolling_beta = rolling_cov / rolling_var

        rolling_results = pd.DataFrame(
            {"date": df.index, "rolling_beta": rolling_beta, "window": window}
        ).dropna()

        self.results[etf_ticker]["rolling_beta"] = rolling_results

        return rolling_results

    def plot_regression(self, etf_ticker, show_ci=True):
        """Plot regression line with confidence intervals"""
        if etf_ticker not in self.results:
            print(f"No results for {etf_ticker}")
            return

        result = self.results[etf_ticker]
        df = result["returns_data"]

        plt.figure(figsize=(10, 6))
        plt.scatter(df["market_excess"], df["etf_excess"], alpha=0.5, s=20)

        # Regression line
        x_line = np.linspace(df["market_excess"].min(), df["market_excess"].max(), 100)
        y_line = result["alpha"] + result["beta"] * x_line

        plt.plot(
            x_line,
            y_line,
            "r-",
            linewidth=2,
            label=f"β = {result['beta']:.3f} [{result['beta_ci_lower']:.3f}, {result['beta_ci_upper']:.3f}]",
        )

        # Add confidence interval band
        if show_ci:
            from scipy import stats as scipy_stats

            n = len(df)
            t_val = scipy_stats.t.ppf(0.975, n - 2)

            # Prediction standard error
            residuals = df["etf_excess"] - (
                result["alpha"] + result["beta"] * df["market_excess"]
            )
            se = np.sqrt(np.sum(residuals**2) / (n - 2))

            ci_upper = y_line + t_val * se
            ci_lower = y_line - t_val * se
            plt.fill_between(
                x_line, ci_lower, ci_upper, alpha=0.2, color="red", label="95% CI"
            )

        plt.xlabel("Market Excess Returns (Ibovespa - SELIC)")
        plt.ylabel(f"ETF Excess Returns ({etf_ticker} - SELIC)")
        plt.title(f"CAPM Beta Estimation: {etf_ticker}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        plt.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

        # Enhanced stats text with confidence intervals
        stats_text = f"β = {result['beta']:.3f}\n"
        stats_text += (
            f"95% CI: [{result['beta_ci_lower']:.3f}, {result['beta_ci_upper']:.3f}]\n"
        )
        stats_text += f"R² = {result['r_squared']:.3f}\n"
        stats_text += f"α = {result['alpha'] * 100:.3f}%\n"
        stats_text += f"p-value: {result['p_value']:.4f}\n"
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

    def plot_rolling_beta(self, etf_ticker):
        """Plot rolling beta over time"""
        if etf_ticker not in self.results:
            print(f"No results for {etf_ticker}")
            return

        if "rolling_beta" not in self.results[etf_ticker]:
            print(
                f"No rolling beta data for {etf_ticker}. Run estimate_rolling_beta first."
            )
            return

        rolling_data = self.results[etf_ticker]["rolling_beta"]
        static_beta = self.results[etf_ticker]["beta"]

        plt.figure(figsize=(12, 6))
        plt.plot(
            rolling_data["date"],
            rolling_data["rolling_beta"],
            linewidth=1.5,
            label=f'Rolling Beta (window={rolling_data["window"].iloc[0]}d)',
        )
        plt.axhline(
            y=static_beta,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Static Beta = {static_beta:.3f}",
        )
        plt.axhline(
            y=1.0,
            color="k",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label="Market Beta = 1.0",
        )

        plt.xlabel("Date")
        plt.ylabel("Beta")
        plt.title(f"Rolling Beta Estimation: {etf_ticker}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def summary_table(self):
        """Create enhanced summary table with confidence intervals and significance"""
        if not self.results:
            print("No results to display")
            return None

        summary = pd.DataFrame(
            [
                {
                    "Ticker": r["ticker"],
                    "Beta": f"{r['beta']:.3f}",
                    "Beta 95% CI": f"[{r['beta_ci_lower']:.3f}, {r['beta_ci_upper']:.3f}]",
                    "Alpha (%)": f"{r['alpha'] * 100:.3f}",
                    "R²": f"{r['r_squared']:.3f}",
                    "P-value": f"{r['p_value']:.4f}",
                    "Sig.": (
                        "***"
                        if r["p_value"] < 0.001
                        else (
                            "**"
                            if r["p_value"] < 0.01
                            else "*" if r["p_value"] < 0.05 else "ns"
                        )
                    ),
                    "ETF Vol (%)": f"{r['etf_volatility'] * 100:.2f}",
                    "Obs": r["observations"],
                }
                for r in self.results.values()
            ]
        )

        print(
            "\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
        )

        return summary


if __name__ == "__main__":
    estimator = BetaEstimator(market_index="^BVSP")

    etfs = [
        "^BVSP",  # Sanity check - should have beta ≈ 1.0
        "BOVA11.SA",
        "SMAL11.SA",
        "IVVB11.SA",
        "DIVO11.SA",
        "GOLD11.SA",
        "HASH11.SA",
    ]

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    print("=" * 70)
    print("COMPREHENSIVE CAPM BETA ESTIMATION")
    print("=" * 70)
    print("\nFetching market and ETF data from Yahoo Finance...")
    etf_data = estimator.fetch_data(etfs, start_date, end_date)

    print("\nFetching SELIC risk-free rate from Brazilian Central Bank...")
    estimator.risk_free_rate = estimator.fetch_risk_free_rate(start_date, end_date)
    print(f"SELIC data points: {len(estimator.risk_free_rate)}")

    print("\n" + "=" * 70)
    print("ESTIMATING BETAS (using excess returns over SELIC)")
    print("=" * 70)

    for ticker, data in etf_data.items():
        result = estimator.estimate_beta(
            ticker,
            data,
            frequency="daily",
            use_robust_se=True,
            outlier_method="winsorize",
        )

        if result:
            print(f"\n{ticker}:")
            print(
                f"  Beta: {result['beta']:.3f} (95% CI: [{result['beta_ci_lower']:.3f}, {result['beta_ci_upper']:.3f}])"
            )
            print(
                f"  Alpha: {result['alpha']*100:.3f}% per day ({result['alpha']*252*100:.2f}% annualized)"
            )
            print(f"  R²: {result['r_squared']:.3f}")
            print(
                f"  P-value: {result['p_value']:.4f} {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else '(ns)'}"
            )
            print(f"  Observations: {result['observations']}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    summary = estimator.summary_table()
    if summary is not None:
        print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("ROLLING BETA ESTIMATION (252-day window)")
    print("=" * 70)
    for ticker in etfs:
        if ticker in estimator.results:
            rolling = estimator.estimate_rolling_beta(ticker, window=252)
            if rolling is not None:
                print(f"{ticker}: {len(rolling)} periods calculated")

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    for etf in etfs:
        if etf in estimator.results:
            print(f"Plotting {etf}...")
            estimator.plot_regression(etf, show_ci=True)
            estimator.plot_rolling_beta(etf)

    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    if "^BVSP" in estimator.results:
        bvsp_beta = estimator.results["^BVSP"]["beta"]
        print(f"✓ Market vs itself (^BVSP): β = {bvsp_beta:.3f}")
        if 0.95 <= bvsp_beta <= 1.05:
            print("  → PASS: Beta is approximately 1.0 as expected")
        else:
            print("  → WARNING: Beta should be close to 1.0")
