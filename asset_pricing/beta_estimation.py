from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats.mstats import winsorize

from utils import fetch_br_risk_free_rate, fetch_yf_close_prices


def calculate_log_returns(prices, frequency="weekly"):
    """
    Calculate log returns from price series.

    Parameters:
    -----------
    prices : pd.Series
        Price time series with DatetimeIndex
    frequency : str
        'daily' or 'weekly'

    Returns:
    --------
    pd.Series : Log returns time series

    Notes:
    ------
    Log returns are preferred because:
    - Time-additive: log(P_t/P_0) = sum of log returns
    - More symmetric (up/down movements)
    - Better statistical properties (closer to normal distribution)
    """
    if frequency == "weekly":
        prices = prices.resample("W").last()

    # Log returns: ln(P_t / P_{t-1})
    returns = np.log(prices / prices.shift(1)).dropna()

    return returns


def prepare_risk_free_rate(risk_free_rate, frequency="weekly"):
    """
    Prepare risk-free rate data for given frequency.

    Parameters:
    -----------
    risk_free_rate : pd.Series
        Daily risk-free rate daily returns from BCB
    frequency : str
        'daily' or 'weekly'

    Returns:
    --------
    pd.Series : Risk-free rate log returns aligned to frequency

    Notes:
    ------
    SELIC data comes as daily simple returns (e.g., 0.0001 for 0.01%).
    We convert to log returns for comparability with market/ETF log returns.
    For weekly frequency, we compound the daily returns then convert to log return.
    """
    # Convert daily simple returns to log returns
    # log(1 + r) where r is the simple return
    rf_log_returns = np.log(1 + risk_free_rate)

    if frequency == "weekly":
        # Resample to weekly by summing log returns (log returns are additive)
        return rf_log_returns.resample("W").sum()

    return rf_log_returns


def detect_outliers(returns_data, method="winsorize", limits=(0.01, 0.01)):
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


def estimate_beta(
    etf_ticker,
    etf_prices,
    market_prices,
    risk_free_rate,
    frequency="weekly",
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
    etf_prices : pd.Series
        ETF price data
    market_prices : pd.Series
        Market index price data
    risk_free_rate : pd.Series
        Daily risk-free rate returns
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
    market_df = calculate_log_returns(market_prices, frequency)
    etf_df = calculate_log_returns(etf_prices, frequency)
    rf_df = prepare_risk_free_rate(risk_free_rate, frequency)

    market_df.columns = ["market"]
    etf_df.columns = ["etf"]
    rf_df.columns = ["rf"]

    df = market_df.join(etf_df, how="inner").join(rf_df, how="inner").dropna()

    df["market_excess"] = df["market"] - df["rf"]
    df["etf_excess"] = df["etf"] - df["rf"]

    if len(df) < 30:
        print(f"Warning: Only {len(df)} observations for {etf_ticker}")
        return None

    if outlier_method:
        df = detect_outliers(df, method=outlier_method)
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

    # XXX check trading weeks per year
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
        "market_volatility": df["market_excess"].std() * np.sqrt(annualization_factor),
        "returns_data": df,
        "model": model,
    }

    return result


def estimate_rolling_beta(returns_data, window=52, min_periods=None):
    """
    Estimate rolling beta over time using a sliding window.

    Useful for detecting changes in systematic risk over time.

    Parameters:
    -----------
    returns_data : pd.DataFrame
        DataFrame with 'market_excess' and 'etf_excess' columns from estimate_beta
    window : int
        Rolling window size in weeks (default 52 weeks = 1 trading year)
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
    if min_periods is None:
        min_periods = window

    # Calculate rolling beta using covariance/variance formula
    # beta = Cov(R_etf - Rf, R_market - Rf) / Var(R_market - Rf)
    rolling_cov = (
        returns_data["etf_excess"]
        .rolling(window=window, min_periods=min_periods)
        .cov(returns_data["market_excess"])
    )

    rolling_var = (
        returns_data["market_excess"]
        .rolling(window=window, min_periods=min_periods)
        .var()
    )

    rolling_beta = rolling_cov / rolling_var

    rolling_results = pd.DataFrame(
        {
            "date": returns_data.index,
            "rolling_beta": rolling_beta,
            "window": window,
        }
    ).dropna()

    return rolling_results


def plot_regression(result, show_ci=True):
    """
    Plot regression line with confidence intervals.

    Parameters:
    -----------
    result : dict
        Beta estimation result from estimate_beta
    show_ci : bool
        Show 95% confidence interval band
    """
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
    plt.ylabel(f"ETF Excess Returns ({result['ticker']} - SELIC)")
    plt.title(f"CAPM Beta Estimation: {result['ticker']}")
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


def plot_rolling_beta(rolling_data, static_beta, etf_ticker):
    """
    Plot rolling beta over time.

    Parameters:
    -----------
    rolling_data : pd.DataFrame
        Rolling beta data from estimate_rolling_beta
    static_beta : float
        Static beta value from estimate_beta
    etf_ticker : str
        ETF ticker symbol for labeling
    """
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


def plot_beta_comparison(results, figsize=(12, 12)):
    """
    Create a horizontal confidence interval plot comparing betas across multiple ETFs.

    Parameters:
    -----------
    results : list of dict
        List of beta estimation results from estimate_beta
    figsize : tuple
        Figure size (width, height)
    """
    if not results:
        print("No results to display")
        return None

    # Sort results by beta (descending - highest first)
    sorted_results = sorted(results, key=lambda x: x["beta"], reverse=True)
    # Remove index from results
    sorted_results = [
        r for r in sorted_results if not r.get("ticker", "").startswith("^")
    ]

    # Extract data
    tickers = [r["ticker"] for r in sorted_results]
    betas = [r["beta"] for r in sorted_results]
    beta_ci_lower = [r["beta_ci_lower"] for r in sorted_results]
    beta_ci_upper = [r["beta_ci_upper"] for r in sorted_results]
    observations = [r["observations"] for r in sorted_results]

    # Create figure with white background
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    fig.set_facecolor("#FAFAFA")  # Very light gray background
    ax.set_facecolor("#FAFAFA")

    # Y-axis positions for each ETF (reversed so highest beta is on top)
    y_pos = np.arange(len(tickers))[::-1]

    # Refined color palette
    ci_color = "#6BA3C5"  # Muted blue for CI lines
    point_color = "#2C5F7D"  # Deep blue for point estimate
    market_color = "#D67873"  # Muted coral for market line

    # Main plot: Beta confidence intervals
    for i, (ticker, beta, ci_low, ci_high) in enumerate(
        zip(tickers, betas, beta_ci_lower, beta_ci_upper)
    ):
        y = y_pos[i]

        # Draw horizontal line from CI lower to upper
        ax.plot(
            [ci_low, ci_high],
            [y, y],
            "-",
            linewidth=3,
            color=ci_color,
            alpha=0.7,
            solid_capstyle="round",
        )

        # Draw markers at CI endpoints
        ax.plot(
            [ci_low, ci_high],
            [y, y],
            "o",
            markersize=5.5,
            color=ci_color,
            alpha=0.7,
            markeredgewidth=0,
        )

        # Highlight the point estimate with a different marker
        ax.plot(
            beta,
            y,
            "o",
            markersize=8,
            color=point_color,
            zorder=5,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )

    # Add reference line at beta = 1
    ax.axvline(
        x=1.0,
        color=market_color,
        linestyle="--",
        linewidth=2.5,
        alpha=0.7,
        label="β = 1.0 (Market)",
        zorder=1,
    )

    # Formatting main plot
    x_max = max(beta_ci_upper) + 0.1
    ax.set_xlim(-0.5, x_max)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tickers, fontsize=11, color="#2C3E50")
    ax.set_xlabel("Beta", fontsize=13, color="#2C3E50", fontweight="500")
    ax.set_title(
        "Beta Estimates with 95% Confidence Intervals",
        fontsize=15,
        pad=20,
        color="#2C3E50",
        fontweight="600",
    )
    ax.grid(True, alpha=0.15, axis="x", linestyle="-", linewidth=0.8, color="#BDC3C7")
    ax.legend(
        loc="upper left",
        fontsize=10,
        framealpha=0.95,
        edgecolor="#BDC3C7",
        fancybox=True,
    )

    # Style spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDC3C7")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("#BDC3C7")
    ax.spines["bottom"].set_linewidth(1.2)

    # Style ticks
    ax.tick_params(colors="#7F8C8D", which="both", labelsize=10)

    # Add observation counts as text on the right side
    x_max = ax.get_xlim()[1]
    x_offset = x_max + (x_max - ax.get_xlim()[0]) * 0.04

    # Add subtle separator line
    ax.axvline(
        x=x_max + (x_max - ax.get_xlim()[0]) * 0.02,
        color="#BDC3C7",
        linestyle="-",
        linewidth=1,
        alpha=0.3,
        zorder=0,
    )

    # Add column header with subtle background
    header_y = y_pos[0] + 0.85
    ax.text(
        x_offset,
        header_y,
        "N obs",
        fontsize=10,
        fontweight="600",
        ha="left",
        va="bottom",
        color="#2C3E50",
    )

    # Add observation counts with alternating subtle backgrounds
    for i, (obs, y) in enumerate(zip(observations, y_pos)):
        # Alternating row backgrounds
        if i % 2 == 0:
            ax.axhspan(
                y - 0.4,
                y + 0.4,
                xmin=0.97,
                xmax=1.0,
                facecolor="#ECF0F1",
                alpha=0.3,
                zorder=0,
            )

        ax.text(
            x_offset,
            y,
            f"{obs}",
            va="center",
            ha="left",
            fontsize=10,
            color="#34495E",
            fontweight="500",
        )

    # Extend x-axis to make room for observation text
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] * 1.18)

    plt.tight_layout()
    return fig


def create_summary_table(results):
    """
    Create enhanced summary table with confidence intervals and significance.

    Parameters:
    -----------
    results : list of dict
        List of beta estimation results from estimate_beta

    Returns:
    --------
    pd.DataFrame : Summary table
    """
    if not results:
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
            for r in results
        ]
    )

    print(
        "\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
    )

    return summary


def main():
    market_index = "^BVSP"

    etfs = [
        "^BVSP",  # Sanity check - should have beta ≈ 1.0
        "BOVA11.SA",
        "IVVB11.SA",
        "HASH11.SA",
        "BOVV11.SA",
        "SPXR11.SA",
        "B5P211.SA",
        "IMAB11.SA",
        "GOLD11.SA",
        "BBOV11.SA",
        "BITH11.SA",
        "LLFT11.SA",
        "SMAL11.SA",
        "BOVB11.SA",
        "TECK11.SA",
        "LFTS11.SA",
        "LFTB11.SA",
        "SPXI11.SA",
        "DIVO11.SA",
        "PIBB11.SA",
        "IMBB11.SA",
        "BITI11.SA",
        "B5MB11.SA",
        "NASD11.SA",
        "QBTC11.SA",
        "BOVX11.SA",
        "IRFM11.SA",
        "PACG11.SA",
        "WRLD11.SA",
        "IBOB11.SA",
        "IB5M11.SA",
        "USAL11.SA",
        "PHIP11.SA",
        "ETHE11.SA",
        "MARG11.SA",
        "DEBB11.SA",
        "BOL511.SA",
        "XRPH11.SA",
        "DOLB11.SA",
        "UTEC11.SA",
        "COIN11.SA",
        "SPYI11.SA",
    ]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 10)

    print("=" * 70)
    print("COMPREHENSIVE CAPM BETA ESTIMATION")
    print("=" * 70)
    print("\nFetching market and ETF data from Yahoo Finance...")

    # Fetch market data
    print(f"Fetching {market_index}...")
    market_prices = fetch_yf_close_prices(market_index, start_date, end_date)

    # Fetch ETF data
    etf_prices_dict = {}
    for ticker in etfs:
        print(f"Fetching {ticker}...")
        etf_prices_dict[ticker] = fetch_yf_close_prices(ticker, start_date, end_date)

    print("\nFetching SELIC risk-free rate from Brazilian Central Bank...")
    risk_free_rate = fetch_br_risk_free_rate(start_date, end_date)
    print(f"SELIC data points: {len(risk_free_rate)}")

    print("\n" + "=" * 70)
    print("ESTIMATING BETAS (using excess returns over SELIC)")
    print("=" * 70)

    # Estimate betas for all ETFs
    results = []
    for ticker, etf_prices in etf_prices_dict.items():
        result = estimate_beta(
            ticker,
            etf_prices,
            market_prices,
            risk_free_rate,
            frequency="weekly",
            use_robust_se=False,
            outlier_method="",
        )

        if result:
            results.append(result)
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
    summary = create_summary_table(results)
    if summary is not None:
        print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("ROLLING BETA ESTIMATION (52-week window)")
    print("=" * 70)

    # Store rolling results alongside results
    rolling_results = {}
    for result in results:
        ticker = result["ticker"]
        rolling = estimate_rolling_beta(result["returns_data"])
        if rolling is not None:
            rolling_results[ticker] = rolling
            print(f"{ticker}: {len(rolling)} periods calculated")

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Beta comparison plot
    print("Plotting beta comparison...")
    plot_beta_comparison(results)
    plt.savefig("beta_comparison.png", dpi=300)
    plt.show()

    for result in results:
        ticker = result["ticker"]
        print(f"Plotting {ticker}...")
        plot_regression(result, show_ci=True)

        if ticker in rolling_results:
            plot_rolling_beta(rolling_results[ticker], result["beta"], ticker)

    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)

    # Find ^BVSP result
    bvsp_result = next((r for r in results if r["ticker"] == "^BVSP"), None)
    if bvsp_result:
        bvsp_beta = bvsp_result["beta"]
        print(f"✓ Market vs itself (^BVSP): β = {bvsp_beta:.3f}")
        if 0.95 <= bvsp_beta <= 1.05:
            print("  → PASS: Beta is approximately 1.0 as expected")
        else:
            print("  → WARNING: Beta should be close to 1.0")


if __name__ == "__main__":
    main()
