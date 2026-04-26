import os
import pandas as pd

from src.data_loader import load_wrds_crsp_data
from src.backtester import StatisticalArbitrageBacktester


def clean_dataframes(returns, prices, volumes, market_caps):
    """
    Clean WRDS/CRSP pivoted data.
    Rows = dates
    Columns = tickers / permnos
    """

    dfs = [returns, prices, volumes, market_caps]

    cleaned = []
    for df in dfs:
        df = df.copy()

        # Remove invalid column names
        df = df.loc[:, df.columns.notna()]

        # Remove duplicated columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Sort index
        df = df.sort_index()

        cleaned.append(df)

    returns, prices, volumes, market_caps = cleaned

    # Keep only common columns across all datasets
    common_cols = (
        returns.columns
        .intersection(prices.columns)
        .intersection(volumes.columns)
        .intersection(market_caps.columns)
    )

    returns = returns[common_cols]
    prices = prices[common_cols]
    volumes = volumes[common_cols]
    market_caps = market_caps[common_cols]

    # Convert returns to numeric
    returns = returns.apply(pd.to_numeric, errors="coerce")
    prices = prices.apply(pd.to_numeric, errors="coerce")
    volumes = volumes.apply(pd.to_numeric, errors="coerce")
    market_caps = market_caps.apply(pd.to_numeric, errors="coerce")

    # Drop columns with almost no return data
    min_obs = int(len(returns) * 0.70)
    valid_cols = returns.columns[returns.notna().sum() >= min_obs]

    returns = returns[valid_cols]
    prices = prices[valid_cols]
    volumes = volumes[valid_cols]
    market_caps = market_caps[valid_cols]

    # Fill missing returns with 0 for backtest stability
    returns = returns.fillna(0.0)

    return returns, prices, volumes, market_caps


if __name__ == "__main__":

    print("Loading WRDS / CRSP data...")

    returns, prices, volumes, market_caps = load_wrds_crsp_data(
        start_date="2002-01-01",
        end_date="2007-12-31"
    )

    print("Data loaded.")
    print("returns shape:", returns.shape)
    print("prices shape:", prices.shape)
    print("volumes shape:", volumes.shape)
    print("market caps shape:", market_caps.shape)

    returns, prices, volumes, market_caps = clean_dataframes(
        returns,
        prices,
        volumes,
        market_caps
    )

    print("Cleaned data shapes:")
    print("returns:", returns.shape)
    print("prices:", prices.shape)
    print("volumes:", volumes.shape)
    print("market_caps:", market_caps.shape)

    print("Running PCA-based statistical arbitrage backtest...")

    backtester = StatisticalArbitrageBacktester(
        returns_lookback=252,
        ou_lookback=60,
        n_factors=15,
        entry_z=1.25,
        exit_z_long=0.50,
        exit_z_short=0.75,
        transaction_cost_bps=0,
        gross_leverage=2.0,
        execution_lag_days=1
    )

    result = backtester.run(returns=returns)

    print("\nBacktest finished.")
    print("\nPerformance summary:")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")

    os.makedirs("data/results", exist_ok=True)

    result["equity_curve"].to_csv("data/results/equity_curve.csv")
    result["daily_returns"].to_csv("data/results/daily_returns.csv")
    result["positions"].to_csv("data/results/positions.csv")
    result["scores"].to_csv("data/results/scores.csv")

    print("\nSaved results to:")
    print("data/results/equity_curve.csv")
    print("data/results/daily_returns.csv")
    print("data/results/positions.csv")
    print("data/results/scores.csv")