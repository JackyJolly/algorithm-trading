import os
import pandas as pd

from src.data_loader import load_wrds_crsp_data
from src.backtester import StatisticalArbitrageBacktester
from src.etf_backtester import ETFStatArbBacktester
from src.etf_data import download_etf_returns
from run_backtest import clean_dataframes


def run_pca_case(returns, entry_z, transaction_cost_bps, n_factors=15):
    print(
        f"\nRunning PCA | entry_z={entry_z} | "
        f"cost={transaction_cost_bps} bps | n_factors={n_factors}"
    )

    backtester = StatisticalArbitrageBacktester(
        returns_lookback=252,
        ou_lookback=60,
        n_factors=n_factors,
        entry_z=entry_z,
        exit_z_long=0.50,
        exit_z_short=0.75,
        transaction_cost_bps=transaction_cost_bps,
        gross_leverage=2.0,
        execution_lag_days=1
    )

    result = backtester.run(returns=returns)

    row = {
        "Strategy": "PCA",
        "Entry Z": entry_z,
        "Transaction Cost bps": transaction_cost_bps,
        "N Factors": n_factors,
        **result["metrics"]
    }

    return row


def run_etf_case(returns, etf_returns, entry_z, transaction_cost_bps):
    print(
        f"\nRunning ETF | entry_z={entry_z} | "
        f"cost={transaction_cost_bps} bps"
    )

    backtester = ETFStatArbBacktester(
        returns_lookback=252,
        ou_lookback=60,
        entry_z=entry_z,
        exit_z_long=0.50,
        exit_z_short=0.75,
        transaction_cost_bps=transaction_cost_bps,
        gross_leverage=2.0,
        execution_lag_days=1
    )

    result = backtester.run(
        returns=returns,
        etf_returns=etf_returns
    )

    row = {
        "Strategy": "ETF",
        "Entry Z": entry_z,
        "Transaction Cost bps": transaction_cost_bps,
        "N Factors": None,
        **result["metrics"]
    }

    return row


if __name__ == "__main__":

    os.makedirs("data/results_sensitivity", exist_ok=True)

    print("Loading WRDS / CRSP stock data...")

    returns, prices, volumes, market_caps = load_wrds_crsp_data(
        start_date="2002-01-01",
        end_date="2007-12-31"
    )

    returns, prices, volumes, market_caps = clean_dataframes(
        returns,
        prices,
        volumes,
        market_caps
    )

    print("Cleaned returns shape:", returns.shape)

    print("Loading ETF returns...")
    etf_returns = download_etf_returns(
        start_date="2002-01-01",
        end_date="2007-12-31"
    )

    rows = []

    # Baseline no-cost cases
    rows.append(run_pca_case(
        returns=returns,
        entry_z=1.25,
        transaction_cost_bps=0,
        n_factors=15
    ))

    rows.append(run_etf_case(
        returns=returns,
        etf_returns=etf_returns,
        entry_z=1.25,
        transaction_cost_bps=0
    ))

    # Low-cost cases
    rows.append(run_pca_case(
        returns=returns,
        entry_z=1.75,
        transaction_cost_bps=1,
        n_factors=15
    ))

    rows.append(run_etf_case(
        returns=returns,
        etf_returns=etf_returns,
        entry_z=1.75,
        transaction_cost_bps=1
    ))

    # Paper-cost cases
    rows.append(run_pca_case(
        returns=returns,
        entry_z=2.00,
        transaction_cost_bps=5,
        n_factors=15
    ))

    rows.append(run_etf_case(
        returns=returns,
        etf_returns=etf_returns,
        entry_z=2.00,
        transaction_cost_bps=5
    ))

    # PCA factor sensitivity
    for n_factors in [5, 10, 20]:
        rows.append(run_pca_case(
            returns=returns,
            entry_z=1.75,
            transaction_cost_bps=1,
            n_factors=n_factors
        ))

    sensitivity_table = pd.DataFrame(rows)

    sensitivity_table.to_csv(
        "data/results_sensitivity/sensitivity_table.csv",
        index=False
    )

    print("\nSensitivity analysis finished.")
    print(sensitivity_table)

    print("\nSaved to:")
    print("data/results_sensitivity/sensitivity_table.csv")