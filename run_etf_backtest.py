import os

from src.data_loader import load_wrds_crsp_data
from src.etf_data import download_etf_returns
from src.etf_backtester import ETFStatArbBacktester
from run_backtest import clean_dataframes


if __name__ == "__main__":

    print("Loading WRDS / CRSP stock data...")

    returns, prices, volumes, market_caps = load_wrds_crsp_data(
        start_date="2002-01-01",
        end_date="2007-12-31"
    )

    print("Stock data loaded.")
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

    print("Cleaned stock data shapes:")
    print("returns:", returns.shape)
    print("prices:", prices.shape)
    print("volumes:", volumes.shape)
    print("market_caps:", market_caps.shape)

    etf_returns = download_etf_returns(
        start_date="2002-01-01",
        end_date="2007-12-31"
    )

    print("Running ETF-factor statistical arbitrage backtest...")

    backtester = ETFStatArbBacktester(
        returns_lookback=252,
        ou_lookback=60,
        entry_z=1.25,
        exit_z_long=0.50,
        exit_z_short=0.75,
        transaction_cost_bps=0,
        gross_leverage=2.0,
        execution_lag_days=1
    )

    result = backtester.run(
        returns=returns,
        etf_returns=etf_returns
    )

    print("\nETF backtest finished.")
    print("\nPerformance summary:")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")

    os.makedirs("data/results_etf", exist_ok=True)

    result["equity_curve"].to_csv("data/results_etf/equity_curve_etf.csv")
    result["daily_returns"].to_csv("data/results_etf/daily_returns_etf.csv")
    result["positions"].to_csv("data/results_etf/positions_etf.csv")
    result["scores"].to_csv("data/results_etf/scores_etf.csv")

    print("\nSaved ETF results to:")
    print("data/results_etf/equity_curve_etf.csv")
    print("data/results_etf/daily_returns_etf.csv")
    print("data/results_etf/positions_etf.csv")
    print("data/results_etf/scores_etf.csv")