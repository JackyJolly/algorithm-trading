import numpy as np
import pandas as pd

from src.signal_engine import SignalEngine


class ETFStatArbBacktester:
    def __init__(
        self,
        returns_lookback=252,
        ou_lookback=60,
        entry_z=1.25,
        exit_z_long=0.50,
        exit_z_short=0.75,
        transaction_cost_bps=0,
        gross_leverage=2.0,
        execution_lag_days=1
    ):
        """
        ETF-factor statistical arbitrage backtester.

        This version uses multiple sector ETFs as observable risk factors.
        For each rolling window:

            stock returns = alpha + ETF factors + residual

        Then the residuals are passed to the OU signal engine.
        """

        self.returns_lookback = returns_lookback
        self.ou_lookback = ou_lookback
        self.entry_z = entry_z
        self.exit_z_long = exit_z_long
        self.exit_z_short = exit_z_short
        self.transaction_cost_bps = transaction_cost_bps
        self.gross_leverage = gross_leverage
        self.execution_lag_days = execution_lag_days

        self.signal_engine = SignalEngine(
            ou_lookback=ou_lookback,
            entry_z=entry_z,
            exit_z_long=exit_z_long,
            exit_z_short=exit_z_short
        )

    def _clean_returns(self, returns):
        returns = returns.copy()
        returns = returns.loc[:, returns.columns.notna()]
        returns = returns.loc[:, ~returns.columns.duplicated()]
        returns = returns.sort_index()
        returns.index = pd.to_datetime(returns.index).tz_localize(None)

        returns = returns.apply(pd.to_numeric, errors="coerce")
        returns = returns.replace([np.inf, -np.inf], np.nan)

        min_obs = int(len(returns) * 0.70)
        valid_cols = returns.columns[returns.notna().sum() >= min_obs]
        returns = returns[valid_cols]

        returns = returns.fillna(0.0)

        return returns

    def _clean_etf_returns(self, etf_returns):
        etf_returns = etf_returns.copy()
        etf_returns = etf_returns.loc[:, etf_returns.columns.notna()]
        etf_returns = etf_returns.loc[:, ~etf_returns.columns.duplicated()]
        etf_returns = etf_returns.sort_index()
        etf_returns.index = pd.to_datetime(etf_returns.index).tz_localize(None)

        etf_returns = etf_returns.apply(pd.to_numeric, errors="coerce")
        etf_returns = etf_returns.replace([np.inf, -np.inf], np.nan)

        return etf_returns

    def _estimate_etf_residuals(self, returns_window, etf_window):
        """
        Estimate ETF-factor residuals.

        returns_window:
            rows = dates
            columns = stocks

        etf_window:
            rows = dates
            columns = ETF factors
        """

        # Align dates
        common_dates = returns_window.index.intersection(etf_window.index)

        if len(common_dates) < 60:
            return pd.DataFrame(index=returns_window.index, columns=returns_window.columns)

        Y_df = returns_window.loc[common_dates].copy()
        X_df = etf_window.loc[common_dates].copy()

        # Drop ETF columns with too many missing values or zero variance
        min_obs = int(len(X_df) * 0.70)
        valid_etfs = X_df.columns[
            (X_df.notna().sum() >= min_obs) &
            (X_df.std(skipna=True) > 1e-8)
        ]

        X_df = X_df[valid_etfs]

        if X_df.shape[1] == 0:
            return pd.DataFrame(index=returns_window.index, columns=returns_window.columns)

        X_df = X_df.fillna(0.0)

        # Drop stock columns with zero variance in this window
        stock_std = Y_df.std(axis=0)
        valid_stocks = stock_std[stock_std > 1e-8].index
        Y_df = Y_df[valid_stocks]

        if Y_df.shape[1] == 0:
            return pd.DataFrame(index=returns_window.index, columns=returns_window.columns)

        Y = Y_df.values.astype(float)
        X = X_df.values.astype(float)

        # Add intercept
        X_design = np.column_stack([
            np.ones(X.shape[0]),
            X
        ])

        try:
            beta = np.linalg.lstsq(X_design, Y, rcond=None)[0]
            fitted = X_design @ beta
            residuals = Y - fitted
        except Exception:
            return pd.DataFrame(index=returns_window.index, columns=returns_window.columns)

        residuals_df = pd.DataFrame(
            residuals,
            index=common_dates,
            columns=valid_stocks
        )

        # Reindex back to full returns_window shape
        residuals_full = pd.DataFrame(
            index=returns_window.index,
            columns=returns_window.columns,
            dtype=float
        )

        residuals_full.loc[common_dates, valid_stocks] = residuals_df

        return residuals_full

    def _signals_to_weights(self, signals):
        signals = pd.Series(signals).copy()
        signals = signals[signals.index.notna()]
        signals = pd.to_numeric(signals, errors="coerce").fillna(0.0)

        longs = signals[signals > 0].index
        shorts = signals[signals < 0].index

        weights = pd.Series(0.0, index=signals.index)

        if len(longs) > 0:
            weights.loc[longs] = (self.gross_leverage / 2.0) / len(longs)

        if len(shorts) > 0:
            weights.loc[shorts] = -(self.gross_leverage / 2.0) / len(shorts)

        return weights

    def _calculate_metrics(self, daily_returns):
        daily_returns = pd.Series(daily_returns).dropna()

        if len(daily_returns) == 0:
            return {
                "Total Return": np.nan,
                "Annualized Return": np.nan,
                "Annualized Volatility": np.nan,
                "Sharpe Ratio": np.nan,
                "Max Drawdown": np.nan
            }

        equity = (1 + daily_returns).cumprod()

        total_return = equity.iloc[-1] - 1
        ann_return = equity.iloc[-1] ** (252 / len(daily_returns)) - 1
        ann_vol = daily_returns.std() * np.sqrt(252)

        if ann_vol == 0 or np.isnan(ann_vol):
            sharpe = np.nan
        else:
            sharpe = ann_return / ann_vol

        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        max_drawdown = drawdown.min()

        return {
            "Total Return": round(float(total_return), 4),
            "Annualized Return": round(float(ann_return), 4),
            "Annualized Volatility": round(float(ann_vol), 4),
            "Sharpe Ratio": round(float(sharpe), 4) if not np.isnan(sharpe) else np.nan,
            "Max Drawdown": round(float(max_drawdown), 4)
        }

    def run(self, returns, etf_returns):
        returns = self._clean_returns(returns)
        etf_returns = self._clean_etf_returns(etf_returns)

        # Align global date range
        common_dates = returns.index.intersection(etf_returns.index)
        returns = returns.loc[common_dates]
        etf_returns = etf_returns.loc[common_dates]

        dates = returns.index
        tickers = returns.columns

        current_signals = pd.Series(0.0, index=tickers)
        current_weights = pd.Series(0.0, index=tickers)

        daily_return_list = []
        equity_list = []
        date_list = []

        positions_history = []
        scores_history = []

        equity = 1.0

        start_idx = self.returns_lookback
        end_idx = len(returns) - self.execution_lag_days

        for t in range(start_idx, end_idx):

            current_date = dates[t]
            next_date = dates[t + self.execution_lag_days]

            # Use only past information
            returns_window = returns.iloc[t - self.returns_lookback:t]
            etf_window = etf_returns.iloc[t - self.returns_lookback:t]

            residuals_window_full = self._estimate_etf_residuals(
                returns_window=returns_window,
                etf_window=etf_window
            )

            residuals_window = residuals_window_full.tail(self.ou_lookback)

            signals, scores = self.signal_engine.generate_signals(
                residuals_window=residuals_window,
                current_positions=current_signals
            )

            signals = signals.reindex(tickers).fillna(0.0)
            scores = scores.reindex(tickers)

            target_weights = self._signals_to_weights(signals)
            target_weights = target_weights.reindex(tickers).fillna(0.0)

            current_weights = current_weights.reindex(tickers).fillna(0.0)

            turnover = (target_weights - current_weights).abs().sum()
            transaction_cost = turnover * (self.transaction_cost_bps / 10000.0)

            realized_returns = returns.iloc[t + self.execution_lag_days]
            realized_returns = pd.Series(realized_returns).copy()
            realized_returns = pd.to_numeric(realized_returns, errors="coerce").fillna(0.0)

            common_tickers = target_weights.index.intersection(realized_returns.index)

            if len(common_tickers) == 0:
                pnl = -transaction_cost
            else:
                pnl = (
                    target_weights.loc[common_tickers]
                    * realized_returns.loc[common_tickers]
                ).sum() - transaction_cost

            equity = equity * (1 + pnl)

            daily_return_list.append(pnl)
            equity_list.append(equity)
            date_list.append(next_date)

            positions_history.append(target_weights.reindex(tickers).fillna(0.0))
            scores_history.append(scores.reindex(tickers))

            current_signals = signals.copy()
            current_weights = target_weights.reindex(tickers).fillna(0.0)

            if (t - start_idx) % 50 == 0:
                print(
                    f"{current_date} | "
                    f"equity={equity:.4f} | "
                    f"active positions={(current_weights != 0).sum()} | "
                    f"daily pnl={pnl:.5f}"
                )

        daily_returns = pd.Series(
            daily_return_list,
            index=date_list,
            name="daily_return"
        )

        equity_curve = pd.Series(
            equity_list,
            index=date_list,
            name="equity"
        )

        positions = pd.DataFrame(
            positions_history,
            index=date_list
        )

        scores = pd.DataFrame(
            scores_history,
            index=date_list
        )

        metrics = self._calculate_metrics(daily_returns)

        return {
            "daily_returns": daily_returns,
            "equity_curve": equity_curve,
            "positions": positions,
            "scores": scores,
            "metrics": metrics
        }