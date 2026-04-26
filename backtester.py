import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.signal_engine import SignalEngine


class StatisticalArbitrageBacktester:
    def __init__(
        self,
        returns_lookback=252,
        ou_lookback=60,
        n_factors=15,
        entry_z=1.25,
        exit_z_long=0.50,
        exit_z_short=0.75,
        transaction_cost_bps=5,
        gross_leverage=2.0,
        execution_lag_days=1
    ):
        """
        PCA-based statistical arbitrage backtester.

        Parameters
        ----------
        returns_lookback : int
            Number of historical days used to estimate PCA factors.

        ou_lookback : int
            Number of residual days used to estimate OU process.

        n_factors : int
            Number of PCA factors.

        entry_z : float
            Entry threshold for s-score.

        exit_z_long : float
            Exit threshold for long positions.

        exit_z_short : float
            Exit threshold for short positions.

        transaction_cost_bps : float
            One-way transaction cost in basis points.

        gross_leverage : float
            Total gross leverage. Example: 2.0 means total absolute weights sum to 2.

        execution_lag_days : int
            Lag between signal generation and realized return.
            1 means signal at close t earns return from t to t+1.
        """

        self.returns_lookback = returns_lookback
        self.ou_lookback = ou_lookback
        self.n_factors = n_factors
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
        """
        Clean returns dataframe before backtest.
        """

        returns = returns.copy()

        # Remove invalid columns
        returns = returns.loc[:, returns.columns.notna()]
        returns = returns.loc[:, ~returns.columns.duplicated()]

        # Sort by date
        returns = returns.sort_index()

        # Convert to numeric
        returns = returns.apply(pd.to_numeric, errors="coerce")

        # Replace inf values
        returns = returns.replace([np.inf, -np.inf], np.nan)

        # Drop columns with too little data
        min_obs = int(len(returns) * 0.70)
        valid_cols = returns.columns[returns.notna().sum() >= min_obs]
        returns = returns[valid_cols]

        # Fill missing values with 0
        returns = returns.fillna(0.0)

        return returns

    def _estimate_pca_residuals(self, returns_window):
        """
        Estimate PCA residuals from a historical returns window.

        returns_window:
            rows = dates
            columns = tickers

        Return:
            residuals dataframe with same shape as returns_window.
        """

        returns_window = returns_window.copy()

        # Remove columns with zero variance
        std = returns_window.std(axis=0)
        valid_cols = std[std > 1e-8].index
        returns_window = returns_window[valid_cols]

        if returns_window.shape[1] < self.n_factors + 5:
            return pd.DataFrame(index=returns_window.index, columns=returns_window.columns)

        # Standardize returns
        mean = returns_window.mean(axis=0)
        std = returns_window.std(axis=0).replace(0, np.nan)

        standardized = (returns_window - mean) / std
        standardized = standardized.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        n_components = min(
            self.n_factors,
            standardized.shape[0] - 1,
            standardized.shape[1] - 1
        )

        if n_components <= 0:
            return pd.DataFrame(index=returns_window.index, columns=returns_window.columns)

        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(standardized.values)
        reconstructed = pca.inverse_transform(factors)

        residuals_standardized = standardized.values - reconstructed

        residuals = pd.DataFrame(
            residuals_standardized,
            index=returns_window.index,
            columns=returns_window.columns
        )

        return residuals

    def _signals_to_weights(self, signals):
        """
        Convert long/short/flat signals to portfolio weights.

        signals:
            1 = long
            -1 = short
            0 = flat
        """

        signals = pd.Series(signals).copy()

        # Remove invalid tickers
        signals = signals[signals.index.notna()]

        # Convert to numeric
        signals = pd.to_numeric(signals, errors="coerce").fillna(0.0)

        longs = signals[signals > 0].index
        shorts = signals[signals < 0].index

        weights = pd.Series(0.0, index=signals.index)

        # Dollar-neutral style:
        # half gross leverage on longs, half on shorts
        if len(longs) > 0:
            weights.loc[longs] = (self.gross_leverage / 2.0) / len(longs)

        if len(shorts) > 0:
            weights.loc[shorts] = -(self.gross_leverage / 2.0) / len(shorts)

        return weights

    def _calculate_metrics(self, daily_returns):
        """
        Calculate common performance metrics.
        """

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

    def run(self, returns):
        """
        Run rolling PCA statistical arbitrage backtest.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns.
            Rows = dates.
            Columns = tickers.

        Returns
        -------
        result : dict
            Contains equity curve, daily returns, positions, scores, and metrics.
        """

        returns = self._clean_returns(returns)

        dates = returns.index
        tickers = returns.columns

        current_signals = pd.Series(0.0, index=tickers)
        current_weights = pd.Series(0.0, index=tickers)

        daily_pnl_list = []
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

            # Historical data only. No lookahead.
            returns_window = returns.iloc[t - self.returns_lookback:t]

            # Estimate PCA residuals using only past window
            residuals_window_full = self._estimate_pca_residuals(returns_window)

            # Use last OU window for signal generation
            residuals_window = residuals_window_full.tail(self.ou_lookback)

            # Generate target signals
            signals, scores = self.signal_engine.generate_signals(
                residuals_window=residuals_window,
                current_positions=current_signals
            )

            # Align signals to full ticker universe
            signals = signals.reindex(tickers).fillna(0.0)
            scores = scores.reindex(tickers)

            # Convert signals to target weights
            target_weights = self._signals_to_weights(signals)
            target_weights = target_weights.reindex(tickers).fillna(0.0)

            # Transaction cost based on turnover
            current_weights = current_weights.reindex(tickers).fillna(0.0)
            turnover = (target_weights - current_weights).abs().sum()

            transaction_cost = turnover * (self.transaction_cost_bps / 10000.0)

            # Realized next-period return
            realized_returns = returns.iloc[t + self.execution_lag_days]
            realized_returns = pd.Series(realized_returns).copy()

            # Clean indexes before PnL calculation
            target_weights = target_weights[target_weights.index.notna()]
            realized_returns = realized_returns[realized_returns.index.notna()]

            target_weights = pd.to_numeric(target_weights, errors="coerce").fillna(0.0)
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

            daily_pnl_list.append(pnl)
            daily_return_list.append(pnl)
            equity_list.append(equity)
            date_list.append(next_date)

            positions_history.append(target_weights.reindex(tickers).fillna(0.0))
            scores_history.append(scores.reindex(tickers))

            # Update current state
            current_signals = signals.copy()
            current_weights = target_weights.reindex(tickers).fillna(0.0)

            # Progress print every 50 days
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

        result = {
            "daily_returns": daily_returns,
            "equity_curve": equity_curve,
            "positions": positions,
            "scores": scores,
            "metrics": metrics
        }

        return result