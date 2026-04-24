"""
Backtesting engine — fixed version.

Bugs fixed:
  BUG-01: Position sizing now correctly uses self.position_size (was using 100% of cash)
  BUG-02: Drawdown circuit-breaker now uses trailing peak (was permanently freezing after any dip)
  BUG-08: Stop-loss P&L now uses slippage-adjusted entry price as cost basis
"""
import pandas as pd
import numpy as np


class BacktestEngine:
    """Realistic backtester with position sizing, stop-loss, transaction costs, slippage."""

    def __init__(self, initial_capital: float = 100_000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 position_size: float = 0.10,
                 stop_loss: float = 0.05):
        """
        Args:
            initial_capital : Starting capital (₹)
            transaction_cost: Per-trade cost as fraction (0.001 = 0.1%)
            slippage        : Slippage as fraction of price (0.0005 = 0.05%)
            position_size   : Fraction of current capital per trade (0.10 = 10%)
            stop_loss       : Hard stop-loss below entry (0.05 = 5%)
        """
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage         = slippage
        self.position_size    = position_size
        self.stop_loss        = stop_loss

    def backtest(self, df: pd.DataFrame, signals: pd.Series,
                 signal_name: str = "signal",
                 min_hold_days: int = 0,
                 size_modifiers: pd.Series = None) -> dict:
        """
        Run backtest with supplied signals.

        Args:
            df            : DataFrame with OHLCV data
            signals       : Series of signals (1=BUY, 0=SELL, NaN=HOLD)
            signal_name   : Label for tracking
            min_hold_days : Minimum days to hold a position before allowing a
                            SELL signal to exit.  Use forward_days here to
                            align the holding period with the prediction horizon
                            (e.g. 3 for a 3-day forward target).  Stop-loss and
                            the 20% drawdown circuit-breaker still fire
                            immediately regardless of this setting.

        Returns dict with portfolio_values, portfolio_series, trades, final_value.
        """
        df = df.copy()
        cash            = self.initial_capital
        position        = 0.0    # units held
        entry_cost      = 0.0    # slippage-adjusted entry price — used for P&L
        entry_price_raw = 0.0    # raw price — used for stop-loss trigger
        days_held       = 0      # bars since entry — enforces min_hold_days
        peak_value      = self.initial_capital   # trailing peak for circuit-breaker

        portfolio_values = []
        trades           = []

        for i in range(len(df)):
            price  = df["Close"].iloc[i]
            signal = signals.iloc[i]

            portfolio_value = cash + position * price

            # ── Trailing peak for circuit-breaker ───────────────────────────
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            stop_trading = portfolio_value < peak_value * 0.80

            if position > 0:
                days_held += 1

            # ── Stop-loss — always fires, ignores min_hold_days ─────────────
            stop_loss_hit = (
                position > 0
                and price < entry_price_raw * (1 - self.stop_loss)
            )
            if stop_loss_hit:
                signal = 0   # force sell regardless of days_held

            # ── Suppress SELL until min_hold_days satisfied ─────────────────
            # Only bypassed when stop_loss_hit (already forced to 0 above)
            elif (
                not pd.isna(signal)
                and int(signal) == 0
                and position > 0
                and days_held < min_hold_days
            ):
                signal = np.nan   # too early — hold on

            # ── No signal or circuit-breaker active ─────────────────────────
            if pd.isna(signal) or stop_trading:
                portfolio_values.append(portfolio_value)
                continue

            signal = int(signal)

            # ── ENTER LONG ──────────────────────────────────────────────────
            if signal == 1 and position == 0 and cash > 0:
                mod = size_modifiers.iloc[i] if size_modifiers is not None else 1.0
                trade_capital = cash * self.position_size * mod

                # Slippage on entry (pay more)
                entry_price_slip  = price * (1 + self.slippage)
                # Transaction cost deducted from trade capital
                trade_capital_net = trade_capital * (1 - self.transaction_cost)

                if trade_capital_net > entry_price_slip:
                    units_bought    = trade_capital_net / entry_price_slip
                    position        = units_bought
                    cash           -= trade_capital
                    entry_cost      = entry_price_slip
                    entry_price_raw = price
                    days_held       = 0   # reset counter on fresh entry

            # ── EXIT LONG ──────────────────────────────────────────────────
            elif signal == 0 and position > 0:
                # Slippage on exit (receive less)
                exit_price_slip = price * (1 - self.slippage)
                proceeds        = position * exit_price_slip
                proceeds_net    = proceeds * (1 - self.transaction_cost)
                cash           += proceeds_net

                pnl_pct = (exit_price_slip - entry_cost) / entry_cost * 100

                trades.append({
                    "entry_price": entry_cost,
                    "exit_price":  exit_price_slip,
                    "pnl_pct":     pnl_pct,
                    "shares":      position,
                    "entry_date":  df.index[i],
                    "days_held":   days_held,
                    "stop_loss":   stop_loss_hit,
                })

                position        = 0.0
                entry_cost      = 0.0
                entry_price_raw = 0.0
                days_held       = 0   # reset counter after exit

            portfolio_values.append(cash + position * price)

        # ── Close open position at period end ──────────────────────────────
        if position > 0:
            final_price = df["Close"].iloc[-1]
            exit_slip   = final_price * (1 - self.slippage)
            proceeds    = position * exit_slip * (1 - self.transaction_cost)
            pnl_pct     = (exit_slip - entry_cost) / entry_cost * 100
            trades.append({
                "entry_price": entry_cost,
                "exit_price":  exit_slip,
                "pnl_pct":     pnl_pct,
                "shares":      position,
                "entry_date":  df.index[-1],
            })
            # Update final portfolio value with closed position
            if portfolio_values:
                portfolio_values[-1] = proceeds + cash

        portfolio_series = pd.Series(portfolio_values, index=df.index[:len(portfolio_values)])

        return {
            "portfolio_values":  portfolio_values,
            "portfolio_series":  portfolio_series,
            "trades":            trades,
            "final_value":       portfolio_values[-1] if portfolio_values else self.initial_capital,
            "signal_name":       signal_name,
        }

    def compute_metrics(self, backtest_result: dict,
                        initial_capital: float = None) -> dict:
        """Compute comprehensive performance metrics from backtest result."""
        if initial_capital is None:
            initial_capital = self.initial_capital

        portfolio_series = backtest_result["portfolio_series"]
        final_value      = backtest_result["final_value"]
        trades           = backtest_result["trades"]

        total_return = (final_value - initial_capital) / initial_capital * 100

        daily_returns = portfolio_series.pct_change().dropna()

        # Sharpe ratio (annualised, risk-free ≈ 4% pa → 0.016% daily)
        RF_DAILY = 0.04 / 252
        sharpe_ratio = 0.0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = float(
                (daily_returns.mean() - RF_DAILY) / daily_returns.std() * np.sqrt(252))

        # Sortino ratio (downside deviation only)
        sortino_ratio = 0.0
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino_ratio = float(
                (daily_returns.mean() - RF_DAILY) / downside.std() * np.sqrt(252))

        # Max drawdown
        running_max  = portfolio_series.expanding().max()
        drawdown     = (portfolio_series - running_max) / running_max
        max_drawdown = float(drawdown.min() * 100)

        # Calmar ratio
        years        = max(len(portfolio_series) / 252, 1e-6)
        annual_return = total_return / years
        calmar_ratio  = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        # Trade stats
        trade_pnls = [t["pnl_pct"] for t in trades]
        win_rate = avg_win = avg_loss = profit_factor = 0.0

        if trade_pnls:
            wins   = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]
            win_rate      = len(wins) / len(trade_pnls) * 100
            avg_win       = float(np.mean(wins))   if wins   else 0.0
            avg_loss      = float(abs(np.mean(losses))) if losses else 0.0
            total_wins    = sum(wins)
            total_losses  = abs(sum(losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        return {
            "final_value":    final_value,
            "total_return":   total_return,
            "sharpe_ratio":   sharpe_ratio,
            "sortino_ratio":  sortino_ratio,
            "max_drawdown":   max_drawdown,
            "calmar_ratio":   calmar_ratio,
            "num_trades":     len(trades),
            "win_rate":       win_rate,
            "avg_win":        avg_win,
            "avg_loss":       avg_loss,
            "profit_factor":  profit_factor,
        }
