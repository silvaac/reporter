"""Hyperliquid-specific reporter implementation."""

from __future__ import annotations

import base64
import logging
import math
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from base.reporter import BaseReporter
from exceptions import ReportGenerationError
from hyperliquid_reporter.monitoring import HyperliquidMonitor
from tz_utils import ensure_utc_index, safe_strip_tz, to_eastern

logger = logging.getLogger(__name__)


class HyperliquidReporter(BaseReporter):
    """Reporter for Hyperliquid trading accounts.
    
    Generates comprehensive performance reports including:
    - AUM (Assets Under Management) tracking
    - Performance metrics ($ and %)
    - Trade cost analysis
    - Funding cost analysis
    - Professional HTML reports with visualizations
    
    Attributes:
        monitor: HyperliquidMonitor instance for data retrieval.
        account_address: The account address being reported on.
    """
    
    def __init__(
        self,
        monitor: HyperliquidMonitor,
        account_address: str,
        pnl_history_file: str = "pnl_history.csv",
        price_cache_dir: str = "./data/hyperliquid"
    ) -> None:
        """Initialize the Hyperliquid reporter.
        
        Args:
            monitor: HyperliquidMonitor instance.
            account_address: The account address to report on.
            pnl_history_file: Path to P&L history CSV file.
            price_cache_dir: Directory for caching price data.
        """
        self.monitor = monitor
        self.account_address = account_address
        self.pnl_history_file = pnl_history_file
        self.price_cache_dir = price_cache_dir
        logger.info("Initialized HyperliquidReporter for account %s", account_address[:10] + "...")
        logger.info("P&L history file: %s", pnl_history_file)
        logger.info("Price cache directory: %s", price_cache_dir)
    
    def generate_aum_data(self, period: str = "allTime") -> pd.DataFrame:
        """Generate AUM (Assets Under Management) data over time.
        
        Args:
            period: Time period for the data (e.g., "day", "week", "month", "allTime").
        
        Returns:
            DataFrame with datetime index and columns:
            - aum_usd: Account value in USD (includes spot + perp)
        
        Raises:
            ReportGenerationError: If unable to generate AUM data.
        
        Note:
            Hyperliquid's accountValueHistory API returns total account value
            including both perpetual and spot balances.
        """
        try:
            df = self.monitor.get_portfolio_dataframe(period=period, data_type="account_value")
            
            if df.empty:
                logger.warning("No AUM data available for period: %s", period)
                return pd.DataFrame(columns=["aum_usd"])
            
            # Ensure UTC-aware index
            ensure_utc_index(df)
            
            df = df.rename(columns={"account_value": "aum_usd"})
            return df
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate AUM data: {e}") from e
    
    def generate_performance_data(self, period: str = "allTime") -> pd.DataFrame:
        """Generate performance data (P&L in $ and %).
        
        Args:
            period: Time period for the data.
        
        Returns:
            DataFrame with datetime index and columns:
            - pnl_usd: Total P&L in USD (realized + unrealized)
            - pnl_pct: Total P&L as percentage
            - aum_usd: Account value for reference
        
        Raises:
            ReportGenerationError: If unable to generate performance data.
        
        Note:
            The pnl_usd includes both realized P&L (from closed trades) and
            unrealized P&L (from open positions). Historical data points show
            realized P&L at that time, while the latest point includes current
            unrealized P&L.
        """
        try:
            # Get account value history (includes total value with unrealized P&L)
            df = self.monitor.get_portfolio_dataframe(period=period, data_type="account_value")
            
            if df.empty:
                logger.warning("No performance data available for period: %s", period)
                return pd.DataFrame(columns=["pnl_usd", "pnl_pct", "aum_usd"])
            
            # Ensure UTC-aware index
            ensure_utc_index(df)
            
            df = df.rename(columns={"account_value": "aum_usd"})
            
            # Get deposit/withdrawal history to calculate net deposits at each point
            # We need to reconstruct net_deposits over time from ledger data
            try:
                from datetime import timedelta
                
                # Get ledger updates for the entire history period
                # Start from 1 year before first account value to catch early deposits
                if not df.empty:
                    first_time_ms = int(df.index[0].timestamp() * 1000)
                    start_time_ms = first_time_ms - (365 * 24 * 60 * 60 * 1000)  # 1 year before
                    end_time_ms = int(df.index[-1].timestamp() * 1000)
                    
                    ledger_updates = self.monitor._info.user_non_funding_ledger_updates(
                        self.monitor._address,
                        start_time_ms,
                        end_time_ms
                    )
                    
                    # Build a time series of cumulative net deposits
                    deposits_timeline = []
                    cumulative_deposits = 0.0
                    
                    for update in ledger_updates:
                        delta = update.get("delta", {})
                        time_ms = update.get("time", 0)
                        
                        if "type" in delta:
                            if delta["type"] == "deposit":
                                usdc_amount = float(delta.get("usdc", 0.0))
                                cumulative_deposits += usdc_amount
                            elif delta["type"] == "withdraw":
                                usdc_amount = float(delta.get("usdc", 0.0))
                                cumulative_deposits -= abs(usdc_amount)
                            elif delta["type"] == "subAccountTransfer":
                                usdc_amount = float(delta.get("usdc", 0.0))
                                cumulative_deposits += usdc_amount
                            
                            deposits_timeline.append({
                                'timestamp': pd.to_datetime(time_ms, unit='ms', utc=True),
                                'net_deposits': cumulative_deposits
                            })
                    
                    # Create deposits dataframe and merge with account value
                    if deposits_timeline:
                        deposits_df = pd.DataFrame(deposits_timeline)
                        # Use merge_asof to get the net_deposits value at or before each timestamp
                        # Reset index to make timestamp a column for merge_asof
                        df_with_time = df.reset_index()
                        df_with_time = pd.merge_asof(
                            df_with_time.sort_values('timestamp'),
                            deposits_df.sort_values('timestamp'),
                            on='timestamp',
                            direction='backward'
                        )
                        # Set index back and handle NaN values
                        df = df_with_time.set_index('timestamp')
                        df['net_deposits'] = df['net_deposits'].fillna(0.0)
                    else:
                        df['net_deposits'] = 0.0
                    
                    logger.info(f"Reconstructed net deposits timeline with {len(deposits_timeline)} deposit/withdrawal events")
                else:
                    df['net_deposits'] = 0.0
                    
            except Exception as e:
                logger.warning(f"Could not reconstruct net deposits timeline: {e}")
                # Fallback: use current net deposits for all points
                try:
                    account_summary = self.monitor.get_account_summary()
                    current_net_deposits = account_summary.get("net_deposits", 0.0)
                    df['net_deposits'] = current_net_deposits
                except Exception:
                    df['net_deposits'] = 0.0
            
            # Resample to daily frequency for uniform time intervals
            df = self._resample_to_daily(df)
            
            # Ensure first row has aum_usd = net_deposits (initial state with no P&L)
            if len(df) > 0 and df["aum_usd"].iloc[0] == 0.0 and df["net_deposits"].iloc[0] > 0:
                # If first AUM is 0 but there are deposits, set AUM to equal deposits
                df.iloc[0, df.columns.get_loc("aum_usd")] = df["net_deposits"].iloc[0]
            
            # Calculate P&L based on period-over-period changes
            # pnl_usd(t) = aum_usd(t) - aum_usd(t-1) - (net_deposits(t) - net_deposits(t-1))
            # pnl_pct(t) = pnl_usd(t) / aum_usd(t-1)
            df["pnl_usd"] = 0.0  # First row is zero
            df["pnl_pct"] = 0.0  # First row is zero
            
            if len(df) > 1:
                # Calculate P&L for each period starting from the second row
                for i in range(1, len(df)):
                    # Change in AUM between consecutive periods
                    aum_change = df["aum_usd"].iloc[i] - df["aum_usd"].iloc[i-1]
                    # Change in deposits between consecutive periods
                    deposit_change = df["net_deposits"].iloc[i] - df["net_deposits"].iloc[i-1]
                    # P&L for this period
                    period_pnl = aum_change - deposit_change
                    
                    df.iloc[i, df.columns.get_loc("pnl_usd")] = period_pnl
                    
                    # P&L percentage based on previous period's AUM
                    if df["aum_usd"].iloc[i-1] > 0:
                        period_pct = (period_pnl / df["aum_usd"].iloc[i-1]) * 100
                        df.iloc[i, df.columns.get_loc("pnl_pct")] = period_pct
            
            return df
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate performance data: {e}") from e
    
    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample DataFrame to uniform daily intervals.
        
        Uses last value of day for aum_usd and net_deposits (forward-fill for gaps).
        
        Args:
            df: DataFrame with DatetimeIndex and at least 'aum_usd' column.
        
        Returns:
            DataFrame resampled to daily frequency.
        """
        if df.empty:
            return df
        
        # Resample to daily: take the last value of each day
        daily = df.resample('D').last()
        
        # Forward-fill gaps (days with no data carry previous day's values)
        daily = daily.ffill()
        
        # Drop any rows that are still NaN (before first data point)
        daily = daily.dropna(subset=['aum_usd'])
        
        return daily
    
    def generate_monthly_performance(self, performance_data: pd.DataFrame) -> pd.DataFrame:
        """Generate monthly performance summary from daily performance data.
        
        Args:
            performance_data: Daily performance DataFrame with columns:
                aum_usd, net_deposits, pnl_usd, pnl_pct.
        
        Returns:
            DataFrame with one row per calendar month and columns:
                month, starting_aum, ending_aum, pnl_usd, pnl_pct,
                cumulative_pnl_usd, cumulative_pnl_pct.
        """
        if performance_data.empty:
            return pd.DataFrame(columns=[
                "month", "starting_aum", "ending_aum",
                "pnl_usd", "pnl_pct",
                "cumulative_pnl_usd", "cumulative_pnl_pct",
            ])
        
        monthly_rows = []
        # Group by year-month (convert to tz-naive to avoid warning)
        perf_data_naive = performance_data.copy()
        perf_data_naive.index = safe_strip_tz(perf_data_naive.index)
        grouped = perf_data_naive.groupby(perf_data_naive.index.to_period('M'))
        
        for period, group in grouped:
            starting_aum = float(group["aum_usd"].iloc[0])
            ending_aum = float(group["aum_usd"].iloc[-1])
            month_pnl_usd = float(group["pnl_usd"].sum())
            
            # Monthly return %: sum of daily P&L / starting AUM
            if starting_aum > 0:
                month_pnl_pct = (month_pnl_usd / starting_aum) * 100
            else:
                month_pnl_pct = 0.0
            
            monthly_rows.append({
                "month": str(period),
                "starting_aum": starting_aum,
                "ending_aum": ending_aum,
                "pnl_usd": month_pnl_usd,
                "pnl_pct": round(month_pnl_pct, 2),
            })
        
        result = pd.DataFrame(monthly_rows)
        if not result.empty:
            result["cumulative_pnl_usd"] = result["pnl_usd"].cumsum()
            result["cumulative_pnl_pct"] = result["pnl_pct"].cumsum()
        
        return result
    
    def generate_weekly_performance(self, performance_data: pd.DataFrame) -> pd.DataFrame:
        """Generate weekly performance summary from daily performance data.
        
        Args:
            performance_data: Daily performance DataFrame with columns:
                aum_usd, net_deposits, pnl_usd, pnl_pct.
        
        Returns:
            DataFrame with one row per ISO week and columns:
                week, starting_aum, ending_aum, pnl_usd, pnl_pct,
                cumulative_pnl_usd, cumulative_pnl_pct.
        """
        if performance_data.empty:
            return pd.DataFrame(columns=[
                "week", "starting_aum", "ending_aum",
                "pnl_usd", "pnl_pct",
                "cumulative_pnl_usd", "cumulative_pnl_pct",
            ])
        
        weekly_rows = []
        perf_data_naive = performance_data.copy()
        perf_data_naive.index = safe_strip_tz(perf_data_naive.index)
        # Use W-SAT: weeks end on Saturday (Sunday through Saturday grouping)
        grouped = perf_data_naive.groupby(perf_data_naive.index.to_period('W-SAT'))
        
        prev_ending_aum = None
        for period, group in grouped:
            ending_aum = float(group["aum_usd"].iloc[-1])
            # starting_aum is the AUM at end of last day of the previous week.
            # group["aum_usd"].iloc[0] is already the closing AUM after the first day's
            # P&L has been applied, so we must back it out using the first day's pnl_usd
            # and the deposit change on that day. Equivalently, this equals the prior
            # week's ending AUM. For the very first week, fall back to first day's AUM.
            if prev_ending_aum is not None:
                starting_aum = prev_ending_aum
            else:
                # First week: back out the first day's P&L from the first day's AUM
                first_day_pnl = float(group["pnl_usd"].iloc[0])
                first_day_deposit_change = float(group["net_deposits"].iloc[0]) - 0.0
                starting_aum = float(group["aum_usd"].iloc[0]) - first_day_pnl - first_day_deposit_change
                if starting_aum <= 0:
                    starting_aum = float(group["aum_usd"].iloc[0])
            prev_ending_aum = ending_aum
            week_pnl_usd = float(group["pnl_usd"].sum())
            
            if starting_aum > 0:
                week_pnl_pct = (week_pnl_usd / starting_aum) * 100
            else:
                week_pnl_pct = 0.0
            
            weekly_rows.append({
                "week": str(period),
                "starting_aum": starting_aum,
                "ending_aum": ending_aum,
                "pnl_usd": week_pnl_usd,
                "pnl_pct": round(week_pnl_pct, 2),
            })
        
        result = pd.DataFrame(weekly_rows)
        if not result.empty:
            result["cumulative_pnl_usd"] = result["pnl_usd"].cumsum()
            result["cumulative_pnl_pct"] = result["pnl_pct"].cumsum()
        
        return result

    def generate_daily_funding(self, funding_analysis: pd.DataFrame) -> pd.DataFrame:
        """Generate daily funding summary by summing hourly funding payments.
        
        Args:
            funding_analysis: Funding DataFrame with columns:
                coin, funding_payment, position_size, funding_rate, etc.
        
        Returns:
            DataFrame with one row per calendar day and columns:
                date, total_funding_usd, plus one column per coin.
        """
        if funding_analysis.empty or "funding_payment" not in funding_analysis.columns:
            return pd.DataFrame(columns=["date", "total_funding_usd"])
        
        # Strip timezone for grouping
        df = funding_analysis.copy()
        df.index = safe_strip_tz(df.index)
        df["date"] = df.index.normalize()
        
        # Total daily funding
        daily_total = df.groupby("date")["funding_payment"].sum().rename("total_funding_usd")
        
        # Per-coin daily funding
        if "coin" in df.columns:
            daily_by_coin = df.pivot_table(
                index="date", columns="coin", values="funding_payment",
                aggfunc="sum", fill_value=0.0
            )
            daily_by_coin.columns = [f"{c}" for c in daily_by_coin.columns]
            result = pd.concat([daily_total, daily_by_coin], axis=1).reset_index()
        else:
            result = daily_total.reset_index()
        
        return result

    def generate_weekly_funding(self, funding_analysis: pd.DataFrame) -> pd.DataFrame:
        """Generate weekly funding summary by summing hourly funding payments.
        
        Args:
            funding_analysis: Funding DataFrame with columns:
                coin, funding_payment, etc.
        
        Returns:
            DataFrame with one row per ISO week and columns:
                week, total_funding_usd.
        """
        if funding_analysis.empty or "funding_payment" not in funding_analysis.columns:
            return pd.DataFrame(columns=["week", "total_funding_usd"])
        
        df = funding_analysis.copy()
        df.index = safe_strip_tz(df.index)
        # Use W-SAT: weeks end on Saturday
        grouped = df.groupby(df.index.to_period('W-SAT'))
        
        weekly_rows = []
        for period, group in grouped:
            weekly_rows.append({
                "week": str(period),
                "total_funding_usd": float(group["funding_payment"].sum()),
            })
        
        return pd.DataFrame(weekly_rows)

    def generate_trade_analysis(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generate trade analysis including costs for each trade.
        
        Args:
            start_time: Optional start time for filtering.
            end_time: Optional end time for filtering.
        
        Returns:
            DataFrame with columns:
            - coin: Asset symbol
            - side: Trade side (buy/sell)
            - price: Fill price
            - size: Fill size
            - notional: Trade value (price * size)
            - fee: Trading fee
            - closed_pnl: Realized P&L from closing position
            - net_pnl: Net P&L (closed_pnl - fee)
            - fee_bps: Fee in basis points
            - feeToken: Token used for fee payment
            - dir: Direction of trade
        
        Raises:
            ReportGenerationError: If unable to generate trade analysis.
        """
        try:
            trades_df = self.monitor.get_trade_history(
                start_time=start_time,
                end_time=end_time,
                as_dataframe=True
            )
            
            if trades_df.empty:
                logger.warning("No trade data available")
                return pd.DataFrame(columns=[
                    "coin", "side", "price", "size", "notional", 
                    "fee", "closed_pnl", "net_pnl", "fee_bps", "feeToken", "dir"
                ])
            
            result_df = pd.DataFrame()
            result_df["coin"] = trades_df["coin"]
            
            result_df["side"] = trades_df["side"].map({"B": "buy", "A": "sell"})
            result_df["price"] = trades_df["px"].astype(float)
            result_df["size"] = trades_df["sz"].astype(float).abs()
            result_df["notional"] = result_df["price"] * result_df["size"]
            result_df["fee"] = trades_df["fee"].astype(float).abs()
            result_df["closed_pnl"] = trades_df["closedPnl"].astype(float)
            result_df["net_pnl"] = result_df["closed_pnl"] - result_df["fee"]
            
            # Add additional columns from token_data
            result_df["fee_bps"] = trades_df["fee_bps"].astype(float)
            result_df["feeToken"] = trades_df["feeToken"]
            result_df["dir"] = trades_df["dir"]
            
            result_df.index = trades_df.index
            
            return result_df
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate trade analysis: {e}") from e
    
    def generate_funding_analysis(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate funding cost analysis for perpetual positions.
        
        Args:
            start_time: Optional start time for filtering.
            end_time: Optional end time for filtering.
            lookback_days: Number of days to look back if start_time is None.
        
        Returns:
            DataFrame with columns:
            - coin: Asset symbol
            - funding_payment: USD value (positive = received, negative = paid)
            - position_size: Position size at time of funding
            - funding_rate: The funding rate applied
            - token_price: Token price at funding time (matched by datetime)
            - calculated_funding: Calculated funding (-1 * size * price * funding_rate)
        
        Raises:
            ReportGenerationError: If unable to generate funding analysis.
        """
        try:
            funding_df = self.monitor.get_funding_history(
                start_time=start_time,
                end_time=end_time,
                lookback=lookback_days,
                as_dataframe=True
            )
            
            if funding_df.empty:
                logger.warning("No funding data available")
                return pd.DataFrame(columns=[
                    "coin", "funding_payment", "position_size", "funding_rate",
                    "token_price", "calculated_funding"
                ])
            
            result_df = pd.DataFrame()
            result_df["coin"] = funding_df["coin"]
            result_df["funding_payment"] = funding_df["usdc"].astype(float)
            result_df["position_size"] = funding_df["szi"].astype(float)
            result_df["funding_rate"] = funding_df["fundingRate"].astype(float)
            
            result_df.index = funding_df.index
            
            # Add token prices matched by datetime
            result_df = self._add_token_prices_to_funding(result_df)
            
            # Calculate funding: -1 * size * price * funding_rate (keep all signs)
            result_df["calculated_funding"] = (
                -1 * 
                result_df["position_size"] * 
                result_df["token_price"] * 
                result_df["funding_rate"]
            )
            
            return result_df
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate funding analysis: {e}") from e
    
    def generate_report_data(
        self,
        period: str = "allTime",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        lookback_days: int = 180,
    ) -> dict[str, Any]:
        """Generate all report data.
        
        Args:
            period: Time period for historical data.
            start_time: Optional start time for trade/funding data.
            end_time: Optional end time for trade/funding data.
            lookback_days: Days to look back for funding data.
        
        Returns:
            Dictionary containing:
            - aum_data: AUM DataFrame
            - performance_data: Performance DataFrame
            - trade_analysis: Trade analysis DataFrame
            - funding_analysis: Funding analysis DataFrame
            - summary_stats: Summary statistics dictionary
            - account_summary: Account summary from monitor
        
        Raises:
            ReportGenerationError: If unable to generate report data.
        """
        try:
            logger.info("Generating report data for period: %s", period)
            
            aum_data = self.generate_aum_data(period=period)
            performance_data = self.generate_performance_data(period=period)
            trade_analysis = self.generate_trade_analysis(
                start_time=start_time,
                end_time=end_time
            )
            # Get all-time funding data (no time constraints)
            funding_analysis = self.generate_funding_analysis()
            
            account_summary = self.monitor.get_account_summary(lookback_days=3650)

            # Enrich account_summary with net-position USD and vol metrics.
            # Assumption: spot and perp positions are for the same underlying token.
            position_token = account_summary.get("position_token")
            if position_token:
                price_series = self._fetch_price_series(position_token)
            else:
                price_series = None
            pos_metrics = self._calculate_net_position_metrics(account_summary, price_series)
            account_summary.update(pos_metrics)

            # Generate funding summaries before saving P&L (need daily funding for snapshot)
            weekly_performance = self.generate_weekly_performance(performance_data)
            daily_funding = self.generate_daily_funding(funding_analysis)
            weekly_funding = self.generate_weekly_funding(funding_analysis)

            # Get today's funding total for the snapshot
            today = pd.Timestamp.now(tz='UTC').normalize()
            today_funding = 0.0
            if not daily_funding.empty and 'date' in daily_funding.columns:
                today_row = daily_funding[daily_funding['date'] == today]
                if not today_row.empty and 'total_funding_usd' in today_row.columns:
                    today_funding = float(today_row['total_funding_usd'].iloc[0])

            # Save current P&L snapshot to history file (with funding and positions)
            current_aum = account_summary.get("current_value", 0.0)
            current_net_deposits = account_summary.get("net_deposits", 0.0)
            self._save_pnl_history(current_aum, current_net_deposits, today_funding)

            # Load P&L history from file (includes newly saved row with exposure_pnl calc)
            pnl_history = self._load_pnl_history()

            summary_stats = self._calculate_summary_stats(
                aum_data=aum_data,
                performance_data=performance_data,
                trade_analysis=trade_analysis,
                funding_analysis=funding_analysis,
                account_summary=account_summary,
                pnl_history=pnl_history,
            )
            
            return {
                "aum_data": aum_data,
                "performance_data": performance_data,
                "trade_analysis": trade_analysis,
                "funding_analysis": funding_analysis,
                "pnl_history": pnl_history,
                "weekly_performance": weekly_performance,
                "daily_funding": daily_funding,
                "weekly_funding": weekly_funding,
                "summary_stats": summary_stats,
                "account_summary": account_summary,
                "period": period,
                "generated_at": datetime.now(timezone.utc),
            }
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate report data: {e}") from e
    
    def _add_token_prices_to_funding(self, funding_df: pd.DataFrame) -> pd.DataFrame:
        """Add token prices to funding dataframe, matched by datetime.
        
        Uses token_data's HyperliquidPerpManager for price data with local caching.
        
        Args:
            funding_df: DataFrame with funding data (must have 'coin' column and datetime index).
        
        Returns:
            DataFrame with added 'token_price' column.
        """
        if funding_df.empty:
            funding_df["token_price"] = 0.0
            return funding_df
        
        try:
            from token_data.hyperliquid import HyperliquidPerpManager
            
            # Get unique coins from funding data
            unique_coins = funding_df["coin"].unique().tolist()
            logger.info(f"Fetching price data for {len(unique_coins)} coins: {unique_coins}")
            
            # Initialize price column with NaN
            funding_df["token_price"] = float('nan')
            
            # Ensure funding_df has UTC timezone-aware datetime index
            ensure_utc_index(funding_df)
            
            # Get date range for price data (add buffer for matching)
            min_date = funding_df.index.min()
            max_date = funding_df.index.max()
            
            # Calculate refresh hours based on date range
            date_range_hours = int((max_date - min_date).total_seconds() / 3600) + 48
            
            # Use HyperliquidPerpManager to fetch and cache price data
            # This will automatically save to local files and reuse them
            price_manager = HyperliquidPerpManager(
                ticker=unique_coins,
                data_dir=self.price_cache_dir,  # Local cache directory
                interval="1h",  # 1-hour candles for matching
                file_type="parquet",  # Use parquet for efficient storage
                update=True,  # Update with new data
                save=True,  # Save to local cache
                refresh_hours=min(date_range_hours, 720),  # Limit to 30 days max per request
                info=self.monitor._info,
                verbose=False  # Reduce logging noise
            )
            
            # Match prices to funding timestamps for each coin
            for coin in unique_coins:
                try:
                    # Get price data for this coin
                    price_data = price_manager.get_data(coin)
                    
                    if price_data is None or price_data.empty:
                        logger.warning(f"No price data available for {coin}")
                        continue
                    
                    # Check if price_data has a datetime column or datetime index
                    if 'datetime' in price_data.columns:
                        # If datetime is a column, set it as index
                        price_df = price_data.set_index('datetime')[["close"]].copy()
                    elif isinstance(price_data.index, pd.DatetimeIndex):
                        # Already has datetime index
                        price_df = price_data[["close"]].copy()
                    else:
                        logger.warning(f"Price data for {coin} has no datetime index or column")
                        continue
                    
                    # Get funding entries for this coin
                    coin_mask = funding_df["coin"] == coin
                    
                    # Ensure price dataframe has UTC timezone-aware datetime index
                    if not isinstance(price_df.index, pd.DatetimeIndex):
                        logger.warning(f"Price data for {coin} index is not DatetimeIndex")
                        continue
                        
                    ensure_utc_index(price_df)
                    
                    # Sort price dataframe by index
                    price_df_sorted = price_df.sort_index()
                    
                    # Get funding timestamps for this coin (already sorted in funding_df)
                    funding_times = funding_df[coin_mask].index
                    
                    logger.info(f"Price data for {coin}: {len(price_df_sorted)} candles from {price_df_sorted.index[0]} to {price_df_sorted.index[-1]}")
                    logger.info(f"Funding data for {coin}: {len(funding_times)} entries from {funding_times[0]} to {funding_times[-1]}")
                    
                    # For each funding timestamp, find the nearest price
                    for funding_time in funding_times:
                        # Find the closest price within tolerance
                        time_diff = abs(price_df_sorted.index - funding_time)
                        min_diff = time_diff.min()
                        
                        # Only match if within 2 hours tolerance
                        if min_diff <= pd.Timedelta(hours=2):
                            closest_idx = time_diff.argmin()
                            price = price_df_sorted.iloc[closest_idx]["close"]
                            funding_df.loc[funding_time, "token_price"] = float(price)
                            logger.debug(f"Matched {coin} at {funding_time}: price={price:.2f}, diff={min_diff}")
                    
                    matched = (funding_df.loc[coin_mask, "token_price"] > 0).sum()
                    logger.info(f"Matched prices for {coin}: {matched}/{coin_mask.sum()} funding entries")
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch/match price data for {coin}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
            
            # Fill any remaining NaN values with 0.0
            funding_df["token_price"] = funding_df["token_price"].fillna(0.0)
            
            # Log summary
            matched_count = (funding_df["token_price"] > 0).sum()
            total_count = len(funding_df)
            logger.info(f"Price matching complete: {matched_count}/{total_count} entries matched")
            
            return funding_df
            
        except ImportError as e:
            logger.warning(f"token_data not available for price data: {e}")
            funding_df["token_price"] = 0.0
            return funding_df
        except Exception as e:
            logger.error(f"Error adding token prices to funding data: {e}")
            funding_df["token_price"] = 0.0
            return funding_df
    
    def _load_market_funding_rates(
        self,
        coins: list[str],
        stale_days: int = 2,
    ) -> pd.DataFrame:
        """Load market-wide hourly funding rates for given coins from local parquet files.
        
        For each coin, reads ``{price_cache_dir}/funding/{coin}.parquet``. If the
        file's most recent datetime is older than ``stale_days``, missing data is
        pulled via ``HyperliquidFundingManager`` and saved back to disk.
        
        Args:
            coins: List of coin symbols (e.g. ["ETH", "HYPE"]).
            stale_days: File is considered stale if last row is older than this many days.
        
        Returns:
            DataFrame with UTC DatetimeIndex and columns: ``funding_rate``, ``coin``.
            Returns empty DataFrame on failure or if no data is available.
        """
        funding_dir = Path(self.price_cache_dir) / "funding"
        if not funding_dir.exists():
            logger.warning(f"Funding directory does not exist: {funding_dir}")
            return pd.DataFrame(columns=["funding_rate", "coin"])
        
        now_utc = pd.Timestamp.now(tz="UTC")
        stale_cutoff = now_utc - pd.Timedelta(days=stale_days)
        
        # Figure out which coins need a refresh
        stale_coins: list[str] = []
        per_coin_df: dict[str, pd.DataFrame] = {}
        
        for coin in coins:
            fpath = funding_dir / f"{coin}.parquet"
            if not fpath.exists():
                logger.info(f"Market funding file missing for {coin}; will fetch")
                stale_coins.append(coin)
                continue
            try:
                df = pd.read_parquet(fpath)
                # Convert datetime column to tz-aware UTC
                if "datetime" in df.columns:
                    dt = pd.to_datetime(df["datetime"], utc=True)
                    last_dt = dt.max()
                    if last_dt < stale_cutoff:
                        logger.info(
                            f"Market funding file for {coin} is stale "
                            f"(last: {last_dt}); will refresh"
                        )
                        stale_coins.append(coin)
                    per_coin_df[coin] = df
                else:
                    logger.warning(f"File {fpath} has no 'datetime' column; refreshing")
                    stale_coins.append(coin)
            except Exception as e:
                logger.warning(f"Failed reading {fpath}: {e}; will refresh")
                stale_coins.append(coin)
        
        # Refresh stale coins using HyperliquidFundingManager
        if stale_coins:
            try:
                from token_data.hyperliquid import HyperliquidFundingManager
                
                manager = HyperliquidFundingManager(
                    ticker=stale_coins,
                    data_dir=self.price_cache_dir,
                    file_type="parquet",
                    update=True,
                    save=True,
                    refresh_hours=24 * (stale_days + 1),
                    info=self.monitor._info,
                    verbose=False,
                )
                for coin in stale_coins:
                    try:
                        fresh = manager.get_data(coin)
                        if fresh is not None and not fresh.empty:
                            per_coin_df[coin] = fresh
                    except Exception as e:
                        logger.warning(f"Failed to refresh funding data for {coin}: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize HyperliquidFundingManager: {e}")
        
        # Combine into a single DataFrame with UTC DatetimeIndex
        frames = []
        for coin, df in per_coin_df.items():
            if df is None or df.empty or "funding_rate" not in df.columns:
                continue
            sub = df[["datetime", "funding_rate"]].copy()
            sub["datetime"] = pd.to_datetime(sub["datetime"], utc=True)
            sub = sub.set_index("datetime")
            sub["coin"] = coin
            frames.append(sub)
        
        if not frames:
            return pd.DataFrame(columns=["funding_rate", "coin"])
        
        combined = pd.concat(frames).sort_index()
        return combined

    def _fetch_price_series(self, coin: str) -> Optional[pd.Series]:
        """Fetch a close-price Series for *coin* using the local cache.

        Returns a pd.Series with a UTC DatetimeIndex and close prices, or None
        on failure.
        """
        try:
            from token_data.hyperliquid import HyperliquidPerpManager

            manager = HyperliquidPerpManager(
                ticker=[coin],
                data_dir=self.price_cache_dir,
                interval="1h",
                file_type="parquet",
                update=True,
                save=True,
                refresh_hours=48,
                info=self.monitor._info,
                verbose=False,
            )
            price_data = manager.get_data(coin)
            if price_data is None or price_data.empty:
                return None
            if "datetime" in price_data.columns:
                price_df = price_data.set_index("datetime")["close"]
            elif isinstance(price_data.index, pd.DatetimeIndex):
                price_df = price_data["close"]
            else:
                return None
            price_df.index = pd.to_datetime(price_df.index, utc=True)
            return price_df.sort_index()
        except Exception as e:
            logger.warning("Failed to fetch price series for %s: %s", coin, e)
            return None

    def _fetch_spot_price_series(self, coin: str) -> Optional[pd.Series]:
        """Fetch a spot close-price Series for *coin* using the local cache.

        Reads from local parquet cache files in the 'spot' subfolder.
        Spot prices are stored separately from perp prices.
        File naming pattern: {coin}_USDC_1h.parquet

        Returns a pd.Series with a UTC DatetimeIndex and close prices, or None
        on failure.
        """
        try:
            from pathlib import Path

            # Use spot subfolder within price_cache_dir
            spot_cache_dir = Path(self.price_cache_dir) / "spot"
            spot_cache_dir.mkdir(parents=True, exist_ok=True)

            # File naming pattern: ETH_USDC_1h.parquet
            file_path = spot_cache_dir / f"{coin}_USDC_1h.parquet"

            if not file_path.exists():
                logger.warning("Spot price file not found: %s", file_path)
                return None

            # Read parquet file - use fastparquet if available (handles older parquet versions)
            try:
                price_data = pd.read_parquet(file_path, engine="fastparquet")
            except Exception:
                # Fall back to pyarrow if fastparquet fails
                price_data = pd.read_parquet(file_path, engine="pyarrow")

            if price_data is None or price_data.empty:
                return None

            # Extract close prices with datetime index
            if "datetime" in price_data.columns:
                price_df = price_data.set_index("datetime")["close"]
            elif isinstance(price_data.index, pd.DatetimeIndex):
                price_df = price_data["close"]
            else:
                return None

            price_df.index = pd.to_datetime(price_df.index, utc=True)
            return price_df.sort_index()
        except Exception as e:
            logger.warning("Failed to fetch spot price series for %s: %s", coin, e)
            return None

    def _calculate_net_position_metrics(
        self,
        positions: dict[str, Any],
        price_series: Optional[pd.Series],
    ) -> dict[str, Any]:
        """Calculate net-position USD exposure and per-period price volatility.

        Uses the last price in *price_series* for the USD conversion and computes
        the std of hourly log-returns over all available price data (no annualisation,
        no window truncation).

        Assumption: spot and perp positions are for the same underlying token.

        Args:
            positions: dict with keys position_token, net_position, spot_position,
                perp_position (as returned by get_positions_summary).
            price_series: pandas Series of hourly close prices with a DatetimeIndex.
                The last value is used as the current price.
                Pass None or an empty Series if no price data is available.

        Returns:
            Dictionary with:
            - last_perp_price: float – most recent price in price_series (0 if unavailable)
            - net_position_usd: float – net_position * last_perp_price
            - position_vol: float – std of hourly log-returns over all available data
              (as fraction per hour, e.g. 0.002).  0.0 if < 2 data points available.
            - net_position_vol_usd: float – |net_position| * last_perp_price * position_vol
        """
        net_position = float(positions.get("net_position", 0.0))
        position_token = positions.get("position_token")

        zero_result: dict[str, Any] = {
            "last_perp_price": 0.0,
            "net_position_usd": 0.0,
            "position_vol": 0.0,
            "net_position_vol_usd": 0.0,
        }

        if position_token is None:
            return zero_result

        # --- Last price ---
        last_price = 0.0
        if price_series is not None and len(price_series) > 0:
            last_price = float(price_series.iloc[-1])

        # --- Per-period (hourly) log-return std over all available data ---
        vol = 0.0
        if price_series is not None and len(price_series) >= 2:
            try:
                log_returns = (price_series / price_series.shift(1)).apply(math.log).dropna()
                if len(log_returns) >= 1:
                    vol = float(log_returns.std())
            except Exception as e:
                logger.warning("Failed to compute vol for %s: %s", position_token, e)

        net_position_usd = net_position * last_price
        net_position_vol_usd = abs(net_position) * last_price * vol

        return {
            "last_perp_price": last_price,
            "net_position_usd": net_position_usd,
            "position_vol": vol,
            "net_position_vol_usd": net_position_vol_usd,
        }

    def _calculate_summary_stats(
        self,
        aum_data: pd.DataFrame,
        performance_data: pd.DataFrame,
        trade_analysis: pd.DataFrame,
        funding_analysis: pd.DataFrame,
        account_summary: dict[str, Any],
        pnl_history: Optional[pd.DataFrame] = None,
    ) -> dict[str, Any]:
        """Calculate summary statistics for the report.
        
        Args:
            aum_data: AUM DataFrame.
            performance_data: Performance DataFrame.
            trade_analysis: Trade analysis DataFrame.
            funding_analysis: Funding analysis DataFrame.
            account_summary: Account summary dictionary.
        
        Returns:
            Dictionary with summary statistics.
        """
        stats = {}
        
        # Use real-time current_value from account_summary for accuracy
        # accountValueHistory can be stale
        stats["current_aum"] = account_summary.get("current_value", 0.0)
        
        if not aum_data.empty and "aum_usd" in aum_data.columns:
            stats["initial_aum"] = float(aum_data["aum_usd"].iloc[0])
            stats["peak_aum"] = float(aum_data["aum_usd"].max())
            stats["min_aum"] = float(aum_data["aum_usd"].min())
        else:
            stats["initial_aum"] = 0.0
            stats["peak_aum"] = 0.0
            stats["min_aum"] = 0.0
        
        # Total P&L = current AUM minus net deposits (matches what the charts show)
        current_aum = stats["current_aum"]
        net_deposits = account_summary.get("net_deposits", 0.0)
        stats["total_pnl_usd"] = current_aum - net_deposits
        if net_deposits > 0:
            stats["total_pnl_pct"] = (stats["total_pnl_usd"] / net_deposits) * 100
        else:
            stats["total_pnl_pct"] = 0.0
        
        if not trade_analysis.empty:
            stats["total_trades"] = len(trade_analysis)
            stats["total_fees"] = float(trade_analysis["fee"].sum())
            stats["total_volume"] = float(trade_analysis["notional"].sum())
            stats["avg_trade_size"] = float(trade_analysis["notional"].mean())
            
            winning_trades = trade_analysis[trade_analysis["net_pnl"] > 0]
            stats["winning_trades"] = len(winning_trades)
            stats["win_rate"] = (len(winning_trades) / len(trade_analysis) * 100) if len(trade_analysis) > 0 else 0.0
            
            if not winning_trades.empty:
                stats["avg_win"] = float(winning_trades["net_pnl"].mean())
            else:
                stats["avg_win"] = 0.0
            
            losing_trades = trade_analysis[trade_analysis["net_pnl"] < 0]
            if not losing_trades.empty:
                stats["avg_loss"] = float(losing_trades["net_pnl"].mean())
            else:
                stats["avg_loss"] = 0.0
        else:
            stats["total_trades"] = 0
            stats["total_fees"] = 0.0
            stats["total_volume"] = 0.0
            stats["avg_trade_size"] = 0.0
            stats["winning_trades"] = 0
            stats["win_rate"] = 0.0
            stats["avg_win"] = 0.0
            stats["avg_loss"] = 0.0
        
        if not funding_analysis.empty:
            stats["total_funding_paid"] = float(funding_analysis["funding_payment"].sum())
            stats["avg_funding_payment"] = float(funding_analysis["funding_payment"].mean())
            
            funding_by_coin = funding_analysis.groupby("coin")["funding_payment"].sum()
            stats["funding_by_coin"] = funding_by_coin.to_dict()
        else:
            stats["total_funding_paid"] = 0.0
            stats["avg_funding_payment"] = 0.0
            stats["funding_by_coin"] = {}
        
        stats["net_deposits"] = account_summary.get("net_deposits", 0.0)
        stats["total_deposits"] = account_summary.get("total_deposits", 0.0)
        stats["total_withdrawals"] = account_summary.get("total_withdrawals", 0.0)
        stats["spot_value"] = account_summary.get("spot_value", 0.0)
        stats["perp_value"] = account_summary.get("perp_value", 0.0)
        stats["unrealized_pnl"] = account_summary.get("unrealized_pnl", 0.0)

        # --- Position fields (spot + perp, assumed same token) ---
        stats["spot_position"] = account_summary.get("spot_position", 0.0)
        stats["perp_position"] = account_summary.get("perp_position", 0.0)
        stats["net_position"] = account_summary.get("net_position", 0.0)
        stats["position_token"] = account_summary.get("position_token", None)
        stats["last_perp_price"] = account_summary.get("last_perp_price", 0.0)
        stats["net_position_usd"] = account_summary.get("net_position_usd", 0.0)
        stats["position_vol"] = account_summary.get("position_vol", 0.0)
        stats["net_position_vol_usd"] = account_summary.get("net_position_vol_usd", 0.0)
        
        # Annualized P&L std dev from pnl_history file
        # Resample to daily (last snapshot per day) to avoid noise from
        # multiple intra-day snapshots, then recompute daily returns.
        if pnl_history is not None and not pnl_history.empty and "aum_usd" in pnl_history.columns:
            daily_hist = pnl_history[["aum_usd", "net_deposits"]].resample("D").last().dropna()
            if len(daily_hist) > 1:
                daily_pnl_pct = []
                for i in range(1, len(daily_hist)):
                    aum_change = daily_hist["aum_usd"].iloc[i] - daily_hist["aum_usd"].iloc[i - 1]
                    dep_change = daily_hist["net_deposits"].iloc[i] - daily_hist["net_deposits"].iloc[i - 1]
                    period_pnl = aum_change - dep_change
                    prev_aum = daily_hist["aum_usd"].iloc[i - 1]
                    if prev_aum > 0:
                        daily_pnl_pct.append((period_pnl / prev_aum) * 100)
                    else:
                        daily_pnl_pct.append(0.0)
                if len(daily_pnl_pct) > 1:
                    s = pd.Series(daily_pnl_pct)
                    daily_std = float(s.std())
                    daily_mean = float(s.mean())
                    stats["pnl_std_ann_pct"] = daily_std * math.sqrt(365)
                    stats["pnl_mean_ann_pct"] = daily_mean * 365
                else:
                    stats["pnl_std_ann_pct"] = 0.0
                    stats["pnl_mean_ann_pct"] = 0.0
            else:
                stats["pnl_std_ann_pct"] = 0.0
                stats["pnl_mean_ann_pct"] = 0.0
        else:
            stats["pnl_std_ann_pct"] = 0.0
            stats["pnl_mean_ann_pct"] = 0.0
        
        # Annualized average funding rate per coin over last 5, 10, 30 calendar days,
        # computed from market-wide hourly funding rate files in
        # {price_cache_dir}/funding/{coin}.parquet. Files that are more than
        # 2 days old are refreshed to fill missing rows.
        # funding_rate is per-hour; annualize by * 24 * 365, reported in %.
        # Keys: funding_rate_{coin}_{days}d_ann_pct  (e.g. funding_rate_ETH_5d_ann_pct)
        # Also stored: funding_rate_coins — ordered list of coin symbols with data
        stats["funding_rate_coins"] = []
        
        # Determine which coins to use: coins the user has held per funding_analysis
        if not funding_analysis.empty and "coin" in funding_analysis.columns:
            coins_to_load = list(funding_analysis["coin"].unique())
        else:
            coins_to_load = []
        
        if coins_to_load:
            try:
                market_rates = self._load_market_funding_rates(
                    coins=coins_to_load, stale_days=2
                )
                if not market_rates.empty and "funding_rate" in market_rates.columns and "coin" in market_rates.columns:
                    now_utc = pd.Timestamp.now(tz="UTC")
                    coins_with_data = []
                    for coin in coins_to_load:
                        coin_rates = market_rates[market_rates["coin"] == coin]
                        has_data = False
                        for days in [5, 10, 30]:
                            cutoff = now_utc - pd.Timedelta(days=days)
                            window = coin_rates[coin_rates.index >= cutoff]["funding_rate"]
                            if len(window) > 0:
                                avg_rate = float(window.mean())
                                stats[f"funding_rate_{coin}_{days}d_ann_pct"] = (
                                    avg_rate * 24 * 365 * 100
                                )
                                has_data = True
                            else:
                                stats[f"funding_rate_{coin}_{days}d_ann_pct"] = 0.0
                        if has_data:
                            coins_with_data.append(coin)
                    stats["funding_rate_coins"] = coins_with_data
            except Exception as e:
                logger.warning(f"Failed to compute market funding rate stats: {e}")
        
        return stats
    
    def _render_position_cards(self, stats: dict) -> str:
        """Render position metric cards for the HTML account summary section.

        Shows spot/perp/net positions, last price, net USD exposure, 30-day vol,
        and position vol (USD).  Assumption noted: spot and perp are the same token.
        """
        token = stats.get("position_token")
        if not token:
            return ""

        spot = stats.get("spot_position", 0.0)
        perp = stats.get("perp_position", 0.0)
        net = stats.get("net_position", 0.0)
        last_price = stats.get("last_perp_price", 0.0)
        net_usd = stats.get("net_position_usd", 0.0)
        vol_30d = stats.get("position_vol", 0.0)
        vol_usd = stats.get("net_position_vol_usd", 0.0)

        net_css = "positive" if net >= 0 else "negative"
        net_usd_css = "positive" if net_usd >= 0 else "negative"

        cards = [
            f'<div class="metric-card">'
            f'<div class="metric-label">Position Token <small style="font-size:0.7em;color:#888">(spot+perp assumed same)</small></div>'
            f'<div class="metric-value">{token}</div>'
            f'</div>',

            f'<div class="metric-card">'
            f'<div class="metric-label">Spot Position ({token})</div>'
            f'<div class="metric-value">{spot:,.4f}</div>'
            f'</div>',

            f'<div class="metric-card">'
            f'<div class="metric-label">Perp Position ({token})</div>'
            f'<div class="metric-value">{perp:,.4f}</div>'
            f'</div>',

            f'<div class="metric-card {net_css}">'
            f'<div class="metric-label">Net Position ({token})</div>'
            f'<div class="metric-value">{net:,.4f}</div>'
            f'</div>',

            f'<div class="metric-card">'
            f'<div class="metric-label">Last Perp Price ({token})</div>'
            f'<div class="metric-value">${last_price:,.2f}</div>'
            f'</div>',

            f'<div class="metric-card {net_usd_css}">'
            f'<div class="metric-label">Net Position (USD)</div>'
            f'<div class="metric-value">${net_usd:,.2f}</div>'
            f'</div>',

            f'<div class="metric-card">'
            f'<div class="metric-label">Vol per Hour ({token})</div>'
            f'<div class="metric-value">{vol_30d*100:.4f}%</div>'
            f'</div>',

            f'<div class="metric-card">'
            f'<div class="metric-label">Position Vol (USD, 1σ)</div>'
            f'<div class="metric-value">${vol_usd:,.2f}</div>'
            f'</div>',
        ]
        return "\n            ".join(cards)

    def _render_funding_rate_cards(self, stats: dict) -> str:
        """Render per-coin funding rate metric cards for the HTML account summary.
        
        Generates one card per coin per window (5d, 10d, 30d) using the keys
        stored by _calculate_summary_stats: funding_rate_{coin}_{days}d_ann_pct.
        """
        coins = stats.get("funding_rate_coins", [])
        if not coins:
            return ""
        cards = []
        for coin in coins:
            for days in [5, 10, 30]:
                key = f"funding_rate_{coin}_{days}d_ann_pct"
                value = stats.get(key, 0.0)
                css = "positive" if value >= 0 else "negative"
                sign = "+" if value >= 0 else ""
                cards.append(
                    f'<div class="metric-card {css}">'
                    f'<div class="metric-label">{coin} Funding {days}d (Ann.)</div>'
                    f'<div class="metric-value">{sign}{value:.2f}%</div>'
                    f"</div>"
                )
        return "\n            ".join(cards)

    def _render_tc_section(self, tc: Optional[dict]) -> str:
        """Render the Transaction Cost (TC) analysis section.

        ``tc`` is the dict returned by ``tc_analysis.generate_tc_analysis``.
        If ``tc`` is missing, or its status is not ``ok``, render a note and
        move on without raising.
        """
        if not tc:
            return """
    <div class="section">
        <h2>📐 TC Analysis</h2>
        <p><em>TC analysis was not produced for this report.</em></p>
    </div>
"""

        status = tc.get("status", "error")
        if status != "ok":
            msg = tc.get("message", "TC analysis unavailable.")
            return f"""
    <div class="section">
        <h2>📐 TC Analysis</h2>
        <p><em>{msg}</em></p>
    </div>
"""

        parts: list[str] = []
        parts.append("""
    <div class="section">
        <h2>📐 TC Analysis</h2>
        <p>Transaction-cost / market-impact analysis produced by <code>analysis.run_analysis</code>.</p>
""")

        metrics_html = tc.get("metrics_html") or ""
        if metrics_html:
            parts.append(f"""
        <h3>Summary Metrics</h3>
        <div style="overflow-x: auto;">{metrics_html}</div>
""")

        impact_table_html = tc.get("impact_table_html") or ""
        if impact_table_html:
            parts.append(f"""
        <h3>Market Impact Summary</h3>
        <div style="overflow-x: auto;">{impact_table_html}</div>
""")

        summary_stats_html = tc.get("summary_stats_html") or ""
        if summary_stats_html:
            parts.append(f"""
        <h3>Execution Summary Statistics</h3>
        <div style="overflow-x: auto;">{summary_stats_html}</div>
""")

        plots = tc.get("plots", {}) or {}
        plot_titles = {
            "slippage": "Market Impact Over Time (vs Close & Mid)",
            "impact_comparison": "Market Impact Comparison Over Time",
            "pnl_vs_impact": "Scaled PnL vs Market Impact (vs Open)",
            "raw_pnl_vs_impact": "Raw PnL Return vs Market Impact (vs Open)",
        }
        for key, title in plot_titles.items():
            b64 = plots.get(key)
            if b64:
                parts.append(f"""
        <h3>{title}</h3>
        <div class="chart">
            <img src="data:image/png;base64,{b64}" alt="{title}">
        </div>
""")

        successful_preview_html = tc.get("successful_preview_html") or ""
        if successful_preview_html:
            parts.append(f"""
        <h3>Successful Trades (last 10)</h3>
        <div style="overflow-x: auto;">{successful_preview_html}</div>
""")

        joined_preview_html = tc.get("joined_preview_html") or ""
        if joined_preview_html:
            parts.append(f"""
        <h3>Joined Trades + Market Data (last 10)</h3>
        <div style="overflow-x: auto;">{joined_preview_html}</div>
""")

        parts.append("""
    </div>
""")
        return "".join(parts)

    def create_visualizations(
        self,
        report_data: dict[str, Any],
        output_dir: str = ".",
    ) -> dict[str, str]:
        """Create visualizations for the report.
        
        Args:
            report_data: Report data from generate_report_data().
            output_dir: Directory to save visualization files.
        
        Returns:
            Dictionary mapping visualization names to base64-encoded image data.
        
        Raises:
            ReportGenerationError: If unable to create visualizations.
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            visualizations = {}
            
            plt.style.use('seaborn-v0_8-darkgrid')
            
            aum_img = self._create_aum_chart(report_data["aum_data"])
            if aum_img:
                visualizations["aum_chart"] = aum_img
            
            perf_img = self._create_performance_chart(report_data["performance_data"])
            if perf_img:
                visualizations["performance_chart"] = perf_img
            
            funding_img = self._create_funding_chart(report_data["funding_analysis"])
            if funding_img:
                visualizations["funding_chart"] = funding_img
            
            funding_by_coin_img = self._create_funding_by_coin_chart(report_data["funding_analysis"])
            if funding_by_coin_img:
                visualizations["funding_by_coin_chart"] = funding_by_coin_img
            
            cumulative_rate_img = self._create_cumulative_funding_rate_chart(report_data["funding_analysis"])
            if cumulative_rate_img:
                visualizations["cumulative_funding_rate_chart"] = cumulative_rate_img
            
            trade_img = self._create_trade_distribution_chart(report_data["trade_analysis"])
            if trade_img:
                visualizations["trade_distribution"] = trade_img
            
            logger.info("Created %d visualizations", len(visualizations))
            return visualizations
        except Exception as e:
            raise ReportGenerationError(f"Failed to create visualizations: {e}") from e
    
    def _create_aum_chart(self, aum_data: pd.DataFrame) -> Optional[str]:
        """Create AUM over time chart.
        
        Args:
            aum_data: AUM DataFrame.
        
        Returns:
            Base64-encoded image string or None if no data.
        """
        if aum_data.empty or "aum_usd" not in aum_data.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(aum_data.index, aum_data["aum_usd"], linewidth=2, color="#2E86AB")
        ax.fill_between(aum_data.index, aum_data["aum_usd"], alpha=0.3, color="#2E86AB")
        
        ax.set_title("Assets Under Management (AUM)", fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("AUM (USD)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _create_performance_chart(self, performance_data: pd.DataFrame) -> Optional[str]:
        """Create performance chart (P&L in $ and %).
        
        Args:
            performance_data: Performance DataFrame.
        
        Returns:
            Base64-encoded image string or None if no data.
        """
        if performance_data.empty:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Calculate cumulative values for the charts
        cumulative_pnl_usd = performance_data["pnl_usd"].cumsum()
        cumulative_pnl_pct = performance_data["pnl_pct"].cumsum()
        
        if "pnl_usd" in performance_data.columns:
            ax1.plot(performance_data.index, cumulative_pnl_usd, 
                    linewidth=2, color="#A23B72", label="Cumulative P&L (USD)")
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax1.fill_between(performance_data.index, cumulative_pnl_usd, 0,
                           where=(cumulative_pnl_usd >= 0), alpha=0.3, color="green")
            ax1.fill_between(performance_data.index, cumulative_pnl_usd, 0,
                           where=(cumulative_pnl_usd < 0), alpha=0.3, color="red")
            ax1.set_title("Cumulative P&L (USD)", fontsize=14, fontweight="bold")
            ax1.set_ylabel("P&L (USD)", fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        if "pnl_pct" in performance_data.columns:
            ax2.plot(performance_data.index, cumulative_pnl_pct, 
                    linewidth=2, color="#F18F01", label="Cumulative P&L (%)")
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax2.fill_between(performance_data.index, cumulative_pnl_pct, 0,
                           where=(cumulative_pnl_pct >= 0), alpha=0.3, color="green")
            ax2.fill_between(performance_data.index, cumulative_pnl_pct, 0,
                           where=(cumulative_pnl_pct < 0), alpha=0.3, color="red")
            ax2.set_title("Cumulative P&L (%)", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Date", fontsize=12)
            ax2.set_ylabel("P&L (%)", fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        fig.autofmt_xdate()
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _save_pnl_history(
        self,
        aum_usd: float,
        net_deposits: float,
        daily_funding_usd: float = 0.0,
    ) -> None:
        """Save current P&L snapshot to history file with position and price data.

        If net_deposits is 0 but the previous snapshot had deposits > 0,
        the last known net_deposits value is carried forward to prevent
        phantom P&L from erroneous zero values (e.g. ledger fetch failure).

        Args:
            aum_usd: Current assets under management.
            net_deposits: Current net deposits.
            daily_funding_usd: Daily cumulative funding paid/received in USD.
        """
        from pathlib import Path

        history_file = Path(self.pnl_history_file)
        current_time = pd.Timestamp.now(tz='UTC')

        # Create parent directories if they don't exist
        history_file.parent.mkdir(parents=True, exist_ok=True)

        # Carry forward last known net_deposits if new value is 0 but previous was > 0
        if net_deposits == 0.0 and history_file.exists():
            try:
                existing = pd.read_csv(history_file)
                if not existing.empty:
                    last_net_deposits = existing['net_deposits'].iloc[-1]
                    if last_net_deposits > 0:
                        logger.warning(
                            "net_deposits=0 but previous was %.2f; carrying forward",
                            last_net_deposits,
                        )
                        net_deposits = last_net_deposits
            except Exception as e:
                logger.warning(f"Could not read previous net_deposits for carry-forward: {e}")

        # Fetch position data and prices
        spot_position = 0.0
        perp_position = 0.0
        spot_price = 0.0
        perp_price = 0.0
        position_token = None

        try:
            positions = self.monitor.get_positions_summary()
            # Validate positions is a dict and extract numeric values
            if isinstance(positions, dict):
                spot_pos = positions.get("spot_position", 0.0)
                perp_pos = positions.get("perp_position", 0.0)
                token = positions.get("position_token")
                # Ensure numeric types (defensive against mocks returning non-numeric)
                spot_position = float(spot_pos) if spot_pos is not None else 0.0
                perp_position = float(perp_pos) if perp_pos is not None else 0.0
                position_token = token if isinstance(token, str) else None
        except Exception as e:
            logger.warning(f"Could not fetch positions for pnl_history: {e}")

        if position_token:
            # Fetch perp price (already cached from funding analysis)
            perp_series = self._fetch_price_series(position_token)
            if perp_series is not None and len(perp_series) > 0:
                perp_price = float(perp_series.iloc[-1])

            # Fetch spot price (may require new download to spot cache)
            spot_series = self._fetch_spot_price_series(position_token)
            if spot_series is not None and len(spot_series) > 0:
                spot_price = float(spot_series.iloc[-1])

        # Create new row with all columns
        new_row = pd.DataFrame({
            'datetime': [current_time],
            'aum_usd': [aum_usd],
            'net_deposits': [net_deposits],
            'spot_position': [spot_position],
            'perp_position': [perp_position],
            'spot_price': [spot_price],
            'perp_price': [perp_price],
            'funding_usd': [daily_funding_usd],
        })

        # Append to file or create new file
        if history_file.exists():
            new_row.to_csv(history_file, mode='a', header=False, index=False)
        else:
            new_row.to_csv(history_file, mode='w', header=True, index=False)

        logger.info(f"Saved P&L snapshot to {history_file}")
    
    def _load_pnl_history(self) -> pd.DataFrame:
        """Load P&L history from file and calculate performance metrics.
        
        Returns:
            DataFrame with historical P&L data including calculated pnl_usd and pnl_pct.
        """
        from pathlib import Path
        
        history_file = Path(self.pnl_history_file)
        
        if not history_file.exists():
            logger.info(f"P&L history file {history_file} does not exist")
            return pd.DataFrame()
        
        try:
            # Load historical data
            df = pd.read_csv(history_file)
            # Use format='ISO8601' to handle both tz-naive and tz-aware timestamps
            df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601', utc=True)
            df.set_index('datetime', inplace=True)
            
            # Sort by datetime
            df = df.sort_index()
            
            # Calculate P&L using same algorithm as performance_data
            # pnl_usd(t) = aum_usd(t) - aum_usd(t-1) - (net_deposits(t) - net_deposits(t-1))
            # pnl_pct(t) = pnl_usd(t) / aum_usd(t-1)
            df["pnl_usd"] = 0.0  # First row is zero
            df["pnl_pct"] = 0.0  # First row is zero
            
            if len(df) > 1:
                for i in range(1, len(df)):
                    aum_change = df["aum_usd"].iloc[i] - df["aum_usd"].iloc[i-1]
                    deposit_change = df["net_deposits"].iloc[i] - df["net_deposits"].iloc[i-1]
                    period_pnl = aum_change - deposit_change
                    
                    df.iloc[i, df.columns.get_loc("pnl_usd")] = period_pnl
                    
                    if df["aum_usd"].iloc[i-1] > 0:
                        period_pct = (period_pnl / df["aum_usd"].iloc[i-1]) * 100
                        df.iloc[i, df.columns.get_loc("pnl_pct")] = period_pct

            # Calculate exposure P&L: day-over-day change in position value
            # Position value = (perp_position × perp_price) + (spot_position × spot_price)
            df["exposure_pnl"] = 0.0

            # Check if position columns exist (backward compatibility with old CSV)
            has_position_data = all(
                col in df.columns
                for col in ["spot_position", "perp_position", "spot_price", "perp_price"]
            )

            if has_position_data and len(df) > 1:
                # Calculate position value for each row
                df["position_value"] = (
                    df["perp_position"] * df["perp_price"]
                    + df["spot_position"] * df["spot_price"]
                )

                # Calculate exposure P&L as day-over-day difference
                for i in range(1, len(df)):
                    exposure_pnl = (
                        df["position_value"].iloc[i] - df["position_value"].iloc[i - 1]
                    )
                    df.iloc[i, df.columns.get_loc("exposure_pnl")] = exposure_pnl

                # Drop temporary calculation column
                df = df.drop(columns=["position_value"])

            logger.info(f"Loaded P&L history with {len(df)} entries from {history_file}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading P&L history from {history_file}: {e}")
            return pd.DataFrame()
    
    def _create_funding_chart(self, funding_data: pd.DataFrame) -> Optional[str]:
        """Create funding costs chart.
        
        Args:
            funding_data: Funding DataFrame.
        
        Returns:
            Base64-encoded image string or None if no data.
        """
        if funding_data.empty or "funding_payment" not in funding_data.columns:
            return None
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        
        cumulative_funding = funding_data["funding_payment"].cumsum()
        ax1.plot(funding_data.index, cumulative_funding, linewidth=2, color="#6A4C93")
        ax1.fill_between(funding_data.index, cumulative_funding, alpha=0.3, color="#6A4C93")
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_title("Cumulative Funding Costs", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative Funding (USD)", fontsize=12)
        ax1.set_xlabel("Date", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _create_funding_by_coin_chart(self, funding_data: pd.DataFrame) -> Optional[str]:
        """Create funding by coin chart.
        
        Args:
            funding_data: Funding DataFrame.
        
        Returns:
            Base64-encoded image string or None if no data.
        """
        if funding_data.empty or "coin" not in funding_data.columns:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        funding_by_coin = funding_data.groupby("coin")["funding_payment"].sum().sort_values()
        
        colors = ['green' if x >= 0 else 'red' for x in funding_by_coin.values]
        funding_by_coin.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax.set_title("Total Funding by Coin", fontsize=14, fontweight="bold")
        ax.set_xlabel("Total Funding (USD)", fontsize=12)
        ax.set_ylabel("Coin", fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _create_cumulative_funding_rate_chart(self, funding_data: pd.DataFrame) -> Optional[str]:
        """Create cumulative funding rate chart in basis points.
        
        Shows the cumulative sum of funding rates (in bps) over the account history.
        
        Args:
            funding_data: Funding DataFrame with 'funding_rate' column.
        
        Returns:
            Base64-encoded image string or None if no data.
        """
        if funding_data.empty or "funding_rate" not in funding_data.columns:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Convert funding rate to basis points (multiply by 10000)
        funding_rate_bps = funding_data["funding_rate"] * 10000
        cumulative_rate_bps = funding_rate_bps.cumsum()
        
        ax.plot(funding_data.index, cumulative_rate_bps, linewidth=2, color="#E67E22")
        ax.fill_between(funding_data.index, cumulative_rate_bps, alpha=0.3, color="#E67E22")
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_title("Cumulative Funding Rate Over Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("Cumulative Funding Rate (bps)", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _create_trade_distribution_chart(self, trade_data: pd.DataFrame) -> Optional[str]:
        """Create trade distribution chart.
        
        Args:
            trade_data: Trade analysis DataFrame.
        
        Returns:
            Base64-encoded image string or None if no data.
        """
        if trade_data.empty:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        if "net_pnl" in trade_data.columns:
            winning_trades = len(trade_data[trade_data["net_pnl"] > 0])
            losing_trades = len(trade_data[trade_data["net_pnl"] < 0])
            breakeven_trades = len(trade_data[trade_data["net_pnl"] == 0])
            
            if winning_trades + losing_trades + breakeven_trades > 0:
                sizes = [winning_trades, losing_trades, breakeven_trades]
                labels = [f'Winning ({winning_trades})', f'Losing ({losing_trades})', 
                         f'Breakeven ({breakeven_trades})']
                colors = ['#2ECC71', '#E74C3C', '#95A5A6']
                
                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                       startangle=90, textprops={'fontsize': 11})
                ax1.set_title("Trade Win/Loss Distribution", fontsize=14, fontweight="bold")
        
        if "coin" in trade_data.columns and "notional" in trade_data.columns:
            volume_by_coin = trade_data.groupby("coin")["notional"].sum().sort_values(ascending=False).head(10)
            
            if not volume_by_coin.empty:
                volume_by_coin.plot(kind='barh', ax=ax2, color='#3498DB', alpha=0.7)
                ax2.set_title("Top 10 Coins by Volume", fontsize=14, fontweight="bold")
                ax2.set_xlabel("Total Volume (USD)", fontsize=12)
                ax2.set_ylabel("Coin", fontsize=12)
                ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig: Figure) -> str:
        """Convert matplotlib figure to base64-encoded string.
        
        Args:
            fig: Matplotlib figure.
        
        Returns:
            Base64-encoded image string.
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return image_base64
    
    def generate_html_report(
        self,
        report_data: dict[str, Any],
        visualizations: dict[str, str],
    ) -> str:
        """Generate HTML report content.
        
        Args:
            report_data: Report data from generate_report_data().
            visualizations: Visualization base64 strings from create_visualizations().
        
        Returns:
            HTML string for the report.
        
        Raises:
            ReportGenerationError: If unable to generate HTML report.
        """
        try:
            stats = report_data["summary_stats"]
            account_summary = report_data["account_summary"]
            generated_at = report_data["generated_at"]
            
            html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Performance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card.positive {{
            border-left-color: #2ECC71;
            background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        }}
        .metric-card.negative {{
            border-left-color: #E74C3C;
            background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .chart {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .positive-value {{
            color: #2ECC71;
            font-weight: 600;
        }}
        .negative-value {{
            color: #E74C3C;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Trading Performance Report</h1>
        <p><strong>Account:</strong> {self.account_address[:10]}...{self.account_address[-8:]}</p>
        <p><strong>Period:</strong> {report_data['period']}</p>
        <p><strong>Generated:</strong> {generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
    
    <div class="section">
        <h2>💰 Account Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card {'positive' if stats.get('current_aum', 0) > stats.get('net_deposits', 0) else 'negative'}">
                <div class="metric-label">Current AUM</div>
                <div class="metric-value">${stats.get('current_aum', 0):,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Net Deposits</div>
                <div class="metric-value">${stats.get('net_deposits', 0):,.2f}</div>
            </div>
            <div class="metric-card {'positive' if stats.get('total_pnl_usd', 0) >= 0 else 'negative'}">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value">${stats.get('total_pnl_usd', 0):,.2f}</div>
            </div>
            <div class="metric-card {'positive' if stats.get('total_pnl_pct', 0) >= 0 else 'negative'}">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{stats.get('total_pnl_pct', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Spot Value</div>
                <div class="metric-value">${stats.get('spot_value', 0):,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Perp Value</div>
                <div class="metric-value">${stats.get('perp_value', 0):,.2f}</div>
            </div>
            <div class="metric-card {'positive' if stats.get('unrealized_pnl', 0) >= 0 else 'negative'}">
                <div class="metric-label">Unrealized P&L</div>
                <div class="metric-value">${stats.get('unrealized_pnl', 0):,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Peak AUM</div>
                <div class="metric-value">${stats.get('peak_aum', 0):,.2f}</div>
            </div>
            {self._render_position_cards(stats)}
            {self._render_funding_rate_cards(stats)}
            <div class="metric-card {'positive' if stats.get('pnl_mean_ann_pct', 0) >= 0 else 'negative'}">
                <div class="metric-label">Avg Return (Ann.)</div>
                <div class="metric-value">{'+'if stats.get('pnl_mean_ann_pct',0)>=0 else ''}{stats.get('pnl_mean_ann_pct', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">P&L Std Dev (Ann.)</div>
                <div class="metric-value">{stats.get('pnl_std_ann_pct', 0):.2f}%</div>
            </div>
        </div>
    </div>
"""
            
            if "aum_chart" in visualizations:
                html += f"""
    <div class="section">
        <h2>📈 Assets Under Management</h2>
        <div class="chart">
            <img src="data:image/png;base64,{visualizations['aum_chart']}" alt="AUM Chart">
        </div>
    </div>
"""
            
            if "performance_chart" in visualizations:
                html += f"""
    <div class="section">
        <h2>💹 Performance Analysis</h2>
        <div class="chart">
            <img src="data:image/png;base64,{visualizations['performance_chart']}" alt="Performance Chart">
        </div>
"""
                
                # Add performance data table
                performance_data = report_data["performance_data"]
                if not performance_data.empty:
                    # Create formatted data for display (don't modify original dataframe)
                    display_data = performance_data.copy()
                    display_data['aum_usd'] = display_data['aum_usd'].round(2)
                    display_data['net_deposits'] = display_data['net_deposits'].round(2)
                    display_data['pnl_pct'] = display_data['pnl_pct'].round(2)
                    
                    # Add cumulative P&L columns for display
                    display_data['cumulative_pnl_usd'] = display_data['pnl_usd'].cumsum()
                    display_data['cumulative_pnl_pct'] = display_data['pnl_pct'].cumsum()
                    
                    html += """
        <h3>Performance Data</h3>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Date (UTC)</th>
                        <th>Date (EST)</th>
                        <th>AUM (USD)</th>
                        <th>Net Deposits (USD)</th>
                        <th>P&L (USD)</th>
                        <th>P&L (%)</th>
                        <th>Cumulative P&L (USD)</th>
                        <th>Cumulative P&L (%)</th>
                    </tr>
                </thead>
                <tbody>
"""
                    # Show last 10 entries in reverse chronological order
                    recent_performance = display_data.tail(10).sort_index(ascending=False)
                    for idx, row in recent_performance.iterrows():
                        pnl_class = "positive-value" if row["pnl_usd"] >= 0 else "negative-value"
                        pnl_pct_class = "positive-value" if row["pnl_pct"] >= 0 else "negative-value"
                        cum_pnl_class = "positive-value" if row["cumulative_pnl_usd"] >= 0 else "negative-value"
                        cum_pct_class = "positive-value" if row["cumulative_pnl_pct"] >= 0 else "negative-value"
                        
                        # Convert UTC to US/Eastern (handles EST/EDT automatically)
                        est_time = to_eastern(idx)
                        
                        html += f"""
                    <tr>
                        <td>{idx.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{est_time.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>${row['aum_usd']:,.2f}</td>
                        <td>${row['net_deposits']:,.2f}</td>
                        <td class="{pnl_class}">${row['pnl_usd']:,.2f}</td>
                        <td class="{pnl_pct_class}">{row['pnl_pct']:,.2f}</td>
                        <td class="{cum_pnl_class}">${row['cumulative_pnl_usd']:,.2f}</td>
                        <td class="{cum_pct_class}">{row['cumulative_pnl_pct']:,.2f}</td>
                    </tr>
"""
                    html += """
                </tbody>
            </table>
        </div>
"""
                
                html += """
    </div>
"""
            
            # Add Monthly Performance section
            performance_data = report_data.get("performance_data", pd.DataFrame())
            if not performance_data.empty:
                monthly_perf = self.generate_monthly_performance(performance_data)
                if not monthly_perf.empty:
                    html += """
    <div class="section">
        <h2>📅 Monthly Performance</h2>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>Starting AUM (USD)</th>
                        <th>Ending AUM (USD)</th>
                        <th>P&L (USD)</th>
                        <th>P&L (%)</th>
                        <th>Cumulative P&L (USD)</th>
                        <th>Cumulative P&L (%)</th>
                    </tr>
                </thead>
                <tbody>
"""
                    for _, mrow in monthly_perf.iterrows():
                        m_pnl_class = "positive-value" if mrow["pnl_usd"] >= 0 else "negative-value"
                        m_pct_class = "positive-value" if mrow["pnl_pct"] >= 0 else "negative-value"
                        m_cum_class = "positive-value" if mrow["cumulative_pnl_usd"] >= 0 else "negative-value"
                        m_cum_pct_class = "positive-value" if mrow["cumulative_pnl_pct"] >= 0 else "negative-value"
                        html += f"""
                    <tr>
                        <td>{mrow['month']}</td>
                        <td>${mrow['starting_aum']:,.2f}</td>
                        <td>${mrow['ending_aum']:,.2f}</td>
                        <td class="{m_pnl_class}">${mrow['pnl_usd']:,.2f}</td>
                        <td class="{m_pct_class}">{mrow['pnl_pct']:.2f}%</td>
                        <td class="{m_cum_class}">${mrow['cumulative_pnl_usd']:,.2f}</td>
                        <td class="{m_cum_pct_class}">{mrow['cumulative_pnl_pct']:.2f}%</td>
                    </tr>
"""
                    html += """
                </tbody>
            </table>
        </div>
    </div>
"""
            
            # Add Weekly Performance section
            weekly_perf = report_data.get("weekly_performance", pd.DataFrame())
            if not weekly_perf.empty:
                html += """
    <div class="section">
        <h2>📅 Weekly Performance</h2>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Week</th>
                        <th>Starting AUM (USD)</th>
                        <th>Ending AUM (USD)</th>
                        <th>P&L (USD)</th>
                        <th>P&L (%)</th>
                        <th>Cumulative P&L (USD)</th>
                        <th>Cumulative P&L (%)</th>
                    </tr>
                </thead>
                <tbody>
"""
                # Show only last 4 weeks, newest first
                for _, wrow in weekly_perf.tail(4).sort_index(ascending=False).iterrows():
                    w_pnl_class = "positive-value" if wrow["pnl_usd"] >= 0 else "negative-value"
                    w_pct_class = "positive-value" if wrow["pnl_pct"] >= 0 else "negative-value"
                    w_cum_class = "positive-value" if wrow["cumulative_pnl_usd"] >= 0 else "negative-value"
                    w_cum_pct_class = "positive-value" if wrow["cumulative_pnl_pct"] >= 0 else "negative-value"
                    html += f"""
                    <tr>
                        <td>{wrow['week']}</td>
                        <td>${wrow['starting_aum']:,.2f}</td>
                        <td>${wrow['ending_aum']:,.2f}</td>
                        <td class="{w_pnl_class}">${wrow['pnl_usd']:,.2f}</td>
                        <td class="{w_pct_class}">{wrow['pnl_pct']:.2f}%</td>
                        <td class="{w_cum_class}">${wrow['cumulative_pnl_usd']:,.2f}</td>
                        <td class="{w_cum_pct_class}">{wrow['cumulative_pnl_pct']:.2f}%</td>
                    </tr>
"""
                html += """
                </tbody>
            </table>
        </div>
    </div>
"""

            # Add Performance from file section (last 30 days only)
            pnl_history = report_data.get("pnl_history", pd.DataFrame())
            if not pnl_history.empty:
                # Filter to last 30 days
                cutoff_30d = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)
                display_history = pnl_history[pnl_history.index >= cutoff_30d].copy()
            else:
                display_history = pd.DataFrame()
            if not display_history.empty:
                display_history['aum_usd'] = display_history['aum_usd'].round(2)
                display_history['net_deposits'] = display_history['net_deposits'].round(2)
                display_history['pnl_pct'] = display_history['pnl_pct'].round(2)
                
                # Add cumulative P&L columns for display
                display_history['cumulative_pnl_usd'] = display_history['pnl_usd'].cumsum()
                display_history['cumulative_pnl_pct'] = display_history['pnl_pct'].cumsum()
                
                html += f"""
    <div class="section">
        <h3>📊 Performance from File (Last 30 Days)</h3>
        <p><strong>Historical P&L data from pnl_history.csv ({len(display_history)} entries)</strong></p>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Date (UTC)</th>
                        <th>Date (EST)</th>
                        <th>AUM (USD)</th>
                        <th>Net Deposits (USD)</th>
                        <th>Spot Price</th>
                        <th>Perp Price</th>
                        <th>Funding ($)</th>
                        <th>Exposure P&L ($)</th>
                        <th>P&L (USD)</th>
                        <th>P&L (%)</th>
                        <th>Cumulative P&L (USD)</th>
                        <th>Cumulative P&L (%)</th>
                    </tr>
                </thead>
                <tbody>
"""
                # Show all entries in reverse chronological order
                recent_history = display_history.sort_index(ascending=False)
                for idx, row in recent_history.iterrows():
                    pnl_class = "positive-value" if row["pnl_usd"] >= 0 else "negative-value"
                    pnl_pct_class = "positive-value" if row["pnl_pct"] >= 0 else "negative-value"
                    cum_pnl_class = "positive-value" if row["cumulative_pnl_usd"] >= 0 else "negative-value"
                    cum_pct_class = "positive-value" if row["cumulative_pnl_pct"] >= 0 else "negative-value"

                    # Handle new columns with backward compatibility (may not exist in old rows)
                    spot_price = row.get('spot_price', 0.0)
                    perp_price = row.get('perp_price', 0.0)
                    funding_usd = row.get('funding_usd', 0.0)
                    exposure_pnl = row.get('exposure_pnl', 0.0)

                    funding_class = "positive-value" if funding_usd >= 0 else "negative-value"
                    exposure_class = "positive-value" if exposure_pnl >= 0 else "negative-value"

                    # Convert UTC to US/Eastern (handles EST/EDT automatically)
                    est_time = to_eastern(idx)

                    html += f"""
                    <tr>
                        <td>{idx.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{est_time.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>${row['aum_usd']:,.2f}</td>
                        <td>${row['net_deposits']:,.2f}</td>
                        <td>${spot_price:,.2f}</td>
                        <td>${perp_price:,.2f}</td>
                        <td class="{funding_class}">${funding_usd:,.2f}</td>
                        <td class="{exposure_class}">${exposure_pnl:,.2f}</td>
                        <td class="{pnl_class}">${row['pnl_usd']:,.2f}</td>
                        <td class="{pnl_pct_class}">{row['pnl_pct']:,.2f}</td>
                        <td class="{cum_pnl_class}">${row['cumulative_pnl_usd']:,.2f}</td>
                        <td class="{cum_pct_class}">{row['cumulative_pnl_pct']:,.2f}</td>
                    </tr>
"""
                html += """
                </tbody>
            </table>
        </div>
    </div>
"""
            
            html += f"""
    <div class="section">
        <h2>🔄 Trading Activity</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{stats.get('total_trades', 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Volume</div>
                <div class="metric-value">${stats.get('total_volume', 0):,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Fees</div>
                <div class="metric-value">${stats.get('total_fees', 0):,.2f}</div>
            </div>
            <div class="metric-card {'positive' if stats.get('win_rate', 0) >= 50 else 'negative'}">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{stats.get('win_rate', 0):.1f}%</div>
            </div>
            <div class="metric-card positive">
                <div class="metric-label">Avg Win</div>
                <div class="metric-value">${stats.get('avg_win', 0):,.2f}</div>
            </div>
            <div class="metric-card negative">
                <div class="metric-label">Avg Loss</div>
                <div class="metric-value">${stats.get('avg_loss', 0):,.2f}</div>
            </div>
        </div>
"""
            
            if "trade_distribution" in visualizations:
                html += f"""
        <div class="chart">
            <img src="data:image/png;base64,{visualizations['trade_distribution']}" alt="Trade Distribution">
        </div>
"""
            
            trade_analysis = report_data["trade_analysis"]
            if not trade_analysis.empty:
                recent_trades = trade_analysis.tail(20).sort_index(ascending=False)
                html += """
        <h3>Recent Trades (Last 20)</h3>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Date (UTC)</th>
                        <th>Date (EST)</th>
                        <th>Coin</th>
                        <th>Side</th>
                        <th>Price</th>
                        <th>Size</th>
                        <th>Notional</th>
                        <th>Fee</th>
                        <th>Fee (bps)</th>
                        <th>Fee Token</th>
                        <th>Direction</th>
                        <th>Net P&L</th>
                    </tr>
                </thead>
                <tbody>
"""
                for idx, row in recent_trades.iterrows():
                    pnl_class = "positive-value" if row["net_pnl"] >= 0 else "negative-value"
                    
                    # Convert UTC to US/Eastern (handles EST/EDT automatically)
                    est_time = to_eastern(idx)
                    
                    html += f"""
                    <tr>
                        <td>{idx.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{est_time.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{row['coin']}</td>
                        <td>{row['side']}</td>
                        <td>${row['price']:,.4f}</td>
                        <td>{row['size']:,.4f}</td>
                        <td>${row['notional']:,.2f}</td>
                        <td>${row['fee']:,.2f}</td>
                        <td>{row['fee_bps']:.1f}</td>
                        <td>{row['feeToken']}</td>
                        <td>{row['dir']}</td>
                        <td class="{pnl_class}">${row['net_pnl']:,.2f}</td>
                    </tr>
"""
                html += """
                </tbody>
            </table>
        </div>
"""
            
                        
            html += """
    </div>
"""
            
            html += f"""
    <div class="section">
        <h2>💸 Funding Costs</h2>
        <div class="metrics-grid">
            <div class="metric-card {'positive' if stats.get('total_funding_paid', 0) >= 0 else 'negative'}">
                <div class="metric-label">Total Funding</div>
                <div class="metric-value">${stats.get('total_funding_paid', 0):,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Funding Payment</div>
                <div class="metric-value">${stats.get('avg_funding_payment', 0):,.4f}</div>
            </div>
        </div>
"""
            
            if "funding_chart" in visualizations:
                html += f"""
        <div class="chart">
            <img src="data:image/png;base64,{visualizations['funding_chart']}" alt="Funding Chart">
        </div>
"""
            
            if "cumulative_funding_rate_chart" in visualizations:
                html += f"""
        <div class="chart">
            <img src="data:image/png;base64,{visualizations['cumulative_funding_rate_chart']}" alt="Cumulative Funding Rate Chart">
        </div>
"""
            
            funding_by_coin = stats.get('funding_by_coin', {})
            if funding_by_coin:
                html += """
        <h3>Funding by Coin</h3>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Coin</th>
                        <th>Total Funding</th>
                    </tr>
                </thead>
                <tbody>
"""
                for coin, funding in sorted(funding_by_coin.items(), key=lambda x: x[1]):
                    funding_class = "positive-value" if funding >= 0 else "negative-value"
                    html += f"""
                    <tr>
                        <td>{coin}</td>
                        <td class="{funding_class}">${funding:,.4f}</td>
                    </tr>
"""
                html += """
                </tbody>
            </table>
        </div>
"""
            
            # Add detailed funding analysis table
            funding_analysis = report_data["funding_analysis"]
            if not funding_analysis.empty:
                # Create formatted data for display (don't modify original dataframe)
                display_funding = funding_analysis.copy()
                display_funding['funding_rate'] = (display_funding['funding_rate'] * 10000).round(2)  # Convert to basis points
                
                html += f"""
        <h3>Funding Analysis Details ({len(funding_analysis)} entries)</h3>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Date (UTC)</th>
                        <th>Date (EST)</th>
                        <th>Coin</th>
                        <th>Funding Payment (USD)</th>
                        <th>Position Size</th>
                        <th>Funding Rate (bps)</th>
                        <th>Token Price (USD)</th>
                        <th>Calculated Funding (USD)</th>
                    </tr>
                </thead>
                <tbody>
"""
                # Show last 20 entries in reverse chronological order
                recent_funding = display_funding.tail(20).sort_index(ascending=False)
                for idx, row in recent_funding.iterrows():
                    funding_class = "positive-value" if row["funding_payment"] >= 0 else "negative-value"
                    rate_class = "positive-value" if row["funding_rate"] >= 0 else "negative-value"
                    calc_funding_class = "positive-value" if row.get("calculated_funding", 0) >= 0 else "negative-value"
                    
                    # idx is already in UTC from token_data API
                    # Convert UTC to US/Eastern (handles EST/EDT automatically)
                    est_time = to_eastern(idx)
                    
                    html += f"""
                    <tr>
                        <td>{idx.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{est_time.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{row['coin']}</td>
                        <td class="{funding_class}">${row['funding_payment']:,.4f}</td>
                        <td>{row['position_size']:,.4f}</td>
                        <td class="{rate_class}">{row['funding_rate']:.2f}</td>
                        <td>${row.get('token_price', 0):,.2f}</td>
                        <td class="{calc_funding_class}">${row.get('calculated_funding', 0):,.4f}</td>
                    </tr>
"""
                html += """
                </tbody>
            </table>
        </div>
"""
            
            # Add Daily Funding table (last 30 days, newest first)
            daily_funding = report_data.get("daily_funding", pd.DataFrame())
            if not daily_funding.empty and "total_funding_usd" in daily_funding.columns:
                # Determine coin columns (all columns except 'date' and 'total_funding_usd')
                coin_cols = [c for c in daily_funding.columns if c not in ("date", "total_funding_usd")]
                # Show last 30 days newest first
                recent_daily = daily_funding.tail(30).sort_values("date", ascending=False)
                coin_headers = "".join(f"<th>{c}</th>" for c in coin_cols)
                html += f"""
        <h3>Daily Funding (Last 30 Days)</h3>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Total Funding (USD)</th>
                        {coin_headers}
                    </tr>
                </thead>
                <tbody>
"""
                for _, drow in recent_daily.iterrows():
                    total_class = "positive-value" if drow["total_funding_usd"] >= 0 else "negative-value"
                    coin_cells = "".join(
                        f'<td class="{"positive-value" if drow.get(c, 0) >= 0 else "negative-value"}">${drow.get(c, 0):,.4f}</td>'
                        for c in coin_cols
                    )
                    date_str = drow["date"].strftime('%Y-%m-%d') if hasattr(drow["date"], 'strftime') else str(drow["date"])
                    html += f"""
                    <tr>
                        <td>{date_str}</td>
                        <td class="{total_class}">${drow['total_funding_usd']:,.4f}</td>
                        {coin_cells}
                    </tr>
"""
                html += """
                </tbody>
            </table>
        </div>
"""

            # Add Weekly Funding table
            weekly_funding = report_data.get("weekly_funding", pd.DataFrame())
            if not weekly_funding.empty and "total_funding_usd" in weekly_funding.columns:
                html += """
        <h3>Weekly Funding</h3>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Week</th>
                        <th>Total Funding (USD)</th>
                    </tr>
                </thead>
                <tbody>
"""
                for _, wfrow in weekly_funding.sort_values("week", ascending=False).iterrows():
                    wf_class = "positive-value" if wfrow["total_funding_usd"] >= 0 else "negative-value"
                    html += f"""
                    <tr>
                        <td>{wfrow['week']}</td>
                        <td class="{wf_class}">${wfrow['total_funding_usd']:,.4f}</td>
                    </tr>
"""
                html += """
                </tbody>
            </table>
        </div>
"""

            html += """
    </div>
"""
            
            # Add funding by coin chart at the end
            if "funding_by_coin_chart" in visualizations:
                html += f"""
    <div class="section">
        <h2>💰 Funding by Coin</h2>
        <div class="chart">
            <img src="data:image/png;base64,{visualizations['funding_by_coin_chart']}" alt="Funding by Coin Chart">
        </div>
    </div>
"""
    
            html += self._render_tc_section(report_data.get("tc_analysis"))

            html += """
    </div>
    
    <div class="footer">
        <p>This report was automatically generated by the Trading Performance Reporter</p>
        <p>For questions or issues, please contact your system administrator</p>
    </div>
</body>
</html>
"""
            
            return html
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate HTML report: {e}") from e
    
    def generate_email_summary(self, report_data: dict[str, Any]) -> str:
        """Generate a plain text summary for email body.
        
        Args:
            report_data: Complete report data dictionary.
            
        Returns:
            Plain text summary string.
        """
        stats = report_data['summary_stats']
        account_summary = report_data['account_summary']
        period = report_data.get('period', 'allTime')
        
        # Format account address for display
        address = self.account_address
        short_address = f"{address[:6]}...{address[-4:]}"
        
        summary_lines = [
            "=" * 70,
            "TRADING PERFORMANCE REPORT",
            "=" * 70,
            "",
            f"Account: {short_address}",
            f"Period: {period}",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "",
            "=" * 70,
            "ACCOUNT SUMMARY",
            "=" * 70,
            "",
            f"Current AUM:        ${stats.get('current_aum', 0):,.2f}",
            f"Initial AUM:        ${stats.get('initial_aum', 0):,.2f}",
            f"Peak AUM:           ${stats.get('peak_aum', 0):,.2f}",
            f"Net Deposits:       ${account_summary.get('net_deposits', 0):,.2f}",
            "",
            f"Total P&L:          ${stats.get('total_pnl_usd', 0):,.2f} ({stats.get('total_pnl_pct', 0):.2f}%)",
            f"Unrealized P&L:     ${account_summary.get('unrealized_pnl', 0):,.2f}",
            "",
            f"Spot Value:         ${account_summary.get('spot_value', 0):,.2f}",
            f"Perpetual Value:    ${account_summary.get('perp_value', 0):,.2f}",
            "",
            "=" * 70,
            "CURRENT POSITION  (spot + perp assumed same token)",
            "=" * 70,
            "",
            f"Token:              {account_summary.get('position_token') or 'None'}",
            f"Spot Position:      {account_summary.get('spot_position', 0):,.4f}",
            f"Perp Position:      {account_summary.get('perp_position', 0):,.4f}",
            f"Net Position:       {account_summary.get('net_position', 0):,.4f}",
            f"Last Price:         ${account_summary.get('last_perp_price', 0):,.2f}",
            f"Net Position (USD): ${account_summary.get('net_position_usd', 0):,.2f}",
            f"Vol per Hour:       {account_summary.get('position_vol', 0)*100:.4f}%",
            f"Position Vol (USD): ${account_summary.get('net_position_vol_usd', 0):,.2f}",
            "",
            "=" * 70,
            "TRADING ACTIVITY",
            "=" * 70,
            "",
            f"Total Trades:       {stats.get('total_trades', 0):,}",
            f"Total Volume:       ${stats.get('total_volume', 0):,.2f}",
            f"Total Fees:         ${stats.get('total_fees', 0):,.2f}",
            "",
            f"Winning Trades:     {stats.get('winning_trades', 0):,}",
            f"Losing Trades:      {stats.get('losing_trades', 0):,}",
            f"Win Rate:           {stats.get('win_rate', 0):.1f}%",
            "",
            f"Average Win:        ${stats.get('avg_win', 0):,.2f}",
            f"Average Loss:       ${stats.get('avg_loss', 0):,.2f}",
            "",
            "=" * 70,
            "FUNDING COSTS",
            "=" * 70,
            "",
            f"Total Funding:      ${stats.get('total_funding_paid', 0):,.2f}",
            f"Avg Funding/Day:    ${stats.get('avg_funding_per_day', 0):,.2f}",
            "",
            "=" * 70,
            "",
            "📎 Full detailed report is attached as HTML file.",
            "   Open the attachment in your browser to view charts and detailed analysis.",
            "",
            "=" * 70,
        ]
        
        return "\n".join(summary_lines)
