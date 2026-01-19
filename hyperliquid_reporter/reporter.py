"""Hyperliquid-specific reporter implementation."""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timedelta
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
    
    def __init__(self, monitor: HyperliquidMonitor, account_address: str) -> None:
        """Initialize the Hyperliquid reporter.
        
        Args:
            monitor: HyperliquidMonitor instance.
            account_address: The account address to report on.
        """
        self.monitor = monitor
        self.account_address = account_address
        logger.info("Initialized HyperliquidReporter for account %s", account_address[:10] + "...")
    
    def generate_aum_data(self, period: str = "allTime") -> pd.DataFrame:
        """Generate AUM (Assets Under Management) data over time.
        
        Args:
            period: Time period for the data (e.g., "day", "week", "month", "allTime").
        
        Returns:
            DataFrame with datetime index and columns:
            - aum_usd: Account value in USD
        
        Raises:
            ReportGenerationError: If unable to generate AUM data.
        """
        try:
            df = self.monitor.get_portfolio_dataframe(period=period, data_type="account_value")
            
            if df.empty:
                logger.warning("No AUM data available for period: %s", period)
                return pd.DataFrame(columns=["aum_usd"])
            
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
            - pnl_usd: Cumulative P&L in USD
            - pnl_pct: Cumulative P&L as percentage
            - aum_usd: Account value for reference
        
        Raises:
            ReportGenerationError: If unable to generate performance data.
        """
        try:
            df = self.monitor.get_portfolio_dataframe(period=period, data_type="both")
            
            if df.empty:
                logger.warning("No performance data available for period: %s", period)
                return pd.DataFrame(columns=["pnl_usd", "pnl_pct", "aum_usd"])
            
            df = df.rename(columns={"account_value": "aum_usd"})
            df = df.rename(columns={"pnl": "pnl_usd"})
            
            if "aum_usd" in df.columns and not df["aum_usd"].empty:
                initial_value = df["aum_usd"].iloc[0]
                if initial_value > 0:
                    df["pnl_pct"] = (df["pnl_usd"] / initial_value) * 100
                else:
                    df["pnl_pct"] = 0.0
            else:
                df["pnl_pct"] = 0.0
            
            return df
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate performance data: {e}") from e
    
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
                    "fee", "closed_pnl", "net_pnl"
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
            
            result_df.index = trades_df.index
            
            return result_df
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate trade analysis: {e}") from e
    
    def generate_funding_analysis(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        lookback_days: int = 30,
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
                    "coin", "funding_payment", "position_size", "funding_rate"
                ])
            
            result_df = pd.DataFrame()
            result_df["coin"] = funding_df["coin"]
            result_df["funding_payment"] = funding_df["usdc"].astype(float)
            result_df["position_size"] = funding_df["szi"].astype(float)
            result_df["funding_rate"] = funding_df["fundingRate"].astype(float)
            
            result_df.index = funding_df.index
            
            return result_df
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate funding analysis: {e}") from e
    
    def generate_report_data(
        self,
        period: str = "allTime",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        lookback_days: int = 30,
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
            funding_analysis = self.generate_funding_analysis(
                start_time=start_time,
                end_time=end_time,
                lookback_days=lookback_days
            )
            
            account_summary = self.monitor.get_account_summary(lookback_days=3650)
            
            summary_stats = self._calculate_summary_stats(
                aum_data=aum_data,
                performance_data=performance_data,
                trade_analysis=trade_analysis,
                funding_analysis=funding_analysis,
                account_summary=account_summary,
            )
            
            return {
                "aum_data": aum_data,
                "performance_data": performance_data,
                "trade_analysis": trade_analysis,
                "funding_analysis": funding_analysis,
                "summary_stats": summary_stats,
                "account_summary": account_summary,
                "period": period,
                "generated_at": datetime.now(),
            }
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate report data: {e}") from e
    
    def _calculate_summary_stats(
        self,
        aum_data: pd.DataFrame,
        performance_data: pd.DataFrame,
        trade_analysis: pd.DataFrame,
        funding_analysis: pd.DataFrame,
        account_summary: dict[str, Any],
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
        
        if not aum_data.empty and "aum_usd" in aum_data.columns:
            stats["current_aum"] = float(aum_data["aum_usd"].iloc[-1])
            stats["initial_aum"] = float(aum_data["aum_usd"].iloc[0])
            stats["peak_aum"] = float(aum_data["aum_usd"].max())
            stats["min_aum"] = float(aum_data["aum_usd"].min())
        else:
            stats["current_aum"] = 0.0
            stats["initial_aum"] = 0.0
            stats["peak_aum"] = 0.0
            stats["min_aum"] = 0.0
        
        if not performance_data.empty and "pnl_usd" in performance_data.columns:
            stats["total_pnl_usd"] = float(performance_data["pnl_usd"].iloc[-1])
            if "pnl_pct" in performance_data.columns:
                stats["total_pnl_pct"] = float(performance_data["pnl_pct"].iloc[-1])
            else:
                stats["total_pnl_pct"] = 0.0
        else:
            stats["total_pnl_usd"] = 0.0
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
        
        return stats
    
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
        
        if "pnl_usd" in performance_data.columns:
            ax1.plot(performance_data.index, performance_data["pnl_usd"], 
                    linewidth=2, color="#A23B72", label="P&L (USD)")
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax1.fill_between(performance_data.index, performance_data["pnl_usd"], 0,
                           where=(performance_data["pnl_usd"] >= 0), alpha=0.3, color="green")
            ax1.fill_between(performance_data.index, performance_data["pnl_usd"], 0,
                           where=(performance_data["pnl_usd"] < 0), alpha=0.3, color="red")
            ax1.set_title("Cumulative P&L (USD)", fontsize=14, fontweight="bold")
            ax1.set_ylabel("P&L (USD)", fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        if "pnl_pct" in performance_data.columns:
            ax2.plot(performance_data.index, performance_data["pnl_pct"], 
                    linewidth=2, color="#F18F01", label="P&L (%)")
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax2.fill_between(performance_data.index, performance_data["pnl_pct"], 0,
                           where=(performance_data["pnl_pct"] >= 0), alpha=0.3, color="green")
            ax2.fill_between(performance_data.index, performance_data["pnl_pct"], 0,
                           where=(performance_data["pnl_pct"] < 0), alpha=0.3, color="red")
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
    
    def _create_funding_chart(self, funding_data: pd.DataFrame) -> Optional[str]:
        """Create funding costs chart.
        
        Args:
            funding_data: Funding DataFrame.
        
        Returns:
            Base64-encoded image string or None if no data.
        """
        if funding_data.empty or "funding_payment" not in funding_data.columns:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        cumulative_funding = funding_data["funding_payment"].cumsum()
        ax1.plot(funding_data.index, cumulative_funding, linewidth=2, color="#6A4C93")
        ax1.fill_between(funding_data.index, cumulative_funding, alpha=0.3, color="#6A4C93")
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_title("Cumulative Funding Costs", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative Funding (USD)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        if "coin" in funding_data.columns:
            funding_by_coin = funding_data.groupby("coin")["funding_payment"].sum().sort_values()
            
            colors = ['green' if x >= 0 else 'red' for x in funding_by_coin.values]
            funding_by_coin.plot(kind='barh', ax=ax2, color=colors, alpha=0.7)
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
            ax2.set_title("Total Funding by Coin", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Total Funding (USD)", fontsize=12)
            ax2.set_ylabel("Coin", fontsize=12)
            ax2.grid(True, alpha=0.3, axis='x')
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
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
        <p><strong>Generated:</strong> {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
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
                        <th>Date</th>
                        <th>Coin</th>
                        <th>Side</th>
                        <th>Price</th>
                        <th>Size</th>
                        <th>Notional</th>
                        <th>Fee</th>
                        <th>Net P&L</th>
                    </tr>
                </thead>
                <tbody>
"""
                for idx, row in recent_trades.iterrows():
                    pnl_class = "positive-value" if row["net_pnl"] >= 0 else "negative-value"
                    html += f"""
                    <tr>
                        <td>{idx.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{row['coin']}</td>
                        <td>{row['side']}</td>
                        <td>${row['price']:,.4f}</td>
                        <td>{row['size']:,.4f}</td>
                        <td>${row['notional']:,.2f}</td>
                        <td>${row['fee']:,.2f}</td>
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
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
