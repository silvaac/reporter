"""Base reporter class for generating performance reports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
import pandas as pd


class BaseReporter(ABC):
    """Abstract base class for generating trading performance reports.
    
    This class provides an interface for generating comprehensive reports
    including AUM tracking, performance metrics, trade analysis, and funding costs.
    """
    
    @abstractmethod
    def generate_aum_data(
        self,
        period: str = "allTime",
    ) -> pd.DataFrame:
        """Generate AUM (Assets Under Management) data over time.
        
        Args:
            period: Time period for the data.
        
        Returns:
            DataFrame with datetime index and AUM values in USD.
        """
        pass
    
    @abstractmethod
    def generate_performance_data(
        self,
        period: str = "allTime",
    ) -> pd.DataFrame:
        """Generate performance data (P&L in $ and %).
        
        Args:
            period: Time period for the data.
        
        Returns:
            DataFrame with datetime index, P&L in USD, and percentage returns.
        """
        pass
    
    @abstractmethod
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
            DataFrame with trade details including costs, P&L, and fees.
        """
        pass
    
    @abstractmethod
    def generate_funding_analysis(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generate funding cost analysis for perpetual positions.
        
        Args:
            start_time: Optional start time for filtering.
            end_time: Optional end time for filtering.
        
        Returns:
            DataFrame with funding payments by coin and time.
        """
        pass
    
    @abstractmethod
    def generate_report_data(
        self,
        period: str = "allTime",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Generate all report data.
        
        Args:
            period: Time period for historical data.
            start_time: Optional start time for trade/funding data.
            end_time: Optional end time for trade/funding data.
        
        Returns:
            Dictionary containing all report components:
            - aum_data: AUM DataFrame
            - performance_data: Performance DataFrame
            - trade_analysis: Trade analysis DataFrame
            - funding_analysis: Funding analysis DataFrame
            - summary_stats: Summary statistics dictionary
        """
        pass
    
    @abstractmethod
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
            Dictionary mapping visualization names to file paths.
        """
        pass
    
    @abstractmethod
    def generate_html_report(
        self,
        report_data: dict[str, Any],
        visualizations: dict[str, str],
    ) -> str:
        """Generate HTML report content.
        
        Args:
            report_data: Report data from generate_report_data().
            visualizations: Visualization file paths from create_visualizations().
        
        Returns:
            HTML string for the report.
        """
        pass
