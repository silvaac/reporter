#!/usr/bin/env python3
"""
Run Trading Performance Report

This script generates and emails a comprehensive trading performance report
for a Hyperliquid account.

Usage:
    python run_report.py

Configuration:
    Edit the CONFIGURATION section below to customize report parameters.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from config import load_config
from hyperliquid_reporter.monitoring import HyperliquidMonitor
from hyperliquid_reporter.reporter import HyperliquidReporter
from email_reporter import send_report_email


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CONFIG_FILE = "config_hyperliquid.json"
TESTNET = False
REPORT_PERIOD = "allTime"
LOOKBACK_DAYS = 30
EMAIL_TO = "silvaac@yahoo.com"
EMAIL_SUBJECT_TEMPLATE = "Performance Report - {account} - {date}"
OUTPUT_DIR = "reports"


def main():
    """Generate and send trading performance report."""
    try:
        logger.info("=" * 80)
        logger.info("TRADING PERFORMANCE REPORT GENERATOR")
        logger.info("=" * 80)
        
        logger.info("Loading configuration from: %s", CONFIG_FILE)
        config = load_config(
            config_path=CONFIG_FILE,
            testnet=TESTNET
        )
        
        account_address = config['account_address']
        info = config['_hyperliquid_info']
        
        logger.info("Account: %s", account_address[:10] + "..." + account_address[-8:])
        logger.info("Network: %s", "Testnet" if TESTNET else "Mainnet")
        
        logger.info("Initializing monitor...")
        monitor = HyperliquidMonitor(info=info, address=account_address)
        
        logger.info("Initializing reporter...")
        reporter = HyperliquidReporter(monitor=monitor, account_address=account_address)
        
        logger.info("Generating report data (period: %s)...", REPORT_PERIOD)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=LOOKBACK_DAYS)
        
        report_data = reporter.generate_report_data(
            period=REPORT_PERIOD,
            start_time=start_time,
            end_time=end_time,
            lookback_days=LOOKBACK_DAYS
        )
        
        logger.info("Creating visualizations...")
        visualizations = reporter.create_visualizations(
            report_data=report_data,
            output_dir=OUTPUT_DIR
        )
        
        logger.info("Generating HTML report...")
        html_content = reporter.generate_html_report(
            report_data=report_data,
            visualizations=visualizations
        )
        
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_filename = f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = output_path / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info("Report saved to: %s", report_path)
        
        logger.info("Generating email summary...")
        email_summary = reporter.generate_email_summary(report_data)
        
        logger.info("Sending report via email...")
        # Create subject with account identifier
        short_address = f"{account_address[:6]}...{account_address[-4:]}"
        network = "Testnet" if TESTNET else "Mainnet"
        subject = EMAIL_SUBJECT_TEMPLATE.format(
            account=short_address,
            date=datetime.now().strftime('%Y-%m-%d')
        )
        subject = f"{subject} [{network}]"
        
        result = send_report_email(
            to=EMAIL_TO,
            subject=subject,
            summary_text=email_summary,
            html_content=html_content,
            attachment_filename=report_filename
        )
        
        if result == 0:
            logger.info("✓ Report sent successfully to: %s", EMAIL_TO)
        else:
            logger.error("✗ Failed to send report via email")
            logger.info("Report is still available at: %s", report_path)
        
        logger.info("=" * 80)
        logger.info("REPORT SUMMARY")
        logger.info("=" * 80)
        stats = report_data['summary_stats']
        logger.info("Current AUM: $%.2f", stats.get('current_aum', 0))
        logger.info("Total P&L: $%.2f (%.2f%%)", 
                   stats.get('total_pnl_usd', 0),
                   stats.get('total_pnl_pct', 0))
        logger.info("Total Trades: %d", stats.get('total_trades', 0))
        logger.info("Win Rate: %.1f%%", stats.get('win_rate', 0))
        logger.info("Total Fees: $%.2f", stats.get('total_fees', 0))
        logger.info("Total Funding: $%.2f", stats.get('total_funding_paid', 0))
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.exception("Failed to generate report: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
