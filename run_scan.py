#!/usr/bin/env python3
"""
Run Market Scanner

Reads funding and perp data, computes summary metrics, and emails
an HTML table with the results.

Usage:
    python run_scan.py
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from scanner import compute_scanner_table, generate_scan_html
from email_reporter import send_report_email


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONFIGURATION – adjust as needed
# ---------------------------------------------------------------------------
DATA_DIR = "./data/hyperliquid"
FUNDING_SUBDIR = "funding"
PERP_SUBDIR = "perp"

LOOKBACK_DAYS = (5, 10, 30)   # windows for mean funding rate
STD_DAYS = 30                  # window for funding rate std
RETURN_DAYS = 30               # window for perp return
VOLUME_DAYS = 30               # window for volume median
HOURS_PER_YEAR = 24 * 365      # annualisation factor for hourly data

EMAIL_TO = "silvaac@yahoo.com"
EMAIL_SUBJECT_TEMPLATE = "Market Scan - {date}"
OUTPUT_DIR = "reports"


def main() -> int:
    """Generate scanner table and email it."""
    try:
        logger.info("=" * 60)
        logger.info("MARKET SCANNER")
        logger.info("=" * 60)

        logger.info("Computing scanner table from %s ...", DATA_DIR)
        df = compute_scanner_table(
            data_dir=DATA_DIR,
            funding_subdir=FUNDING_SUBDIR,
            perp_subdir=PERP_SUBDIR,
            lookback_days=LOOKBACK_DAYS,
            std_days=STD_DAYS,
            return_days=RETURN_DAYS,
            volume_days=VOLUME_DAYS,
            hours_per_year=HOURS_PER_YEAR,
        )

        logger.info("Tokens found: %s", ", ".join(df["Token"].tolist()))
        logger.info("\n%s", df.to_string(index=False))

        logger.info("Generating HTML ...")
        html_content = generate_scan_html(df)

        # Save a local copy
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"scan_{timestamp}.html"
        report_file.write_text(html_content, encoding="utf-8")
        logger.info("Saved HTML to %s", report_file)

        # Email
        subject = EMAIL_SUBJECT_TEMPLATE.format(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
        summary_text = (
            "Hyperliquid Funding & Perp Scanner\n"
            "Open the attached HTML file for the full table.\n"
        )

        logger.info("Sending email to %s ...", EMAIL_TO)
        result = send_report_email(
            to=EMAIL_TO,
            subject=subject,
            summary_text=summary_text,
            html_content=html_content,
            attachment_filename=f"scan_{timestamp}.html",
        )

        if result == 0:
            logger.info("Email sent successfully to %s", EMAIL_TO)
        else:
            logger.error("Failed to send email. Report saved at %s", report_file)

        logger.info("=" * 60)
        return result

    except Exception as e:
        logger.exception("Scanner failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
