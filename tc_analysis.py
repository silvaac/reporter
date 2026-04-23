"""Transaction Cost (TC) analysis wrapper for the report.

This module is a thin wrapper around ``analysis.run_analysis`` whose logic
must NOT be modified. It:

- Validates input paths (trades csv, market parquet).
- Calls ``run_analysis`` with all analyses + plots enabled.
- Collects the generated PNG plots and HTML tables into a dictionary that
  the HTML report can embed directly.
- Returns a ``status`` flag so the report can render a friendly note and
  keep going when inputs are missing or invalid.

The ``run_analysis`` function writes its artifacts into the current working
directory with hardcoded filenames. To avoid polluting the repo root, this
wrapper temporarily chdir's into an output directory and restores cwd after.
"""

from __future__ import annotations

import base64
import logging
import os
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


PLOT_FILES = {
    "slippage": "plot_slippage.png",
    "impact_comparison": "plot_impact_comparison.png",
    "pnl_vs_impact": "plot_pnl_vs_impact.png",
    "raw_pnl_vs_impact": "plot_raw_pnl_vs_impact.png",
}


def _png_to_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _describe_to_html(series_map: dict[str, Any]) -> str:
    """Convert dict of {col: pandas Series from describe()} to HTML."""
    frames = []
    for col, stats in series_map.items():
        if isinstance(stats, pd.Series):
            frames.append(stats.rename(col))
        else:
            frames.append(pd.Series(stats, name=col))
    if not frames:
        return ""
    combined = pd.concat(frames, axis=1)
    return combined.to_html(classes="tc-table", border=0, float_format=lambda v: f"{v:,.6f}")




def _summary_metrics_html(results: dict[str, Any]) -> str:
    rows = [
        ("Mean Impact vs Close (bps)", results.get("mean_impact_vs_close_bps")),
        ("Mean Impact vs Mid (bps)", results.get("mean_impact_vs_mid_bps")),
        ("Mean Impact vs Open (bps)", results.get("mean_impact_vs_open_bps")),
        ("Mean Overnight Impact (bps)", results.get("mean_impact_overnight_bps")),
        ("Cumulative PnL ($)", results.get("cumulative_pnl")),
        ("Cumulative Raw PnL Return", results.get("cumulative_raw_pnl")),
    ]
    body_rows = []
    for label, val in rows:
        if val is None:
            continue
        try:
            val_str = f"{float(val):,.6f}"
        except (TypeError, ValueError):
            val_str = str(val)
        body_rows.append(f"<tr><td>{label}</td><td>{val_str}</td></tr>")
    if not body_rows:
        return ""
    return (
        '<table class="tc-table"><thead><tr><th>Metric</th><th>Value</th>'
        '</tr></thead><tbody>' + "".join(body_rows) + '</tbody></table>'
    )


def generate_tc_analysis(
    trades_csv_path: str,
    market_parquet_path: str,
    output_dir: str | Path = "reports/tc",
) -> dict[str, Any]:
    """Run TC analysis and package results for the HTML report.

    Parameters
    ----------
    trades_csv_path:
        Path to the cycle/trades CSV (e.g. ``ringo_trades.csv``). If empty
        string or missing, analysis is skipped and a note is returned.
    market_parquet_path:
        Path to the market parquet file (e.g. ``ETH-USD.parquet``). If
        empty string or missing, analysis is skipped and a note is returned.
    output_dir:
        Directory where TC plots/tables will be written.

    Returns
    -------
    dict with keys:
        status: "ok" | "skipped_empty_path" | "file_not_found" | "error"
        message: human-readable note (for non-ok statuses)
        impact_table_html, summary_stats_html, metrics_html,
        joined_preview_html, successful_preview_html: str
        plots: dict[str, base64 png str]
    """
    result: dict[str, Any] = {
        "status": "ok",
        "message": "",
        "impact_table_html": "",
        "summary_stats_html": "",
        "metrics_html": "",
        "joined_preview_html": "",
        "successful_preview_html": "",
        "plots": {},
    }

    # ── Path validation ──────────────────────────────────────────────────
    if not trades_csv_path or not market_parquet_path:
        result["status"] = "skipped_empty_path"
        missing = []
        if not trades_csv_path:
            missing.append("trades_csv_path")
        if not market_parquet_path:
            missing.append("market_parquet_path")
        result["message"] = (
            f"TC analysis skipped: empty path provided for {', '.join(missing)}."
        )
        logger.info(result["message"])
        return result

    trades_abs = Path(trades_csv_path).resolve()
    market_abs = Path(market_parquet_path).resolve()

    missing_files = []
    if not trades_abs.is_file():
        missing_files.append(str(trades_abs))
    if not market_abs.is_file():
        missing_files.append(str(market_abs))
    if missing_files:
        result["status"] = "file_not_found"
        result["message"] = (
            "TC analysis skipped: file(s) not found: " + ", ".join(missing_files)
        )
        logger.warning(result["message"])
        return result

    # ── Run analysis (analysis.py writes artifacts to CWD) ──────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_abs = out_path.resolve()

    original_cwd = os.getcwd()
    try:
        from analysis import run_analysis  # imported here to surface import errors cleanly
        from great_tables import GT as _GT

        _original_gt_save = _GT.save

        def _gt_save_no_selenium(self, output_path, *args, **kwargs):
            try:
                _original_gt_save(self, output_path, *args, **kwargs)
            except ImportError:
                logger.warning("GT.save skipped (selenium not available): %s", output_path)

        _GT.save = _gt_save_no_selenium

        os.chdir(out_abs)
        results = run_analysis(
            trades_csv_path=str(trades_abs),
            market_parquet_path=str(market_abs),
            run_summary_stats=True,
            run_slippage_analysis=True,
            run_market_join=True,
            run_plots=True,
        )
    except Exception as exc:  # pragma: no cover - surfaces to report
        result["status"] = "error"
        result["message"] = f"TC analysis failed: {exc}"
        logger.error("TC analysis error:\n%s", traceback.format_exc())
        return result
    finally:
        os.chdir(original_cwd)
        try:
            _GT.save = _original_gt_save
        except Exception:
            pass

    # ── Package outputs ──────────────────────────────────────────────────
    impact_table = results.get("impact_table")
    if impact_table is not None:
        try:
            result["impact_table_html"] = impact_table.as_raw_html()
        except Exception as exc:
            logger.warning("Could not render impact_table html: %s", exc)

    result["summary_stats_html"] = _describe_to_html(results.get("summary_stats", {}))
    result["metrics_html"] = _summary_metrics_html(results)
    for key, df_key in [("successful_preview_html", "successful_trades"), ("joined_preview_html", "joined_df")]:
        df = results.get(df_key)
        if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
            result[key] = df.tail(10).to_html(classes="tc-table", border=0, index=False,
                                              float_format=lambda v: f"{v:,.6f}")

    plots: dict[str, str] = {}
    for key, fname in PLOT_FILES.items():
        b64 = _png_to_base64(out_abs / fname)
        if b64:
            plots[key] = b64
    result["plots"] = plots

    return result
