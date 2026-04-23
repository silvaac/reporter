"""
Market scanner: reads funding and perp parquet files, computes summary metrics,
and produces an HTML table suitable for emailing.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults (all overridable via function arguments)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "./data/hyperliquid"
DEFAULT_FUNDING_SUBDIR = "funding"
DEFAULT_PERP_SUBDIR = "perp"
DEFAULT_HOURS_PER_YEAR = 24 * 365          # annualisation factor for hourly data
DEFAULT_LOOKBACK_DAYS = (5, 10, 30)        # windows for mean funding
DEFAULT_STD_DAYS = 30                      # window for std calculations
DEFAULT_RETURN_DAYS = 30                   # window for perp return
DEFAULT_VOLUME_DAYS = 30                   # window for volume median


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _discover_paired_tokens(
    data_dir: str | Path,
    funding_subdir: str = DEFAULT_FUNDING_SUBDIR,
    perp_subdir: str = DEFAULT_PERP_SUBDIR,
) -> list[str]:
    """Return sorted list of tokens that exist in both funding/ and perp/."""
    data_dir = Path(data_dir)
    funding_dir = data_dir / funding_subdir
    perp_dir = data_dir / perp_subdir

    funding_tokens = {p.stem for p in funding_dir.glob("*.parquet")}
    # perp files are named like BTC_1h.parquet → strip the suffix
    perp_tokens = {p.stem.split("_")[0] for p in perp_dir.glob("*.parquet")}

    paired = sorted(funding_tokens & perp_tokens)
    return paired


def _load_funding(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _load_perp(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_scanner_table(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    funding_subdir: str = DEFAULT_FUNDING_SUBDIR,
    perp_subdir: str = DEFAULT_PERP_SUBDIR,
    lookback_days: tuple[int, ...] = DEFAULT_LOOKBACK_DAYS,
    std_days: int = DEFAULT_STD_DAYS,
    return_days: int = DEFAULT_RETURN_DAYS,
    volume_days: int = DEFAULT_VOLUME_DAYS,
    hours_per_year: int = DEFAULT_HOURS_PER_YEAR,
) -> pd.DataFrame:
    """Build the scanner summary DataFrame.

    Funding and perp data are inner-joined on datetime so that every
    calculation uses the same aligned history.

    Returns a DataFrame with one row per token.
    """
    data_dir = Path(data_dir)
    tokens = _discover_paired_tokens(data_dir, funding_subdir, perp_subdir)

    if not tokens:
        raise ValueError(
            f"No paired tokens found in {data_dir / funding_subdir} and "
            f"{data_dir / perp_subdir}"
        )

    rows: list[dict] = []

    for token in tokens:
        funding_path = data_dir / funding_subdir / f"{token}.parquet"
        perp_path = data_dir / perp_subdir / f"{token}_1h.parquet"

        funding = _load_funding(funding_path)
        perp = _load_perp(perp_path)

        # Inner-join so both series share the same datetime range
        merged = pd.merge(funding, perp, on="datetime", how="inner", suffixes=("_fund", "_perp"))
        merged = merged.sort_values("datetime").reset_index(drop=True)

        first_dt = merged["datetime"].iloc[0]
        last_dt = merged["datetime"].iloc[-1]
        span_days = (last_dt - first_dt).total_seconds() / 86400

        row: dict = {
            "Token": token,
            "First Date": first_dt.strftime("%Y-%m-%d"),
            "Last Date": last_dt.strftime("%Y-%m-%d"),
            "Days": round(span_days, 1),
        }

        # --- funding rate metrics ---
        for days in lookback_days:
            n_hours = days * 24
            tail = merged["funding_rate"].tail(n_hours)
            ann_mean = tail.mean() * hours_per_year * 100  # percentage
            row[f"{days}d Mean FR (Ann%)"] = round(ann_mean, 1)

        n_std_hours = std_days * 24
        tail_std = merged["funding_rate"].tail(n_std_hours)
        ann_std = tail_std.std() * np.sqrt(hours_per_year) * 100
        row[f"{std_days}d Std FR (Ann%)"] = round(ann_std, 1)

        # --- perp close-to-close metrics ---
        n_return_hours = return_days * 24
        perp_tail = merged.tail(n_return_hours + 1)  # +1 to get n returns
        close_returns = perp_tail["close"].pct_change().dropna()

        ann_return = close_returns.mean() * hours_per_year * 100
        ann_return_std = close_returns.std() * np.sqrt(hours_per_year) * 100
        row[f"{return_days}d Perp Return (Ann%)"] = round(ann_return, 1)
        row[f"{return_days}d Perp Std (Ann%)"] = round(ann_return_std, 1)

        # --- dollar volume median in millions (NOT annualised) ---
        n_vol_hours = volume_days * 24
        vol_tail = merged.tail(n_vol_hours)
        dollar_vol = vol_tail["close"] * vol_tail["volume"]
        row[f"{volume_days}d Med Hourly $Vol (M)"] = round(dollar_vol.median() / 1e6, 1)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# HTML generation  (mirrors existing report styling)
# ---------------------------------------------------------------------------

def generate_scan_html(
    df: pd.DataFrame,
    title: str = "Hyperliquid Funding & Perp Scanner (Hourly Data)",
) -> str:
    """Render the scanner DataFrame as a styled HTML document.

    Uses the same CSS palette as the existing trading report so that it
    looks consistent when sent in an email.
    """

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build table header
    headers = "".join(f"<th>{col}</th>" for col in df.columns)

    # Columns that are plain text (no colour / no % suffix)
    text_cols = {"Token", "First Date", "Last Date", "Days"}

    # Build table body
    body_rows = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            val = row[col]
            if col == "Token":
                cells += f"<td><strong>{val}</strong></td>"
            elif col in text_cols:
                cells += f"<td>{val}</td>"
            elif "$Vol" in col or "Vol" in col:
                cells += f"<td>{val:,.1f}</td>"
            else:
                css = "positive-value" if val >= 0 else "negative-value"
                cells += f'<td class="{css}">{val:,.1f}%</td>'
        body_rows += f"<tr>{cells}</tr>\n"

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
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
            font-size: 2em;
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
        <h1>{title}</h1>
        <p><strong>Generated:</strong> {generated_at}</p>
    </div>

    <div class="section">
        <h2>Scanner Results</h2>
        <div style="overflow-x: auto;">
            <table>
                <thead><tr>{headers}</tr></thead>
                <tbody>
{body_rows}
                </tbody>
            </table>
        </div>
    </div>

    <div class="footer">
        <p>This scan was automatically generated by the Market Scanner</p>
    </div>
</body>
</html>
"""
    return html
