import numpy as np
import pandas as pd
from great_tables import GT, style, loc, google_font
from plotnine import ggplot, aes, geom_line, geom_point, geom_smooth, geom_hline, ggtitle, labs, scale_color_manual, theme, element_text

from read_csv_columns import load_csv, convert_datetime_columns, build_cycle_execution_df


def _build_impact_table(metrics: dict, output_path: str = "table_impact_summary.png") -> GT:
    rows = [
        {
            "Metric": "Impact vs Close",
            "Benchmark": "Execution vs API Close",
            "Mean (bps)": round(metrics["mean_impact_vs_close_bps"], 4),
            "N Trades": metrics["n_trades"],
        },
        {
            "Metric": "Impact vs Mid",
            "Benchmark": "Execution vs Mid at Submit",
            "Mean (bps)": round(metrics["mean_impact_vs_mid_bps"], 4),
            "N Trades": metrics["n_trades"],
        },
        {
            "Metric": "Impact vs Open",
            "Benchmark": "Execution vs Next Bar Open",
            "Mean (bps)": round(metrics["mean_impact_vs_open_bps"], 4),
            "N Trades": metrics["n_trades_joined"],
        },
        {
            "Metric": "Overnight Impact",
            "Benchmark": "Open vs API Close (overnight gap)",
            "Mean (bps)": round(metrics["mean_impact_overnight_bps"], 4),
            "N Trades": metrics["n_trades_joined"],
        },
    ]

    df = pd.DataFrame(rows)

    gt = (
        GT(df)
        .tab_header(
            title="Market Impact Summary",
            subtitle="Mean impact in basis points across all successful trades",
        )
        .fmt_number(columns="Mean (bps)", decimals=4)
        .fmt_integer(columns="N Trades")
        .cols_align(align="center", columns=["Mean (bps)", "N Trades"])
        .cols_align(align="left", columns=["Metric", "Benchmark"])
        .tab_style(
            style=style.fill(color="#e8f4e8"),
            locations=loc.body(columns="Mean (bps)", rows=df.index[df["Mean (bps)"] > 0].tolist()),
        )
        .tab_style(
            style=style.fill(color="#fde8e8"),
            locations=loc.body(columns="Mean (bps)", rows=df.index[df["Mean (bps)"] < 0].tolist()),
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .tab_source_note("Green = positive impact (favorable). Red = negative impact (cost).")
        .opt_table_font(font=google_font("IBM Plex Mono"))
    )

    gt.save(output_path)
    print(f"Saved: {output_path}")

    html_path = output_path.rsplit(".", 1)[0] + ".html"
    with open(html_path, "w") as f:
        f.write(gt.as_raw_html())
    print(f"Saved: {html_path}")

    return gt


def run_analysis(
    trades_csv_path: str = "ringo_trades.csv",
    market_parquet_path: str = "ETH-USD.parquet",
    run_summary_stats: bool = True,
    run_slippage_analysis: bool = True,
    run_market_join: bool = True,
    run_plots: bool = True,
) -> dict:
    # ── Load & prepare trades ──────────────────────────────────────────────
    raw_df = load_csv(trades_csv_path)
    converted_df = convert_datetime_columns(raw_df)
    execution_df = build_cycle_execution_df(converted_df)

    results = {}

    # ── Summary statistics ─────────────────────────────────────────────────
    if run_summary_stats:
        numeric_cols = ["execution_price", "mid_price_at_submit", "trade_size_signed"]
        summary = {
            col: execution_df[col].describe()
            for col in numeric_cols
            if col in execution_df.columns
        }
        results["summary_stats"] = summary
        for col, stats in summary.items():
            print(f"\n{col}:")
            print(stats)

    # ── Slippage / market-impact analysis ─────────────────────────────────
    if run_slippage_analysis:
        successful_trades = execution_df[execution_df["success"] == True].copy()

        successful_trades["execution_latency"] = (
            successful_trades["execution_datetime_utc"] - successful_trades["submission_datetime_utc"]
        )
        successful_trades["slippage_vs_close"] = (
            -1 * (successful_trades["execution_price"] - successful_trades["api_Close"])
            * successful_trades["trade_size_signed"]
        )
        successful_trades["slippage_vs_mid"] = (
            -1 * (successful_trades["execution_price"] - successful_trades["mid_price_at_submit"])
            * successful_trades["trade_size_signed"]
        )
        successful_trades["trade_notional"] = (
            np.abs(successful_trades["trade_size_signed"]) * successful_trades["api_Close"]
        )
        successful_trades["impact_vs_close"] = (
            successful_trades["slippage_vs_close"] / successful_trades["trade_notional"]
        )
        successful_trades["impact_vs_mid"] = (
            successful_trades["slippage_vs_mid"] / successful_trades["trade_notional"]
        )

        mean_impact_vs_close = 1e4 * successful_trades["impact_vs_close"].mean()
        mean_impact_vs_mid = 1e4 * successful_trades["impact_vs_mid"].mean()
        print(f"\nMean impact vs close (bps): {mean_impact_vs_close:.4f}")
        print(f"Mean impact vs mid   (bps): {mean_impact_vs_mid:.4f}")

        results["successful_trades"] = successful_trades
        results["mean_impact_vs_close_bps"] = mean_impact_vs_close
        results["mean_impact_vs_mid_bps"] = mean_impact_vs_mid

        if run_plots:
            slippage_long = (
                successful_trades[["execution_datetime_utc", "impact_vs_mid", "impact_vs_close"]]
                .melt(id_vars="execution_datetime_utc", var_name="series", value_name="impact")
            )
            slippage_long["series"] = slippage_long["series"].replace({
                "impact_vs_mid": "vs Mid",
                "impact_vs_close": "vs Close",
            })

            slippage_plot = (
                ggplot(slippage_long, aes("execution_datetime_utc", "impact", color="series"))
                + geom_line()
                + scale_color_manual(values={"vs Mid": "steelblue", "vs Close": "red"})
                + ggtitle("Market Impact Over Time")
                + labs(x="Execution Date (UTC)", y="Impact (raw)", color="Benchmark")
                + theme(plot_title=element_text(weight="bold"))
            )
            slippage_plot.save("plot_slippage.png", dpi=150, verbose=False)
            print("Saved: plot_slippage.png")
            results["slippage_plot"] = slippage_plot

    # ── Market-data join & PnL ─────────────────────────────────────────────
    if run_market_join:
        if "successful_trades" not in results:
            raise ValueError(
                "run_market_join requires run_slippage_analysis=True to produce successful_trades"
            )

        trades_rounded = results["successful_trades"].copy()
        trades_rounded["api_run_time"] = (
            pd.to_datetime(trades_rounded["api_run_time"], utc=True)
            .dt.floor("1h")
            .dt.tz_localize(None)
        )

        market_df = pd.read_parquet(market_parquet_path)
        market_df["datetime"] = pd.to_datetime(market_df["datetime"], utc=True).dt.tz_localize(None)
        market_df["open_lag1"] = market_df["open"].shift(-1)

        joined_df = trades_rounded.merge(
            market_df,
            left_on="api_run_time",
            right_on="datetime",
            how="inner",
        )

        joined_df["slippage_vs_open"] = (
            -1 * (joined_df["execution_price"] - joined_df["open"]) * joined_df["trade_size_signed"]
        )
        joined_df["impact_vs_open"] = joined_df["slippage_vs_open"] / joined_df["trade_notional"]
        joined_df["overnight_slippage"] = (
            -1 * (joined_df["open"] - joined_df["api_Close"]) * joined_df["trade_size_signed"]
        )
        joined_df["impact_overnight"] = joined_df["overnight_slippage"] / joined_df["trade_notional"]
        joined_df["pnl"] = joined_df["target_position_scaled"] * (joined_df["close"] - joined_df["open"])
        joined_df["raw_pnl_return"] = (
            np.sign(joined_df["api_N"]) * (joined_df["close"] - joined_df["open"]) / joined_df["open"]
        )

        mean_impact_vs_open = 1e4 * joined_df["impact_vs_open"].mean()
        mean_impact_overnight = 1e4 * joined_df["impact_overnight"].mean()
        cumulative_pnl = joined_df["pnl"].sum()
        cumulative_raw_pnl = (joined_df["raw_pnl_return"] + 1).prod() - 1

        print(f"\nMean impact vs open  (bps): {mean_impact_vs_open:.4f}")
        print(f"Mean overnight impact(bps): {mean_impact_overnight:.4f}")
        print(f"Cumulative PnL ($):         {cumulative_pnl:.6f}")
        print(f"Cumulative raw PnL return (%): {cumulative_raw_pnl:.6f}")

        results["joined_df"] = joined_df
        results["mean_impact_vs_open_bps"] = mean_impact_vs_open
        results["mean_impact_overnight_bps"] = mean_impact_overnight
        results["cumulative_pnl"] = cumulative_pnl
        results["cumulative_raw_pnl"] = cumulative_raw_pnl

        impact_table = _build_impact_table({
            "mean_impact_vs_close_bps": results["mean_impact_vs_close_bps"],
            "mean_impact_vs_mid_bps": results["mean_impact_vs_mid_bps"],
            "mean_impact_vs_open_bps": mean_impact_vs_open,
            "mean_impact_overnight_bps": mean_impact_overnight,
            "n_trades": len(results["successful_trades"]),
            "n_trades_joined": len(joined_df),
        })
        results["impact_table"] = impact_table

        if run_plots:
            impact_long = (
                joined_df[["execution_datetime_utc", "impact_vs_close", "impact_vs_open", "impact_overnight"]]
                .melt(id_vars="execution_datetime_utc", var_name="series", value_name="impact")
            )
            impact_long["series"] = impact_long["series"].replace({
                "impact_vs_close": "vs Close",
                "impact_vs_open": "vs Open",
                "impact_overnight": "Overnight",
            })

            impact_comparison_plot = (
                ggplot(impact_long, aes("execution_datetime_utc", "impact", color="series"))
                + geom_line()
                + scale_color_manual(values={"vs Close": "black", "vs Open": "red", "Overnight": "steelblue"})
                + ggtitle("Market Impact Comparison Over Time")
                + labs(x="Execution Date (UTC)", y="Impact (raw)", color="Benchmark")
                + theme(plot_title=element_text(weight="bold"))
            )
            impact_comparison_plot.save("plot_impact_comparison.png", dpi=150, verbose=False)
            print("Saved: plot_impact_comparison.png")

            pnl_vs_impact_plot = (
                ggplot(joined_df, aes("impact_vs_open", "pnl"))
                + geom_point(color="steelblue", alpha=0.7)
                + geom_smooth(method="lm", color="red")
                + geom_hline(yintercept=20e-4, linetype="dashed", color="gray")
                + ggtitle("Scaled PnL vs Market Impact (vs Open)")
                + labs(x="Impact vs Open (raw)", y="Scaled PnL")
                + theme(plot_title=element_text(weight="bold"))
            )
            pnl_vs_impact_plot.save("plot_pnl_vs_impact.png", dpi=150, verbose=False)
            print("Saved: plot_pnl_vs_impact.png")

            raw_pnl_vs_impact_plot = (
                ggplot(joined_df, aes("impact_vs_open", "raw_pnl_return"))
                + geom_point(color="steelblue", alpha=0.7)
                + geom_smooth(method="lm", color="red")
                + geom_hline(yintercept=20e-4, linetype="dashed", color="gray")
                + ggtitle("Raw PnL Return vs Market Impact (vs Open)")
                + labs(x="Impact vs Open (raw)", y="Raw PnL Return (sign(N) * Δprice / open)")
                + theme(plot_title=element_text(weight="bold"))
            )
            raw_pnl_vs_impact_plot.save("plot_raw_pnl_vs_impact.png", dpi=150, verbose=False)
            print("Saved: plot_raw_pnl_vs_impact.png")

            results["impact_comparison_plot"] = impact_comparison_plot
            results["pnl_vs_impact_plot"] = pnl_vs_impact_plot
            results["raw_pnl_vs_impact_plot"] = raw_pnl_vs_impact_plot

    return results


if __name__ == "__main__":
    run_analysis()
