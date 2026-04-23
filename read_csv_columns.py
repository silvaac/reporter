import sys
import pandas as pd


def load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    datetime_like_columns = [
        col
        for col in df.columns
        if df[col].dtype == object and ("time" in col or "datetime" in col)
    ]

    df = df.copy()
    for col in datetime_like_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    return df


def build_cycle_execution_df(df: pd.DataFrame, include_all_columns: bool = True) -> pd.DataFrame:
    df = convert_datetime_columns(df)

    df = df.copy()
    df["execution_price"] = pd.to_numeric(df["execution_price"], errors="coerce")
    df["mid_price_at_submit"] = pd.to_numeric(df["mid_price_at_submit"], errors="coerce")
    df["trade_size_signed"] = pd.to_numeric(df["trade_size_signed"], errors="coerce")

    if not include_all_columns:
        datetime_columns = [
            col for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col])
        ]
        output_columns = datetime_columns + [
            "execution_price",
            "mid_price_at_submit",
            "trade_size_signed",
            "order_type",
        ]
        df = df[output_columns]

    return df[df["execution_price"].notna()].copy()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python read_csv_columns.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    df = load_csv(csv_path)

    print("Columns in CSV:")
    for col in df.columns:
        print(f"- {col}")

    result_df = build_cycle_execution_df(df)
    non_null_execution_prices = result_df["execution_price"].notna().sum()

    print(f"Non-null execution_price rows: {non_null_execution_prices}")

    print(result_df)
    
if __name__ == "__main__":
    main()
