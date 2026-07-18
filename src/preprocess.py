import pandas as pd

def get_series(df, store_nbr, family, freq="W"):
    """
    Filters the merged dataset down to one store + one product family,
    then aggregates to weekly totals for sales, promotion, and oil price.

    Returns a dataframe ready for Prophet with columns:
    ds, y, onpromotion, oil_price, is_holiday
    """
    subset = df[
        (df["store_nbr"] == store_nbr) & (df["family"] == family)
    ].copy()

    subset["date"] = pd.to_datetime(subset["date"])
    subset = subset.set_index("date")

    # Resample to weekly: sum sales/promo, average oil price, max holiday flag
    weekly = subset.resample(freq).agg({
        "sales": "sum",
        "onpromotion": "sum",
        "oil_price": "mean",
        "is_holiday": "max"  # if any day that week was a holiday, flag the week
    }).reset_index()

    weekly.rename(columns={"date": "ds", "sales": "y"}, inplace=True)

    return weekly


if __name__ == "__main__":
    # Quick manual test
    df = pd.read_csv("data/merged_data.csv")
    sample = get_series(df, store_nbr=1, family="GROCERY I")
    print(sample.head())
    print(sample.shape)