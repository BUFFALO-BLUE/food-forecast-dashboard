import pandas as pd
from prophet import Prophet

def forecast_series(weekly_df, periods=8):
    """
    Trains Prophet on a weekly series that includes external regressors
    (onpromotion, oil_price) and returns the fitted model + forecast.

    IMPORTANT: Prophet needs FUTURE values for onpromotion and oil_price
    too, not just historical ones. Since we don't know the true future
    values, we carry forward the last known value as a simple assumption.
    """
    model = Prophet(interval_width=0.95)

    # Register external regressors BEFORE fitting
    model.add_regressor("onpromotion")
    model.add_regressor("oil_price")

    # Use Prophet's built-in national holiday calendar for Ecuador
    model.add_country_holidays(country_name="EC")

    model.fit(weekly_df)

    # Build future dataframe
    future = model.make_future_dataframe(periods=periods, freq="W")

    # Carry forward last known regressor values into the future
    # (a simple assumption — no promotion data exists for unseen future weeks)
    last_promo = weekly_df["onpromotion"].iloc[-1]
    last_oil = weekly_df["oil_price"].iloc[-1]

    future = future.merge(weekly_df[["ds", "onpromotion", "oil_price"]], on="ds", how="left")
    future["onpromotion"] = future["onpromotion"].fillna(last_promo)
    future["oil_price"] = future["oil_price"].fillna(last_oil)

    forecast = model.predict(future)

    return model, forecast


if __name__ == "__main__":
    from preprocess import get_series

    df = pd.read_csv("data/merged_data.csv")
    weekly = get_series(df, store_nbr=1, family="GROCERY I")

    model, forecast = forecast_series(weekly)
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(8))