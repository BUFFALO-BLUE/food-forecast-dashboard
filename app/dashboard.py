import streamlit as st
import pandas as pd
from prophet import Prophet

# ---------------------------
# Page setup
# ---------------------------
st.title("Favorita Store Sales Forecasting")

st.write(
    "This dashboard forecasts weekly sales for products at Favorita stores "
    "in Ecuador, using historical sales, promotions, oil prices, and holidays."
)

with st.expander("What this dashboard can and can't tell you"):
    st.markdown("""
    - **Store and product scope:** This dataset includes multiple real stores 
      and product families, so forecasts are specific to the store/family you select.
    - **Promotions:** Included directly as a factor in the forecast (`onpromotion`).
    - **Oil prices:** Ecuador's economy is oil-dependent; oil price is included 
      as a factor, since it may relate to consumer spending.
    - **Holidays:** National Ecuadorian holidays are included in the forecast.
    - **Known anomaly:** A magnitude 7.8 earthquake struck Ecuador on April 16, 2016, 
      significantly disrupting supermarket sales for several weeks afterward. 
      Forecasts around this period may be less reliable.
    - **Prices:** This dataset does not include per-unit prices. Dollar estimates 
      shown below use assumed prices, not confirmed data.
    """)

# ---------------------------
# Load merged data
# ---------------------------
df = pd.read_csv("data/merged_data.csv")
df["date"] = pd.to_datetime(df["date"])

# ---------------------------
# Sidebar: store + product family selection
# ---------------------------
store_list = sorted(df["store_nbr"].unique())
family_list = sorted(df["family"].unique())

selected_store = st.sidebar.selectbox("Choose a store", store_list)
selected_family = st.sidebar.selectbox("Choose a product family", family_list)
inventory = st.sidebar.number_input("Current Inventory", value=100, min_value=0)

# ---------------------------
# Preprocess: filter + weekly aggregation
# ---------------------------
def get_series(df, store_nbr, family, freq="W"):
    subset = df[(df["store_nbr"] == store_nbr) & (df["family"] == family)].copy()
    subset = subset.set_index("date")
    weekly = subset.resample(freq).agg({
        "sales": "sum",
        "onpromotion": "sum",
        "oil_price": "mean",
        "is_holiday": "max"
    }).reset_index()
    weekly.rename(columns={"date": "ds", "sales": "y"}, inplace=True)
    weekly["oil_price"] = weekly["oil_price"].ffill().bfill()  # patch any remaining gaps
    return weekly

weekly_sales = get_series(df, selected_store, selected_family)



# ---------------------------
# Forecast (cached as a resource so the model only trains once)
# ---------------------------
@st.cache_resource
def run_forecast(weekly_df):
    model = Prophet(interval_width=0.90)
    model.add_regressor("onpromotion")
    model.add_regressor("oil_price")
    model.add_country_holidays(country_name="EC")
    model.fit(weekly_df)

    future = model.make_future_dataframe(periods=8, freq="W")
    last_promo = weekly_df["onpromotion"].iloc[-1]
    last_oil = weekly_df["oil_price"].iloc[-1]

    future = future.merge(weekly_df[["ds", "onpromotion", "oil_price"]], on="ds", how="left")
    future["onpromotion"] = future["onpromotion"].fillna(last_promo)
    future["oil_price"] = future["oil_price"].fillna(last_oil)

    forecast = model.predict(future)
    return model, forecast

model, forecast = run_forecast(weekly_sales)

st.subheader(f"Forecast: Store {selected_store} — {selected_family}")

next_week_forecast = forecast.iloc[-8]["yhat"]        # realistic/typical estimate
next_week_upper = forecast.iloc[-8]["yhat_upper"]      # conservative ceiling, for ordering only

st.metric("Next Week Forecast (typical)", f"{next_week_forecast:,.0f} units")
st.caption(
    f"90% confidence: actual demand is expected to stay below "
    f"**{next_week_upper:,.0f} units** in most weeks."
)

# ---------------------------
# Inventory decision
# Ordering uses the UPPER bound (conservative) — this is the one place
# yhat_upper should be used, since the goal here is avoiding stockouts.
# ---------------------------
recommended_order = max(next_week_upper - inventory, 0)

st.subheader("📦 Recommended Order")
if recommended_order > 0:
    st.warning(
        f"Order **{recommended_order:,.0f} units**. This uses the 90% demand "
        f"ceiling ({next_week_upper:,.0f} units) to reduce the chance of running out."
    )
else:
    st.success("Current inventory already covers the 90% demand ceiling.")

# ---------------------------
# Expected leftover / lost sales
# These use yhat (the realistic estimate), NOT yhat_upper — we want the
# most likely outcome here, not the conservative ceiling.
# ---------------------------
expected_leftover = max(inventory - next_week_forecast, 0)
expected_lost_sales = max(next_week_forecast - inventory, 0)
fill_rate = (min(inventory, next_week_forecast) / next_week_forecast * 100) if next_week_forecast > 0 else 100

st.subheader("Estimated Inventory Impact")
c1, c2, c3 = st.columns(3)
c1.metric("Expected Leftover", f"{expected_leftover:,.0f} units")
c2.metric("Expected Lost Sales", f"{expected_lost_sales:,.0f} units")
c3.metric("Expected Fill Rate", f"{fill_rate:.1f}%")

# ---------------------------
# Estimated price per unit, by product family
# Anchored to real Safeway prices where available (eggs, dairy, produce,
# meat); other categories use a reasonable estimate. These are category
# averages, not exact per-SKU prices, since "family" is a broad category.
# ---------------------------
estimated_family_prices = {
    "EGGS": 0.55, "BREAD/BAKERY": 3.50, "DAIRY": 4.50, "PRODUCE": 2.50,
    "MEATS": 6.50, "POULTRY": 6.00, "SEAFOOD": 9.00, "DELI": 7.00,
    "PREPARED FOODS": 6.00, "FROZEN FOODS": 4.50, "BEVERAGES": 3.00,
    "LIQUOR,WINE,BEER": 12.00, "GROCERY I": 3.50, "GROCERY II": 3.00,
    "CLEANING": 5.00, "HOME CARE": 5.00, "PERSONAL CARE": 6.00,
    "BEAUTY": 8.00, "BABY CARE": 9.00, "PET SUPPLIES": 10.00,
    "CELEBRATION": 5.00, "LADIESWEAR": 15.00, "LINGERIE": 15.00,
    "HOME AND KITCHEN I": 8.00, "HOME AND KITCHEN II": 8.00,
    "HOME APPLIANCES": 30.00, "HARDWARE": 10.00, "LAWN AND GARDEN": 12.00,
    "AUTOMOTIVE": 15.00, "PLAYERS AND ELECTRONICS": 25.00,
    "SCHOOL AND OFFICE SUPPLIES": 4.00, "BOOKS": 10.00, "MAGAZINES": 5.00,
}
default_family_price = 5.00
price_per_unit = estimated_family_prices.get(selected_family, default_family_price)

st.subheader("Estimated Dollar Impact")
st.caption(
    f"Estimated price for **{selected_family}**: \\${price_per_unit:.2f}/unit "
    f"(anchored to real Safeway pricing where available; category-level estimate)."
)

if expected_leftover > 0:
    st.info(f"Potential waste value: **\\${expected_leftover * price_per_unit:,.2f}**")

if expected_lost_sales > 0:
    st.warning(f"Potential lost revenue: **\\${expected_lost_sales * price_per_unit:,.2f}**")

if expected_leftover == 0 and expected_lost_sales == 0:
    st.success("Inventory is expected to closely match forecast demand.")


# ---------------------------
# Actual vs Forecast chart
# ---------------------------
combined = forecast.merge(weekly_sales, on="ds", how="left")
combined.rename(columns={"y": "actual", "yhat": "predicted"}, inplace=True)
st.subheader("Actual vs Forecast")
st.line_chart(combined.set_index("ds")[["actual", "predicted"]])

# ---------------------------
# Model reliability check
# Compares actual sales to the REALISTIC forecast (yhat), not the
# inflated ordering ceiling — this gives a true measure of accuracy.
# ---------------------------
st.subheader("How reliable is this model?")
historical = combined.dropna(subset=["actual", "predicted"]).copy()
historical["error"] = historical["predicted"] - historical["actual"]

mae = historical["error"].abs().mean()
mean_bias = historical["error"].mean()
over_weeks = (historical["predicted"] > historical["actual"]).sum()
under_weeks = (historical["predicted"] < historical["actual"]).sum()

c1, c2, c3 = st.columns(3)
c1.metric("Mean Absolute Error (MAE)", f"{mae:,.1f}")
c2.metric("Weeks Over-Predicted", f"{over_weeks}/{len(historical)}")
c3.metric("Weeks Under-Predicted", f"{under_weeks}/{len(historical)}")

if mean_bias < 0:
    st.warning(f"This model tends to UNDER-predict by ~{abs(mean_bias):,.1f} units on average.")
elif mean_bias > 0:
    st.info(f"This model tends to OVER-predict by ~{mean_bias:,.1f} units on average.")
else:
    st.success("No consistent bias detected.")

# ---------------------------
# Forecast table
# ---------------------------
st.subheader("Next Eight Weeks")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(8))

# ---------------------------
# Seasonality / component breakdown
# ---------------------------
st.subheader("What's driving this forecast?")
st.markdown("""
Prophet breaks demand into: **trend** (long-term direction), **weekly pattern** 
(which days see more/less sales), **yearly pattern** (seasonal effects), plus the 
**promotion**, **oil price**, and **holiday** regressors added to this model.
""")

# Reuse the already-trained model and forecast — no need to retrain
fig = model.plot_components(forecast)
st.pyplot(fig)