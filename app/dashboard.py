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
# Forecast (cached so switching dropdowns doesn't retrain unnecessarily)
# ---------------------------
@st.cache_data
def run_forecast(weekly_df):
    model = Prophet()
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
    return forecast

forecast = run_forecast(weekly_sales)

# ---------------------------
# Headline metric + recommendation
# ---------------------------
# ---------------------------
# Headline metric + recommendation
# ---------------------------
st.subheader(f"Forecast: Store {selected_store} — {selected_family}")

next_week_forecast = forecast.iloc[-8]["yhat"]
st.metric("Next Week Forecast (units)", f"{next_week_forecast:,.0f}")

# --- Asymmetric safety buffer ---
# Understocking (running out) risks losing a customer entirely — a worse
# outcome than overstocking (a bit of extra inventory sitting around).
# To reflect that, we deliberately bias the recommendation UPWARD by a
# safety margin, rather than treating both risks as equally costly.
safety_margin_pct = st.sidebar.slider(
    "Safety buffer (%) — extra stock to guard against under-predicting",
    min_value=0, max_value=50, value=15
)
buffered_forecast = next_week_forecast * (1 + safety_margin_pct / 100)

difference = buffered_forecast - inventory
if difference > 0:
    st.warning(
        f"Recommended order: {difference:,.0f} units "
        f"(includes a {safety_margin_pct}% buffer above the {next_week_forecast:,.0f}-unit "
        f"forecast, since running out risks losing customers)."
    )
else:
    st.success(
        f"Current inventory ({inventory:,.0f}) covers the buffered demand estimate "
        f"of {buffered_forecast:,.0f} units."
    )
# ---------------------------
# Cost of overstock/understock
# ---------------------------
# ---------------------------
# Was before the cost of the overstock/understock but the units we are working with doesnt have formal prices.
# Optional: estimate dollar impact (user-provided assumption)
# ---------------------------
st.subheader("💰 Estimate dollar impact (optional)")
st.caption(
    "This dataset does not include real per-unit prices. If you'd like to see "
    "an estimated dollar impact, enter your own price assumptions below."
)

show_cost = st.checkbox("Estimate dollar impact")
if show_cost:
    col1, col2 = st.columns(2)
    with col1:
        price_per_unit = st.number_input("Assumed price per unit ($)", min_value=0.0, step=0.10)
    with col2:
        spoilage_cost_per_unit = st.number_input("Assumed cost if unsold ($)", min_value=0.0, step=0.10)

    if difference > 0:
        st.info(f"Estimated lost sales if not restocked: ${difference * price_per_unit:,.2f}")
    else:
        st.info(f"Estimated waste cost of excess units: ${(-difference) * spoilage_cost_per_unit:,.2f}")

# ---------------------------
# Actual vs Forecast chart
# ---------------------------
combined = forecast.merge(weekly_sales, on="ds", how="left")
combined.rename(columns={"y": "actual", "yhat": "predicted"}, inplace=True)

st.subheader("Actual vs Forecast")
st.line_chart(combined.set_index("ds")[["actual", "predicted"]])

# ---------------------------
# Model reliability check
# ---------------------------
st.subheader("How reliable is this model?")
historical = combined.dropna(subset=["actual", "predicted"]).copy()
historical["error"] = historical["predicted"] - historical["actual"]

mae = historical["error"].abs().mean()
mean_bias = historical["error"].mean()
over_weeks = (historical["error"] > 0).sum()
under_weeks = (historical["error"] < 0).sum()

c1, c2, c3 = st.columns(3)
c1.metric("Average Error (MAE)", f"{mae:,.1f}")
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

model_for_plot = Prophet()
model_for_plot.add_regressor("onpromotion")
model_for_plot.add_regressor("oil_price")
model_for_plot.add_country_holidays(country_name="EC")
model_for_plot.fit(weekly_sales)
future_plot = model_for_plot.make_future_dataframe(periods=8, freq="W")
future_plot = future_plot.merge(weekly_sales[["ds", "onpromotion", "oil_price"]], on="ds", how="left")
future_plot["onpromotion"] = future_plot["onpromotion"].fillna(weekly_sales["onpromotion"].iloc[-1])
future_plot["oil_price"] = future_plot["oil_price"].fillna(weekly_sales["oil_price"].iloc[-1])
full_forecast = model_for_plot.predict(future_plot)

fig = model_for_plot.plot_components(full_forecast)
st.pyplot(fig)