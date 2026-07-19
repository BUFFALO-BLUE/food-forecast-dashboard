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


# ---------------------------
# Actual vs Forecast chart
# ---------------------------
combined = forecast.merge(weekly_sales, on="ds", how="left")
combined.rename(columns={"y": "actual", "yhat": "predicted"}, inplace=True)
st.subheader("Actual vs Forecast")
st.line_chart(combined.set_index("ds")[["actual", "predicted"]])

# ---------------------------
# Model reliability, reframed as business impact rather than raw stats.
# Compares actual sales to the REALISTIC forecast (yhat), not the
# inflated ordering ceiling.
# ---------------------------
historical = combined.dropna(subset=["actual", "predicted"]).copy()
historical["error"] = historical["predicted"] - historical["actual"]

mae = historical["error"].abs().mean()
mean_bias = historical["error"].mean()

under_predicted = historical[historical["predicted"] < historical["actual"]]
over_predicted = historical[historical["predicted"] > historical["actual"]]

avg_shortage = (under_predicted["actual"] - under_predicted["predicted"]).mean()
avg_excess = (over_predicted["predicted"] - over_predicted["actual"]).mean()

# --- Model Accuracy ---
st.subheader("Model Accuracy")
c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:,.1f} units")
c2.metric("Average Bias", f"{mean_bias:,.1f} units")

st.caption(
    f"The forecast differs from actual sales by about {mae:,.1f} units "
    f"on an average week."
)

if abs(mean_bias) < 1:
    st.write(
        "The model shows almost no consistent bias, meaning over-predictions "
        "and under-predictions balance out over time."
    )
elif mean_bias < 0:
    st.write(
        f"Overall, the model leans toward under-predicting by about "
        f"{abs(mean_bias):,.1f} units on average."
    )
else:
    st.write(
        f"Overall, the model leans toward over-predicting by about "
        f"{mean_bias:,.1f} units on average."
    )


st.subheader("Estimated Dollar Impact")
st.caption(
    f"Estimated price for **{selected_family}**: \\${price_per_unit:.2f}/unit "
    f"(anchored to real Safeway pricing where available; category-level estimate)."
)

# ---------------------------
# Build year and month columns from historical actual-vs-forecast data
# (historical = combined.dropna(subset=["actual", "predicted"]), defined earlier)
# ---------------------------
historical["year"] = historical["ds"].dt.year
historical["month"] = historical["ds"].dt.month_name()

# Per-row excess/shortage units (0 if not applicable that week)
historical["excess_units"] = (historical["predicted"] - historical["actual"]).clip(lower=0)
historical["shortage_units"] = (historical["actual"] - historical["predicted"]).clip(lower=0)

# Convert to dollars using this family's price
historical["excess_value"] = historical["excess_units"] * price_per_unit
historical["shortage_value"] = historical["shortage_units"] * price_per_unit

# ---------------------------
# Clearly labeled overall totals, with formulas shown
# ---------------------------
total_excess_units = historical["excess_units"].sum()
total_excess_value = historical["excess_value"].sum()
total_shortage_units = historical["shortage_units"].sum()
total_shortage_value = historical["shortage_value"].sum()

c1, c2 = st.columns(2)
with c1:
    st.metric("Over-Prediction Value (Excess Inventory)", f"\\${total_excess_value:,.2f}")
    st.caption(
        f"Formula: {total_excess_units:,.0f} total excess units × \\${price_per_unit:.2f}/unit "
        f"= \\${total_excess_value:,.2f}. This is the value of inventory that historically "
        f"went unsold because the forecast was higher than actual demand."
    )
with c2:
    st.metric("Under-Prediction Value (Lost Sales)", f"\\${total_shortage_value:,.2f}")
    st.caption(
        f"Formula: {total_shortage_units:,.0f} total shortage units × \\${price_per_unit:.2f}/unit "
        f"= \\${total_shortage_value:,.2f}. This is the value of sales historically missed "
        f"because actual demand exceeded the forecast."
    )

# ---------------------------
# Yearly breakdown
# ---------------------------
st.subheader("Yearly Breakdown")
yearly = historical.groupby("year").agg(
    excess_units=("excess_units", "sum"),
    excess_value=("excess_value", "sum"),
    shortage_units=("shortage_units", "sum"),
    shortage_value=("shortage_value", "sum"),
).reset_index()

yearly_display = yearly.rename(columns={
    "year": "Year",
    "excess_units": "Excess Units",
    "excess_value": "Excess Value ($)",
    "shortage_units": "Shortage Units",
    "shortage_value": "Shortage Value ($)",
})
st.dataframe(yearly_display.style.format({
    "Excess Units": "{:,.0f}",
    "Excess Value ($)": "${:,.2f}",
    "Shortage Units": "{:,.0f}",
    "Shortage Value ($)": "${:,.2f}",
}))


# ---------------------------
# Year x Month breakdown — kept SEPARATE per year, not blended together,
# since averaging Decembers across 2013-2017 hides year-specific patterns
# (e.g. the 2016 earthquake).
# ---------------------------
st.subheader("Month-by-Month Breakdown, Per Year")

month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

pivot_excess = historical.pivot_table(
    index="year", columns="month", values="excess_value", aggfunc="sum", fill_value=0
).reindex(columns=month_order, fill_value=0)

pivot_shortage = historical.pivot_table(
    index="year", columns="month", values="shortage_value", aggfunc="sum", fill_value=0
).reindex(columns=month_order, fill_value=0)

st.markdown("**Over-Prediction Value ($) — by Year and Month**")
st.dataframe(pivot_excess.style.format("${:,.2f}"))

st.markdown("**Under-Prediction Value ($) — by Year and Month**")
st.dataframe(pivot_shortage.style.format("${:,.2f}"))

# ---------------------------
# Weekly drill-down — pick a specific year and month to see individual
# weekly rows underneath the monthly total.
# ---------------------------
st.subheader("Weekly Drill-Down")

years_available = sorted(historical["year"].unique())
drill_year = st.selectbox("Select year", years_available, key="drill_year")

months_available = [m for m in month_order if m in historical[historical["year"] == drill_year]["month"].unique()]
drill_month = st.selectbox("Select month", months_available, key="drill_month")

weekly_detail = historical[
    (historical["year"] == drill_year) & (historical["month"] == drill_month)
][["ds", "actual", "predicted", "excess_units", "excess_value", "shortage_units", "shortage_value"]]

weekly_detail_display = weekly_detail.rename(columns={
    "ds": "Week Of", "actual": "Actual", "predicted": "Predicted",
    "excess_units": "Excess Units", "excess_value": "Excess Value ($)",
    "shortage_units": "Shortage Units", "shortage_value": "Shortage Value ($)",
})

st.dataframe(weekly_detail_display.style.format({
    "Actual": "{:,.0f}", "Predicted": "{:,.1f}",
    "Excess Units": "{:,.0f}", "Excess Value ($)": "${:,.2f}",
    "Shortage Units": "{:,.0f}", "Shortage Value ($)": "${:,.2f}",
}))

# ---------------------------
# Product tiers — grouped by how often a typical customer buys this
# category. This grouping and the purchase frequency are business
# judgment calls, not derived from the data (this dataset has no
# individual customer purchase history).
# ---------------------------
product_tiers = {
    "Tier 1 (Weekly purchase)": [
        "DAIRY", "EGGS", "BREAD/BAKERY", "PRODUCE", "MEATS", "POULTRY",
        "BEVERAGES", "GROCERY I", "GROCERY II", "FROZEN FOODS", "DELI",
        "PREPARED FOODS", "SEAFOOD", "CLEANING", "HOME CARE",
        "PERSONAL CARE", "BABY CARE", "PET SUPPLIES",
    ],
    "Tier 2 (Monthly purchase)": [
        "BEAUTY", "CELEBRATION", "MAGAZINES", "LIQUOR,WINE,BEER",
        "LAWN AND GARDEN",
    ],
    "Tier 3 (Yearly purchase)": [
        "LADIESWEAR", "LINGERIE", "HOME APPLIANCES", "HARDWARE",
        "AUTOMOTIVE", "PLAYERS AND ELECTRONICS", "BOOKS",
        "SCHOOL AND OFFICE SUPPLIES", "HOME AND KITCHEN I",
        "HOME AND KITCHEN II",
    ],
}

# How many times per year a typical customer in this tier buys at all
purchase_frequency = {
    "Tier 1 (Weekly purchase)": 52,
    "Tier 2 (Monthly purchase)": 12,
    "Tier 3 (Yearly purchase)": 1,
}

# Assumed units a typical customer buys in ONE visit, by tier.
# This is the key assumption in this model — everything below scales
# from this number. Adjust here if a more informed estimate is available.
units_per_customer_order = {
    "Tier 1 (Weekly purchase)": 10,
    "Tier 2 (Monthly purchase)": 5,
    "Tier 3 (Yearly purchase)": 2,
}

def get_tier(family):
    for tier, families in product_tiers.items():
        if family in families:
            return tier
    return "Tier 1 (Weekly purchase)"  # fallback if a family isn't explicitly listed

selected_tier = get_tier(selected_family)
freq = purchase_frequency[selected_tier]
order_size = units_per_customer_order[selected_tier]

# ---------------------------
# Step 1: What is one loyal customer worth per year, in this category?
# One order's worth of units x how many times/year they'd normally buy x price.
# ---------------------------
value_per_customer_per_year = order_size * freq * price_per_unit

# ---------------------------
# Step 2: How many "customer-equivalents" of demand went unmet historically?
# Each week's shortage is expressed as a FRACTION (or multiple) of one
# customer's typical order size, rather than always counting as exactly
# one lost customer. A small shortage might be a fraction of a customer;
# a big shortage might represent several customers' worth of unmet demand.
# ---------------------------
total_customers_affected = total_shortage_units / order_size

# ---------------------------
# Step 3: Total estimated value of lost future business, across all
# historical shortage events, based on the customers affected and what
# each one is worth per year.
# ---------------------------
total_annual_customer_loss_value = total_customers_affected * value_per_customer_per_year

# ---------------------------
# Combine with the immediate lost-sales value (the actual missed
# transaction itself, calculated earlier) for a full picture.
# ---------------------------
total_under_prediction_impact = total_shortage_value + total_annual_customer_loss_value

st.subheader("Understanding the Under-Prediction Risk")

st.markdown(f"""
**How this is calculated:** Under-prediction means actual demand exceeded the
forecast — the store likely didn't have enough stock to meet demand that week.
This section estimates two things: the value of sales missed in the moment,
and the value of customer relationships that may be lost long-term if the
store can't meet demand reliably.

**Product tier:** {selected_family} is classified as **{selected_tier}**,
assuming a typical customer buys about **{order_size} units** per visit,
**{freq} times per year**.
""")

# ---------------------------
# Per-year under-prediction impact, instead of one historical total —
# this reveals whether specific years drove most of the impact.
# ---------------------------
yearly_under = historical.groupby("year").agg(
    shortage_units=("shortage_units", "sum"),
    shortage_value=("shortage_value", "sum"),
).reset_index()

yearly_under["customers_affected"] = yearly_under["shortage_units"] / order_size
yearly_under["customer_loss_value"] = yearly_under["customers_affected"] * value_per_customer_per_year
yearly_under["total_impact"] = yearly_under["shortage_value"] + yearly_under["customer_loss_value"]

st.subheader("Understanding the Under-Prediction Risk")

st.markdown(f"""
**How this is calculated:** Under-prediction means actual demand exceeded the
forecast — the store likely didn't have enough stock to meet demand that week.
Below is a year-by-year breakdown of two things: the value of sales missed in
the moment, and the estimated value of customer relationships that may be lost
long-term if the store can't meet demand reliably.

**Product tier:** {selected_family} is classified as **{selected_tier}**,
assuming a typical customer buys about **{order_size} units** per visit,
**{freq} times per year** — making one loyal customer worth about
**\\${value_per_customer_per_year:,.2f}/year** in this category.
""")

yearly_under_display = yearly_under.rename(columns={
    "year": "Year",
    "shortage_units": "Shortage Units",
    "shortage_value": "Immediate Lost Sales ($)",
    "customers_affected": "Customer-Equivalents Affected",
    "customer_loss_value": "Long-Term Customer Loss ($)",
    "total_impact": "Total Under-Prediction Impact ($)",
})

st.dataframe(yearly_under_display.style.format({
    "Shortage Units": "{:,.0f}",
    "Immediate Lost Sales ($)": "${:,.2f}",
    "Customer-Equivalents Affected": "{:,.2f}",
    "Long-Term Customer Loss ($)": "${:,.2f}",
    "Total Under-Prediction Impact ($)": "${:,.2f}",
}))

st.caption("""
**Column formulas:**
- **Shortage Units** = total units short that year (actual − forecast, summed across weeks where actual exceeded forecast)
- **Immediate Lost Sales ($)** = Shortage Units × price per unit
- **Customer-Equivalents Affected** = Shortage Units ÷ units per typical order for this tier
- **Long-Term Customer Loss ($)** = Customer-Equivalents Affected × value of one customer per year
- **Total Under-Prediction Impact ($)** = Immediate Lost Sales + Long-Term Customer Loss
""")

st.info(
    "**Note on assumptions:** This dataset does not include individual "
    "customer purchase history, so 'units per typical order' and purchase "
    "frequency are estimated assumptions based on how often each product "
    "category is normally bought, not measured directly from the data. "
    "Shortage units and immediate lost sales are grounded in real historical "
    "sales data; the customer loss estimate is a business projection built "
    "on top of that data."
)

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