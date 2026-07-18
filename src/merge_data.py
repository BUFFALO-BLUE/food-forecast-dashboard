import pandas as pd

# ---------------------------
# Load all 4 files
# ---------------------------
train = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
oil = pd.read_csv("data/oil.csv", parse_dates=["date"])
holidays = pd.read_csv("data/holidays_events.csv", parse_dates=["date"])

# ---------------------------
# Step 1: Join train + stores (adds city, state, type, cluster)
# ---------------------------
merged = train.merge(stores, on="store_nbr", how="left")

# ---------------------------
# Step 2: Join oil price by date
# Oil data has gaps (no price on weekends/holidays), so we
# reindex to include every date, then forward-fill missing prices.
# ---------------------------
full_dates = pd.DataFrame({"date": pd.date_range(oil["date"].min(), oil["date"].max())})
oil_filled = full_dates.merge(oil, on="date", how="left")
oil_filled["dcoilwtico"] = oil_filled["dcoilwtico"].ffill()  # carry last known price forward
oil_filled.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)

merged = merged.merge(oil_filled, on="date", how="left")

# ---------------------------
# Step 3: Build a clean holiday flag
# ---------------------------
# Drop rows where the holiday was transferred AWAY from this date —
# the original date becomes a normal day when transferred.
non_transferred = holidays[holidays["transferred"] == False].copy()

# Days that behave like a real holiday/atypical shopping day:
# - Holiday: standard holiday
# - Additional: extra day added onto a calendar holiday (e.g. Christmas Eve)
# - Bridge: extra day off added to extend a long weekend
# - Transfer: the row showing where a transferred holiday was ACTUALLY observed
# NOTE: "Work Day" is deliberately excluded — it's a normally-off day (e.g. Saturday)
# turned INTO a working day to pay back a Bridge day, so it behaves like a normal day.
holiday_types = ["Holiday", "Additional", "Bridge", "Transfer"]
actual_holidays = non_transferred[non_transferred["type"].isin(holiday_types)]

holiday_dates = set(actual_holidays["date"])

merged["is_holiday"] = merged["date"].isin(holiday_dates)
# ---------------------------
# Step 4: Save the combined file
# ---------------------------
merged.to_csv("data/merged_data.csv", index=False)

print("Merge complete.")
print("Final shape:", merged.shape)
print(merged.head())
print("Missing oil prices remaining:", merged["oil_price"].isna().sum())