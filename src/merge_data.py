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
# Drop rows where the holiday was transferred away from this date —
# a "transferred" holiday is treated like a normal working day.
holidays_clean = holidays[holidays["transferred"] == False].copy()

# Keep only actual holiday-type rows (ignore "Work Day" and "Bridge" for a simple flag)
holidays_clean = holidays_clean[holidays_clean["type"] == "Holiday"]

# Just need the set of dates that are real holidays
holiday_dates = set(holidays_clean["date"])

merged["is_holiday"] = merged["date"].isin(holiday_dates)

# ---------------------------
# Step 4: Save the combined file
# ---------------------------
merged.to_csv("data/merged_data.csv", index=False)

print("Merge complete.")
print("Final shape:", merged.shape)
print(merged.head())
print("Missing oil prices remaining:", merged["oil_price"].isna().sum())