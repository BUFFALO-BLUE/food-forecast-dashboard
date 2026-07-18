import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt

# Will be used to explore all 5 files of the STORE SALES - TIME SERIES FORECASTING
# Feom Kaggle and is to use machine learning to predict grocery sales.

# Load all 4 source files
train = pd.read_csv("data/train.csv")
stores = pd.read_csv("data/stores.csv")
oil = pd.read_csv("data/oil.csv")
holidays = pd.read_csv("data/holidays_events.csv")

# --- TRAIN ---
print("=== TRAIN.CSV ===")
print(train.head())
print(train.info())
print("Date range:", train["date"].min(), "to", train["date"].max())
print("Unique stores:", train["store_nbr"].nunique())
print("Unique product families:", train["family"].nunique())
print(train["family"].unique())  # see exact family names

# --- STORES ---
print("\n=== STORES.CSV ===")
print(stores.head())
print(stores["city"].nunique(), "unique cities")
print(stores["type"].unique())  # store types (A, B, C, etc.)
print(stores["cluster"].nunique(), "unique clusters")

# --- OIL ---
print("\n=== OIL.CSV ===")
print(oil.head())
print("Missing oil price rows:", oil["dcoilwtico"].isna().sum())
print("Date range:", oil["date"].min(), "to", oil["date"].max())
# NOTE: oil.csv likely has gaps (weekends/holidays have no oil price recorded)
# We'll forward-fill these gaps in merge_data.py

# --- HOLIDAYS ---
print("\n=== HOLIDAYS_EVENTS.CSV ===")
print(holidays.head())
print(holidays["type"].unique())      # Holiday, Transfer, Bridge, Work Day, etc.
print(holidays["locale"].unique())    # National, Regional, Local
print("Transferred holidays:", holidays["transferred"].sum())

# people are lazy are would like to the least work to get the most result I think that this is human nature.
# After fiinishing the coding hackerrank assessment for optiver I realized how much
# I am not skilled as a data scientist and how much I need to learn. I will continue to learn and practice more to become a better data scientist.
# obviously I will not be able to become a data scientist in a day or two but I will continue to learn and practice more to become a better data scientist.
# one thing I realized is that I need to learn more about data visualization and how to use it effectively to communicate my findings. I will continue to learn and practice more to become a better data scientist.
# i HINK THAT THIS IS NOT A GAME BUT A JOURNEY AND I WILL CONTINUE TO LEARN AND PRACTICE MORE TO BECOME A BETTER DATA SCIENTIST.
# WHY AM I OBSESED WITH DATA SCIENCE? I THINK THAT THIS IS BECAUSE I WANT TO MAKE A DIFFERENCE IN THE WORLD AND I THINK THAT DATA SCIENCE IS THE BEST WAY TO DO THAT. I WILL CONTINUE TO LEARN AND PRACTICE MORE TO BECOME A BETTER DATA SCIENTIST.

