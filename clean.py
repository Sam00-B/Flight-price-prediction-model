"""
STEP 1: Clean the raw data and turn it into "features" a model can learn from.
 
Nothing ML-y happens in this file yet. This is just: take messy real-world
data, and shape it into a clean table of (features -> price).
"""
import pandas as pd

df =pd.read_csv("itineraries-min-100k.csv")
print(f"Loaded {len(df):,} rows")
 
# ---------------------------------------------------------------
# 2. ENGINEER THE "BOOKING VELOCITY" FEATURE
# We have searchDate (when the customer searched) and flightDate
# (when the flight departs). The gap between them is exactly the
# "how far out is this booking" signal from the hotel project.
# ---------------------------------------------------------------
df["searchDate"] = pd.to_datetime(df["searchDate"])
df["flightDate"] = pd.to_datetime(df["flightDate"])
df["daysUntilDeparture"] = (df["flightDate"] - df["searchDate"]).dt.days
df["cabin"] = df["segmentsCabinCode"].str.split("\\|\\|").str[0]
 
# ---------------------------------------------------------------
# 3. ENGINEER SEASONALITY-STYLE FEATURES
# Models can't understand a date like "2022-04-17" directly.
# We break it into pieces a model CAN find patterns in.
# ---------------------------------------------------------------
df["dayOfWeek"] = df["flightDate"].dt.dayofweek     # 0=Monday ... 6=Sunday
df["month"] = df["flightDate"].dt.month
df["isWeekendFlight"] = df["dayOfWeek"].isin([5, 6])  # Sat/Sun
 
# ---------------------------------------------------------------
# 4. CLEAN UP MULTI-SEGMENT AIRLINE NAMES
# "American||American" just means a connecting flight on the same
# airline for both legs. We only need the airline once.
# ---------------------------------------------------------------
#add route features
df["airline"] = df["segmentsAirlineName"].str.split("\\|\\|").str[0]
df["numSegments"] = df["segmentsAirlineName"].str.split("\\|\\|").str.len()
df["route"] = df["startingAirport"] + "_" + df["destinationAirport"]
 
# ---------------------------------------------------------------
# 5. HANDLE MISSING VALUES
# totalTravelDistance has ~5% missing. For a first model, the
# simplest honest fix is dropping those rows rather than guessing.
# ---------------------------------------------------------------
before = len(df)
df = df.dropna(subset=["totalTravelDistance"])
print(f"Dropped {before - len(df):,} rows with missing distance")
 
# ---------------------------------------------------------------
# 6. SELECT THE FEATURES + TARGET
# This is the actual (features -> target) table the model will see.
# ---------------------------------------------------------------
feature_cols = [
    "daysUntilDeparture",
    "dayOfWeek",
    "month",
    "isWeekendFlight",
    "startingAirport",
    "destinationAirport",
    "cabin",
    "route",
    "numSegments",
    "airline",
    "isNonStop",
    "isBasicEconomy",
    "seatsRemaining",
    "totalTravelDistance",
]
target_col = "totalFare"
 
model_df = df[feature_cols + [target_col]].copy()
 
print("\nFinal shape:", model_df.shape)
print("\nFeature preview:")
print(model_df.head())
print("\nMissing values left:")
print(model_df.isnull().sum())
 
model_df.to_csv("clean_flight_data.csv", index=False)
print("\nSaved -> clean_flight_data.csv")