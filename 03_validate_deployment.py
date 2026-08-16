"""
Validate the DEPLOYED pipeline against real historical data.

Goal: reload the exact same held-out test set from training, run it through
the SAME feature-prep logic the Lambda function uses (get_dummies + reindex),
and confirm MAE matches what  seen during training ($46.66).

This proves the deployment (S3 -> Lambda -> reindex -> predict) is faithful
to the model  actually evaluated -- not testing the model itself again,
testing that nothing broke in translation to production.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------
# 1. Load the SAME clean data used in training.
# ---------------------------------------------------------------
df = pd.read_csv("clean_flight_data.csv")

# ---------------------------------------------------------------
# 2. Recreate the EXACT same train/test split.
# Same random_state=42 means  get back the identical 18,992 test rows
# your original 02_train_model.py evaluated against.
# ---------------------------------------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Recovered {len(test_df):,} test rows (should match original test set size)")

# Keep the real prices aside before  touch features.
actual_prices = test_df["totalFare"].values
test_features_raw = test_df.drop(columns=["totalFare"])

# ---------------------------------------------------------------
# 3. Load the DEPLOYED model + column list -- same files sitting in S3,
# same files Lambda downloads at runtime.
# ---------------------------------------------------------------
model = joblib.load("flight_price_model.joblib")
model_columns = joblib.load("model_columns.joblib")

# ---------------------------------------------------------------
# 4. Prepare features using the SAME logic as lambda_function.py's
# prepare_features(): one-hot encode, then reindex to match training columns.
# ---------------------------------------------------------------
text_cols = ["startingAirport", "destinationAirport", "airline", "route", "cabin"]
encoded = pd.get_dummies(test_features_raw, columns=text_cols)
aligned = encoded.reindex(columns=model_columns, fill_value=0)

# ---------------------------------------------------------------
# 5. Predict and compare to the REAL historical prices.
# ---------------------------------------------------------------
predictions = model.predict(aligned)
mae = mean_absolute_error(actual_prices, predictions)

print(f"\nDeployed-pipeline MAE: ${mae:.2f}")
print(f"Original training-time MAE: $46.66")
print(f"Difference: ${abs(mae - 46.66):.4f}")
"""
Run the ACTUAL lambda_function.py logic locally, and print predictions
in the same format DynamoDB stores them, so you can directly compare against
what's sitting in your FlightPrice table.

Note: this only matches DynamoDB's values if run on the SAME DAY as your
Lambda invocation -- daysUntilDeparture is computed from "today," so a
different day shifts every prediction.
"""
import joblib
from lambda_function import build_prediction_rows, prepare_features

# Load the same model files Lambda uses (must be in this same folder).
model = joblib.load("flight_price_model.joblib")
model_columns = joblib.load("model_columns.joblib")

# Reuse the EXACT same functions lambda_function.py runs.
raw_df = build_prediction_rows()
features_df, flight_dates = prepare_features(raw_df, model_columns)

predictions = model.predict(features_df)

# Print in the same "route#date -> price" format as your DynamoDB routeDate key.
print(f"{'routeDate':<25} {'predictedPrice'}")
for i in range(len(raw_df)):
    route = raw_df.iloc[i]["route"]
    flight_date = raw_df.iloc[i]["flightDate"]
    price = round(float(predictions[i]), 2)
    airline = raw_df.iloc[i]["airline"]
    print(f"{route}#{flight_date:<15} {price} {airline}")