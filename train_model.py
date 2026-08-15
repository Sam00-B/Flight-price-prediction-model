import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
df = pd.read_csv("clean_flight_data.csv")
df_encoded = pd.get_dummies(df, columns=["startingAirport", "destinationAirport", "airline", "route", "cabin"])
print(f"Went from {df.shape[1]} columns to {df_encoded.shape[1]} columns")
# X = everything the model is allowed to see. y = the answer key (price).
X = df_encoded.drop(columns=["totalFare"])
y = df_encoded["totalFare"]
# Split into training rows (model learns from these) and test rows (hidden, used only to check honestly).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model trained.")
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Average fare in test set: ${y_test.mean():.2f}")
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(10))
#testing where the model is wrong
results = df.loc[X_test.index, ["airline", "route", "daysUntilDeparture", "seatsRemaining", "isNonStop", "cabin","dayOfWeek","month","numSegments"]].copy()
results["actual"] = y_test.values
results["predicted"] = predictions.round(2)
results["error"] = (results["actual"] - results["predicted"]).abs().round(2)
results["signedError"] = results["actual"] - results["predicted"]

print(results.groupby("month")["signedError"].agg(["mean", "count"]))
print(results.groupby("cabin")["signedError"].agg(["mean", "count"]))
print(results.groupby("dayOfWeek")["signedError"].agg(["mean", "count"]))
print(results.groupby("route")["signedError"].agg(["mean", "count"]))
print(results.groupby("airline")["signedError"].agg(["mean", "count"]))
worst = results.sort_values("error", ascending=False).head(20)
print(worst)
joblib.dump(model, "flight_price_model.joblib")
joblib.dump(list(X.columns), "model_columns.joblib")

print("Saved model -> flight_price_model.joblib")
print("Saved column list -> model_columns.joblib")
