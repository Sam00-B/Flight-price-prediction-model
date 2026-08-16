"""
Nightly flight pricing Lambda.

Triggered by EventBridge once a night. Loads the trained model from S3,
generates price predictions for a fixed set of routes over the next 30
days, and writes the results into DynamoDB.
"""
import boto3
import joblib
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------
# CONFIG: names of  actual AWS resources.
# ---------------------------------------------------------------
S3_BUCKET = "saumik-flight-pricing-model"
MODEL_KEY = "flight_price_model.joblib"
COLUMNS_KEY = "model_columns.joblib"
DYNAMO_TABLE = "FlightPrice"

# ---------------------------------------------------------------
# HARDCODED ROUTES: a small, fixed set to start with.
# Values are typical/average, based on what we saw in training data.
# ---------------------------------------------------------------
ROUTES = [
    {"route": "ATL_BOS", "airline": "Delta", "totalTravelDistance": 947,  "cabin": "coach"},
    {"route": "LAX_JFK", "airline": "American Airlines", "totalTravelDistance": 2475, "cabin": "coach"},
    {"route": "SFO_EWR", "airline": "United", "totalTravelDistance": 2565, "cabin": "coach"},
    {"route": "ORD_DEN", "airline": "United", "totalTravelDistance": 888,  "cabin": "coach"},
    {"route": "CLT_SFO", "airline": "American Airlines", "totalTravelDistance": 2296, "cabin": "coach"},
    {"route": "DFW_PHL", "airline": "American Airlines", "totalTravelDistance": 1300, "cabin": "coach"},
    {"route": "MIA_LGA", "airline": "American Airlines", "totalTravelDistance": 1096, "cabin": "coach"},
    {"route": "DEN_SFO", "airline": "United", "totalTravelDistance": 967,  "cabin": "coach"},
]

# Assumed-typical values for features we can't know without live booking data.
DEFAULT_NUM_SEGMENTS = 1
DEFAULT_IS_NONSTOP = True
DEFAULT_IS_BASIC_ECONOMY = False
DEFAULT_SEATS_REMAINING = 7  # a middling, non-alarming value


def load_model_from_s3():
    """Download the model + column list from S3 into /tmp, then load them."""
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, MODEL_KEY, "/tmp/flight_price_model.joblib")
    s3.download_file(S3_BUCKET, COLUMNS_KEY, "/tmp/model_columns.joblib")

    model = joblib.load("/tmp/flight_price_model.joblib")
    model_columns = joblib.load("/tmp/model_columns.joblib")
    return model, model_columns


def build_prediction_rows():
    """
    Build one row per (route, day-out) combination for the next 30 days.
    This mirrors the same feature columns used during training.
    """
    rows = []
    today = datetime.utcnow().date()

    for days_out in range(1, 31):
        flight_date = today + timedelta(days=days_out)

        for r in ROUTES:
            rows.append({
                "route": r["route"],
                "airline": r["airline"],
                "cabin": r["cabin"],
                "totalTravelDistance": r["totalTravelDistance"],
                "daysUntilDeparture": days_out,
                "dayOfWeek": flight_date.weekday(),
                "month": flight_date.month,
                "isWeekendFlight": flight_date.weekday() in (5, 6),
                "numSegments": DEFAULT_NUM_SEGMENTS,
                "isNonStop": DEFAULT_IS_NONSTOP,
                "isBasicEconomy": DEFAULT_IS_BASIC_ECONOMY,
                "seatsRemaining": DEFAULT_SEATS_REMAINING,
                "flightDate": flight_date.isoformat(),
            })
    return pd.DataFrame(rows)


def prepare_features(df, model_columns):
    """
    One-hot encode text columns, then reindex to match the EXACT columns
    the model was trained on (same order, same set) -- filling any column
    the model expects but this batch doesn't have with 0.
    """
    text_cols = ["route", "airline", "cabin"]
    encoded = pd.get_dummies(df, columns=text_cols)

    # Keep only flightDate aside -- it's not a model feature, just metadata
    # we want to carry through for writing to DynamoDB afterward.
    flight_dates = encoded["flightDate"]
    encoded = encoded.drop(columns=["flightDate"])

    # Align to the model's expected columns exactly. Any column the model
    # expects but isn't present here (e.g. an airline not in this batch)
    # gets filled with 0 -- correctly saying "not this airline" for every row.
    aligned = encoded.reindex(columns=model_columns, fill_value=0)

    return aligned, flight_dates


def lambda_handler(event, context):
    model, model_columns = load_model_from_s3()

    raw_df = build_prediction_rows()
    features_df, flight_dates = prepare_features(raw_df, model_columns)

    predictions = model.predict(features_df)

    # Write results, using route + flightDate as the unique key.
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(DYNAMO_TABLE)

    with table.batch_writer() as batch:
        for i in range(len(raw_df)):
            route = raw_df.iloc[i]["route"]
            flight_date = raw_df.iloc[i]["flightDate"]
            price = round(float(predictions[i]), 2)

            batch.put_item(Item={
                "routeDate": f"{route}#{flight_date}",
                "route": route,
                "flightDate": flight_date,
                "predictedPrice": str(price),
                "generatedAt": datetime.utcnow().isoformat(),
            })

    return {
        "statusCode": 200,
        "body": f"Wrote {len(raw_df)} price predictions to DynamoDB."
    }
