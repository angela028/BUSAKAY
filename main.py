from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# ===== Load Trained Models =====
trips_model = joblib.load("trips_model.pkl")
passengers_model = joblib.load("passengers_model.pkl")

# ===== FastAPI Initialization =====
app = FastAPI(title="Bus Prediction API")

# ===== CORS Configuration =====
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://busakay-pasiguenio.web.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ===== Request Schemas =====
class PredictionRequest(BaseModel):
    year: int
    month: int
    day: int
    holiday: int
    weather: str
    route_id: str

class ForecastRequest(BaseModel):
    start_date: str           # 'YYYY-MM-DD'
    months_ahead: int = 5
    routes: list[str] = ["Route_1", "Route_2"]
    weather: str = "Clear"
    holiday: int = 0

# ===== Helper Function =====
def prepare_features(df: pd.DataFrame):
    """Match incoming data to model feature columns."""
    df_encoded = pd.get_dummies(df, columns=["day_of_week", "weather", "route_id"], drop_first=True)
    df_encoded = df_encoded.reindex(columns=trips_model.feature_names_in_, fill_value=0)
    return df_encoded

# ===== Root Endpoint =====
@app.get("/")
def root():
    return {"message": "Bus Prediction API is running."}

# ===== /predict Endpoint =====
@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        date_obj = datetime(req.year, req.month, req.day)
        day_of_week = date_obj.strftime("%A")

        # 🔹 Skip weekends (Saturday=5, Sunday=6)
        if date_obj.weekday() in [5, 6]:
            return {"error": "No trips on weekends (Saturday and Sunday)."}

        # Prepare input data
        df = pd.DataFrame([{
            "year": req.year,
            "month": req.month,
            "day": req.day,
            "day_of_week": day_of_week,
            "holiday": req.holiday,
            "weather": req.weather,
            "route_id": req.route_id
        }])

        features = prepare_features(df)

        trips_pred = int(round(trips_model.predict(features)[0]))
        passengers_pred = int(round(passengers_model.predict(features)[0]))

        return {
            "year": req.year,
            "month": req.month,
            "day": req.day,
            "day_of_week": day_of_week,
            "route_id": req.route_id,
            "weather": req.weather,
            "holiday": req.holiday,
            "predicted_trips": trips_pred,
            "predicted_passengers": passengers_pred
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ===== /forecast Endpoint =====
@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        start_date = datetime.strptime(req.start_date, "%Y-%m-%d")
        end_date = start_date + pd.DateOffset(months=req.months_ahead)
        date_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq="D")

        rows = []
        for date in date_range:
            # 🔹 Skip weekends
            if date.weekday() in [5, 6]:
                continue

            for route in req.routes:
                rows.append({
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "day_of_week": date.strftime("%A"),
                    "holiday": req.holiday,
                    "weather": req.weather,
                    "route_id": route
                })

        df_forecast = pd.DataFrame(rows)
        if df_forecast.empty:
            return {"error": "No valid weekdays found in forecast period."}

        features = prepare_features(df_forecast)

        df_forecast["predicted_trips"] = trips_model.predict(features).round().astype(int)
        df_forecast["predicted_passengers"] = passengers_model.predict(features).round().astype(int)

        return df_forecast.to_dict(orient="records")

    except Exception as e:
        return {"error": f"Forecast failed: {str(e)}"}
