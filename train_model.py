import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. LOAD DATA
# =====================================
df = pd.read_csv("ETO.csv")

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# =====================================
# 2. ADD TIME-BASED FEATURES
# =====================================
# Lag and rolling average (use shift to avoid data leakage)
df["trips_prev_7"] = df["trips"].shift(7)
df["passengers_prev_7"] = df["passengers"].shift(7)
df["trips_avg_7"] = df["trips"].rolling(7).mean().shift(1)
df["passengers_avg_7"] = df["passengers"].rolling(7).mean().shift(1)

# Drop first few rows with NaN (due to rolling/shift)
df = df.dropna().reset_index(drop=True)

# =====================================
# 3. ENCODE CATEGORICAL COLUMNS
# =====================================
df_encoded = pd.get_dummies(df, columns=["day_of_week", "holiday", "weather", "route_id"], drop_first=True)

# =====================================
# 4. DEFINE FEATURES AND TARGETS
# =====================================
X = df_encoded.drop(["trips", "passengers", "date"], axis=1)
y_trips = df_encoded["trips"]
y_passengers = df_encoded["passengers"]

# =====================================
# 5. CHRONOLOGICAL SPLIT (train = past years, test = latest year)
# =====================================
train_mask = df["year"] < df["year"].max()
X_train, X_test = X[train_mask], X[~train_mask]
y_trips_train, y_trips_test = y_trips[train_mask], y_trips[~train_mask]
y_pass_train, y_pass_test = y_passengers[train_mask], y_passengers[~train_mask]

# =====================================
# 6. TRAIN RANDOM FOREST MODELS
# =====================================
trips_model = RandomForestRegressor(n_estimators=200, random_state=42)
trips_model.fit(X_train, y_trips_train)

passengers_model = RandomForestRegressor(n_estimators=200, random_state=42)
passengers_model.fit(X_train, y_pass_train)

# =====================================
# 7. EVALUATE MODELS
# =====================================
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"\n {name} Model Evaluation:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    return mae, rmse, r2

# Trips
y_trips_pred = trips_model.predict(X_test)
evaluate_model("Trips", y_trips_test, y_trips_pred)

# Passengers
y_pass_pred = passengers_model.predict(X_test)
evaluate_model("Passengers", y_pass_test, y_pass_pred)

# =====================================
# 8. FEATURE IMPORTANCE VISUALIZATION
# =====================================
def show_importances(model, X, label):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(15)
    plt.figure(figsize=(8, 6))
    top_features.plot(kind='barh')
    plt.title(f"{label} Model Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

show_importances(trips_model, X, "Trips")
show_importances(passengers_model, X, "Passengers")

# =====================================
# 9. SAVE TRAINED MODELS
# =====================================
joblib.dump(trips_model, "trips_model.pkl")
joblib.dump(passengers_model, "passengers_model.pkl")

print("\n✅ Random Forest Models trained and saved successfully with lag features!")
