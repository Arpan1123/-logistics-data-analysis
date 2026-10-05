import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("hypothetical_logistics_week4.csv")
X = df.drop(columns=["delivery_time_days"])
y = df["delivery_time_days"]

num = ["shipment_volume_units", "distance_km"]
cat = ["region", "transport_mode", "priority", "traffic_level"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

params = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [8, 12, None]
}

grid = GridSearchCV(model, params, cv=3,
                    scoring="neg_mean_absolute_error")
grid.fit(X_train, y_train)

pred = grid.best_estimator_.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("Best parameters:", grid.best_params_)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
