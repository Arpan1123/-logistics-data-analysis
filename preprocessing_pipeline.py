import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load
orders = pd.read_csv("../data/olist_orders_dataset.csv")
items = pd.read_csv("../data/olist_order_items_dataset.csv")

# Convert timestamps
date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Feature engineering
orders["delivery_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_purchase_timestamp"]
).dt.total_seconds() / 86400

orders["delay_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

orders["is_late"] = orders["delay_days"] > 0

# Outlier flag using IQR
q1 = items["freight_value"].quantile(0.25)
q3 = items["freight_value"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

items["freight_outlier"] = (
    (items["freight_value"] < lower) |
    (items["freight_value"] > upper)
)

# Example scaling
numeric = items[["price", "freight_value"]].fillna(
    items[["price", "freight_value"]].median()
)
scaler = StandardScaler()
scaled = scaler.fit_transform(numeric)

print("Orders:", orders.shape)
print("Items:", items.shape)
print("Duplicate orders:", orders["order_id"].duplicated().sum())
print("Potential freight outliers:", items["freight_outlier"].sum())
