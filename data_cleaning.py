import pandas as pd


def load_data(data_dir="../data"):
    """Load the main Olist logistics tables."""
    orders = pd.read_csv(f"{data_dir}/olist_orders_dataset.csv")
    items = pd.read_csv(f"{data_dir}/olist_order_items_dataset.csv")
    customers = pd.read_csv(f"{data_dir}/olist_customers_dataset.csv")
    return orders, items, customers


def prepare_orders(orders):
    """Convert timestamps and create delivery-performance features."""
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_cols:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    delivered = orders.dropna(
        subset=["order_delivered_customer_date"]
    ).copy()

    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    delivered["delay_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    delivered["is_late"] = delivered["delay_days"] > 0

    return delivered
