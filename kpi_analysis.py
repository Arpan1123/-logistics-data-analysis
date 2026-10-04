import pandas as pd


def calculate_kpis(orders, items):
    """Calculate headline logistics KPIs."""
    delivered = orders.dropna(
        subset=["order_delivered_customer_date"]
    ).copy()

    if "delivery_days" not in delivered.columns:
        delivered["delivery_days"] = (
            delivered["order_delivered_customer_date"]
            - delivered["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

    if "delay_days" not in delivered.columns:
        delivered["delay_days"] = (
            delivered["order_delivered_customer_date"]
            - delivered["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86400

    delivered["is_late"] = delivered["delay_days"] > 0

    on_time_rate = (~delivered["is_late"]).mean() * 100
    avg_delivery_days = delivered["delivery_days"].mean()

    freight_by_order = items.groupby("order_id")["freight_value"].sum()
    avg_freight_per_order = freight_by_order.mean()

    return {
        "on_time_delivery_rate_percent": on_time_rate,
        "average_delivery_lead_time_days": avg_delivery_days,
        "average_freight_cost_per_order": avg_freight_per_order,
    }
