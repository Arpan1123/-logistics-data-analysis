import matplotlib.pyplot as plt
import seaborn as sns


def plot_delivery_distribution(orders):
    """Plot the distribution of delivery lead time."""
    plt.figure(figsize=(9, 5))
    sns.histplot(
        orders["delivery_days"].dropna(),
        bins=40,
        kde=True
    )
    plt.title("Distribution of Delivery Lead Time")
    plt.xlabel("Delivery time (days)")
    plt.ylabel("Number of orders")
    plt.tight_layout()
    plt.show()


def plot_late_orders(orders):
    """Plot late vs on-time deliveries."""
    counts = orders["is_late"].value_counts()
    labels = ["On Time", "Late"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, [counts.get(False, 0), counts.get(True, 0)])
    plt.title("On-Time vs Late Deliveries")
    plt.ylabel("Number of orders")
    plt.tight_layout()
    plt.show()
