import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("hypothetical_logistics.csv")
print(df.describe())
print(df.isna().sum())

avg = df.groupby("region")["delivery_time_days"].mean()
print(avg)

plt.hist(df["delivery_time_days"], bins=30)
plt.title("Distribution of Delivery Time")
plt.show()

plt.scatter(df["distance_km"], df["transport_cost"], alpha=.4)
plt.title("Transportation Cost vs Distance")
plt.show()

corr = df[["shipment_volume_units","distance_km","delivery_time_days","transport_cost"]].corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()
