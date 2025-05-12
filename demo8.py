import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
df=pd.read_csv("accidental-deaths-in-usa-monthly.csv",parse_dates=["Month"])
print(df)
df["Month"]=df["Month"].dt.month
print(df)
plt.scatter(df.Month,df.Year)
scaler=MinMaxScaler()
df["Month"]=scaler.fit_transform(df["Month"])
