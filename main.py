import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df = yf.download('ETH-USD', start='2022-1-1', end='2022-1-31')
data = df

df.reset_index(inplace=True)
df.drop('Open', inplace=True, axis=1)
df.drop('High', inplace=True, axis=1)
df.drop('Low', inplace=True, axis=1)
df.drop('Adj Close', inplace=True, axis=1)
df.drop('Volume', inplace=True, axis=1)

df['Date'] = [d.date() for d in df['Date']]
df["day"] = df['Date'].map(lambda x: x.day)
df["month"] = df['Date'].map(lambda x: x.month)
df["year"] = df['Date'].map(lambda x: x.year)

df.drop(['day', 'month', 'year'], axis=1, inplace=True) #not used in this particular project
df.columns = ['ds', 'y']

m = Prophet(interval_width=0.95, daily_seasonality=True)
model = m.fit(df)

future = m.make_future_dataframe(periods=100, freq='D')
forecast = m.predict(future)

# print(forecast.head())
plot1 = m.plot(forecast)
plot2 = m.plot_components(forecast)