# predict house prices given area in sqr.ft.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/home/tim/Datasets/house_prices.csv')
feature = df[['area']]
target = df.prices
model = LinearRegression()

model.fit(df[['area']], df.prices)
print("model score: ", model.score(feature, target))
pred = model.predict([[3700]])
print(np.round(pred))

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area, df.prices, marker='+', color='red')
plt.plot(df.area, model.predict(df[['area']]), color='blue')
plt.show()