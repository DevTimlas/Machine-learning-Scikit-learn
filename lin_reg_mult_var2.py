# predict houses
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/home/tim/Datasets/data2.csv')

dummy = pd.get_dummies(df.town)

merged = pd.concat([df,  dummy], axis=1)
final = merged.drop(['town'], axis=1)

features = final.drop('price', axis=1)
target = final.price

model = LinearRegression()
model.fit(features, target)
print(model.score(features, target))
