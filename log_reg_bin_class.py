# predict if a person have insurance or not

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

df = pd.read_csv('/home/tim/Datasets/insurance.csv')
feature = df[['age']]

target = df.have_insurance

model = LogisticRegression()
model.fit(feature, target)
print(model.score(feature, target))
