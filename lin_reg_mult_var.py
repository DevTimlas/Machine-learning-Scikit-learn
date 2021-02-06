import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
data = iris.data
target = iris.target

# X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(data, target)

score = model.score(data, target)

print('model score:', score)

pred = model.predict([[5, 6, 4, 1]])
print(pred)