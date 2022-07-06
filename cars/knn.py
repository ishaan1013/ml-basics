import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style

data=pd.read_csv("cars/car.data")
print(data.head())

convert = preprocessing.LabelEncoder()
buying = convert.fit_transform(list(data["buying"]))
maint = convert.fit_transform(list(data["maint"]))
door = convert.fit_transform(list(data["door"]))
persons = convert.fit_transform(list(data["persons"]))
lug_boot = convert.fit_transform(list(data["lug_boot"]))
safety = convert.fit_transform(list(data["safety"]))
# class
cls = convert.fit_transform(list(data["class"]))

predict=cls
X = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

nbs=7
model = KNeighborsClassifier(n_neighbors=nbs)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
# print(accuracy)

p = model.predict(x_test)
names=["unacc", "acc", "good", "vgood"]
for i, v in enumerate(p):
    print("Predicted: ", names[v], "Data: ", x_test[i], "Actual: ", names[y_test[i]])

    n=model.kneighbors([x_test[i]], nbs, True)
    print("N: ",n)