from tkinter import Y
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
# from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data=pd.read_csv("student/student-mat.csv",sep=";")
data=data[["G1","G2","G3","studytime","goout","Dalc","Walc","failures","absences"]]
predict="G3"

X = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# best=0
# for a in range(10000):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     # print(accuracy)
#     if accuracy > best:
#         best = accuracy
#         with open("student/student.pickle","wb") as f:
#             pickle.dump(linear,f)

# print(best)

pickle_in = open("student/student.pickle","rb")
linear = pickle.load(pickle_in)

# print("accuracy: ", accuracy)
# print("coefficient: ", linear.coef_)
# print("intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

# for i, v in enumerate(predictions):
#     print(v, x_test[i], y_test[i])