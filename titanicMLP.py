import numpy as np
import pandas as pd  # data processing, CSV file I/O

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./titanic_latest.csv').fillna(1e6).astype(np.float64)

y = df['survived']
x = df.drop(['survived'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=27)
print(y_test)

scaler = StandardScaler()
scaler.fit(x_train)  # fit only on training data
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)  # apply same transformation to test data

clf = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=10000, alpha=0.0001, solver='sgd', verbose=10,
                    random_state=21, tol=0.0000001, n_iter_no_change=20)

clf.fit(x_train, y_train)  # fit the model to data matrix X and target(s) y

y_pred = clf.predict(x_test)  # predict using the multi-layer perceptron classifier


print("\n Accuracy Score")
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix")
print(cm)
