import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

data_train = np.loadtxt('data.csv', delimiter=',')
X = data_train[:, 0:]
y = data_train[:, 0].astype(np.int)
clf = ExtraTreesClassifier(n_estimators=100).fit(X, y)

data_test = np.loadtxt('data_test.csv', delimiter=',')
print(clf.predict(data_test))