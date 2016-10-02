import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
  
gender_dataframe = pd.read_csv('main_data.csv')

print  gender_dataframe.shape
gender_labels = gender_dataframe.label	
labels = list(set(gender_labels))
gender_labels = np.array([labels.index(x) for x in gender_labels])
gender_features = gender_dataframe.iloc[:,:20]
gender_features = np.array(gender_features)


gender_features_scaled = preprocessing.scale(gender_features)
print "gender labels: "
print gender_labels #this is the labels vector Y
print 
print "gender features:"
print gender_features #this is the feature vector X
print "Yuhuuh"
print gender_features_scaled[1]
classifier = svm.SVC()
classifier.fit(gender_features_scaled, gender_labels)

test_dataframe = pd.read_csv('data_test.csv')

gender_labels = gender_dataframe.label
labels = list(set(gender_labels))
test_labels = np.array([labels.index(x) for x in gender_labels])

test_features = test_dataframe.iloc[:,:20]
test_features = np.array(test_features)



results = classifier.predict(test_features)
print "Test features"
print test_features	

num_correct = (results == test_labels)
print "Hehhe"
print num_correct.sum()
print len(test_labels)
total_correct = num_correct.sum()


recall = float(total_correct)/(len(test_labels))
print recall
print "model accuracy (%): ", recall * 100, "%"