import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Load the training data
training_dataframe = pd.read_csv('main_data.csv')

# Print the size of the feature vector
print "The size of the feature vector is:"
print  training_dataframe.shape
print 

# Getting the labels from the feature vector that is creating the Y array
training_labels = training_dataframe.label	
labels = list(set(training_labels))
training_labels = np.array([labels.index(x) for x in training_labels])
training_features = training_dataframe.iloc[:,:20]
training_features = np.array(training_features)

print "_________"


# Normalizing the data by applyinfeature scaling
scaler = preprocessing.StandardScaler().fit(training_features)
# print "Scaler"
# print scaler
# print scaler.mean_ 
# print scaler.scale_
normalized_features = scaler.transform(training_features) 


# Run the SVM classifier
classifier = svm.SVC()
classifier.fit(normalized_features, training_labels)

# Load the testing data
testing_dataframe = pd.read_csv('data_test.csv')

# Extract Y
training_labels = training_dataframe.label
labels = list(set(training_labels))
testing_labels = np.array([labels.index(x) for x in training_labels])
testing_features = testing_dataframe.iloc[:,:20]
testing_features = np.array(testing_features)

# Normalizing the testing data as well
scaler = preprocessing.StandardScaler().fit(testing_features)
normalized_testing = scaler.transform(testing_features)

# the prediction from the classifier are stored in the results variable
results = classifier.predict(normalized_testing)
	
#counting the number of correctly predicted labels by SVM
num_correct = (results == testing_labels)
print "The number of correctly predicted outcomes"
print num_correct.sum()
print
print "The total input data"
print len(testing_labels)
print
total_correct = num_correct.sum()

print "The output labels of testing data"
print testing_labels
recall = float(total_correct)/(len(testing_labels))

print recall
print
print "Model accuracy (%): ", recall * 100, "%"