from sklearn import tree
from sklearn import ensemble
import csv
import numpy as np
import matplotlib.pyplot as plt

# cross validation
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.feature_extraction import DictVectorizer


from collections import Counter


import pandas as pd 


ordering = ["TripType_3","TripType_4","TripType_5","TripType_6","TripType_7","TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18","TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24","TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30","TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36","TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42","TripType_43","TripType_44","TripType_999"]

ordering = [int(x.split('_')[1]) for x in ordering]

# Error function for the multi-class log loss
def cross_entropy(pred, label):
    """
    :param pred: prediction matrix with shape (n_pred, n_class),
                 each row is rescaled to sum to 1
    :param label: a list/array with label[i] = index of correct class,
                  each element shoudl be on the interval [0, n_class - 1]
    :return: cross entropy / multiclass logarithmic loss
    """
    tiny = np.ones(pred.shape) * 1e-15
    pred = np.max((np.min((pred, 1-tiny), 0), tiny), 0)
    pred = (pred.T / pred.sum(1)).T
    return -np.mean([np.log(p[label[i]]) for i, p in enumerate(pred)])



train = pd.read_csv('train.csv')
numTrain = train.shape[0]
test = pd.read_csv('test.csv')

allData = train.append(test)

dummiesAll = pd.get_dummies(allData, dummy_na=True)
# Don't really need these. Too much info
dummiesAll = dummiesAll.drop(['Upc', 'ScanCount', 'FinelineNumber'], axis=1)

dummiesTrain = dummiesAll.iloc[:numTrain]
dummiesTest = dummiesAll.iloc[numTrain:]

# Get rid of the TripType category for the test
dummiesTest = dummiesTest.drop(['TripType'], axis=1)

groupedTrain = dummiesTrain.groupby(['VisitNumber', 'TripType'])
groupedTest = dummiesTest.groupby(['VisitNumber'])

X_train = groupedTrain.sum()
X_test = groupedTest.sum()


Y_train = np.array([int(i[1]) for i in X_train.index.values])
train_VisitNumbers = [int(i[0]) for i in X_train.index.values]

test_VisitNumbers = X_test.index.values


truth_values = np.zeros(Y_train.shape[0])
for i, e in enumerate(Y_train):
    truth_values[i] = ordering.index(e)

print "Training Decision Tree..."

clf = tree.DecisionTreeClassifier()


class_w = Counter(Y_train)
num_forests = [10]
for n in num_forests:
    print "Forests: ", n
    clf = SVC(probability = True)

    #clf = RandomForestClassifier(n_estimators = n, criterion = 'gini', class_weight = None)
    #K = 2
    #scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=K, scoring='log_loss', verbose = 1, n_jobs = -1)
    #print('Scores = {}'.format(scores))

    clf = clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    training_probabilities = clf.predict_proba(X_train)

    class_probabilities = clf.predict_proba(X_test)

    #cross_entropy(truth_values, training_probabilities)
    print "Training Loss: ", metrics.log_loss(Y_train, training_probabilities)




"""
f = open('walmartDTree.csv','w')
f.write('"VisitNumber","TripType_3","TripType_4","TripType_5","TripType_6","TripType_7","TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18","TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24","TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30","TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36","TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42","TripType_43","TripType_44","TripType_999"\n')
for (i, e) in enumerate(class_probabilities):
    visitNumber = test_VisitNumbers[i]
    a = ['{:.2f}'.format(x) for x in e] # format floats
    for i,e in enumerate(a):
        if e == '0.00':
            a[i] = '0'
    a_join = ",".join(a) # join by commas
    print (visitNumber, a_join)
    f.write('%d,%s\n' % (visitNumber, a_join))
f.close()
"""

"""
numVisitsTrain = sumVisitsTrain.shape[0]
Y_train = np.zeros(numVisitsTrain)
for i in range(numVisitsTrain):
    key = sumVisitsTrain.index[i]  #(VisitNumber, TripType)
    Y_train[i] = int(key[1])
"""

"""
# Don't really need these. Too much info
dummiesTrain = dummiesTrain.drop(['Upc', 'ScanCount', 'FinelineNumber'], axis=1)


#train.groupby(['Weekday', 'TripType'], as_index=False)['VisitNumber'].agg(uniqueCount)


groupedVisitsTrain = dummiesTrain.groupby(['VisitNumber', 'TripType'])
sumVisitsTrain = groupedVisitsTrain.sum()

numVisitsTrain = sumVisitsTrain.shape[0]
Y_train = np.zeros(numVisitsTrain)
for i in range(numVisitsTrain):
    key = sumVisitsTrain.index[i]  #(VisitNumber, TripType)
    Y_train[i] = int(key[1])


test = pd.read_csv('test.csv')

dummiesTest = pd.get_dummies(test, dummy_na=True)
# Don't really need these. Too much info
dummiesTest = dummiesTest.drop(['Upc', 'ScanCount', 'FinelineNumber'], axis=1)

groupedVisitsTest = dummiesTest.groupby(['VisitNumber'])
sumVisitsTest = groupedVisitsTest.sum()
"""


"""
fin_name = 'train.csv'
fout_name = 'test.csv'

# Read in the training data.
with open(fin_name) as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:
        print row
"""

"""
# Read in the training data.
with open('train.csv') as csvfile:

    reader = csv.DictReader(csvfile)

    parsedTrainingData = []
    parsedTrainingLabels = []

    for row in reader:

        #Remove the label and record it seperately.
        parsedTrainingLabels.append(int(row['TripType']))
        del row['TripType']

        # Convert the continuous fields to floats
        for key, value in row.iteritems():
            try:
                row[key] = float(value)
            except ValueError:
                pass

        parsedTrainingData.append(row)

"""

"""
# Read in the training data.
with open('test.csv') as csvfile:

    reader = csv.DictReader(csvfile)

    parsedTestData = []

    for row in reader:

        # Convert the continuous fields to floats
        for key, value in row.iteritems():
            try:
                row[key] = float(value)
            except ValueError:
                pass

        parsedTestData.append(row)




print "Done reading in data. Vectorizing..."
# Pass all the data into the Dict Vectorizer to do One-Hot encoding
v = DictVectorizer(sparse=False)

# Fit the training and test data to the DictVectorizer.
#v.fit(parsedTrainingData)
#X_train = v.transform(parsedTrainingData)


Y_train = np.array(parsedTrainingLabels)
X_test = v.transform(parsedTestData)

labelDist = Counter(Y_train)

print "Done vectorizing. Training..."
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

class_probabilities = clf.predict_proba(X_test)

with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(class_probabilities)

"""


"""
f = open('walmartDTree.csv','w')
f.write('"VisitNumber","TripType_3","TripType_4","TripType_5","TripType_6","TripType_7","TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18","TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24","TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30","TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36","TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42","TripType_43","TripType_44","TripType_999"\n')
for (i, e) in enumerate(predictions):
    #print i, e
    f.write('%d,%d\n' % (i+1, e))
"""


'''
clf.fit(X_train, Y_train)
G_testFile = clf.predict(X_testFile)
f = open('predictionsSvm.csv','w')
f.write('Id,Prediction\n')
for (i, e) in enumerate(G_testFile):
    #print i, e
    f.write('%d,%d\n' % (i+1, e))
'''


"""
input_file = csv.DictReader(open("train.csv"))

#for i in input_file:
#    print i

v = DictVectorizer(sparse=False)
#v.restrict(support)
X = v.fit_transform(input_file)
"""

"""
# Pass all the data into the Dict Vectorizer to do One-Hot encoding
v = DictVectorizer(sparse=False)

# Fit the training and test data to the DictVectorizer.
v.fit(parsedTrainingData)
X_train = v.transform(parsedTrainingData)
X_test = v.transform(parsedTestData)
"""