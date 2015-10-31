
from sklearn import tree
from sklearn import ensemble
import csv
import numpy as np
import matplotlib.pyplot as plt

# cross validation
from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing


def get_error(G, Y):
    error = 0
    for i in range(len(G)):
        if G[i] != Y[i]:
            error += 1
    return 1.0 * error / len(G)

# NOTE: Decrease if you want to do your own CV on the trianing data
NUM_TRAININGS = 15120

fin_name = 'train.csv'
fout_name = 'test.csv'


# grab the real training data
with open(fin_name, 'r') as fin:
    next(fin)
    trainingData = np.array(list(csv.reader(fin)))

print trainingData.shape

# grab the real test data
with open(fout_name, 'r') as fout:
    next(fout)
    testData = np.array(list(csv.reader(fout)))
print testData.shape


X_train = trainingData[:NUM_TRAININGS, 1:-1]
Y_train = trainingData[:NUM_TRAININGS, -1]

# these will be empty unless you do some cross validation
X_validation = trainingData[NUM_TRAININGS:, 1:-1]
Y_validation = trainingData[NUM_TRAININGS:, -1]

X_testfile = testData[:, 1:]

min_samples_leafs = range(1,15)

for min_samples_leaf in min_samples_leafs:

    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf)
    # train the model
    """
    clf = clf.fit(X_train, Y_train)

    # make prediction
    G_train = clf.predict(X_train)
    G_test = clf.predict(X_test)

    train_error = get_error(G_train, Y_train)
    #train_errors.append(train_error)
    test_error = get_error(G_test, Y_test)
    #test_errors.append(test_error)
    print train_error
    print test_error

    """

    # Run some cross validation
    K = 10
    scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=K, scoring='accuracy', verbose = 0, n_jobs = -1)
    #print('Scores = {}'.format(scores))
    print 'Min_samples = ', min_samples_leaf, ' Avg score: ', sum(scores)/len(scores)


