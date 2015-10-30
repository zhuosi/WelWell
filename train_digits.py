
from scipy.io import loadmat

import numpy as np

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

train = loadmat('train.mat')
train_images = train['train_images']
train_labels = train['train_labels']

# Reshape the arrays. Note that both need to be transposed
X = train_images[:,:, :60000].reshape(28*28, 60000).transpose()
y = train_labels[:60000].reshape(1,60000).transpose()


# Custom split function that splits X_data and y_data into training and test
# sets. Specify the size of the desired test set as the third argument. 
def split_data(X_data, y_data, testSize):

    # Generate a random shuffling of the number of rows
    indicies = np.arange(X_data.shape[0])
    permutation = np.random.permutation(indicies)

    # Pull out the number of test cases
    test_indicies = permutation[:testSize]
    test_set = X_data[test_indicies]
    test_labels = y_data[test_indicies]

    # The rest of the rows are for training
    training_indicies = np.setdiff1d(permutation, test_indicies)
    # Make sure the training set is shuffled as well
    training_indicies = np.random.permutation(training_indicies)

    training_set = X_data[training_indicies]
    training_labels = y_data[training_indicies]

    return (training_set, test_set, training_labels, test_labels)


# Split into 50000 training examples and 10000 test examples
X_training_set, X_validation, y_training_set, y_validation = split_data(X, y, 10000)


# For Problem 1, plot the validation error rate of 7 different sized training sets
def plotErrorRates():
    training_set_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    error_rates = []

    # Set up the classifer, a linear SVM
    classifier = svm.SVC(kernel='linear')

    print ("Running Problem 1...")

    for t in training_set_sizes:
        
        # Of the 50000 training points left, split the data into a training set and a test set
        X_train, X_test, y_train, y_test = split_data(X_training_set, y_training_set, 50000 - t)

        classifier.fit(X_train, y_train.ravel())

        print ("Training set size: ", t)
        #print (classifier.score(X_train, y_train))

        # Compute the error on the validation set of size 10000
        print (classifier.score(X_validation, y_validation))
        error = 1 - classifier.score(X_validation, y_validation)
        error_rates.append(error)

    plt.scatter(training_set_sizes, error_rates)
    plt.xlabel('Training set size')
    plt.ylabel('Error on validation set')
    plt.xlim(0, 10500)
    plt.title('Error Rate vs Training Size')
    plt.show()

plotErrorRates()


# For Problem 2, draws 7 confusion matrices for various size training sets
def drawConfusionMatrices():
    training_set_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    error_rates = []

    # Set up the classifer, a linear SVM
    classifier = svm.SVC(kernel='linear')

    print ("Running Problem 2...")

    for index, t in enumerate(training_set_sizes):

       # Of the 50000 training points left, split the data into a training set and a test set
       X_train, X_test, y_train, y_test = split_data(X_training_set, y_training_set, 50000 - t)

       classifier.fit(X_train, y_train.ravel())

       print ("Training set size: ", t)

       # Compute the error on the validation set of size 10000
       print (classifier.score(X_validation, y_validation))
       error = 1 - classifier.score(X_validation, y_validation)
       error_rates.append(error)

       # Compute confusion matrix
       y_pred = classifier.predict(X_validation)
       cm = confusion_matrix(y_validation, y_pred)

       # Show confusion matrix in a separate window
       plt.subplot(2, 4, index+1)
       plt.matshow(cm, 0)
       plt.title('Training Size = ' + str(t))
       plt.colorbar()
       plt.ylabel('True label')
       plt.xlabel('Predicted label')

    plt.show()

drawConfusionMatrices()


# Helper function for Problem 3.
# Use X and y as training in 10 fold cross validation to determine the optimal C value
def find_optimal_C(C_values, X, y):

    # 10 fold cross validation
    K = 10

    avg_scores = []
    for i in range(len(C_values)):

        C = C_values[i] 

        indicies = np.arange(X.shape[0])
        permutation = np.random.permutation(indicies)

        scores = []

        classifier = svm.SVC(kernel='linear', C=C)

        for k in range(K):
            startIndex = k * 1000
            endIndex = k * 1000 + 1000

            test_indicies = permutation[startIndex:endIndex]
            test_set = X[test_indicies]
            test_labels = y[test_indicies]

            training_indicies = np.setdiff1d(permutation, test_indicies)
            # Make sure the training indicies are shuffled as well
            training_indicies = np.random.permutation(training_indicies)

            k_set = X[training_indicies]
            k_labels = y[training_indicies]

            classifier.fit(k_set, k_labels.ravel())

            score = classifier.score(test_set, test_labels)
            print (score)
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        print ("C value:", C, "  avg_score: ", avg_score)
        avg_scores.append(avg_score)

    # Find the best averaged cross validation score
    max_index = avg_scores.index(max(avg_scores))
    max_C = C_values[max_index]

    return max_C

# Pull out 10000 points to use for cross validation.
X_cross_validation, _, y_cross_validation, _ = split_data(X_training_set, y_training_set, 40000)


# Trains the final SVM for Problem 3, training on a 10000 set and testing 
# on the validation set of 100000.
def trainFinalSVM(C_value):
    classifier = svm.SVC(kernel='linear', C = best_C)
    classifier.fit(X_cross_validation, y_cross_validation.ravel())

    score = classifier.score(X_validation, y_validation)
    return score

print "Running Problem 3..."
# Find the optimal C value through 10 Fold cross validation
possible_C = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
possible_C = [0.00000001, 0.0000001, 0.000001]
best_C = find_optimal_C(possible_C, X_cross_validation, y_cross_validation)

# Through testing, found that the best value of C was 10^-6
#best_C = 0.000001 

print "Best C value: ", best_C

# Train an SVM with the best value of C on 10000 data points, and report the score on a 
# validation set of 10000 points.
final_score = trainFinalSVM(best_C)
print "Final SVM validation score: ", final_score


def writePredictions(C_value):

    # Load the test data from test.mat
    test = loadmat('test.mat')
    test_images = test['test_images']

    # Reshape the test array
    X_test = test_images[:,:, :10000].reshape(28*28, 10000).transpose()

    classifier = svm.SVC(kernel='linear', C=C_value)
    # Train on the entire training set (60000 examples)
    classifier.fit(X, y.ravel()) 
    predictions = classifier.predict(X_test)


	# Write out the predictions to a CSV file
    f = open('predictions.csv','w')
    f.write('Id,Category\n')
    for (i, e) in enumerate(predictions):
        f.write('%d,%d\n' % (i+1, e))
    f.close()

writePredictions(best_C)

