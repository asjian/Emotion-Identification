"""EECS 445 - Winter 2022.

Project 1
"""

import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt


from helper import *

import warnings
from sklearn.exceptions import ConvergenceWarning

import random
import math

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)


def extract_word(input_string):
    """Preprocess review into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    # TODO: Implement this function
    lemmatizer = WordNetLemmatizer()
    specialpunct = ['!']
    emotions = [':)', ':D', ':(', ':/',':|', ':$']
    
    '''
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
    wordlist = input_string.lower().translate(translator).strip().split()
    '''
    
    input_string = input_string.strip()

    for i in range(len(input_string)):
        char = input_string[i]

        if input_string[i:i+2] in emotions:
            input_string = input_string.replace(input_string[i:i+2], ' ' + input_string[i:i+2] + ' ')

        elif char in specialpunct:
            input_string = input_string.replace(char, ' ' + char + ' ')

        elif char in string.punctuation:
            input_string = input_string.replace(char, ' ')

    input_string = input_string.lower()
    wordlist = input_string.strip().split()

    for k in range(len(wordlist)):
        if not wordlist[k] in emotions and not wordlist[k] in specialpunct:
            wordlist[k] = lemmatizer.lemmatize(wordlist[k])

    refinedlist = []
    for word in wordlist:
        if not word in stop_words:
            refinedlist.append(word)

    #random insertion
    if len(refinedlist) > 15:
        rindex = random.randint(0, len(refinedlist)-1)
        iword = refinedlist[rindex]
        refinedlist.append(iword)

        #random deletion
        dindex = random.randint(0, len(refinedlist)-1)
        refinedlist.pop(dindex)

    return refinedlist

def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | text                          | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    # TODO: Implement this function
    index = 0
    for text in df['text']:
        wordlist = extract_word(text)
        for word in wordlist:
            if not word in word_dict:
                word_dict[word] = index
                index += 1
    
    return word_dict

def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words+1))
    
    rownum = 0
    for text in df['text']:
        wordlist = extract_word(text)
        for word in wordlist:
            if word in word_dict:
                currval = feature_matrix[rownum][word_dict[word]]
                if currval == 0:
                    feature_matrix[rownum][word_dict[word]] = 1
                else:
                    feature_matrix[rownum][word_dict[word]] = math.sqrt(1+currval**2) #repeats considered

        #feature_matrix[rownum][-2] = len(wordlist)/10 #consider length
        feature_matrix[rownum][-1] = 1 #augmentation
        rownum += 1

    return feature_matrix

def performance(y_true, y_pred, metric="accuracy", multi_class = False):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.

    if metric == 'accuracy':
        return metrics.accuracy_score(y_true, y_pred)

    elif metric == 'f1_score':
        return metrics.f1_score(y_true, y_pred)
    
    elif metric == 'precision':
        return metrics.precision_score(y_true, y_pred)
    
    elif metric == 'auroc':
        if multi_class:
            return metrics.roc_auc_score(y_true, y_pred, multi_class = 'ovo')
        return metrics.roc_auc_score(y_true, y_pred)
    
    elif metric == 'sensitivity':
        mat = metrics.confusion_matrix(y_true, y_pred)
        return mat[1,1]/(mat[1,0]+mat[1,1])
    
    elif metric == 'specificity':
        mat = metrics.confusion_matrix(y_true, y_pred)
        return mat[0,0]/(mat[0,0] + mat[0,1])

def cv_performance(clf, X, y, k=5, metric="accuracy", multi_class = False):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

    # Put the performance of the model on each fold in the scores array
    scores = []

    skf = StratifiedKFold(n_splits = k, shuffle = False)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)

        if metric == 'auroc':
            val = performance(y_test, clf.decision_function(X_test), metric, multi_class)
        else:
            val = performance(y_test, clf.predict(X_test), metric, multi_class)

        scores.append(val)
        
    return np.array(scores).mean()


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """Search for hyperparameters of linear SVM with best k-fold CV performance.
    
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1"ß)
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    maxC = C_range[0]
    maxperf = 0

    for c in C_range:
        svc = LinearSVC(penalty=penalty, loss = loss, dual = dual, C=c, random_state = 445)
        perfval = cv_performance(svc,X,y,k,metric)
        if perfval > maxperf:
            maxperf = perfval
            maxC = c

    print('Metric: ' + metric)
    print('Best C: ' + str(maxC))
    print('CV Score: ' + str(maxperf))
    return maxC
    

def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: penalty to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for c in C_range:
        counter = 0
        svc = LinearSVC(penalty = penalty, loss = loss, C=c, dual = dual, random_state = 445)
        svc.fit(X,y)

        for component in svc.coef_[0]: #L0 norm, the number of nonzero components
            if not component == 0:
                counter += 1

        norm0.append(counter)

    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()

def find_important_words(X,y, word_dict, penalty = 'l2', loss = 'hinge', C=0.1, dual = True, random_state = 445):

    svc = LinearSVC(penalty=penalty, loss = loss, dual = dual, C=C, random_state = 445)
    svc.fit(X,y)

    indtheta = sorted(zip(svc.coef_[0], [a for a in range(0, len(svc.coef_[0]))]))
    results = []

    for tup in indtheta[0:5]:
        for key in word_dict:
            if word_dict[key] == tup[1]:
                results.append([tup[0],key])
                break
    
    for tup2 in indtheta[-5:]:
        for key in word_dict:
            if word_dict[key] == tup2[1]:
                results.append([tup2[0],key])
                break
    
    return results

def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[], mode = 'grid'):
    """Search for hyperparameters of quadratic SVM with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    maxperf = -10000

    for pair in param_range:
        c = pair[0]
        r = pair[1]
        print(c,r)

        svc = SVC(kernel='poly', degree=2, C=c, coef0=r, gamma = 'auto')
        perfval = cv_performance(svc, X, y, k, metric)

        print(perfval)
        print()

        if perfval > maxperf:
            maxperf = perfval
            best_C_val = c
            best_r_val = r
    
    print()
    print('Quadratic Kernel, ' + mode + ' search, ' + metric + ' metric:')
    print('Best C: ' + str(best_C_val) + ' Best r: ' + str(best_r_val))
    return best_C_val, best_r_val

def select_proper_weights(X, y, k=5, metric = 'f1_score', weights = []):
    maxperf = 0
    maxweights = []

    for pair in weights:
        wn = pair[0]
        wp = pair[1]

        svc = LinearSVC(penalty = 'l2', loss = 'hinge', C = 0.01, class_weight = {-1:wn, 1:wp}, dual = True)
        perfval = cv_performance(svc, X, y, k, metric)

        if perfval > maxperf:
            maxperf = perfval
            maxweights = pair
    
    print('Best Weights for Imbalanced Data: Wn = ' + str(maxweights[0]) + ', Wp= ' + str(maxweights[1]))
    print('Performance (' + metric + '): ' + str(maxperf))
    return maxweights

def choose_hp_multi(X, y, k=5, metric = 'accuracy', hps = []):
    maxperf = 0
    maxC = 0
    maxweights = []


    for quad in hps:
        C = quad[0]
        wn = quad[1]
        w0 = quad[2]
        wp = quad[3]

        svc = LinearSVC(penalty = 'l2', loss = 'squared_hinge', C = C, class_weight = {-1:wn, 0:w0, 1:wp}, dual = True, multi_class = 'ovr')
        perfval = cv_performance(svc, X, y, k, metric)

        if perfval > maxperf:
            maxperf = perfval
            maxC = C 
            maxweights = [wn,w0,wp]
    
    print('Best Weights for multiclass data: ' + str(maxweights))
    print('Performance (Accuracy): ' + str(maxperf))
    return [maxC, maxweights[0], maxweights[1], maxweights[2]]

def main():
    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    (multiclass_features,
    multiclass_labels,
    multiclass_dictionary) = get_multiclass_training_data()
    
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    '''
    #baseline
    nsvc = LinearSVC(penalty = 'l2', loss = 'hinge', C = 0.1, dual = True, multi_class = 'ovr')
    print(str(cv_performance(nsvc, multiclass_features, multiclass_labels, 5, 'accuracy')))
    generate_challenge_labels(nsvc.predict(heldout_features), 'basetest')
    '''

    hpmat = []

    for i in range(100):
        logC = random.uniform(-3, 3)
        logWn = random.uniform(-1.5,1.5)
        logW0 = random.uniform(-1.5,1.5)
        logWp = random.uniform(-1.5,1.5)
        hpmat.append([10**logC, 10**logWn, 10**logW0, 10**logWp])

    settings = choose_hp_multi(multiclass_features, multiclass_labels, hps = hpmat)
    print(settings)
    
    #bsvc = LinearSVC(penalty = 'l2', loss = 'squared_hinge', C = 0.301, class_weight = {-1:0.0851, 0:0.104, 1:0.084}, dual = True, multi_class = 'ovr')
    bsvc = LinearSVC(penalty = 'l2', loss = 'squared_hinge', C = settings[0], class_weight = {-1:settings[1], 0:settings[2], 1:settings[3]}, dual = True, multi_class = 'ovr')
    bsvc.fit(multiclass_features, multiclass_labels)

    cpn = 0
    cp0 = 0
    cpp = 0
    lcount = 0

    for label in multiclass_labels:
        if label == -1:
            cpn += 1
        elif label == 0:
            cp0 += 1
        elif label == 1:
            cpp += 1
        lcount += 1
    
    print("correct proportions:")
    print(cpn/lcount)
    print(cp0/lcount)
    print(cpp/lcount)
    print(lcount)

    results = bsvc.predict(heldout_features)

    pn = 0
    p0 = 0
    pp = 0
    rcount = 0

    for label2 in results:
        if label2 == -1:
            pn += 1
        elif label2 == 0:
            p0 += 1
        elif label2 == 1:
            pp += 1
        rcount += 1

    print("actual proportions:")
    print(pn/rcount)
    print(p0/rcount)
    print(pp/rcount)
    print(rcount)

    generate_challenge_labels(results, 'asjian')
    
    '''
    rbfsvc = SVC(kernel='rbf', C=100, gamma = 'scale')
    initperf = cv_performance(rbfsvc, multiclass_features, multiclass_labels, 5, 'accuracy')
    print('Initial Performance (accuracy): ' + str(initperf))
    '''
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary

    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )
    '''
    k=5
    metrics = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']
    #part 3 below
    
    xtrainshape = X_train.shape
    featuresum = 0
    for row in X_train:
        for val in row:
            featuresum += val
    
    word_freq = {}
    maxfreq = 1
    mostword = 'and'

    for word in dictionary_binary:
        index = dictionary_binary[word]
        for row in X_train:
            if row[index] == 1:
                if word in word_freq:
                    word_freq[word] += 1
                    if word_freq[word] > maxfreq:
                        mostword = word
                        maxfreq = word_freq[word]
                else:
                    word_freq[word] = 1
    
    print('3(a):')
    print('Processed Sentence: ' + str(extract_word('BEST book ever! It\'s great')))
    print('d: ' + str(len(dictionary_binary)))
    print('Average number of nonzero features: ' + str(featuresum/xtrainshape[0]))
    print('Most common word: ' + mostword)
    print()

    # part 3 above, part 4 below

    bestparam = {}

    #4.1b
    for metric in metrics:
        bestparam[metric] = select_param_linear(X_train, Y_train, k, metric, [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3])
    print()
    #4.1c, chose AUROC
    print('4.1 c) Performance scores for all metrics for C optimizing auroc')
    cauroc = bestparam['auroc']
    svcauroc = LinearSVC(penalty = 'l2', loss = 'hinge', C = cauroc, dual = True, random_state = 445)
    svcauroc.fit(X_train, Y_train)
    
    svcscores = {}
    for metric2 in metrics:
        if metric2 == 'auroc':
            svcscores[metric2] = performance(Y_test, svcauroc.decision_function(X_test), metric2)
        else:
            svcscores[metric2] = performance(Y_test, svcauroc.predict(X_test), metric2)
    
    for metric3 in svcscores:
        print(metric3 + ': ' + str(svcscores[metric3]))

    #4.1d
    plot_weight(X_train, Y_train, 'l2', [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3], 'hinge',True)
    print()

    #4.1e
    indicators = find_important_words(X_train, Y_train, dictionary_binary)
    for pair in indicators:
        print('coeff: ' + str(pair[0]) + ' word: ' + str(pair[1]))
    print()

    negativew = [indicators[e][1] for e in range(0,5)]
    negativeco = [indicators[e1][0] for e1 in range(0,5)]
    positivew = [indicators[f][1] for f in range(5,10)]
    positiveco = [indicators[f2][0] for f2 in range(5,10)]

    plt.bar(positivew, positiveco)
    plt.title('5 Most Positive Words')
    plt.ylabel('Component Value in Optimal Theta')
    plt.savefig("postivewordsbargraph.png")
    plt.clf()

    plt.bar(negativew, negativeco)
    plt.title('5 Most Negative Words')
    plt.ylabel('Component Value in Optimal Theta')
    plt.savefig("negativewordsbargraph.png")
    plt.clf()

    #4.2a

    l1C = select_param_linear(X_train, Y_train, k, 'auroc', [10**-3, 10**-2, 10**-1, 10**0], 'squared_hinge', 'l1', dual = False)
    svc2 = LinearSVC(penalty = 'l1', loss = 'squared_hinge', C = l1C, dual = False, random_state = 445)
    svc2.fit(X_train, Y_train)
    score = performance(Y_test, svc2.decision_function(X_test), 'auroc')

    print('L1 Penalty Squared Hinge Loss Score on Test Data: ' + str(score))

    #4.2b
    plot_weight(X_train, Y_train, 'l1', [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3], 'squared_hinge', False)
    
    #4.3a(i) Grid Search
    param_grid = []
    for x in range(-2, 4):
        c = 10**x
        for y in range(-2, 4):
            r = 10**y
            param_grid.append([c,r])

    print()
    gridbest = select_param_quadratic(X_train, Y_train, k, 'auroc', param_grid, 'grid')
    quadsvcg = SVC(kernel='poly', degree=2, C=gridbest[0], coef0=gridbest[1], gamma = 'auto')
    quadsvcg.fit(X_train, Y_train)
    print('Test Performance: ' + str(performance(Y_test, quadsvcg.decision_function(X_test), 'auroc')))
    print()
    
    #4.3a(ii) Random Search
    param_rand = []

    for a in range(25):
        c = np.random.uniform(-2.0, 3.0)
        r = np.random.uniform(-2.0, 3.0)
        param_rand.append([10**c, 10**r])
    
    param_rand.sort(key = lambda x: x[0])
    randbest = select_param_quadratic(X_train, Y_train, k, 'auroc', param_rand, 'random')
    
    quadsvcr = SVC(kernel='poly', degree=2, C=randbest[0], coef0=randbest[1], gamma = 'auto')
    quadsvcr.fit(X_train, Y_train)
    print('Test Performance: ' + str(performance(Y_test, quadsvcr.decision_function(X_test), 'auroc')))
    print()

    #5.1c)
    weightedscores = {}
    weightedsvc = LinearSVC(penalty = 'l2', loss = 'hinge', C = 0.01, class_weight = {-1:1, 1:10}, dual = True)
    weightedsvc.fit(X_train, Y_train)

    for metric4 in metrics:
        if metric4 == 'auroc':
            weightedscores[metric4] = performance(Y_test, weightedsvc.decision_function(X_test), metric4)
        else:
            weightedscores[metric4] = performance(Y_test, weightedsvc.predict(X_test), metric4)
    
    for metric5 in weightedscores:
        print(metric5 + ': ' + str(weightedscores[metric5]))
    print()

    #5.2a)
    equalsvc = LinearSVC(penalty = 'l2', loss = 'hinge', C = 0.01, class_weight = {-1:1, 1:1}, dual = True)
    equalsvc.fit(IMB_features, IMB_labels)

    equalscores = {}
    for metric6 in metrics:
        if metric6 == 'auroc':
            equalscores[metric6] = performance(IMB_test_labels, equalsvc.decision_function(IMB_test_features), metric6)
        else:
            equalscores[metric6] = performance(IMB_test_labels, equalsvc.predict(IMB_test_features), metric6)

    for metric7 in equalscores:
        print(metric7 + ': ' + str(equalscores[metric7]))
    print()
    
    #5.3a)
    pos_weights = [] 
    for tr in range(200):
        wn = random.uniform(0.1, 15)
        wp = random.uniform(0.1, 15)
        pos_weights.append([wn,wp])
    
    bestweights = select_proper_weights(X=IMB_features, y=IMB_labels, weights=pos_weights)
    print()
    
    #5.3b)
    optweightsvc = LinearSVC(penalty = 'l2', loss = 'hinge', C = 0.01, class_weight = {-1:bestweights[0], 1:bestweights[1]}, dual = True)
    optweightsvc.fit(IMB_features, IMB_labels)

    
    optweightscores = {}
    for metric8 in metrics:
        if metric8 == 'auroc':
            optweightscores[metric8] = performance(IMB_test_labels, optweightsvc.decision_function(IMB_test_features), metric8)
        else:
            optweightscores[metric8] = performance(IMB_test_labels, optweightsvc.predict(IMB_test_features), metric8)
    
    for metric9 in optweightscores:
        print(metric9 + ': ' + str(optweightscores[metric9]))
    

    #5.4)
    svcroc = LinearSVC(penalty='l2', loss='hinge', C=0.01, class_weight={-1:1, 1:1}, dual = True)
    svcroc.fit(IMB_features, IMB_labels)

    fpr, tpr, _ = metrics.roc_curve(IMB_test_labels, svcroc.decision_function(IMB_test_features))
    plt.plot(fpr, tpr, label='Wn = 1, Wp = 1')

    svcroc2 = LinearSVC(penalty='l2', loss='hinge', C=0.01, class_weight={-1:12.862, 1:12.403}, dual = True)
    svcroc2.fit(IMB_features, IMB_labels)

    fpr2, tpr2, _ = metrics.roc_curve(IMB_test_labels, svcroc2.decision_function(IMB_test_features))
    plt.plot(fpr2, tpr2, label='Wn = 12.862' + ', Wp = 12.403')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc=4)

    plt.savefig("roccurve.png")
    '''

if __name__ == "__main__":
    main()
