# EECS 445 - Winter 2022
# Project 1 - helper.py

import pandas as pd
import numpy as np

import project1


def load_data(fname):
    """
    Reads in a csv file and return a dataframe. A dataframe df is similar to dictionary.
    You can access the label by calling df['label'], the content by df['content']
    the rating by df['rating']
    """
    return pd.read_csv(fname)


def get_split_binary_data(fname="data/dataset.csv", n=None, neutral = False):
    """
    Reads in the data from fname and returns it using
    extract_dictionary and generate_feature_matrix split into training and test sets.
    The binary labels take two values:
        -1: poor/average
         1: good
    Also returns the dictionary used to create the feature matrices.
    Input:
        fname: name of the file to be read from.
    """
    dataframe = load_data(fname)

    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    if n!=None: class_size=n
    else: class_size = 2 * positiveDF.shape[0] // 3
    X_train = (
        pd.concat([positiveDF[:class_size], negativeDF[:class_size]])
        .reset_index(drop=True)
        .copy()
    )
    dictionary = project1.extract_dictionary(X_train)
    X_test = (
        pd.concat([positiveDF[class_size:], negativeDF[class_size:]])
        .reset_index(drop=True)
        .copy()
    )
    Y_train = X_train["label"].values.copy()
    Y_test = X_test["label"].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test, dictionary)


def get_imbalanced_data(dictionary, fname="data/dataset.csv", ratio=0.25):
    """
    Reads in the data from fname and returns imbalanced dataset using
    extract_dictionary and generate_feature_matrix split into training and test sets.
    The binary labels take two values:
        -1: poor/average
         1: good
    Input:
        dictionary: dictionary to create feature matrix from
        fname: name of the file to be read from.
        ratio: ratio of positive to negative samples
    """
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe["label"] != 0]
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    negativeDF = negativeDF[: int(ratio * positiveDF.shape[0])]
    positive_class_size = 2 * positiveDF.shape[0] // 3
    negative_class_size = 2 * negativeDF.shape[0] // 3
    positiveDF = positiveDF.sample(frac=1, random_state=445)
    negativeDF = negativeDF.sample(frac=1, random_state=445)
    X_train = (
        pd.concat([positiveDF[:positive_class_size], negativeDF[:negative_class_size]])
        .reset_index(drop=True)
        .copy()
    )
    X_test = (
        pd.concat([positiveDF[positive_class_size:], negativeDF[negative_class_size:]])
        .reset_index(drop=True)
        .copy()
    )
    Y_train = X_train["label"].values.copy()
    Y_test = X_test["label"].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test)


def get_multiclass_training_data(class_size=750):
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are multiclass as follows
        -1: poor
         0: average
         1: good
    Also returns the dictionary used to create X_train.
    Input:
        class_size: Size of each class (pos/neg/neu) of training dataset.
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    neutralDF = dataframe[dataframe["label"] == 0].copy()
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    X_train = (
        pd.concat(
            [positiveDF[:class_size], negativeDF[:class_size], neutralDF[:class_size]]
        )
        .reset_index(drop=True)
        .copy()
    )
    dictionary = project1.extract_dictionary(X_train)
    Y_train = X_train["label"].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)

    return (X_train, Y_train, dictionary)


def get_heldout_reviews(dictionary):
    """
    Reads in the data from data/heldout.csv and returns it as a feature
    matrix based on the functions extract_dictionary and generate_feature_matrix
    Input:
        dictionary: the dictionary created by get_multiclass_training_data
    """
    fname = "data/heldout.csv"
    dataframe = load_data(fname)
    X = project1.generate_feature_matrix(dataframe, dictionary)
    return X


def generate_challenge_labels(y, uniqname):
    """
    Takes in a numpy array that stores the prediction of your multiclass
    classifier and output the prediction to held_out_result.csv. Please make sure that
    you do not change the order of the ratings in the heldout dataset since we will use
    this file to evaluate your classifier.
    """
    pd.Series(np.array(y)).to_csv(uniqname + ".csv", header=["label"], index=False)
    return
