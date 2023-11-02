# -*- coding: utf-8 -*-
"""BoW + Synonym Augmentation.ipynb

Automatically generated by Colaboratory.

"""
import os

###
### Import libraries
###

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
import joblib
###
### Load dataset
###
# dataset = "Automatability-Data-Multiclass-noExp"
#
# tech_impact_columns_mapping = {
#     'CONFIDENTIALITY': {'NONE': 0, 'LOW': 1, 'HIGH': 2},
#     'INTEGRITY': {'NONE': 0, 'LOW': 1, 'HIGH': 2},
#     'TECH IMPACT': {'partial': 0, 'total': 1},
# }
#
# automatability_columns_mapping = {
#     'Automatable? (Final Decision)': {'NO': 0, 'YES': 1},
# }
#
# dataset_configurations = {
#     "Rawdata364partial364total": {
#         "file_name": "Rawdata364partial364total.csv",
#         "binary_classification": 'binary',
#         "multi_classification": 'weighted',
#         "downsample": False,
#         "columns_mapping": tech_impact_columns_mapping
#     },
#     "10810-data-CI": {
#         "file_name": "10810-data-CI.csv",
#         "binary_classification": 'none',
#         "multi_classification": 'weighted',
#         "downsample": False,
#         "columns_mapping": tech_impact_columns_mapping
#     },
#     "Automatability-Data-Multiclass-noExp": {
#         "file_name": "Automatability-Data-Multiclass-noExp.csv",
#         "binary_classification": 'binary',
#         "multi_classification": 'none',
#         "downsample": True,
#         "columns_mapping": automatability_columns_mapping
#     },
#     "Old-and-new-tech-impact-trimmed": {
#         "file_name": "Old-and-new-tech-impact-trimmed.csv",
#         "binary_classification": 'binary',
#         "multi_classification": 'weighted',
#         "downsample": True,
#         "columns_mapping": tech_impact_columns_mapping
#     }
# }
#
# # Read the dataset
# if dataset in dataset_configurations:
#     print(dataset)
#     config = dataset_configurations[dataset]
#     df = pd.read_csv(config["file_name"])
#     binary_classification = config["binary_classification"]
#     multi_classification = config["multi_classification"]
#     for column, mapping in config["columns_mapping"].items():
#         df[column].replace(mapping, inplace=True)
# else:
#     print("No Dataset")
#
# # Downsampling
# # Note: Can only run this block once b/c it reference and sets df
# if config["downsample"]:
#     if dataset == "Old-and-new-tech-impact-trimmed":
#         zero_class = df[df["TECH IMPACT"] == 0]
#         one_class = df[df["TECH IMPACT"] == 1]
#     if dataset == "Automatability-Data-Multiclass-noExp":
#         zero_class = df[df["Automatable? (Final Decision)"] == 0]
#         one_class = df[df["Automatable? (Final Decision)"] == 1]
#     print("Sum of Zero: {}".format(len(zero_class)))
#     print("Sum of One: {}".format(len(one_class)))
#
#     # Total Class length is 13976 if you want to keep the entire thing for old-and-new dataset
#     # Total Class length is 167 if you want to keep the entire thing for automatability
#     num_samples_to_keep = 167  # len(total_class)
#
#     print("Number of Samples to Keep: {}".format(num_samples_to_keep))
#
#     # Randomly sample the class to create the balanced dataset
#     zero_downsampled = zero_class.sample(n=num_samples_to_keep, random_state=42)
#     one_downsampled = one_class.sample(n=num_samples_to_keep, random_state=42)
#
#     # Combine the minority class and the downsampled majority class
#     balanced_df = pd.concat([zero_downsampled, one_downsampled])
#
#     # Shuffle the balanced DataFrame to randomize the order of samples
#     df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

###
### Word snyonym data augmentation
###
import nltk

nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import wordnet


# Identify relevant synonyms
def get_synonyms(word):
    synonyms = []

    # retrieves a list of synsets(groups of synonymous words) assocciated with the word
    for syn in wordnet.synsets(word):

        # for each synset, it will iterate over its individual synonym
        # lemma refers to the base or canonical form of a word (the lemma of cats is cat)
        for lemma in syn.lemmas():
            # append to list
            synonyms.append(lemma.name())

    # filter and return unique synonyms
    unique_synonyms = [x for i, x in enumerate(synonyms) if x not in synonyms[:i]]
    return unique_synonyms


# Perform word synonym substitution
def augment_with_synonyms(document):
    # tokenize a given text into individual words or tokens by splitting the text into words
    tokens = nltk.word_tokenize(document)
    augmented_doc = []
    for token in tokens:
        synonyms = get_synonyms(token)
        if synonyms:
            augmented_doc.append(synonyms[0])
        else:
            augmented_doc.append(token)
    return ' '.join(augmented_doc)


def synonym_augmentation(X):
    augmented_set = []
    for document in X:
        augmented_doc = augment_with_synonyms(document)
        augmented_set.append(augmented_doc)
    expanded_text = pd.concat([X, pd.Series(augmented_set)])
    return expanded_text


# @title
###
### Calculate Function
###
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import CountVectorizer

stopwords = ['to', 'the', 'in', 'of', 'and', 'an', 'this', 'is', 'that', 'or', 'with', 'are', 'can', 'for', 'as',
             'could', 'on', 'allow', 'all', 'by', 'which', 'from', 'be', 'was', 'when', 'it', 'has']


def perform(X_description, y, model, classification, augment_synonyms=True, X_integrity=None, X_confidentiality=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    average_precision = []
    average_recall = []
    average_F1_score = []
    average_auroc = []

    for train_idx, test_idx in kf.split(X_description):
        # Split data into training and validation sets for the current fold
        X_train, X_test = X_description[train_idx], X_description[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # augmentation here, double dataset to match number of rows
        if augment_synonyms:
            X_train = synonym_augmentation(X_train)
            y_train = pd.concat([y_train, y_train])

        # bag of words here
        v = CountVectorizer(lowercase=True, stop_words=stopwords)
        X_train = v.fit_transform(X_train)
        X_test = v.transform(X_test)

        # if model 3 inputs: combining integrity and confidentiality together w/ description as inputs
        if X_integrity is not None and X_confidentiality is not None:
            X_integrity_train, X_integrity_test = X_integrity[train_idx], X_integrity[test_idx]
            X_confidentiality_train, X_confidentiality_test = X_confidentiality[train_idx], X_confidentiality[test_idx]

            # if augmentation, double training dataset to match number of rows
            if augment_synonyms:
                X_integrity_train = pd.concat([X_integrity_train, X_integrity_train])
                X_confidentiality_train = pd.concat([X_confidentiality_train, X_confidentiality_train])

            # Reshape integrity and confidentiality into the a 2-dimensional column vector to align with X_text
            X_integrity_train = np.array(X_integrity_train).reshape(-1, 1)
            X_integrity_test = np.array(X_integrity_test).reshape(-1, 1)

            X_confidentiality_train = np.array(X_confidentiality_train).reshape(-1, 1)
            X_confidentiality_test = np.array(X_confidentiality_test).reshape(-1, 1)

            # combine the three inputs
            X_train = np.concatenate((X_train.toarray(), X_integrity_train, X_confidentiality_train), axis=1)
            X_test = np.concatenate((X_test.toarray(), X_integrity_test, X_confidentiality_test), axis=1)

        # fit and transform model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # saving model weight for predictions
        joblib.dump(model, f"auto_forest_bow_synaug.joblib")
        joblib.dump(v, "auto_forest_bow_vectorizer.pkl")
        # calculate metrics
        precision = precision_score(y_test, y_pred, average=classification)
        recall = recall_score(y_test, y_pred, average=classification)
        f1 = f1_score(y_test, y_pred, average=classification)

        average_precision.append(precision)
        average_recall.append(recall)
        average_F1_score.append(f1)

        if classification == 'binary':
            auroc = roc_auc_score(y_test, y_pred)
            average_auroc.append(auroc)

    print("Mean - F1 Score: {:.4f} +/- {:.4f}".format(np.mean(average_F1_score), np.std(average_F1_score)))
    print("Mean - Precision: {:.4f} +/- {:.4f}".format(np.mean(average_precision), np.std(average_precision)))
    print("Mean - Recall: {:.4f} +/- {:.4f}".format(np.mean(average_recall), np.std(average_recall)))
    print("Mean - Auroc: {:.4f} +/- {:.4f}".format(np.mean(average_auroc), np.std(average_auroc)))


def predict_description(model_name, cve_description, working_dir=os.path.dirname(__file__)):
    """
    The predict data function will load an already trained Logistic Regression model with synonym augmentatoin and Bag of Words
    implementation, and process and predict a cve_descrpition that is passed into the function.
    :param cve_description: description of cve being determined
    :return: prediction string
    """
    model = joblib.load(os.path.join(working_dir, f"models/{model_name}.joblib"))
    vectorizer = joblib.load(os.path.join(working_dir, f"models/{model_name}_vectorizer.pkl"))
    # Augment the text with synonyms
    df = pd.Series([cve_description])

    # augmented_test_text = synonym_augmentation(df)
    # Transform the test text using the saved vectorizer
    X_test = vectorizer.transform(df)

    # Make predictions using the pre-trained model
    predictions = model.predict(X_test)

    data = []
    for i, cve in enumerate(predictions):
        decision = "total" if predictions[i] == 1 else "partial"
        data.append(decision)  # add to list of predictions to add to df.

    return data[0]

# models = [MultinomialNB(), KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), SVC(probability=True), GradientBoostingClassifier(), MLPClassifier()]
models = [RandomForestClassifier()]

#
# Description --> Automatable
#
# warnings.filterwarnings("ignore")
# #
# X_description = df['CVE Description']
# y = df['Automatable? (Final Decision)']
#
# for model in models:
#   print(model)
#   perform(X_description=X_description, y=y, model=model, classification=binary_classification, augment_synonyms=False)
#   print("========================================================")
#
# Reset warnings filter
# warnings.resetwarnings()

#
# Description + Confidentiality + Integrity -> Impact
#
# warnings.filterwarnings("ignore")
# print("COMPARE: IMPACT vs. Text + INTEGRITY + CONFIDENTIALITY")
#
# X_description = df['DESCRIPTION']
# X_integrity = df['INTEGRITY']
# X_confidentiality = df['CONFIDENTIALITY']
# y = df['TECH IMPACT']
#
# for model in models:
#   print(model)
#   perform(X_description=X_description, y=y, model=model, X_integrity=X_integrity, X_confidentiality=X_confidentiality, classification=binary_classification)
#   print("========================================================")
#
# # Reset warnings filter
# warnings.resetwarnings()
#
# #
# # Description --> Impact
# #
#
# warnings.filterwarnings("ignore")
#
# UNCOMMENT HERE TO PRINT STATEMENT FOR RUNNING MODELS
# print("COMPARE: IMPACT vs. Text")
#
# X_description = df['DESCRIPTION']
# y = df['TECH IMPACT']
#
# for model in models:
#     print(model)
#     perform(X_description=X_description, y=y, model=model, classification=binary_classification)
#     print("========================================================")


# Reset warnings filter
# warnings.resetwarnings()

#
# #
# # Description --> Confidentiality
# #
# warnings.filterwarnings("ignore")
# print("COMPARE: CONFIDENTIALITY vs. Text")
#
# X_description = df['DESCRIPTION']
# y = df['CONFIDENTIALITY']
# #
# for model in models:
#   print(model)
#   perform(X_description=X_description, y=y, model=model, classification=multi_classification)
#   print("========================================================")
#
# # Reset warnings filter
# warnings.resetwarnings()
#
# #
# # Description --> Integrity
# #
# warnings.filterwarnings("ignore")
# print("COMPARE: INTEGRITY vs. Text")
#
# X_description = df['DESCRIPTION']
# y = df['INTEGRITY']
#
# for model in models:
#   print(model)
#   perform(X_description=X_description, y=y, model=model, classification=multi_classification)
#   print("========================================================")
#
# # Reset warnings filter
# warnings.resetwarnings()
