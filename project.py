"""
Name: Chaim Hoch & Aviv Yaish
Filename: project.py
Usage: python2 project.py
"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.metrics import roc_curve, auc

from collections import Counter
import numpy as np
import pandas as pd
import json
import itertools
import matplotlib.pyplot as plt
import warnings

dataset_filename = 'roam_prescription_based_prediction.jsonl'

# A delimiter for the various text prints
TEXT_DELIMITER = "========================================="


def print_header(header):
    """
    Prints a header with the given text.
    :param header: the header to print.
    """
    print "\n\n" + TEXT_DELIMITER + "\n" + header + "\n" + TEXT_DELIMITER


def load_dataset(drug_mincount=50, specialty_mincount=50):
    """
    Load the data.
    :param drug_mincount: keep only doctors who prescribed enough drugs. 
    :param specialty_mincount: keep only doctors who specialize in popular enough specialties.
    """
    def iter_dataset():
        """
        An iterator for the dataset.
        """
        with open(dataset_filename, 'rt') as f:
            for line in f:
                ex = json.loads(line)
                yield (ex['cms_prescription_counts'],
                       ex['provider_variables'])

    # filter data according to the mincounts
    data = [(phi_dict, y_dict) for phi_dict, y_dict in iter_dataset() if len(phi_dict) >= drug_mincount]
    specialties = Counter([y_dict['specialty'] for _, y_dict in data])
    specialties = set([s for s, c in specialties.items() if c >= specialty_mincount])
    feats, ys = zip(*[(phi, ys) for phi, ys in data if ys['specialty'] in specialties])
    vectorizer = DictVectorizer(sparse=True)
    X = TfidfTransformer().fit_transform(vectorizer.fit_transform(feats))

    # ys fields:
    # brand_name_rx_count
    # gender
    # generic_rx_count
    # region
    # settlement_type
    # specialty
    # years_practicing
    ys = pd.DataFrame(list(ys))
    ys['gender'] = ys['gender'].astype('category')
    ys['region'] = ys['region'].astype('category')
    ys['specialty'] = ys['specialty'].astype('category')
    ys['settlement_type'] = ys['settlement_type'].astype('category')

    ys['years_practicing'] = ys['years_practicing'].astype(int)
    ys['brand_name_rx_count'] = ys['brand_name_rx_count'].astype(int)
    ys['generic_rx_count'] = ys['generic_rx_count'].astype(int)

    return X, ys, vectorizer


def plot_count_graph(series, kind="bar"):
    """
    Visualizes the given data using a count plot.
    """
    plt.figure()
    counts = series.value_counts()

    ax = plt.subplot()
    counts.plot(title=series.name, kind=kind, ax=ax)

    vals = counts.values
    total = vals.sum()
    for i, val in enumerate(vals * 1.01):
        formatted_per = '{:0.01%}'.format(float(val) / total)
        ax.text(val, i, formatted_per)

    plt.tight_layout()
    return ax


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_simple_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_mosaic(series_a, series_b, xlabel, ylabel):
    """
    Prints a mosaic graph.
    """
    fig, _ = mosaic(pd.crosstab(series_a, series_b).unstack(), labelizer=(lambda key: key[1]))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()


def plot_roc(title, y_true, probabilities):
    """
    Prints the ROC plot for the given labels and probabilities.
    """
    plt.figure()
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.title(title + ': Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tight_layout()


def output_results_binary(title, clf, x_train, y_train, x_test, y_test, classes, vectorizer, is_tree):
    """
    Outputs all the results for binary classification.
    :param title: the title to print.
    :param clf: the classifier to use.
    :param x_train: X data for training.
    :param y_train: labels for training.
    :param x_valid: X data for validation.
    :param y_valid: labels for validation.
    :param classes: the label classes.
    :param vectorizer: the vectorizer used on the data.
    :param is_tree: True if clf is a tree-based classifier, False otherwise.
    :return: the cross validation scores, the score on the test data, and the confusion matrix.
    """
    print_header(title)

    scores = cross_val_score(clf, x_train, y_train, cv=9)
    print "Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    print "Cross validation median: %0.2f" % np.median(scores)

    fit = clf.fit(x_train, y_train)
    final_score = fit.score(x_test, y_test)
    print "After training on all data: %0.2f" % final_score

    cm = confusion_matrix(y_test, fit.predict(x_test))
    print "Confusion matrix:\n%r" % cm
    plot_confusion_matrix(cm, classes=classes, title=title + ': confusion matrix')

    dummy = pd.get_dummies(y_test)
    plot_roc(title, dummy[dummy.columns[0]], fit.predict_proba(x_test)[:, 0])

    if is_tree:
        print_important_features(clf, x_train, vectorizer, title)

    return scores, final_score, cm


def output_results_non_binary(title, clf, x_train, y_train, x_valid, y_valid, classes, vectorizer, is_tree):
    """
    Outputs all the results for non binary classification.
    :param title: the title to print.
    :param clf: the classifier to use.
    :param x_train: X data for training.
    :param y_train: labels for training.
    :param x_valid: X data for validation.
    :param y_valid: labels for validation.
    :param classes: the label classes.
    :param vectorizer: the vectorizer used on the data.
    :param is_tree: True if clf is a tree-based classifier, False otherwise.
    :return: mean score for test, median score for test, mean confusion matrix for test, 
    median confusion for test, validation accuracy, validation confusion matrix.
    """
    print_header(title)

    scores = []
    confusion_matrices = []

    kf = KFold(n_splits=9)
    for train_index, test_index in kf.split(x_train):
        # split data
        cross_x_train, cross_x_test = x_train[train_index], x_train[test_index]
        cross_y_train, cross_y_test = y_train[train_index], y_train[test_index]

        # ternary classification
        clf.fit(cross_x_train, cross_y_train)
        prediction = clf.predict(cross_x_test)
        scores.append(np.mean(prediction == cross_y_test))
        confusion_matrices.append(confusion_matrix(cross_y_test, prediction))

    mean_score_test = np.mean(scores)
    median_score_test = np.median(scores)
    mean_confusion_test = np.mean(confusion_matrices, axis=0)
    median_confusion_test = np.median(confusion_matrices, axis=0)
    print "The mean test accuracy of the cross-validation classification is: %r." % mean_score_test
    print "The median test accuracy of the cross-validation classification is: %r." % median_score_test
    print "The mean test confusion matrix of the cross-validation classification is:\n%r." % mean_confusion_test
    print "The median test confusion matrix of the cross-validation classification is:\n%r." % median_confusion_test

    clf.fit(x_train, y_train)
    prediction = clf.predict(x_valid)
    validation_accuracy = np.mean(prediction == y_valid)
    validation_confusion = confusion_matrix(y_valid, prediction)
    print "The validation accuracy of the classification is: %r." % validation_accuracy
    print "The validation confusion matrix of the classification is:\n%r." % validation_confusion

    if classes is None:
        plot_simple_confusion_matrix(validation_confusion, title=title + ': confusion matrix')
    else:
        plot_confusion_matrix(validation_confusion, classes=classes, title=title + ': confusion matrix')

    if is_tree:
        print_important_features(clf, x_train, vectorizer, title)

    return (mean_score_test, median_score_test, mean_confusion_test,
            median_confusion_test, validation_accuracy, validation_confusion)


def data_statistics(ys):
    """
    Shows all sorts of nice statistics about the data.
    :param ys: the labels of the data.
    """
    plot_mosaic(ys['gender'], ys['years_practicing'], 'years practicing', 'gender')

    ys_top_specialties = ys.loc[ys['specialty'].isin(ys['specialty'].value_counts().head(5).index.tolist())]
    ys_top_specialties['specialty'] = ys_top_specialties['specialty'].cat.remove_unused_categories()
    plot_mosaic(ys_top_specialties['gender'], ys_top_specialties['specialty'], 'specialities', 'gender')

    plot_count_graph(ys['gender'])
    plot_count_graph(ys['region'], kind="barh")
    plot_count_graph(ys['specialty'], kind="barh")


def print_important_features(clf, x, vectorizer, title='', n=10):
    """
    Prints the feature importances for the given classifier.
    :param clf: a tree based classifier.
    :param x: the training data the classifier was fitted on.
    :param vectorizer: the vectorizer used to process x.
    :param title: title for the various prints.
    :param n: the number of features to print.
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = vectorizer.get_feature_names()

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(n):
        print("%d. feature %r (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances
    plt.figure()
    ax = plt.axes()

    ax.set_title("Feature importances for " + title)
    ax.bar(range(x.shape[1]), importances[indices], color="r", align="center")
    ax.set_xticks(range(x.shape[1]))
    ax.set_xlim([-1, x.shape[1]])

    plt.tight_layout()


def classifications(X_train, y_train, X_test, y_test, vectorizer):
    """
    Performs all kinds of classifications on the given data.
    """
    classifiers = [
        ("Logistic Regression", LogisticRegression(), False),
        ("Adaboosted Logistic Regression", AdaBoostClassifier(LogisticRegression()), False),
        ("Random Forest", RandomForestClassifier(), True),
        ("Adaboosted Random Forest", AdaBoostClassifier(RandomForestClassifier()), False),
        ("Bagged Random Forest", BaggingClassifier(RandomForestClassifier(),
                                                   max_features=0.5, max_samples=0.5, n_jobs=8), False),
        ("SVM", svm.SVC(probability=True), False),
        ("Multi-Layer Perceptron, alpha=1", MLPClassifier(), False),
    ]

    print_header("Gender classification")
    for title, clf, is_tree in classifiers:
        output_results_binary(title + ' (gender)', clf, X_train, y_train['gender'], X_test, y_test['gender'],
                              ['F', 'M'], vectorizer, is_tree)

    print_header("Region classification")
    for title, clf, is_tree in classifiers:
        output_results_non_binary(title + ' (region)', clf, X_train, y_train['region'], X_test, y_test['region'],
                                  ['Midwest', 'Northeast', 'South', 'West'], vectorizer, is_tree)

    print_header("Specialty classification")
    for title, clf, is_tree in classifiers:
        output_results_non_binary(title + ' (specialty)', clf, X_train, y_train['specialty'],
                                  X_test, y_test['specialty'], None, vectorizer, is_tree)


def print_top_words(model, topics, n_prescriptions, n_specilaties, trans, y):
    """
    Prints the top words (medications) for each topic (specialty).
    :param n_prescriptions: the number of prescriptions to print for each specialty,
    :param n_specilaties: the number of candidate specialties to print for each topic.
    """
    for topic_idx, topic in enumerate(model.components_):
        cur_specilaties = Counter(find_speciality_by_topic(trans, y, topic_idx)).most_common(n_specilaties)
        print "Topic #%r, candidates: %r, prescriptions:" % (topic_idx, cur_specilaties)
        print ", ".join([topics[i]
                         for i in topic.argsort()[:-n_prescriptions - 1:-1]])


def find_speciality_by_topic(trans, y, topic):
    """
    Given a topic and the data, finds the specialty for the topic.
    """
    max_count = []
    for i in range(trans.shape[0]):
        if trans[i].argmax() == topic:
            max_count.append(y[i])
    return max_count


def learn_specialties(X_train, y_train, X_test, y_test, vectorizer):
    """
    Associates prescriptions with specialties.
    """
    print_header("Learning specialties using LDA")

    lda = LatentDirichletAllocation(n_topics=35, n_jobs=8)
    lda_fit = lda.fit(X_train, y_train['specialty'])

    topics = vectorizer.get_feature_names()
    trans_X_train = lda_fit.transform(X_train)
    print_top_words(lda_fit, topics, 5, 3, trans_X_train, y_train['specialty'])


def main():
    """
    Our final project. Enjoy!
    """
    X, ys, vectorizer = load_dataset(drug_mincount=30, specialty_mincount=50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, ys, test_size=0.25, random_state=42)
    y_train, y_test = y_train.reset_index(), y_test.reset_index()

    # all graphs are shown only after ALL the code has finished running
    data_statistics(ys)
    classifications(X_train, y_train, X_test, y_test, vectorizer)
    learn_specialties(X_train, y_train, X_test, y_test, vectorizer)

    plt.show()
    return


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
