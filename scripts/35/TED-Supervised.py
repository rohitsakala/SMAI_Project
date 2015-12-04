from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
import json
from time import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC,NuSVC
from sklearn.naive_bayes import  MultinomialNB, GaussianNB
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from fastica import fastica
from sklearn.decomposition.truncated_svd import TruncatedSVD
from scipy.io import mmread,mmwrite
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

###############################################################################
# Load some categories from the training set

(opts, args) = op.parse_args()


print("Loading dataset ")
bunch = load_files('../global/')
X_train, X_test, y_train, y_test = train_test_split(bunch.data, bunch.target, test_size=.16)


categories = [bunch.target_names[idx] for idx in y_test]    # Load Categories
target_files = [bunch.filenames[idx] for idx in y_test]

print(target_files[0:5])
with open("question_tags.json","rb") as ques_tags:
    question_tags = json.load(ques_tags)

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(X_train)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')#,max_features=10000)
    X_train = vectorizer.fit_transform(X_train)
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()

#X_test,y_test = get_test_data()

X_test = vectorizer.transform(X_test)
duration = time() - t0
print(X_train.shape)

#x,z, X_train = fastica(X_train.toarray())
svd = TruncatedSVD(n_components=1000)
print(X_train)

#print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


print(X_train.shape)
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)
#u,o,X_train = fastica(X_train.toarray(),n_comp=1000)
print(X_train)
print(X_train.shape)
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


our_accuracies = []
def final_accuracy(predicted):
    count_same = 0
    total = 0
    for i in xrange(0,len(predicted)):
        tags_needed =  question_tags[(target_files[i].split("/")[-1])]
        count_same += len(list(set(predicted[i]) & set(tags_needed)))
	print(predicted[i],"========",tags_needed)
        total+=len(tags_needed)
    return float(count_same)/total



###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    predictions = clf.predict_proba(X_test)
    fin_predict = []
    for i in xrange(0,len(predictions)):
        x = np.argpartition(predictions[i],-5)[-5:]
        x = clf.classes_[x]
        fin_predict.append([bunch.target_names[e] for e in x])
    
    our_accuracies.append(final_accuracy(fin_predict))
    print(our_accuracies[-1])
    # print("------------predictions------------")
    # print(pred)
    # print("-------------------------")
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
print('=' * 80)
for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
#results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
#                                            dual=False, tol=1e-3)))

# Train RBF SVM model
results.append(benchmark(NuSVC(cache_size=1000,probability=True)))
#train sigmoid SVM model
results.append(benchmark(NuSVC(kernel='sigmoid',cache_size=1000,probability = True)))
results.append(benchmark(NuSVC(kernel='linear',cache_size=1000,probability = True)))
# Train sparse Naive Bayes classifiers
print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
#results.append(benchmark(Pipeline([
#  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
#  ('classification', LinearSVC())
#])))

#gaussian naiave bayes training
results.append(benchmark(GaussianNB()))
#results.append(benchmark(MultinomialNB()))
# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
