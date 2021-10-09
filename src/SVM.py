from sklearn import svm


def create_SVM_classifier(x, y, c):
    clf = svm.SVC(kernel='poly', C=c)
    clf.fit(x, y)
    return clf
# call clf.predict(new_input) to get the output on a new input
