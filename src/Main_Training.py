from src.NaiveBayes import *
from src.SVM import *
from src.Random_Forest import *
from src.Vector_Creation import *
from src.Knn import KNN
import json


def classifier_evaluation(clf, X_test, y_test):
    classifier_outputs = clf.predict(X_test)
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(classifier_outputs)):
        if y_test[i] == 1 and classifier_outputs[i] == 1:
            true_positives += 1
        elif y_test[i] == -1 and classifier_outputs[i] == 1:
            false_positives += 1
        elif y_test[i] == -1 and classifier_outputs[i] == -1:
            true_negatives += 1
        elif y_test[i] == 1 and classifier_outputs[i] == -1:
            false_negatives += 1
    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    F1_score = (2 * precision * recall)/(precision + recall)
    print("Accuracy : ", accuracy)
    print("Precision : ", precision)
    print("Recall : ", recall)
    print("F1 score : ", F1_score)
    print("********************")
    return accuracy, precision, recall, F1_score


X, y, idf, all_tokens = train_documents_to_vectors('data/train.csv')
X_test, y_test = other_documents_to_vectors('data/test.csv', idf, all_tokens, True)


#############################
#   SVM Training
#############################

svm_classifiers = []
c = 0.5
for i in range(4):
    svm_classifier = create_SVM_classifier(X[0:int(0.9 * len(X))], y[0:int(0.9 * len(y))], c)
    # evaluating the SVM classifier based on validation set
    print("Evaluating SVM for C = ", c)
    classifier_evaluation(svm_classifier, X[int(0.9 * len(X)):], y[int(0.9 * len(y)):])
    c += 0.5
    svm_classifiers.append(svm_classifier)

# using the results, we use SVM classifier with C = 1
svm_classifier = svm_classifiers[1]
# evaluating the SVM classifier based on test set

print("Evaluating SVM classifier with C = 1 on test set :")
classifier_evaluation(svm_classifier, X_test, y_test)

#############################
#   Random Forest training
#############################
random_forest_classifier = create_Random_Forest_classifier(X,y)
print("Evaluating Random Forest classifier with test set :")
classifier_evaluation(random_forest_classifier,X_test,y_test)


#############################
#   Naive Bayes training
#############################
naive_Bayes_classifier = Naive_Bayes_classifier()
naive_Bayes_classifier.fit(X, y)
print("Evaluating Naive Bayes classifier with test set :")
classifier_evaluation(naive_Bayes_classifier, X_test, y_test)

#############################
#   KNN training
#############################
Ks = [1, 5, 9]
Knn_classifiers = []
for k in Ks:
    knn_classifier = KNN(train_x=X[0:int(0.9 * len(X))], train_y=y[0:int(0.9 * len(y))], k=k)
    print("Evaluating Knn for k =", k, "on validation set:")
    classifier_evaluation(knn_classifier, X_test, y_test)
    Knn_classifiers.append(knn_classifier)

# using the results, we use KNN classifier with K = 9
knn_classifier = Knn_classifiers[2]
# evaluating the KNN classifier based on test set

print("Evaluating Knn classifier with K = 9 on test set :")
classifier_evaluation(knn_classifier, X_test, y_test)


#############################
#   Labeling The Unlabeled data
#############################
chosen_classifier = naive_Bayes_classifier
X_Data,temp = other_documents_to_vectors('data/ted_talks.csv',idf,all_tokens,False)

document_classification = {}#doc id to class
for i,data in enumerate(X_Data):
    document_classification[i+1] = chosen_classifier.predict([data])[0]
with open('../Phase1/document_classifications.txt', 'w',encoding='utf-8') as f:
    f.write(json.dumps(document_classification))



