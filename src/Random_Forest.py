from sklearn.ensemble import RandomForestClassifier


def create_Random_Forest_classifier(x, y):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x, y)
    return clf

# call clf.predict(new_input) to get the output on a new input
