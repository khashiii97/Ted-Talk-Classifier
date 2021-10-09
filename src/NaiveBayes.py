import math


class Naive_Bayes_classifier:
    def __init__(self):
        self.idf = {}
        self.Pt1 = {}  # P(t|view = +1)
        self.Pt0 = {}  # P(t|view = -1)
        self.P_c1 = 0  # probability of view = +1
        self.P_c0 = 0  # probability of view = -1

    def fit(self, X, y):

        B = len(X[0])  # B = |V|  >>  |V| is number of unique tokens in all docs
        T_ctp1 = 0  # number of token in doc with view +1
        T_ctp0 = 0  # number of token in doc with view -1
        NumToken1 = {}  # dictionary of tokens and numbers in class view = +1
        NumToken0 = {}  # dictionary of tokens and numbers in class view = -1

        Nc1 = len([i for i in  y if i == 1])  # number of docs in class view = +1
        Nc0 = len([i for i in  y if i == -1])  # number of docs in class view = -1
        N = len(X)  # number of all docs
        df = {}
        for i in range(len(X[0])):
            df[i] = 0
        for x in X:
            for i in range(len(x)):
                if x[i] != 0:
                    df[i] += 1
        for i in range(len(X[0])):
            self.idf[i] = math.log(N/df[i])
        for i,training_sample in enumerate(X):
            for index,tf_mult_idf in enumerate(training_sample):
                if y[i] == 1:
                    if tf_mult_idf != 0:
                        tf = int(tf_mult_idf/self.idf[index])
                        T_ctp1 += tf
                        if index not in NumToken1.keys():
                            NumToken1[index] = tf
                        else:
                            NumToken1[index] += tf
                else:
                    if tf_mult_idf != 0:
                        tf = int(tf_mult_idf/self.idf[index])
                        T_ctp0 += tf
                        if index not in NumToken0.keys():
                            NumToken0[index] = tf
                        else:
                            NumToken0[index] += tf
        self.P_c0 = Nc0/N
        self.P_c1 = Nc1 / N
        for t in range(len(X[0])):
            p_tc1 = (NumToken1.get(t, 0) + 1) / (T_ctp1 + B)
            p_tc0 = (NumToken0.get(t, 0) + 1) / (T_ctp0 + B)

            self.Pt1[t] = p_tc1
            self.Pt0[t] = p_tc0

    def predict(self, X_test):
        results = []
        for test_sample in X_test:
            temp1 = 0
            temp0 = 0
            for t,tf_mult_idf in enumerate(test_sample):
                if tf_mult_idf != 0:
                    tf = int(tf_mult_idf / self.idf[t])
                    temp1 += tf * math.log(self.Pt1.get(t, 1))
                    temp0 += tf * math.log(self.Pt0.get(t, 1))
            P_ct1 = temp1 + math.log(self.P_c1)
            P_ct0 = temp0 + math.log(self.P_c0)
            if P_ct1 > P_ct0:
                results.append(1)
            else:
                results.append(-1)
        return results
