import numpy as np


class KNN:
    def __init__(self, train_x, train_y, k):
        self.train_x = np.array(train_x)
        self.train_y = train_y
        self.K = k

    def predict(self, test_x):
        test_x = np.array(test_x)
        results = []

        for i in range(len(test_x)):
            distances = []  # tuple of index and distance
            for j in range(len(self.train_x)):
                dist = np.linalg.norm(test_x[i] - self.train_x[j])
                distances.append((j, dist))
            distances.sort(key=lambda x: x[1])
            distances = distances[:self.K]

            num_positives = 0
            num_negatives = 0
            for (index, dis) in distances:
                if self.train_y[index] == +1:
                    num_positives += 1
                else:
                    num_negatives += 1

            if num_positives > num_negatives:
                results.append(1)
            else:
                results.append(-1)
        return results
