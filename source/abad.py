import numpy as np
from sklearn.utils.validation import check_array

def _cos(x, a, b):
    v1 = a-x
    v2 = b-x
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

class ABAD():
    def fit(self, X):
        self.X_train = check_array(X)
        return self

    def decision(self,X):
        X_test = check_array(X)
        if X_test.shape[1] != self.X_train.shape[1]:
            raise Exception("Test data format error!")
        self.anomalyScores=[]
        for ix in range(X_test.shape[0]):
            cos = 1
            for ia in range (self.X_train.shape[0]):
                if np.array_equal(X_test[ix],self.X_train[ia]):
                    continue
                for ib in range(ia+1, self.X_train.shape[0]):
                    if np.array_equal(X_test[ix], self.X_train[ib]):
                        continue
                    cos = min(cos,_cos(X_test[ix],self.X_train[ia],self.X_train[ib]))
            self.anomalyScores.append(cos)
        return self.anomalyScores





