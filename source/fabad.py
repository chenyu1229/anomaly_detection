import numpy as np
from sklearn.utils.validation import check_array
from sklearn.neighbors import NearestNeighbors

def _cos(x, a, b):
    v1 = a-x
    v2 = b-x
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

class FastABAD():
    def __init__(self,n_neighbors=5):
        self.n_neighbors = n_neighbors
        

    def fit(self, X):
        self.X_train = check_array(X)
        self.X_train = np.unique(self.X_train, axis=0)
        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.neigh.fit(self.X_train)

        return self

    def decision(self,X):
        X_test = check_array(X)
        if X_test.shape[1] != self.X_train.shape[1]:
            raise Exception("Test data format error!")

        ind_arr = self.neigh.kneighbors(X_test,return_distance=False)
        print(ind_arr)
        self.anomalyScores=[]
        for ix in range(X_test.shape[0]):
            cos = 1
            for ia in range (len(ind_arr[ix])):
                for ib in range(ia+1, len(ind_arr[ix])):
                    cos = min(cos,_cos(X_test[ix],self.X_train[ind_arr[ix][ia]],self.X_train[ind_arr[ix][ib]]))
            self.anomalyScores.append(cos)
        return self.anomalyScores