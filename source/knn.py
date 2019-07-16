import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

class KNN():
    def __init__(self, method = 'mean', n_neighbors=5,radius=1.0, algorithm='auto', leaf_size=30,
                metric='minkowski', p=2, metric_params=None, n_jobs=1):
        self.method = method
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X):
        self.X_train = check_array(X)
        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors,
                                radius=self.radius,
                                algorithm=self.algorithm,
                                leaf_size=self.leaf_size,
                                metric=self.metric,
                                p=self.p,
                                metric_params=self.metric_params,
                                n_jobs=self.n_jobs)
        self.neigh.fit(self.X_train)

    def decision(self,X):
        X_test = check_array(X)
        if X_test.shape[1] != self.X_train.shape[1]:
            raise Exception("Test data format error!")
        dis_arr, ind_arr = self.neigh.kneighbors(X_test,return_distance=True)
        if self.method = 'mean':
            return np.mean(dis_arr,axis=1)
        elif self.method = 'median':
            return np.median(dis_arr,axis=1)
        elif self.method = 'largest':
            return np.max(dis_arr,axis=1)
        elif self.method = 'smallest':
            return np.min(dis_arr[:,1:],axis=1)

    