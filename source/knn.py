import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

class KNN():
    def __init__(self, n_neighbors=5,radius=1.0, algorithm='auto', leaf_size=30,
                metric='minkowski', p=2, metric_params=None, n_jobs=1):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X):
        X = check_array(X)
        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors,
                                radius=self.radius,
                                algorithm=self.algorithm,
                                leaf_size=self.leaf_size,
                                metric=self.metric,
                                p=self.p,
                                metric_params=self.metric_params,
                                n_jobs=self.n_jobs)
        self.neigh.fit(X)

    def decision(self,X):
        pass
    