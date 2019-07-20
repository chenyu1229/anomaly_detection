import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

class KNN():
    """KNN Anomaly Detection, for each point calcuate mean, media or maximum of KNN, and sort them,
    choose the largest values as outliers.
    Parameters
    ----------
    method: {'mean', 'media','largest', 'smallest'}, optional (default = 'mean')
        The calculation method of k nearest point:
        -'mean': calculate mean value of k nearest point
        -'median': calculate median value of k nearest point
        -'largest': the kth neighbor's distance
        -'smallest': the first neighbour's distance
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    metric : string or callable, default 'minkowski'
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.
        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
        Distance matrices are not supported.
        Valid values for metric are:
        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']
        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']
        See the documentation for scipy.spatial.distance for details on these
        metrics.
    p : integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`

    """
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
        """Fit detector.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """
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
        """Predict anomaly score of each element.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ll : array, shape (n_samples,)
            KNN distance of each sample under the current model, which is the anomaly score of each element.
        """
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

    