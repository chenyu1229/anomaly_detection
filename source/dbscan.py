import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array,check_consistent_length
from sklearn.cluster._dbscan_inner import dbscan_inner

class DBSCAN():
    """
    DBSCAN algorithm is a density-based clustering algorithm that has the capability of discovering anomalous data
    Parameters:
    -----------
    eps : float, optional
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. This is not a maximum bound
            on the distances of points within a cluster. This is the most
            important DBSCAN parameter to choose appropriately for your data set
            and distance function.
        min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself.
        metric : string, or callable
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string or callable, it must be one of
            the options allowed by :func:`sklearn.metrics.pairwise_distances` for
            its metric parameter.
            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square. X may be a sparse matrix, in which case only "nonzero"
            elements may be considered neighbors for DBSCAN.
        metric_params : dict, optional
            Additional keyword arguments for the metric function.
            .. versionadded:: 0.19
        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
            The algorithm to be used by the NearestNeighbors module
            to compute pointwise distances and find nearest neighbors.
            See NearestNeighbors module documentation for details.
        leaf_size : int, optional (default = 30)
            Leaf size passed to BallTree or cKDTree. This can affect the speed
            of the construction and query, as well as the memory required
            to store the tree. The optimal value depends
            on the nature of the problem.
        p : float, optional
            The power of the Minkowski metric to be used to calculate distance
            between points.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.
        n_jobs : int or None, optional (default=None)
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
    """
    def __init__(self, eps=0.5, min_samples=5, metric='minkowski', metric_params=None,
            algorithm='auto', leaf_size=30, p=2, sample_weight=None,
            n_jobs=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, sample_weight=None):
        """Fit detector.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """
        X = check_array(X, accept_sparse='csr')
        clust = self.dbscan(X, sample_weight=sample_weight,eps=self.eps, min_samples=self.min_samples, metric=self.metric, metric_params=self.metric_params,
            algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p,n_jobs=self.n_jobs)
        self.core_sample_indices_, self.labels_ = clust
        return self

    def fit_predict(self, X, sample_weight=None):
        """Performs clustering on X and returns cluster labels.
        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.
        Returns
        -------
        y : ndarray, shape (n_samples,)
             Anomaly points are given the label -1.
        """
        self.fit(X, sample_weight=sample_weight)
        return self.labels_


    def dbscan(self, X, eps=0.5, min_samples=5, metric='minkowski', metric_params=None,
           algorithm='auto', leaf_size=30, p=2, sample_weight=None,
           n_jobs=None):
        """Perform DBSCAN clustering from vector array or distance matrix.
        Read more in the :ref:`User Guide <dbscan>`.
        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        eps : float, optional
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. This is not a maximum bound
            on the distances of points within a cluster. This is the most
            important DBSCAN parameter to choose appropriately for your data set
            and distance function.
        min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself.
        metric : string, or callable
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string or callable, it must be one of
            the options allowed by :func:`sklearn.metrics.pairwise_distances` for
            its metric parameter.
            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square. X may be a sparse matrix, in which case only "nonzero"
            elements may be considered neighbors for DBSCAN.
        metric_params : dict, optional
            Additional keyword arguments for the metric function.
            .. versionadded:: 0.19
        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
            The algorithm to be used by the NearestNeighbors module
            to compute pointwise distances and find nearest neighbors.
            See NearestNeighbors module documentation for details.
        leaf_size : int, optional (default = 30)
            Leaf size passed to BallTree or cKDTree. This can affect the speed
            of the construction and query, as well as the memory required
            to store the tree. The optimal value depends
            on the nature of the problem.
        p : float, optional
            The power of the Minkowski metric to be used to calculate distance
            between points.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.
        n_jobs : int or None, optional (default=None)
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        Returns
        -------
        core_samples : array [n_core_samples]
            Indices of core samples.
        labels : array [n_samples]
            Cluster labels for each point. 
        """
        if not eps > 0.0:
            raise ValueError("eps must be positive.")

        X = check_array(X, accept_sparse='csr')
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            check_consistent_length(X, sample_weight)

        # Calculate neighborhood for all samples. This leaves the original point
        # in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if metric == 'precomputed' and sparse.issparse(X):
            neighborhoods = np.empty(X.shape[0], dtype=object)
            X.sum_duplicates()  # XXX: modifies X's internals in-place

            # set the diagonal to explicit values, as a point is its own neighbor
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())  # XXX: modifies X's internals in-place

            X_mask = X.data <= eps
            masked_indices = X.indices.astype(np.intp, copy=False)[X_mask]
            masked_indptr = np.concatenate(([0], np.cumsum(X_mask)))
            masked_indptr = masked_indptr[X.indptr[1:-1]]

            # split into rows
            neighborhoods[:] = np.split(masked_indices, masked_indptr)
        else:
            neighbors_model = NearestNeighbors(radius=eps, algorithm=algorithm,
                                            leaf_size=leaf_size,
                                            metric=metric,
                                            metric_params=metric_params, p=p,
                                            n_jobs=n_jobs)
            neighbors_model.fit(X)
            # This has worst case O(n^2) memory complexity
            neighborhoods = neighbors_model.radius_neighbors(X, eps,
                                                            return_distance=False)

        if sample_weight is None:
            n_neighbors = np.array([len(neighbors)
                                    for neighbors in neighborhoods])
        else:
            n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                    for neighbors in neighborhoods])

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, labels)
        return np.where(core_samples)[0], labels


