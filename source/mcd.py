from sklearn.utils.validation import check_array
from sklearn.covariance import MinCovDet
class MCD():
    def __init__():
        """
        Minimum Covariance Determinant (MCD) based anomaly detection is based on that Mahalanobis-type distances 
        in which the shape matrix is derived from a consistent high breakdown robust multivariate location and 
        scale estimator can be used to find anomaly points.
        The Minimum Covariance Determinant covariance estimator is to be applied on Gaussian-distributed data, 
        but could still be relevant on data drawn from a unimodal, symmetric distribution. 

        Parameters
        ----------
        store_precision : bool
            Specify if the estimated precision is stored.
        assume_centered : bool
            If True, the support of the robust location and the covariance
            estimates is computed, and a covariance estimate is recomputed from
            it, without centering the data.
            Useful to work with data whose mean is significantly equal to
            zero but is not exactly zero.
            If False, the robust location and covariance are directly computed
            with the FastMCD algorithm without additional treatment.
        support_fraction : float, 0 < support_fraction < 1
            The proportion of points to be included in the support of the raw
            MCD estimate. Default is None, which implies that the minimum
            value of support_fraction will be used within the algorithm:
            [n_sample + n_features + 1] / 2
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        """
        def __init__(self, store_precision=True, assume_centered=False,
                 support_fraction=None, random_state=None):
            self.store_precision = store_precision
            self.assume_centered = assume_centered
            self.support_fraction = support_fraction
            self.random_state = random_state

        def fit(self, X):
            """Fit detector.
            Parameters
            ----------
            X : numpy array of shape (n_samples, n_features)
                The input samples.
            """
            self.X_train = check_array(X)
            self.mcd = MinCovDet(store_precision=self.store_precision,
                            assume_centered=self.assume_centered,
                            support_fraction=self.support_fraction,
                            random_state=self.random_state)
            self.mcd.fit(X=X, y=y)
            pass

        def decision(X):
            pass
