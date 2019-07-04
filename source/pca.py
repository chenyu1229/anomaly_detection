import numpy as np
from sklearn.utils.validation import check_array
from sklearn.decomposition import PCA as sklearnPCA
class PCA():
    """Principal component analysis (PCA) anomaly detection

    Parameters
    ----------
    n_components : int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept::
            n_components == min(n_samples, n_features)
        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.
        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.
        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.
        Hence, the None case results in::
        n_components == min(n_samples, n_features) - 1

    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        randomized :
            run randomized SVD by the method of Halko et al.

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.
        """
        
    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X):
        """Fit detector.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """
        self.X = check_array(X)
        self.pca = sklearnPCA(n_components=self.n_components,
                        copy=self.copy,
                        whiten=self.whiten,
                        svd_solver=self.svd_solver,
                        tol=self.tol,
                        iterated_power=self.iterated_power,
                        random_state=self.random_state)
        self.pca.fit(self.X)
        # X_train_PCA = self.pca.transform(self.X)
        # self.X_train_PCA_inverse = self.pca.inverse_transform(X_train_PCA)
        return self

    def anomalyScores(self,X1, X2):
        """Calculate anomaly scores, range from 0 to 1, the larger values, the more chance to be a anomaly
        Parameters
        ----------
        X1 : numpy array of shape (n_samples, n_features)
            The input samples after pca and inverse pca process.
        X2 : numpy array of shape (n_samples, n_features)
            The input samples.
        """
        loss = np.sqrt(np.sum((np.array(X1)-np.array(X2))**2, axis=1))
        loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
        return loss

    def decision(self,X):
        """Predict anomaly score of each element.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ll : array, shape (n_samples,)
            Log-likelihood of each sample under the current model, which is the anomaly score of each element.
        """
        X_test_PCA = self.pca.transform(X)
        X_test_PCA_inverse = self.pca.inverse_transform(X_test_PCA)
        return self.anomalyScores(self.X, X_test_PCA_inverse)