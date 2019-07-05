import numpy as np
from sklearn.utils.validation import check_array
from sklearn.decomposition import SparsePCA as sklearnSparsePCA
class SparsePCA():
    """Sparse Principal Components Analysis (SparsePCA)
    Finds the set of sparse components that can optimally reconstruct
    the data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha.
    Read more in the :ref:`User Guide <SparsePCA>`.
    Parameters
    ----------
    n_components : int,
        Number of sparse atoms to extract.
    alpha : float,
        Sparsity controlling parameter. Higher values lead to sparser
        components.
    ridge_alpha : float,
        Amount of ridge shrinkage to apply in order to improve
        conditioning when calling the transform method.
    max_iter : int,
        Maximum number of iterations to perform.
    tol : float,
        Tolerance for the stopping condition.
    method : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.
    n_jobs : int or None, optional (default=None)
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    U_init : array of shape (n_samples, n_components),
        Initial values for the loadings for warm restart scenarios.
    V_init : array of shape (n_components, n_features),
        Initial values for the components for warm restart scenarios.
    verbose : int
        Controls the verbosity; the higher, the more messages. Defaults to 0.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    normalize_components : boolean, optional (default=False)
        - if False, use a version of Sparse PCA without components
          normalization and without data centering. This is likely a bug and
          even though it's the default for backward compatibility,
          this should not be used.
        - if True, use a version of Sparse PCA with components normalization
          and data centering.
           ``normalize_components`` was added and set to ``False`` for
           backward compatibility. It would be set to ``True`` from 0.22
           onwards.
"""
    def __init__(self, n_components=None, alpha=1, ridge_alpha=0.01,
                 max_iter=1000, tol=1e-8, method='lars', n_jobs=None,
                 U_init=None, V_init=None, verbose=False, random_state=None,
                 normalize_components=True):
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_jobs = n_jobs
        self.U_init = U_init
        self.V_init = V_init
        self.verbose = verbose
        self.random_state = random_state
        self.normalize_components = normalize_components

    def fit(self, X):
        """Fit detector.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """
        self.X = check_array(X)
        self.pca = sklearnSparsePCA(n_components = self.n_components,
                                    alpha = self.alpha,
                                    ridge_alpha = self.ridge_alpha,
                                    max_iter = self.max_iter,
                                    tol = self.tol,
                                    method = self.method,
                                    n_jobs = self.n_jobs,
                                    U_init = self.U_init,
                                    V_init = self.V_init,
                                    verbose = self.verbose,
                                    random_state = self.random_state,
                                    normalize_components = self.normalize_components,
                        )
        self.pca.fit(self.X)
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
        # loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
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
        X_test_PCA_inverse = np.array(X_test_PCA).dot(self.pca.components_) + np.array(self.X.mean(axis=0))
        return self.anomalyScores(self.X, X_test_PCA_inverse)

