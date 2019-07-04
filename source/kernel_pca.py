import numpy as np
from sklearn.utils.validation import check_array
from sklearn.decomposition import KernelPCA as sklearnKernelPCA
from pca import PCA
class KernelPCA(PCA):
"""Kernel Principal component analysis (KPCA)
    Non-linear dimensionality reduction through the use of kernels (see
    :ref:`metrics`).
    Read more in the :ref:`User Guide <kernel_PCA>`.
    Parameters
    ----------
    n_components : int, default=None
        Number of components. If None, all non-zero components are kept.
    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel. Default="linear".
    gamma : float, default=1/n_features
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels.
    degree : int, default=3
        Degree for poly kernels. Ignored by other kernels.
    coef0 : float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : mapping of string to any, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.
    alpha : int, default=1.0
        Hyperparameter of the ridge regression that learns the
        inverse transform (when fit_inverse_transform=True).
    fit_inverse_transform : bool, default=False
        Learn the inverse transform for non-precomputed kernels.
        (i.e. learn to find the pre-image of a point)
    eigen_solver : string ['auto'|'dense'|'arpack'], default='auto'
        Select eigensolver to use. If n_components is much less than
        the number of training samples, arpack may be more efficient
        than the dense eigensolver.
    tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.
    max_iter : int, default=None
        Maximum number of iterations for arpack.
        If None, optimal value will be chosen by arpack.
    remove_zero_eig : boolean, default=False
        If True, then all components with zero eigenvalues are removed, so
        that the number of components in the output may be < n_components
        (and sometimes even zero due to numerical instability).
        When n_components is None, this parameter is ignored and components
        with zero eigenvalues are removed regardless.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``eigen_solver`` == 'arpack'.

    copy_X : boolean, default=True
        If True, input X is copied and stored by the model in the `X_fit_`
        attribute. If no further changes will be done to X, setting
        `copy_X=False` saves memory by storing a reference.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        """

    def __init__(self, n_components=None, kernel="linear",
                 gamma=None, degree=3, coef0=1, kernel_params=None,
                 alpha=1.0, fit_inverse_transform=False, eigen_solver='auto',
                 tol=0, max_iter=None, remove_zero_eig=False,
                 random_state=None, copy_X=True, n_jobs=None):
        if fit_inverse_transform and kernel == 'precomputed':
            raise ValueError(
                "Cannot fit_inverse_transform with a precomputed kernel.")
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.remove_zero_eig = remove_zero_eig
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.copy_X = copy_X   

    def fit(self, X):
        """Fit detector.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """
        self.X = check_array(X)
        self.pca = KernelPCA(n_components = self.n_components, 
                            kernel = self.kernel,
                            kernel_params = self.kernel_params,
                            gamma = self.gamma,
                            degree = self.degree,
                            coef0 = self.coef0,
                            alpha = self.alpha,
                            fit_inverse_transform = self.fit_inverse_transform,
                            eigen_solver = self.eigen_solver,
                            remove_zero_eig = self.remove_zero_eig,
                            tol = self.tol,
                            max_iter = self.max_iter,
                            random_state = self.random_state,
                            n_jobs = self.n_jobs,
                            copy_X = self.copy_X)
        self.pca.fit(self.X)
        return self  