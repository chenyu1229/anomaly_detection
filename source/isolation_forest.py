from sklearn.ensemble import IsolationForest as sklearn_iforest
import numpy as np
class IsolationForest():
        def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination="auto",
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=None,
                 behaviour='deprecated',
                 random_state=None,
                 verbose=0,
                 warm_start=False):
            super().__init__(
                base_estimator=ExtraTreeRegressor(
                    max_features=1,
                    splitter='random',
                    random_state=random_state),
                # here above max_features has no links with self.max_features
                bootstrap=bootstrap,
                bootstrap_features=False,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                warm_start=warm_start,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose)

            self.behaviour = behaviour
            self.contamination = contamination

        def fit(self, X):
            # validate inputs X and y (optional)
            X = check_array(X)
            self.iforest = sklearn_iforest(n_estimators=self.n_estimators,
                                    max_samples=self.max_samples,
                                    contamination=self.contamination,
                                    max_features=self.max_features,
                                    bootstrap=self.bootstrap,
                                    n_jobs=self.n_jobs,
                                    behaviour=self.behaviour,
                                    random_state=self.random_state,
                                    verbose=self.verbose)