import numpy as np
from scipy.sparse import issparse, vstack
from scipy.stats import lognorm

from sklearn.naive_bayes import _BaseDiscreteNB
from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted



class SS_NB_logGMM(_BaseDiscreteNB):
    """The SS_EM part is inspired by: https://github.com/jmatayoshi/semi-supervised-naive-bayes/blob/master/semi_supervised_naive_bayes.py
    """
    # alpha=.01, beta=.25, eps=1e-5, initialization=0, max_iter=100
    def __init__(self,
                 alpha=1.,
                 beta=.05,
                 gamma=1.,
                 eps=0.,
                 initialization=0,
                 class_prior=None,
                 tol=1e-3,
                 max_iter=100,
                 unsupervised_perc=1.,
                 force_alpha=False,
                 X_u=None,
                 threshold=0.,
                 verbose=False):
        """alpha: float [0, 1] - laplace smoothing count
           beta: float [0, 1] - weight unlabeled likelihood
           gamma: float [0, 1] - weight of log likelihood of class dependent activation
           eps: float - range extreme of the initialization of probabiliti score in unsupervised data
           initialization: int {0, 1} - uses only supervised data to initialize variables
           unsupervised_perc: floar {0, 1} - percentage of unsupervised data used in train
           X_u: array (samples x feat) - matrix with unlabeled data
           threshold: float [0, 1], minimum proba score associated with unlabel sample used in EM 
        """
        self.X_u = None

        # Hyper-params
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.initialization = initialization
        self.unsupervised_perc = unsupervised_perc
        self.threshold = threshold
        
        # Prior
        self.class_prior = class_prior

        # Learnable params
        self.class_count_ = None # raw counts per class: n_classes x
        self.class_log_prior_ = None # fitted log prior
        self.feature_log_prob_ = None # fitted feature prob
        self.means_ = None # fitted mean of lognormal
        self.sdeviations_ = None # fitted std of lognormal
        
        # Setting
        self.fit_prior = True
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

        if X_u is not None:
            self.set_unsupervised_data(X_u)


    def set_unsupervised_data(self, X_u = None):
        """Setting unsupervised matrix attribute
        """
        if X_u is not None and not issparse(X_u):
            raise ValueError("X_u should be a scipy sparse matrix")
        self.X_u = X_u[: int(self.unsupervised_perc * X_u.shape[0]), :]

    
    def _count(self, X, Y):
        """Count and return the number of occurrences of each feature per class.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        Y : array-like of shape (n_samples, n_classes)
            Label matrix with a binary representation of the classes.

        """
        # You would typically compute feature counts here
        # feature_count_ should be of shape (n_classes, n_features)
        self.feature_count_ = safe_sparse_dot(Y.T, X) 
        return self
    
    
    def _estimate_class_log_prior_(self, class_count_, class_prior):
        """Mirror of _update_class_log_prior: 
        https://github.com/scikit-learn/scikit-learn/blob/4af30870b0a09bf0a04d704bea4c5d861eae7c83/sklearn/naive_bayes.py#L109:~:text=def-,_update_class_log_prior,-(self%2C
        
            Return class log priors.
            The class log priors are based on `class_prior`, class count or the number of classes. 
        """
        n_classes = len(self.classes_)
        if class_prior is not None:
            class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            log_class_count = np.log(class_count_)
            class_log_prior_ = log_class_count - np.log(class_count_.sum())
        else:
            class_log_prior_ = np.full(n_classes, -np.log(n_classes))
        
        return class_log_prior_
    
    def _update_feature_log_prob(self, Y, X, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        feature_count_ = safe_sparse_dot(Y.T, X)
        smoothed_fc = feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        return (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)))
    
    def _estimate_means_(self, Y, X):
        """Compute mean estimator for each class in the log space of activity"""
        log_activity = np.log1p(np.atleast_2d(np.asarray(X.sum(axis=1))).T)
        return safe_sparse_dot(Y.T, log_activity.T) / self.class_count_[:, np.newaxis]
    
    
    
    def _estimate_sdeviations_(self, Y, X):
        """Compute stdv estimator for each class in the log space of activity"""
        log_activity = np.log1p(np.atleast_2d(np.asarray(X.sum(axis=1))).T)

        avg_X2 = safe_sparse_dot(Y.T,  np.power(log_activity, 2).T) / self.class_count_[:, np.newaxis]
        avg_means2 = self.means_ ** 2
        avg_X_means = self.means_ * safe_sparse_dot(Y.T, log_activity.T) / self.class_count_[:, np.newaxis]
        
        return avg_X2 - 2 * avg_X_means + avg_means2
    
    
    def _get_activity_log_prob(self, x):
        """Calculate the log prob of activity for each class"""
        classes_proba = []
        n_classes = len(self.classes_)
        for k in range(n_classes):
            params = (self.means_[k], self.sdeviations_[k], np.exp(self.means_[k]))
            class_proba = lognorm.cdf(x + 1, *params) - lognorm.cdf(x, *params)
            classes_proba.append(class_proba)
        return np.log1p(np.vstack(classes_proba)).T
    

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")
        X = check_array(X, accept_sparse='csr')
        log_activity = np.log1p(np.atleast_2d(np.asarray(X.sum(axis=1))).T)
        
        return (safe_sparse_dot(X, self.feature_log_prob_.T) +
                self.class_log_prior_ + 
                self.gamma * self._get_activity_log_prob(log_activity))
    
    def binarize_labeled_data(self, y_labeled):
        """Return binarized version of Y: n_samples x n_classes
        """
        labelbin = LabelBinarizer()
        Y_l = labelbin.fit_transform(y_labeled) # returns dtype=np.int64.
        if Y_l.shape[1] == 1:
            Y_l = np.concatenate((1 - Y_l, Y_l), axis = 1)
        Y_l = Y_l.astype(np.float64, copy=False) # to be consistent with dtypes
        return Y_l, labelbin.classes_
        
    def initialize_unlabeled_data(self, n_unlabeled, n_classes, eps = 0.):
        """Inizializing unlabeled probabilities almost uniformly by eps
        """
        if n_unlabeled > 0:
            initial_proba = (np.random.uniform(- eps, + eps, size = (n_unlabeled, n_classes)) + 
                            np.ones(shape = (n_unlabeled, n_classes)))
            Y_u = normalize(initial_proba, axis=1, norm="l1")
            return Y_u
        return None

    def fit(self, X, y):
        assert self.X_u is not None, "X_u is None, please provide unsupervised data using method set_unsupervised_data"
        # Hyperparameters
        alpha = self._check_alpha()
        
        X = vstack([X, self.X_u])
        y_u = np.repeat(-1, self.X_u.shape[0])
        y = np.concatenate([y, y_u])

        # Dataset
        X, y = check_X_y(X, y, 'csr')
        unlabeled = np.flatnonzero(y == -1)
        labeled = np.setdiff1d(np.arange(len(y)), unlabeled)
        
        Y_labeled, self.classes_ = self.binarize_labeled_data(y[labeled])
        Y_unlabeled = self.initialize_unlabeled_data(len(unlabeled), len(self.classes_), self.eps)
        Y = np.zeros((X.shape[0], len(self.classes_)))
        Y[labeled, :] = Y_labeled
        Y[unlabeled, :] = Y_unlabeled
        

        # M-step (initialization==0 -> only supervised data are used in first M step)
        first_indeces = labeled if self.initialization == 0 else np.concatenate([labeled, unlabeled])
        self.class_count_ = Y[first_indeces, :].sum(axis=0)
        self.class_log_prior_ = self._estimate_class_log_prior_(self.class_count_, self.class_prior)
        self.feature_log_prob_ = self._update_feature_log_prob(Y[first_indeces, :], X[first_indeces, :], alpha) 
        self.means_ = self._estimate_means_(Y[first_indeces, :], X[first_indeces, :])
        self.sdeviations_ = self._estimate_sdeviations_(Y[first_indeces, :], X[first_indeces, :])
        
        jll = self._joint_log_likelihood(X)
        sum_jll = jll.sum()
        if self.verbose:
            print(f"jll before EM: {sum_jll}")

        # EM algorithm
        if len(unlabeled) > 0:
            num_iter = 0
            while num_iter < self.max_iter:
                num_iter += 1
                prev_sum_jll = sum_jll
                
                # E-step with threshold:
                proba_unlabeled = self.predict_proba(X[unlabeled])
                high_confidence_select = proba_unlabeled.max(axis=1) > self.threshold
                unlabeled_with_high_conf = unlabeled[high_confidence_select]
                Y[unlabeled_with_high_conf, :] = (1 - self.beta) * Y[unlabeled_with_high_conf, :] + self.beta * proba_unlabeled[high_confidence_select]

                # M-step:
                m_step_indices = np.concatenate([labeled, unlabeled_with_high_conf])
                self.class_count_ = Y[m_step_indices, :].sum(axis=0)
                self.class_log_prior_ = self._estimate_class_log_prior_(self.class_count_, self.class_prior)
                self.feature_log_prob_ = self._update_feature_log_prob(Y[m_step_indices, :], X[m_step_indices, :], alpha)
                self.means_ = self._estimate_means_(Y[m_step_indices, :], X[m_step_indices, :])
                self.sdeviations_ = self._estimate_sdeviations_(Y[m_step_indices, :], X[m_step_indices, :])
                
                jll = self._joint_log_likelihood(X)
                sum_jll = jll.sum()
                
                # Prints
                if self.verbose:
                    print('Step {}: jll = {:f}'.format(num_iter, sum_jll))
                if num_iter > 1 and prev_sum_jll - sum_jll < self.tol:
                    break
            if self.verbose:
                end_text = 's.' if num_iter > 1 else '.'
                print(f'Optimization converged after {num_iter} iteration' + end_text)

        return self
    

    # Add sklearn set_params method
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    # Add sklearn get_params method
    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'eps': self.eps,
            'initialization': self.initialization,
            'class_prior': self.class_prior,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'unsupervised_perc': self.unsupervised_perc,
            'force_alpha': self.force_alpha,
            'X_u': self.X_u,
            'verbose': self.verbose
        }
