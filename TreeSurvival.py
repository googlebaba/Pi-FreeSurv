import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


class RFSurvival(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.hyperparams = {
            'n_estimators': 100, 
            'max_depth': 10, 
            'min_samples_split': 10, 
            'min_samples_leaf': 15, 
            'n_jobs': -1, 
            'random_state': 0
        }
        for key, val in kwargs.items():
            self.hyperparams[key] = val
        self.model = RandomSurvivalForest(**self.hyperparams)
        
    
    def fit(self, X, duration_col='Survival.months', event_col='Survival.status'):
        # X_train, X_val = train_test_split(X, test_size=0.2)
        X_train = X.copy()
        X_train = X_train.reset_index(drop=True)
        X_train_pd = X_train.drop([event_col, duration_col], axis = 1)
        y_train = Surv.from_dataframe(event_col, duration_col, X_train) 
        self.model.fit(X_train_pd, y_train)
        return self
    
    def score(self, X, duration_col='Survival.months', event_col='Survival.status'):
        X_test_pd = X.drop([event_col, duration_col], axis = 1)
        y_test = Surv.from_dataframe(event_col, duration_col, X)
        c_index = self.model.score(X_test_pd, y_test)
        return c_index
    def predict(self, X):
        return self.model.predict(X)


