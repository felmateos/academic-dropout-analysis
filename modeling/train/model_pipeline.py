import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold
import joblib

seed = 777
rng = np.random.default_rng(seed)

def rng_int():
    return rng.integers(1, 10000)


def get_best_params(pipeline: Pipeline, param_dist: dict, kfold: KFold, X_train: pd.DataFrame, y_train: pd.Series):
    """
    
    """

    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, cv=kfold, scoring='recall', error_score='raise', random_state=rng_int())
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    return best_model, best_params, best_score


def dump_model(best_clf, model_path, best_score):
    """

    """

    joblib.dump(best_clf, f'{model_path}_r{int(100*best_score)}.pkl')