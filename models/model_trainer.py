"""
Module untuk training dan evaluasi model machine learning (efisien).
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from config import CV_N_SPLITS, CV_N_REPEATS, RANDOM_STATE, MODEL_PARAMS


class ModelTrainer:
    """Class untuk training dan evaluasi model."""

    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.cv = RepeatedStratifiedKFold(
            n_splits=CV_N_SPLITS,
            n_repeats=CV_N_REPEATS,
            random_state=random_state
        )
        self.best_model = None
        self.best_params = None

    def train_svm(self, X_train, y_train):
        param_grid = {'C': MODEL_PARAMS['SVM']['C_values']}

        svm = SVC(
            kernel=MODEL_PARAMS['SVM']['kernel'],
            probability=MODEL_PARAMS['SVM']['probability'],
            random_state=self.random_state
        )

        grid_search = GridSearchCV(
            svm, param_grid, cv=self.cv, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        return {
            'model': grid_search.best_estimator_,
            'params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }

    def train_knn(self, X_train, y_train):
        param_grid = {'n_neighbors': MODEL_PARAMS['KNN']['n_neighbors']}
        knn = KNeighborsClassifier()

        grid_search = GridSearchCV(
            knn, param_grid, cv=self.cv, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        return {
            'model': grid_search.best_estimator_,
            'params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }

    def train_random_forest(self, X_train, y_train, n_iter_search=10):
        """
        Training Random Forest dengan RandomizedSearchCV untuk efisiensi.
        """
        param_dist = {
            'n_estimators': MODEL_PARAMS['Random Forest']['n_estimators'],
            'max_depth': MODEL_PARAMS['Random Forest']['max_depth']
        }

        rf = RandomForestClassifier(random_state=self.random_state)

        search = RandomizedSearchCV(
            rf,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state
        )
        search.fit(X_train, y_train)

        self.best_model = search.best_estimator_
        self.best_params = search.best_params_

        return {
            'model': search.best_estimator_,
            'params': search.best_params_,
            'cv_results': search.cv_results_
        }

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        try:
            y_score = model.predict_proba(X_test)
        except:
            y_score = None

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        roc_auc = None
        if y_score is not None:
            try:
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                roc_auc = roc_auc_score(y_test_bin, y_score, average='macro')
            except:
                roc_auc = None

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_auc': roc_auc,
            'predictions': y_pred
        }

    def get_feature_importance(self, model, feature_names):
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            return importance_df
        elif hasattr(model, 'coef_'):
            coef_abs = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': coef_abs
            }).sort_values('Importance', ascending=False)
            return importance_df
        return None
