import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, ConfusionMatrixDisplay
)
from time import time
import optuna
from ax.service.managed_loop import optimize

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class prediction:

    def __init__(self, path, target, label0, label1, features=[], cv_folds=5):
        self.path = path
        self.target = target
        self.label0 = label0
        self.label1 = label1
        self.features = features
        self.cv_folds = cv_folds
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def confusionmat(self, y_test, y_pred, label0, label1):
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=[label0, label1],
            cmap=plt.cm.Blues
        )
        plt.title("Confusion Matrix")

    def datasetsetup(self, path, target, features=[]):
        df = pd.read_csv(path)

        if features == []:
            X = df.drop(target, axis=1)
        else:
            X = df.drop(target, axis=1)[features]

        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def cross_validate_model(self, model, X_train, y_train):
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        scoring = {
            'roc_auc':     'roc_auc',
            'f1':          'f1',
            'sensitivity': 'recall',
            'accuracy':    'accuracy',
        }

        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        summary = {}
        for metric, key in [
            ('roc_auc',     'test_roc_auc'),
            ('f1',          'test_f1'),
            ('sensitivity', 'test_sensitivity'),
            ('accuracy',    'test_accuracy'),
        ]:
            scores = cv_results[key]
            summary[metric] = {
                'mean':   float(np.mean(scores)),
                'std':    float(np.std(scores)),
                'scores': scores.tolist(),
            }

        print(
            f"  CV AUC-ROC:     {summary['roc_auc']['mean']:.4f} ± {summary['roc_auc']['std']:.4f}\n"
            f"  CV F1:          {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}\n"
            f"  CV Sensitivity: {summary['sensitivity']['mean']:.4f} ± {summary['sensitivity']['std']:.4f}\n"
            f"  CV Accuracy:    {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}"
        )

        return summary

    # ------------------------------------------------------------------ #
    #  XGBoost                                                             #
    # ------------------------------------------------------------------ #

    def xgboost(self, label0, label1, X_train, X_test, y_train, y_test):
        xgb_model = xgb.XGBClassifier(random_state=42)

        print(f"[XGBoost] {self.cv_folds}-fold CV on training set:")
        cv_summary = self.cross_validate_model(xgb_model, X_train, y_train)

        start = time()
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        y_prob = xgb_model.predict_proba(X_test)[:, 1]
        end = time()

        a1       = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred)
        sens     = recall_score(y_test, y_pred)
        auc      = roc_auc_score(y_test, y_prob)
        time_dur = (end - start) / 60

        print(
            f"[XGBoost] Holdout — AUC-ROC: {auc:.4f} | F1: {f1:.4f} | "
            f"Sensitivity: {sens:.4f} | Time: {time_dur:.4f} min"
        )

        return "XGboost_norm", a1, f1, sens, auc, cv_summary

    def xgboost_hp(self, label0, label1, X_train, X_test, y_train, y_test):

        def xgb_eval(parameterization):
            params = {
                'max_depth':        parameterization.get('max_depth'),
                'learning_rate':    parameterization.get('learning_rate'),
                'subsample':        parameterization.get('subsample'),
                'colsample_bytree': parameterization.get('colsample_bytree'),
                'n_estimators':     100,
                'objective':        'binary:logistic',
                'random_state':     42,
            }
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return recall_score(y_test, y_pred)

        parameters = [
            {"name": "max_depth",        "type": "range", "bounds": [3, 10],   "value_type": "int"},
            {"name": "learning_rate",    "type": "range", "bounds": [0.01, 0.3]},
            {"name": "subsample",        "type": "range", "bounds": [0.5, 1.0]},
            {"name": "colsample_bytree", "type": "range", "bounds": [0.5, 1.0]},
        ]

        best_parameters, best_values, experiment, model = optimize(
            parameters=parameters,
            evaluation_function=xgb_eval,
            objective_name='sens',
            total_trials=10,
            random_seed=42,
        )

        best_xgb = XGBClassifier(
            max_depth=best_parameters['max_depth'],
            learning_rate=best_parameters['learning_rate'],
            subsample=best_parameters['subsample'],
            colsample_bytree=best_parameters['colsample_bytree'],
            n_estimators=100,
            objective='binary:logistic',
            random_state=42,
        )

        print(f"[XGBoost-HP] {self.cv_folds}-fold CV on training set:")
        cv_summary = self.cross_validate_model(best_xgb, X_train, y_train)

        start = time()
        best_xgb.fit(X_train, y_train)
        y_proba = best_xgb.predict_proba(X_test)[:, 1]
        y_pred  = best_xgb.predict(X_test)
        end = time()

        a1       = accuracy_score(y_test, y_pred)
        auc      = roc_auc_score(y_test, y_proba)
        f1       = f1_score(y_test, y_pred)
        sens     = recall_score(y_test, y_pred)
        time_dur = (end - start) / 60

        print(
            f"[XGBoost-HP] Holdout — AUC-ROC: {auc:.4f} | F1: {f1:.4f} | "
            f"Sensitivity: {sens:.4f} | Time: {time_dur:.4f} min"
        )

        return "XGboost_hp", a1, f1, sens, auc, cv_summary

    # ------------------------------------------------------------------ #
    #  Decision Tree                                                       #
    # ------------------------------------------------------------------ #

    def decision_tree(self, label0, label1, X_train, X_test, y_train, y_test):
        dt_model = DecisionTreeClassifier(random_state=42)

        print(f"[DecisionTree] {self.cv_folds}-fold CV on training set:")
        cv_summary = self.cross_validate_model(dt_model, X_train, y_train)

        start = time()
        dt_model.fit(X_train, y_train)
        y_pred  = dt_model.predict(X_test)
        y_prob  = dt_model.predict_proba(X_test)[:, 1]
        end = time()

        a1       = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred)
        sens     = recall_score(y_test, y_pred)
        auc      = roc_auc_score(y_test, y_prob)
        time_dur = (end - start) / 60

        print(
            f"[DecisionTree] Holdout — AUC-ROC: {auc:.4f} | F1: {f1:.4f} | "
            f"Sensitivity: {sens:.4f} | Time: {time_dur:.4f} min"
        )

        return "DecisionTree_norm", a1, f1, sens, auc, cv_summary

    def decision_tree_hp(self, label0, label1, X_train, X_test, y_train, y_test):

        def dt_eval(parameterization):
            params = {
                'max_depth':        parameterization.get('max_depth'),
                'min_samples_split': parameterization.get('min_samples_split'),
                'min_samples_leaf':  parameterization.get('min_samples_leaf'),
                'max_features':      parameterization.get('max_features'),
                'random_state':      42,
            }
            model = DecisionTreeClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return recall_score(y_test, y_pred)

        parameters = [
            {"name": "max_depth",          "type": "range", "bounds": [2, 20],    "value_type": "int"},
            {"name": "min_samples_split",  "type": "range", "bounds": [2, 20],    "value_type": "int"},
            {"name": "min_samples_leaf",   "type": "range", "bounds": [1, 10],    "value_type": "int"},
            {"name": "max_features",       "type": "range", "bounds": [0.1, 1.0]},
        ]

        best_parameters, best_values, experiment, model = optimize(
            parameters=parameters,
            evaluation_function=dt_eval,
            objective_name='sens',
            total_trials=10,
            random_seed=42,
        )

        best_dt = DecisionTreeClassifier(
            max_depth=best_parameters['max_depth'],
            min_samples_split=best_parameters['min_samples_split'],
            min_samples_leaf=best_parameters['min_samples_leaf'],
            max_features=best_parameters['max_features'],
            random_state=42,
        )

        print(f"[DecisionTree-HP] {self.cv_folds}-fold CV on training set:")
        cv_summary = self.cross_validate_model(best_dt, X_train, y_train)

        start = time()
        best_dt.fit(X_train, y_train)
        y_proba = best_dt.predict_proba(X_test)[:, 1]
        y_pred  = best_dt.predict(X_test)
        end = time()

        a1       = accuracy_score(y_test, y_pred)
        auc      = roc_auc_score(y_test, y_proba)
        f1       = f1_score(y_test, y_pred)
        sens     = recall_score(y_test, y_pred)
        time_dur = (end - start) / 60

        print(
            f"[DecisionTree-HP] Holdout — AUC-ROC: {auc:.4f} | F1: {f1:.4f} | "
            f"Sensitivity: {sens:.4f} | Time: {time_dur:.4f} min"
        )

        return "DecisionTree_hp", a1, f1, sens, auc, cv_summary

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    def run(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.datasetsetup(
            self.path, self.target, features=self.features
        )
        print("Dataset prepared")

        _, a1_xg, f1_xg, sens_xg, auc_xg, cv_xg = self.xgboost(
            self.label0, self.label1,
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("xgboost done")

        _, a1_xg_hp, f1_xg_hp, sens_xg_hp, auc_xg_hp, cv_xg_hp = self.xgboost_hp(
            self.label0, self.label1,
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("hp_xgboost done")

        _, a1_dt, f1_dt, sens_dt, auc_dt, cv_dt = self.decision_tree(
            self.label0, self.label1,
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("decision_tree done")

        _, a1_dt_hp, f1_dt_hp, sens_dt_hp, auc_dt_hp, cv_dt_hp = self.decision_tree_hp(
            self.label0, self.label1,
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("hp_decision_tree done")

        results = {
            "XGBoost": {
                "before": {
                    "roc_auc":     auc_xg,
                    "sensitivity": sens_xg,
                    "accuracy":    a1_xg,
                    "f1_score":    f1_xg,
                    "cv":          cv_xg,
                },
                "after": {
                    "roc_auc":     auc_xg_hp,
                    "sensitivity": sens_xg_hp,
                    "accuracy":    a1_xg_hp,
                    "f1_score":    f1_xg_hp,
                    "cv":          cv_xg_hp,
                }
            },
            "DecisionTree": {
                "before": {
                    "roc_auc":     auc_dt,
                    "sensitivity": sens_dt,
                    "accuracy":    a1_dt,
                    "f1_score":    f1_dt,
                    "cv":          cv_dt,
                },
                "after": {
                    "roc_auc":     auc_dt_hp,
                    "sensitivity": sens_dt_hp,
                    "accuracy":    a1_dt_hp,
                    "f1_score":    f1_dt_hp,
                    "cv":          cv_dt_hp,
                }
            }
        }

        return results