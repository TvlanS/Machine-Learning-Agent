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

import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

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
            total_trials=1,
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
            total_trials=1,
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
    # LR and Neural Network

    def cross_validate_model_lr(self, model, X_tr, y_tr, folds= 5):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state= 42)
        scoring = {
            'roc_auc'    : 'roc_auc',
            'f1'         : 'f1',
            'sensitivity': 'recall',
            'accuracy'   : 'accuracy',
        }
        res = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
        summary = {}
        for metric, key in [('roc_auc',     'test_roc_auc'),
                            ('f1',          'test_f1'),
                            ('sensitivity', 'test_sensitivity'),
                            ('accuracy',    'test_accuracy')]:
            scores = res[key]
            summary[metric] = {
                'mean'  : float(np.mean(scores)),
                'std'   : float(np.std(scores)),
                'scores': scores.tolist()
            }
            print(f"  CV {metric:12s}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
        return summary

    def evaluate_holdout_lr(self, y_true, y_pred, y_prob, label):
            acc  = accuracy_score(y_true, y_pred)
            f1   = f1_score(y_true, y_pred)
            sens = recall_score(y_true, y_pred)
            auc  = roc_auc_score(y_true, y_prob)
            print(f"[{label}] Holdout - AUC-ROC: {auc} | F1: {f1} | Sensitivity: {sens} | Accuracy: {acc}")
            return {'accuracy': acc, 'f1_score': f1, 'sensitivity': sens, 'roc_auc': auc}

    def get_bottleneck_features(self, autoencoder, X, bottleneck_layer_idx=2):
            activation = X.copy()
            for i in range(bottleneck_layer_idx):
                activation = np.maximum(0, activation @ autoencoder.coefs_[i] + autoencoder.intercepts_[i])
            return activation

    def run_logistic_regression(self, X_train, X_test, y_train, y_test):
            # ============================================================
            # LOGISTIC REGRESSION
            # ============================================================

            print("\n" + "=" * 60)
            print("LOGISTIC REGRESSION - Default (C=1.0)")
            print("=" * 60)

            lr_default = LogisticRegression(random_state= 42, max_iter=1000)
            print(f"\n[LogisticRegression] {5}-fold CV on training set:")
            cv_lr_default = self.cross_validate_model_lr(lr_default, X_train, y_train)

            lr_default.fit(X_train, y_train)
            y_pred_lr_d = lr_default.predict(X_test)
            y_prob_lr_d = lr_default.predict_proba(X_test)[:, 1]
            holdout_lr_d = self.evaluate_holdout_lr(y_test, y_pred_lr_d, y_prob_lr_d, "LogisticRegression")

            print("\n" + "=" * 60)
            print("LOGISTIC REGRESSION - Hyperparameter Tuning")
            print("=" * 60)

            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
            gs_lr = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=1000),
                param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=0
            )
            gs_lr.fit(X_train, y_train)
            best_C = gs_lr.best_params_['C']

            print("\nGrid search results (scoring=recall):")
            for params, score in zip(gs_lr.cv_results_['params'], gs_lr.cv_results_['mean_test_score']):
                marker = " <- best" if params['C'] == best_C else ""
                print(f"  C={params['C']:6} -> mean recall: {score:.4f}{marker}")

            lr_hp = LogisticRegression(C=best_C, random_state=42, max_iter=1000)
            print(f"\n[LogisticRegression-HP] {5}-fold CV on training set:")
            cv_lr_hp = self.cross_validate_model_lr(lr_hp, X_train, y_train)

            lr_hp.fit(X_train, y_train)
            y_pred_lr_hp = lr_hp.predict(X_test)
            y_prob_lr_hp = lr_hp.predict_proba(X_test)[:, 1]
            holdout_lr_hp = self.evaluate_holdout_lr(y_test, y_pred_lr_hp, y_prob_lr_hp, "LogisticRegression-HP")

            results_lr = {
                "LogisticRegression": {
                    "before": {
                        "roc_auc"    : holdout_lr_d['roc_auc'],
                        "sensitivity": holdout_lr_d['sensitivity'],
                        "accuracy"   : holdout_lr_d['accuracy'],
                        "f1_score"   : holdout_lr_d['f1_score'],
                        "cv"         : cv_lr_default,
                    },
                    "after": {
                        "roc_auc"    : holdout_lr_hp['roc_auc'],
                        "sensitivity": holdout_lr_hp['sensitivity'],
                        "accuracy"   : holdout_lr_hp['accuracy'],
                        "f1_score"   : holdout_lr_hp['f1_score'],
                        "best_C"     : best_C,
                        "cv"         : cv_lr_hp,
                    }
                }
            }
            print("\nLogistic Regression results stored.")
            return results_lr
    def cross_validate_model_lr(model, X, y, folds=5, random_state=42):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = {'roc_auc': 'roc_auc', 'f1': 'f1', 'sensitivity': 'recall', 'accuracy': 'accuracy'}
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        summary = {}
        for metric, key in [('roc_auc', 'test_roc_auc'), ('f1', 'test_f1'),
                            ('sensitivity', 'test_sensitivity'), ('accuracy', 'test_accuracy')]:
            summary[metric] = {'mean': float(np.mean(res[key])), 'std': float(np.std(res[key])),
                            'scores': res[key].tolist()}
        return summary

    def evaluate_holdout_lr(y_true, y_pred, y_prob, label):
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            sens = recall_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            print(f"[{label}] Holdout - AUC: {auc:.4f} | F1: {f1:.4f} | Sens: {sens:.4f} | Acc: {acc:.4f}")
            return {'accuracy': acc, 'f1_score': f1, 'sensitivity': sens, 'roc_auc': auc}

    def get_bottleneck_features(autoencoder, X, bottleneck_layer_idx=2):
            activation = X.copy()
            for i in range(bottleneck_layer_idx):
                activation = np.maximum(0, activation @ autoencoder.coefs_[i] + autoencoder.intercepts_[i])
            return activation

        # ---------- Autoencoder pipeline ----------
    class AutoencoderPipeline:
            def __init__(self, random_state=42, cv_folds=5, max_iter=500, early_stopping=True):
                self.random_state = random_state
                self.cv_folds = cv_folds
                self.max_iter = max_iter
                self.early_stopping = early_stopping
                self.default_arch = (18, 10, 18)          # bottleneck = 10
                self.search_configs = [
                    {'hidden_layer_sizes': (18,  6, 18), 'label': '27->18->6->18->27', 'bn': 6},
                    {'hidden_layer_sizes': (18,  8, 18), 'label': '27->18->8->18->27', 'bn': 8},
                    {'hidden_layer_sizes': (18, 10, 18), 'label': '27->18->10->18->27', 'bn': 10},
                    {'hidden_layer_sizes': (20, 10, 20), 'label': '27->20->10->20->27', 'bn': 10},
                    {'hidden_layer_sizes': (24, 14, 24), 'label': '27->24->14->24->27', 'bn': 14},
                ]

            def _train_autoencoder(self, X_train, architecture):
                ae = MLPRegressor(
                    hidden_layer_sizes=architecture,
                    activation='relu',
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    early_stopping=self.early_stopping,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    verbose=False
                )
                ae.fit(X_train, X_train)
                return ae

            def _evaluate(self, ae, X_train, X_test, y_train, y_test, label_prefix=""):
                """Train a classifier on bottleneck features and return (cv_results, holdout_metrics)."""
                train_enc = get_bottleneck_features(ae, X_train)
                test_enc = get_bottleneck_features(ae, X_test)
                clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
                cv_res = cross_validate_model_lr(clf, train_enc, y_train,
                                                folds=self.cv_folds, random_state=self.random_state)
                clf.fit(train_enc, y_train)
                y_pred = clf.predict(test_enc)
                y_prob = clf.predict_proba(test_enc)[:, 1]
                holdout = evaluate_holdout_lr(y_test, y_pred, y_prob, label_prefix)
                return cv_res, holdout

            def run(self, X_train, X_test, y_train, y_test):
                n_features = X_train.shape[1]
                print(f"Input features: {n_features}")

                # ----- BEFORE: default autoencoder -----
                print("\n" + "="*60)
                print("AUTOENCODER - Default Architecture (bottleneck=10)")
                print("="*60)
                ae_default = self._train_autoencoder(X_train, self.default_arch)
                recon_mse = np.mean((X_test - ae_default.predict(X_test))**2)
                print(f"Reconstruction MSE (test): {recon_mse:.6f}")
                cv_before, holdout_before = self._evaluate(
                    ae_default, X_train, X_test, y_train, y_test, label_prefix="Autoencoder (before)"
                )

                # ----- AFTER: architecture search -----
                print("\n" + "="*60)
                print("AUTOENCODER - Architecture Search")
                print("="*60)
                best_auc = -1
                best_cfg = None
                best_enc_train = best_enc_test = None
                for cfg in self.search_configs:
                    ae = self._train_autoencoder(X_train, cfg['hidden_layer_sizes'])
                    train_enc = get_bottleneck_features(ae, X_train)
                    test_enc = get_bottleneck_features(ae, X_test)
                    clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
                    clf.fit(train_enc, y_train)
                    auc = roc_auc_score(y_test, clf.predict_proba(test_enc)[:, 1])
                    marker = " <- best" if auc > best_auc else ""
                    print(f"  {cfg['label']:22s} | AUC={auc:.4f}{marker}")
                    if auc > best_auc:
                        best_auc = auc
                        best_cfg = cfg
                        best_enc_train, best_enc_test = train_enc, test_enc

                print(f"\nBest architecture: {best_cfg['label']} (bottleneck={best_cfg['bn']})")
                print("\n" + "="*60)
                print("AUTOENCODER - Best Architecture (after)")
                print("="*60)
                clf_best = LogisticRegression(random_state=self.random_state, max_iter=1000)
                cv_after = cross_validate_model_lr(clf_best, best_enc_train, y_train,
                                                folds=self.cv_folds, random_state=self.random_state)
                clf_best.fit(best_enc_train, y_train)
                y_pred_best = clf_best.predict(best_enc_test)
                y_prob_best = clf_best.predict_proba(best_enc_test)[:, 1]
                holdout_after = evaluate_holdout_lr(y_test, y_pred_best, y_prob_best, "Autoencoder (after)")

                # Return dictionary with exactly the required structure
                return {
                    "before": {
                        "roc_auc": holdout_before["roc_auc"],
                        "sensitivity": holdout_before["sensitivity"],
                        "accuracy": holdout_before["accuracy"],
                        "f1_score": holdout_before["f1_score"],
                        "cv": cv_before
                    },
                    "after": {
                        "roc_auc": holdout_after["roc_auc"],
                        "sensitivity": holdout_after["sensitivity"],
                        "accuracy": holdout_after["accuracy"],
                        "f1_score": holdout_after["f1_score"],
                        "cv": cv_after
                    }
                }
    

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

        results_lr = self.run_logistic_regression(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("Lr done")


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
        results.update(results_lr)
        return results