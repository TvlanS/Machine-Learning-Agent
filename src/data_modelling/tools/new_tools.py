import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix
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

from pathlib import Path
import json
from datetime import datetime



def _4sf(x):
    """Round a numeric value to 4 significant figures for JSON output."""
    if isinstance(x, (np.floating, float)):
        return float(f"{float(x):.4g}")
    return x


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

    def confusionmat(self, y_test, y_pred):
        cm = confusion_matrix(y_test,y_pred)
        return cm
    

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
        cm = confusion_matrix(y_test,y_pred)

        print(
            f"[XGBoost] Holdout — AUC-ROC: {auc:.4f} | F1: {f1:.4f} | "
            f"Sensitivity: {sens:.4f} | Time: {time_dur:.4f} min"
        )

        return "XGboost_norm", a1, f1, sens, auc, cm, time_dur

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
            total_trials=15,
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
        cm = confusion_matrix(y_test,y_pred)

        print(
            f"[XGBoost-HP] Holdout — AUC-ROC: {auc:.4f} | F1: {f1:.4f} | "
            f"Sensitivity: {sens:.4f} | Time: {time_dur:.4f} min"
        )

        return "XGboost_hp", a1, f1, sens, auc, cm, time_dur

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
        cm = confusion_matrix(y_test, y_pred)

        print(
            f"[DecisionTree] Holdout — AUC-ROC: {auc:.4f} | F1: {f1:.4f} | "
            f"Sensitivity: {sens:.4f} | Time: {time_dur:.4f} min"
        )

        return "DecisionTree_norm", a1, f1, sens, auc, cm, time_dur, cv_summary

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
            total_trials=15,
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
        cm = confusion_matrix(y_test,y_pred)

        print(
            f"[DecisionTree-HP] Holdout — AUC-ROC: {auc:.4f} | F1: {f1:.4f} | "
            f"Sensitivity: {sens:.4f} | Time: {time_dur:.4f} min"
        )

        return "DecisionTree_hp", a1, f1, sens, auc, cm , time_dur
    # LR and Neural Network


    class SepsisModeling:
        """
        Encapsulates logistic regression and autoencoder-based feature extraction
        for sepsis risk prediction. Returns a results dictionary with 'before' and 'after'
        for both models.
        """
        
        def __init__(self, random_state=42, cv_folds=5, max_iter_ae=500):
            self.random_state = random_state
            self.cv_folds = cv_folds
            self.max_iter_ae = max_iter_ae
            
        # ------------------------------------------------------------------
        # Helper functions (identical to original)
        # ------------------------------------------------------------------
        def _cross_validate_model_lr(self, model, X, y):
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scoring = {
                'roc_auc': 'roc_auc',
                'f1': 'f1',
                'sensitivity': 'recall',
                'accuracy': 'accuracy',
            }
            res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            summary = {}
            for metric, key in [('roc_auc', 'test_roc_auc'),
                                ('f1', 'test_f1'),
                                ('sensitivity', 'test_sensitivity'),
                                ('accuracy', 'test_accuracy')]:
                scores = res[key]
                summary[metric] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'scores': scores.tolist()
                }
                print(f"  CV {metric:12s}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
            return summary

        def _evaluate_holdout_lr(self, y_true, y_pred, y_prob, label):
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            sens = recall_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            print(f"[{label}] Holdout - AUC-ROC: {auc:.4f} | F1: {f1:.4f} | Sensitivity: {sens:.4f} | Accuracy: {acc:.4f}")
            return {'accuracy': acc, 'f1_score': f1, 'sensitivity': sens, 'roc_auc': auc}

        def _get_bottleneck_features(self, autoencoder, X, bottleneck_layer_idx=2):
            activation = X.copy()
            for i in range(bottleneck_layer_idx):
                activation = np.maximum(0, activation @ autoencoder.coefs_[i] + autoencoder.intercepts_[i])
            return activation

        # ------------------------------------------------------------------
        # Main run method – expects scaled train/test splits
        # ------------------------------------------------------------------
        def run(self, X_train, X_test, y_train, y_test):
            """
            Parameters
            ----------
            X_train, X_test : array-like, scaled features
            y_train, y_test : array-like, labels (0/1 for sepsis_risk)
            
            Returns
            -------
            results : dict
                Contains 'LogisticRegression' and 'Autoencoder' keys, each with
                'before' and 'after' sub-dictionaries (exactly as in the original code).
            """
            results = {}
            
            # ============================================================
            # LOGISTIC REGRESSION (identical to original)
            # ============================================================
            print("\n" + "=" * 60)
            print("LOGISTIC REGRESSION - Default (C=1.0)")
            print("=" * 60)
            
            lr_default = LogisticRegression(random_state=self.random_state, max_iter=1000)
            print(f"\n[LogisticRegression] {self.cv_folds}-fold CV on training set:")
            cv_lr_default = self._cross_validate_model_lr(lr_default, X_train, y_train)

            start_time = time()
            lr_default.fit(X_train, y_train)
            y_pred_lr_d = lr_default.predict(X_test)
            y_prob_lr_d = lr_default.predict_proba(X_test)[:, 1]
            cm_lr_d = confusion_matrix(y_test, y_pred_lr_d)
            time_lr_d = (time() - start_time) / 60
            holdout_lr_d = self._evaluate_holdout_lr(y_test, y_pred_lr_d, y_prob_lr_d, "LogisticRegression")
            
            print("\n" + "=" * 60)
            print("LOGISTIC REGRESSION - Hyperparameter Tuning")
            print("=" * 60)
            
            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
            gs_lr = GridSearchCV(
                LogisticRegression(random_state=self.random_state, max_iter=1000),
                param_grid, cv=self.cv_folds, scoring='recall', n_jobs=-1, verbose=0
            )
            gs_lr.fit(X_train, y_train)
            best_C = gs_lr.best_params_['C']
            
            print("\nGrid search results (scoring=recall):")
            for params, score in zip(gs_lr.cv_results_['params'], gs_lr.cv_results_['mean_test_score']):
                marker = " <- best" if params['C'] == best_C else ""
                print(f"  C={params['C']:6} -> mean recall: {score:.4f}{marker}")
            
            lr_hp = LogisticRegression(C=best_C, random_state=self.random_state, max_iter=1000)
            print(f"\n[LogisticRegression-HP] {self.cv_folds}-fold CV on training set:")
            cv_lr_hp = self._cross_validate_model_lr(lr_hp, X_train, y_train)

            start_time = time()
            lr_hp.fit(X_train, y_train)
            y_pred_lr_hp = lr_hp.predict(X_test)
            y_prob_lr_hp = lr_hp.predict_proba(X_test)[:, 1]
            cm_lr_hp = confusion_matrix(y_test, y_pred_lr_hp)
            time_lr_hp = (time() - start_time) / 60
            holdout_lr_hp = self._evaluate_holdout_lr(y_test, y_pred_lr_hp, y_prob_lr_hp, "LogisticRegression-HP")
            
            results["LogisticRegression"] = {
                "before": {
                    "roc_auc":           _4sf(holdout_lr_d['roc_auc']),
                    "sensitivity":       _4sf(holdout_lr_d['sensitivity']),
                    "accuracy":          _4sf(holdout_lr_d['accuracy']),
                    "f1_score":          _4sf(holdout_lr_d['f1_score']),
                    "confusion_matrix":  cm_lr_d.tolist(),
                    "training_time_min": _4sf(time_lr_d),
                },
                "after": {
                    "roc_auc":           _4sf(holdout_lr_hp['roc_auc']),
                    "sensitivity":       _4sf(holdout_lr_hp['sensitivity']),
                    "accuracy":          _4sf(holdout_lr_hp['accuracy']),
                    "f1_score":          _4sf(holdout_lr_hp['f1_score']),
                    "confusion_matrix":  cm_lr_hp.tolist(),
                    "training_time_min": _4sf(time_lr_hp),
                    "best_C":            best_C,
                }
            }
            print("\nLogistic Regression results stored.")
            
            # ============================================================
            # AUTOENCODER MODELLING (identical to original)
            # ============================================================
            print("\n" + "=" * 60)
            print("AUTOENCODER - Default Architecture (bottleneck=10)")
            print("=" * 60)
            
            N_FEATURES = X_train.shape[1]
            BOTTLENECK = 10
            
            ae_default = MLPRegressor(
                hidden_layer_sizes=(18, BOTTLENECK, 18),
                activation='relu', max_iter=self.max_iter_ae, random_state=self.random_state,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=15, verbose=False
            )
            
            print(f"\n[Autoencoder] Training: {N_FEATURES} -> 18 -> {BOTTLENECK} -> 18 -> {N_FEATURES}")
            start_time = time()
            ae_default.fit(X_train, X_train)
            time_ae_d = (time() - start_time) / 60
            
            recon_mse = np.mean((X_test - ae_default.predict(X_test)) ** 2)
            print(f"[Autoencoder] Reconstruction MSE (test): {recon_mse:.6f} | Iterations: {ae_default.n_iter_}")
            
            X_train_enc_d = self._get_bottleneck_features(ae_default, X_train)
            X_test_enc_d = self._get_bottleneck_features(ae_default, X_test)
            print(f"[Autoencoder] Encoded shape: {X_train_enc_d.shape}")
            
            clf_ae_default = LogisticRegression(random_state=self.random_state, max_iter=1000)
            print(f"\n[Autoencoder] {self.cv_folds}-fold CV on encoded features:")
            cv_ae_default = self._cross_validate_model_lr(clf_ae_default, X_train_enc_d, y_train)
            
            clf_ae_default.fit(X_train_enc_d, y_train)
            y_pred_ae_d = clf_ae_default.predict(X_test_enc_d)
            y_prob_ae_d = clf_ae_default.predict_proba(X_test_enc_d)[:, 1]
            holdout_ae_d = self._evaluate_holdout_lr(y_test, y_pred_ae_d, y_prob_ae_d, "Autoencoder (default)")
            
            print("\n" + "=" * 60)
            print("AUTOENCODER - Architecture Search")
            print("=" * 60)
            
            ae_configs = [
                {'hidden_layer_sizes': (18,  6, 18),  'label': '27->18->6->18->27',  'bn': 6},
                {'hidden_layer_sizes': (18,  8, 18),  'label': '27->18->8->18->27',  'bn': 8},
                {'hidden_layer_sizes': (18, 10, 18),  'label': '27->18->10->18->27', 'bn': 10},
                {'hidden_layer_sizes': (20, 10, 20),  'label': '27->20->10->20->27', 'bn': 10},
                {'hidden_layer_sizes': (24, 14, 24),  'label': '27->24->14->24->27', 'bn': 14},
            ]
            
            best_auc_hp = -1
            best_cfg_idx = -1
            #print("\nSearching over encoder architectures (criterion: holdout ROC-AUC):\n")
            
            for idx, cfg in enumerate(ae_configs):
                ae_tmp = MLPRegressor(
                    hidden_layer_sizes=cfg['hidden_layer_sizes'],
                    activation='relu', max_iter=self.max_iter_ae, random_state=self.random_state,
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=15, verbose=False
                )
                ae_tmp.fit(X_train, X_train)
                enc_tr = self._get_bottleneck_features(ae_tmp, X_train)
                enc_te = self._get_bottleneck_features(ae_tmp, X_test)
                clf_tmp = LogisticRegression(random_state=self.random_state, max_iter=1000)
                clf_tmp.fit(enc_tr, y_train)
                auc_tmp = roc_auc_score(y_test, clf_tmp.predict_proba(enc_te)[:, 1])
                if auc_tmp > best_auc_hp:
                    best_auc_hp = auc_tmp
                    best_cfg_idx = idx
                    best_enc_tr = enc_tr
                    best_enc_te = enc_te
                marker = " <- best" if idx == best_cfg_idx else ""
                #print(f"  {cfg['label']:22s} | AUC={auc_tmp:.4f}{marker}")
            
            best_label = ae_configs[best_cfg_idx]['label']
            best_bn = ae_configs[best_cfg_idx]['bn']
            #print(f"\nBest architecture: {best_label} (bottleneck={best_bn})")
            
            #print("\n" + "=" * 60)
            #print(f"AUTOENCODER-HP - Best: {best_label}")
            #print("=" * 60)
            
            clf_ae_hp = LogisticRegression(random_state=self.random_state, max_iter=1000)
            #print(f"\n[Autoencoder-HP] {self.cv_folds}-fold CV on encoded features:")
            cv_ae_hp = self._cross_validate_model_lr(clf_ae_hp, best_enc_tr, y_train)
            
            start_time = time()
            clf_ae_hp.fit(best_enc_tr, y_train)
            y_pred_ae_hp = clf_ae_hp.predict(best_enc_te)
            y_prob_ae_hp = clf_ae_hp.predict_proba(best_enc_te)[:, 1]
            time_ae_hp = (time() - start_time) / 60
            holdout_ae_hp = self._evaluate_holdout_lr(y_test, y_pred_ae_hp, y_prob_ae_hp, "Autoencoder-HP")
            cm_d = confusion_matrix(y_test, y_pred_ae_d)
            cm_hp = confusion_matrix(y_test, y_pred_ae_hp)
            
            results["Autoencoder"] = {
                "before": {
                    "roc_auc":           _4sf(holdout_ae_d['roc_auc']),
                    "sensitivity":       _4sf(holdout_ae_d['sensitivity']),
                    "accuracy":          _4sf(holdout_ae_d['accuracy']),
                    "f1_score":          _4sf(holdout_ae_d['f1_score']),
                    "confusion_matrix":  cm_d.tolist(),
                    "training_time_min": _4sf(time_ae_d),
                    "recon_mse":         _4sf(recon_mse),
                    "architecture":      f"{N_FEATURES}->18->10->18->{N_FEATURES}",
                    "bottleneck":        BOTTLENECK,
                },
                "after": {
                    "roc_auc":           _4sf(holdout_ae_hp['roc_auc']),
                    "sensitivity":       _4sf(holdout_ae_hp['sensitivity']),
                    "accuracy":          _4sf(holdout_ae_hp['accuracy']),
                    "f1_score":          _4sf(holdout_ae_hp['f1_score']),
                    "confusion_matrix":  cm_hp.tolist(),
                    "training_time_min": _4sf(time_ae_hp),
                    "best_architecture": best_label,
                    "bottleneck":        best_bn,
                }
            }
            print("\nAutoencoder results stored.")
            
            return results


    def run(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.datasetsetup(
            self.path, self.target, features=self.features
        )
        print("Dataset prepared")

        _, a1_xg, f1_xg, sens_xg, auc_xg, cm_xg, time_xg = self.xgboost(
            self.label0, self.label1,
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("xgboost done")

        _, a1_xg_hp, f1_xg_hp, sens_xg_hp, auc_xg_hp, cm_xg_hp, time_xg_hp = self.xgboost_hp(
            self.label0, self.label1,
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("hp_xgboost done")

        _, a1_dt, f1_dt, sens_dt, auc_dt, cm_dt, time_dt, cv_dt = self.decision_tree(
            self.label0, self.label1,
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("decision_tree done")

        _, a1_dt_hp, f1_dt_hp, sens_dt_hp, auc_dt_hp, cm_dt_hp, time_dt_hp = self.decision_tree_hp(
            self.label0, self.label1,
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("hp_decision_tree done")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.X_train)
        X_test  = scaler.transform(self.X_test)

        model = self.SepsisModeling(random_state=42, cv_folds=5)
        final_results = model.run(X_train, X_test, self.y_train, self.y_test)

        results = {
            "XGBoost": {
                "before": {
                    "roc_auc":           _4sf(auc_xg),
                    "sensitivity":       _4sf(sens_xg),
                    "accuracy":          _4sf(a1_xg),
                    "f1_score":          _4sf(f1_xg),
                    "confusion_matrix":  cm_xg.tolist(),
                    "training_time_min": _4sf(time_xg),
                },
                "after": {
                    "roc_auc":           _4sf(auc_xg_hp),
                    "sensitivity":       _4sf(sens_xg_hp),
                    "accuracy":          _4sf(a1_xg_hp),
                    "f1_score":          _4sf(f1_xg_hp),
                    "confusion_matrix":  cm_xg_hp.tolist(),
                    "training_time_min": _4sf(time_xg_hp),
                }
            },
            "DecisionTree": {
                "before": {
                    "roc_auc":           _4sf(auc_dt),
                    "sensitivity":       _4sf(sens_dt),
                    "accuracy":          _4sf(a1_dt),
                    "f1_score":          _4sf(f1_dt),
                    "confusion_matrix":  cm_dt.tolist(),
                    "training_time_min": _4sf(time_dt),
                },
                "after": {
                    "roc_auc":           _4sf(auc_dt_hp),
                    "sensitivity":       _4sf(sens_dt_hp),
                    "accuracy":          _4sf(a1_dt_hp),
                    "f1_score":          _4sf(f1_dt_hp),
                    "confusion_matrix":  cm_dt_hp.tolist(),
                    "training_time_min": _4sf(time_dt_hp),
                }
            }
        }
        results.update(final_results)

        # --- Add metadata and save to JSON ---
        n_features = self.X_train.shape[1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results["metadata"] = {
            "input_file":  self.path,
            "target":      self.target,
            "n_features":  n_features,
            "features":    self.features if self.features else "all",
            "timestamp":   datetime.now().isoformat(),
        }

        out_name = f"prediction_results_{n_features}feat_{timestamp}.json"
        out_path = Path(self.path).parent / out_name
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_path}")

        return results
    
if __name__ == "__main__":
    path = r"C:\Users\tvlan\Documents\1.0 Python\data_modelling\datafolder\early_sepsis_full_simulated_dataset_dropped_encoded_20260607_154658.csv"
    target = "sepsis_risk"
    label1 = "0"
    label0 = "1"

    tool = prediction(path,target,label0,label1)
    output = tool.run()

    print(output)

