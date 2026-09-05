import optuna
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from ml.pipeline.preprocessing import build_preprocessing_pipeline


def optimize_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_trials: int = 20,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Optimize XGBoost with leakage-safe preprocessing."""

    if task_type not in {"classification", "regression"}:
        raise ValueError(
            f"Unsupported task type: {task_type}"
        )

    if n_trials < 1:
        raise ValueError(
            "n_trials must be at least 1."
        )

    if cv < 2:
        raise ValueError(
            "cv must be at least 2."
        )

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if task_type == "classification":
        splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

        scoring = "f1_weighted"

    else:
        splitter = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

        scoring = "neg_root_mean_squared_error"

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int(
            "n_estimators",
            50,
            300,
            step=50,
        )

        max_depth = trial.suggest_int(
            "max_depth",
            3,
            12,
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True,
        )

        min_child_weight = trial.suggest_int(
            "min_child_weight",
            1,
            10,
        )

        subsample = trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        )

        colsample_bytree = trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        )

        gamma = trial.suggest_float(
            "gamma",
            0.0,
            5.0,
        )

        reg_alpha = trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        )

        reg_lambda = trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True,
        )

        if task_type == "classification":
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
            )

        else:
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                n_jobs=-1,
            )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    build_preprocessing_pipeline(X),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=splitter,
            scoring=scoring,
            n_jobs=-1,
        )

        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=random_state,
        ),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    best_params = study.best_params

    if task_type == "classification":
        best_model = XGBClassifier(
            **best_params,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )

    else:
        best_model = XGBRegressor(
            **best_params,
            random_state=random_state,
            n_jobs=-1,
        )

    return {
        "model": best_model,
        "best_params": best_params,
        "best_score": study.best_value,
        "n_trials": len(study.trials),
    }