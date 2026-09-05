import optuna
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
)


def optimize_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_trials: int = 20,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Optimize a Random Forest model using Optuna."""

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

    if task_type == "classification":
        splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

        scoring = "f1_weighted"
        direction = "maximize"

    else:
        splitter = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

        scoring = "neg_root_mean_squared_error"
        direction = "maximize"

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int(
            "n_estimators",
            50,
            300,
            step=50,
        )

        max_depth = trial.suggest_int(
            "max_depth",
            2,
            20,
        )

        min_samples_split = trial.suggest_int(
            "min_samples_split",
            2,
            10,
        )

        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf",
            1,
            5,
        )

        max_features = trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None],
        )

        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1,
            )

        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1,
            )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=splitter,
            scoring=scoring,
            n_jobs=-1,
        )

        return scores.mean()

    study = optuna.create_study(
        direction=direction,
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
        best_model = RandomForestClassifier(
            **best_params,
            random_state=random_state,
            n_jobs=-1,
        )

    else:
        best_model = RandomForestRegressor(
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