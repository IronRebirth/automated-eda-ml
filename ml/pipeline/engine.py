import pandas as pd

from ml.models import (
    build_model_leaderboard,
    cross_validate_models,
    evaluate_models,
    evaluate_optimized_models,
    select_best_model,
    train_models,
)
from ml.optimization import (
    optimize_random_forest,
    optimize_xgboost,
)
from ml.pipeline.preprocessing import build_preprocessing_pipeline
from ml.pipeline.splitter import split_features_target
from ml.pipeline.task_detection import detect_task_type


class MLPipeline:
    """Run the complete automated machine-learning pipeline."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        cv: int = 5,
        optimization_trials: int = 20,
    ):
        self.df = dataframe
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.optimization_trials = optimization_trials

    def run(self) -> dict:
        """Execute the complete automated ML pipeline."""

        task_type = detect_task_type(
            self.df,
            self.target_column,
        )

        X_train, X_test, y_train, y_test = split_features_target(
            self.df,
            self.target_column,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        preprocessing_pipeline = build_preprocessing_pipeline(
            X_train
        )

        X_train_transformed = preprocessing_pipeline.fit_transform(
            X_train
        )

        X_test_transformed = preprocessing_pipeline.transform(
            X_test
        )

        models = train_models(
            X_train_transformed,
            y_train,
            task_type,
        )

        evaluation_results = evaluate_models(
            models,
            X_test_transformed,
            y_test,
            task_type,
        )

        cross_validation_results = cross_validate_models(
            models,
            X_train_transformed,
            y_train,
            task_type,
            cv=self.cv,
            random_state=self.random_state,
        )

        leaderboard = build_model_leaderboard(
            evaluation_results,
            task_type,
            cross_validation_results,
        )

        optimization_results = {}

        for model_name, optimizer in {
            "random_forest": optimize_random_forest,
            "xgboost": optimize_xgboost,
        }.items():
            optimization_results[model_name] = optimizer(
                X_train_transformed,
                y_train,
                task_type,
                n_trials=self.optimization_trials,
                cv=self.cv,
                random_state=self.random_state,
            )

        optimized_evaluation = evaluate_optimized_models(
            optimization_results,
            X_train_transformed,
            y_train,
            X_test_transformed,
            y_test,
            task_type,
        )

        best_model = select_best_model(
            optimization_results,
            optimized_evaluation,
            task_type,
        )

        return {
            "task_type": task_type,
            "target_column": self.target_column,
            "models": models,
            "preprocessing": preprocessing_pipeline,
            "evaluation": evaluation_results,
            "cross_validation": cross_validation_results,
            "leaderboard": leaderboard,
            "optimization": optimization_results,
            "optimized_evaluation": optimized_evaluation,
            "best_model": best_model,
        }