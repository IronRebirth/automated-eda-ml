import pandas as pd

from ml.explainability import (
    build_explainability_metadata,
    build_explainability_summary,
    explain_preprocessed_model,
    generate_explainability_insights,
    validate_explainability_output,
)
from ml.models import (
    build_model_leaderboard,
    cross_validate_models,
    evaluate_models,
    evaluate_optimized_models,
    get_classification_models,
    get_regression_models,
    save_model_artifact,
    select_best_model,
    train_models,
)
from ml.optimization import (
    optimize_random_forest,
    optimize_xgboost,
)
from ml.pipeline.preprocessing import (
    build_preprocessing_pipeline,
)
from ml.pipeline.splitter import (
    split_features_target,
)
from ml.pipeline.task_detection import (
    detect_task_type,
)
from ml.reporting import build_analysis_report


class MLPipeline:
    """End-to-end automated machine learning pipeline."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        cv: int = 5,
        optimization_trials: int = 20,
        artifact_path: str | None = None,
    ):
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.optimization_trials = optimization_trials
        self.artifact_path = artifact_path

    def run(
        self,
        artifact_path: str | None = None,
    ) -> dict:
        """Run the complete automated ML pipeline."""

        final_artifact_path = (
            artifact_path
            if artifact_path is not None
            else self.artifact_path
        )

        task_type = detect_task_type(
            self.df,
            self.target_column,
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = split_features_target(
            self.df,
            self.target_column,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        preprocessing_pipeline = (
            build_preprocessing_pipeline(X_train)
        )

        X_train_transformed = (
            preprocessing_pipeline.fit_transform(
                X_train
            )
        )

        X_test_transformed = (
            preprocessing_pipeline.transform(
                X_test
            )
        )

        if task_type == "classification":
            models = get_classification_models()

        elif task_type == "regression":
            models = get_regression_models()

        else:
            raise ValueError(
                f"Unsupported task type: {task_type}"
            )

        trained_models = train_models(
            X_train_transformed,
            y_train,
            task_type,
        )

        evaluation_results = evaluate_models(
            trained_models,
            X_test_transformed,
            y_test,
            task_type,
        )

        cross_validation_results = (
            cross_validate_models(
                models,
                X_train,
                y_train,
                task_type,
                cv=self.cv,
                random_state=self.random_state,
            )
        )

        leaderboard = build_model_leaderboard(
            evaluation_results,
            task_type,
            cross_validation_results,
        )

        optimization_results = {}

        optimization_results[
            "random_forest"
        ] = optimize_random_forest(
            X_train,
            y_train,
            task_type,
            n_trials=self.optimization_trials,
            cv=self.cv,
            random_state=self.random_state,
        )

        optimization_results[
            "xgboost"
        ] = optimize_xgboost(
            X_train,
            y_train,
            task_type,
            n_trials=self.optimization_trials,
            cv=self.cv,
            random_state=self.random_state,
        )

        optimized_evaluation_results = (
            evaluate_optimized_models(
                optimization_results,
                X_train,
                y_train,
                X_test,
                y_test,
                task_type,
                preprocessing_pipeline=preprocessing_pipeline,
            )
        )

        best_model = select_best_model(
            optimization_results,
            optimized_evaluation_results,
            task_type,
        )

        explainability = explain_preprocessed_model(
            best_model["model"],
            preprocessing_pipeline,
            X_test,
        )

        explainability["summary"] = (
            build_explainability_summary(
                explainability[
                    "feature_importance"
                ]
            )
        )

        explainability["insights"] = (
            generate_explainability_insights(
                explainability[
                    "feature_importance"
                ]
            )
        )

        explainability["metadata"] = (
            build_explainability_metadata(
                explainability[
                    "feature_importance"
                ],
                model_name=best_model[
                    "model_name"
                ],
                task_type=task_type,
                preprocessing_applied=True,
                top_features_count=(
                    explainability[
                        "summary"
                    ]["top_n"]
                ),
            )
        )

        validate_explainability_output(
            explainability
        )

        if final_artifact_path is not None:
            save_model_artifact(
                best_model["model"],
                preprocessing_pipeline,
                final_artifact_path,
            )

        result = {
            "task_type": task_type,
            "target_column": self.target_column,
            "models": trained_models,
            "evaluation": evaluation_results,
            "cross_validation": (
                cross_validation_results
            ),
            "leaderboard": leaderboard,
            "optimization": optimization_results,
            "optimized_evaluation": (
                optimized_evaluation_results
            ),
            "best_model": best_model,
            "explainability": explainability,
            "artifact_path": (
                str(final_artifact_path)
                if final_artifact_path is not None
                else None
            ),
        }

        result["report"] = build_analysis_report(
            result,
            dataset_shape=self.df.shape,
        )

        return result