from collections.abc import Callable

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
from ml.reporting import (
    build_analysis_report,
    serialize_analysis_report,
    validate_serialized_report,
)

ProgressCallback = Callable[
    [str, int, str],
    None,
]


def log_stage(stage: str) -> None:
    """Print a visible pipeline progress checkpoint."""

    print(
        f"[ML PIPELINE] {stage}",
        flush=True,
    )


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
        progress_callback: ProgressCallback | None = None,
    ):
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.optimization_trials = optimization_trials
        self.artifact_path = artifact_path
        self.progress_callback = progress_callback

    def report_progress(
        self,
        stage: str,
        progress: int,
        message: str,
    ) -> None:
        """Report pipeline progress when a callback is configured."""

        if self.progress_callback is not None:
            self.progress_callback(
                stage,
                progress,
                message,
            )

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

        log_stage(
            "Starting pipeline."
        )

        self.report_progress(
            "loading",
            5,
            "Loading dataset into the ML pipeline.",
        )

        log_stage(
            f"Dataset shape: {self.df.shape}"
        )

        log_stage(
            f"Target column: {self.target_column}"
        )

        self.report_progress(
            "profiling",
            10,
            "Detecting the machine-learning task type.",
        )

        log_stage(
            "Detecting task type..."
        )

        task_type = detect_task_type(
            self.df,
            self.target_column,
        )

        log_stage(
            f"Task type detected: {task_type}"
        )

        self.report_progress(
            "preprocessing",
            20,
            "Splitting features and target data.",
        )

        log_stage(
            "Splitting features and target..."
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

        log_stage(
            f"Split complete: train={X_train.shape}, "
            f"test={X_test.shape}"
        )

        log_stage(
            "Building preprocessing pipeline..."
        )

        preprocessing_pipeline = (
            build_preprocessing_pipeline(X_train)
        )

        self.report_progress(
            "preprocessing",
            25,
            "Fitting preprocessing pipeline.",
        )

        log_stage(
            "Fitting preprocessing pipeline..."
        )

        X_train_transformed = (
            preprocessing_pipeline.fit_transform(
                X_train
            )
        )

        log_stage(
            "Transforming test data..."
        )

        X_test_transformed = (
            preprocessing_pipeline.transform(
                X_test
            )
        )

        log_stage(
            "Preprocessing complete."
        )

        if task_type == "classification":
            models = get_classification_models()

        elif task_type == "regression":
            models = get_regression_models()

        else:
            raise ValueError(
                f"Unsupported task type: {task_type}"
            )

        self.report_progress(
            "training",
            35,
            "Training baseline machine-learning models.",
        )

        log_stage(
            "Training baseline models..."
        )

        trained_models = train_models(
            X_train_transformed,
            y_train,
            task_type,
        )

        log_stage(
            "Baseline model training complete."
        )

        self.report_progress(
            "evaluation",
            45,
            "Evaluating baseline model performance.",
        )

        log_stage(
            "Evaluating baseline models..."
        )

        evaluation_results = evaluate_models(
            trained_models,
            X_test_transformed,
            y_test,
            task_type,
        )

        log_stage(
            "Baseline evaluation complete."
        )

        self.report_progress(
            "cross_validation",
            55,
            f"Running {self.cv}-fold cross-validation.",
        )

        log_stage(
            f"Starting {self.cv}-fold cross-validation..."
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

        log_stage(
            "Cross-validation complete."
        )

        self.report_progress(
            "evaluation",
            63,
            "Building the model performance leaderboard.",
        )

        log_stage(
            "Building model leaderboard..."
        )

        leaderboard = build_model_leaderboard(
            evaluation_results,
            task_type,
            cross_validation_results,
        )

        log_stage(
            "Leaderboard complete."
        )

        self.report_progress(
            "optimization",
            68,
            (
                "Optimizing Random Forest and XGBoost "
                "hyperparameters."
            ),
        )

        optimization_results = {}

        log_stage(
            f"Starting Random Forest optimization "
            f"({self.optimization_trials} trials)..."
        )

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

        log_stage(
            "Random Forest optimization complete."
        )

        self.report_progress(
            "optimization",
            76,
            "Optimizing XGBoost hyperparameters.",
        )

        log_stage(
            f"Starting XGBoost optimization "
            f"({self.optimization_trials} trials)..."
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

        log_stage(
            "XGBoost optimization complete."
        )

        self.report_progress(
            "evaluation",
            82,
            "Evaluating the optimized models.",
        )

        log_stage(
            "Evaluating optimized models..."
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

        log_stage(
            "Optimized model evaluation complete."
        )

        log_stage(
            "Selecting best model..."
        )

        best_model = select_best_model(
            optimization_results,
            optimized_evaluation_results,
            task_type,
        )

        log_stage(
            f"Best model: {best_model['model_name']}"
        )

        self.report_progress(
            "explainability",
            87,
            "Generating SHAP explanations for the best model.",
        )

        log_stage(
            "Starting SHAP explainability..."
        )

        explainability = explain_preprocessed_model(
            best_model["model"],
            preprocessing_pipeline,
            X_test,
        )

        log_stage(
            "SHAP explainability complete."
        )

        log_stage(
            "Building explainability summary..."
        )

        explainability["summary"] = (
            build_explainability_summary(
                explainability[
                    "feature_importance"
                ]
            )
        )

        log_stage(
            "Generating explainability insights..."
        )

        explainability["insights"] = (
            generate_explainability_insights(
                explainability[
                    "feature_importance"
                ]
            )
        )

        log_stage(
            "Building explainability metadata..."
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

        log_stage(
            "Validating explainability output..."
        )

        validate_explainability_output(
            explainability
        )

        log_stage(
            "Explainability validation complete."
        )

        if final_artifact_path is not None:
            self.report_progress(
                "saving",
                94,
                "Saving the trained model artifact.",
            )

            log_stage(
                "Saving model artifact..."
            )

            save_model_artifact(
                best_model["model"],
                preprocessing_pipeline,
                final_artifact_path,
            )

            log_stage(
                "Model artifact saved."
            )

        self.report_progress(
            "saving",
            97,
            "Building and validating the analysis report.",
        )

        log_stage(
            "Building final result..."
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

        log_stage(
            "Building analysis report..."
        )

        report = build_analysis_report(
            result,
            dataset_shape=self.df.shape,
        )

        result["report"] = report

        log_stage(
            "Serializing analysis report..."
        )

        result["serialized_report"] = (
            serialize_analysis_report(
                report
            )
        )

        log_stage(
            "Validating serialized report..."
        )

        validate_serialized_report(
            result["serialized_report"]
        )

        self.report_progress(
            "saving",
            99,
            "Finalizing persisted analysis results.",
        )

        log_stage(
            "Pipeline completed successfully."
        )

        return result