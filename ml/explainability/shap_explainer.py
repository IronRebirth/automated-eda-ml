import pandas as pd
import shap
from scipy import sparse


def _get_feature_names(
    preprocessing_pipeline,
) -> list[str]:
    """Get feature names produced by the preprocessing pipeline."""

    return (
        preprocessing_pipeline
        .get_feature_names_out()
        .tolist()
    )


def explain_model(
    model,
    X: pd.DataFrame,
) -> dict:
    """Generate global SHAP feature importance for a trained model."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "X must not be empty."
        )

    numeric_X = X.astype(float)

    explainer = shap.Explainer(
        model,
        numeric_X,
    )

    shap_values = explainer(
        numeric_X
    )

    values = shap_values.values

    if values.ndim == 3:
        values = values.mean(
            axis=2
        )

    feature_importance = pd.DataFrame(
        {
            "feature": numeric_X.columns,
            "importance": (
                abs(values)
                .mean(axis=0)
            ),
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return {
        "feature_importance": feature_importance,
        "shap_values": shap_values,
    }


def explain_preprocessed_model(
    model,
    preprocessing_pipeline,
    X: pd.DataFrame,
) -> dict:
    """Explain a trained model using its fitted preprocessing pipeline."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "X must not be empty."
        )

    X_transformed = preprocessing_pipeline.transform(
        X
    )

    feature_names = _get_feature_names(
        preprocessing_pipeline
    )

    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.tocsr()

        transformed_data = pd.DataFrame.sparse.from_spmatrix(
            X_transformed,
            columns=feature_names,
            index=X.index,
        )
    else:
        transformed_data = pd.DataFrame(
            X_transformed,
            columns=feature_names,
            index=X.index,
        )

    result = explain_model(
        model,
        transformed_data,
    )

    return {
        "feature_importance": result[
            "feature_importance"
        ],
        "shap_values": result[
            "shap_values"
        ],
        "transformed_data": transformed_data,
        "feature_names": feature_names,
    }