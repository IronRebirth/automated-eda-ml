__all__ = [
    "MLPipeline",
    "build_preprocessing_pipeline",
    "detect_task_type",
    "split_features_target",
]


def __getattr__(name):
    if name == "MLPipeline":
        from .engine import MLPipeline

        return MLPipeline

    if name == "build_preprocessing_pipeline":
        from .preprocessing import build_preprocessing_pipeline

        return build_preprocessing_pipeline

    if name == "detect_task_type":
        from .task_detection import detect_task_type

        return detect_task_type

    if name == "split_features_target":
        from .splitter import split_features_target

        return split_features_target

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )