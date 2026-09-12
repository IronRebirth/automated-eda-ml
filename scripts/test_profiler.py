from ml.profiling.loader import load_dataset
from ml.profiling.profiler import DatasetProfiler


def main():
    dataframe = load_dataset("datasets/sample/customer_churn.csv")

    profiler = DatasetProfiler(dataframe)

    profile = profiler.profile()

    for key, value in profile.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
