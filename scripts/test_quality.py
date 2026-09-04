from ml.profiling import load_dataset
from ml.quality import DataQualityAnalyzer


def main():
    dataframe = load_dataset(
        "datasets/sample/customer_churn.csv"
    )

    analyzer = DataQualityAnalyzer(dataframe)

    result = analyzer.analyze()

    print("Data Quality Report:")
    print(result)


if __name__ == "__main__":
    main()