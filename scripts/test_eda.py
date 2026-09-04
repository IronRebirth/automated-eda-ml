from ml.eda import EDAAnalyzer
from ml.profiling import load_dataset


def main():
    dataframe = load_dataset(
        "datasets/sample/customer_churn.csv"
    )

    analyzer = EDAAnalyzer(
        dataframe,
        target_column="churn",
    )

    result = analyzer.analyze()

    print("EDA Report")
    print("=" * 40)

    print("\nNumerical columns:")
    print(list(result["numerical"].keys()))

    print("\nCategorical columns:")
    print(list(result["categorical"].keys()))

    print("\nTarget:")
    print(result["target"])

    print("\nInsights:")
    for insight in result["insights"]:
        print(f"- {insight}")

    print("\nNumerical visualizations:")
    print(list(result["visualizations"]["numerical"].keys()))

    print("\nCategorical visualizations:")
    print(list(result["visualizations"]["categorical"].keys()))


if __name__ == "__main__":
    main()