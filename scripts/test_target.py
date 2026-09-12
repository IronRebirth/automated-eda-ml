import pandas as pd

from ml.eda import analyze_target


def main():
    dataframe = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "income": [30000, 40000, 50000, 60000, 70000],
            "churn": [0, 0, 1, 0, 1],
        }
    )

    result = analyze_target(dataframe, "churn")

    print("Target Analysis:")
    print(result)


if __name__ == "__main__":
    main()