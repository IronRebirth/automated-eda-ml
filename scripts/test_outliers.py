import pandas as pd

from ml.quality.outliers import analyze_outliers


def main():
    dataframe = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 100],
            "salary": [30000, 32000, 31000, 33000, 34000, 150000],
        }
    )

    result = analyze_outliers(dataframe)

    print("Outliers:")
    print(result)


if __name__ == "__main__":
    main()