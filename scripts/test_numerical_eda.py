import pandas as pd

from ml.eda import analyze_numerical_columns


def main():
    dataframe = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "salary": [30000, 40000, 50000, 60000, 70000],
            "city": ["Dhaka", "Dhaka", "Chittagong", "Dhaka", "Sylhet"],
        }
    )

    result = analyze_numerical_columns(dataframe)

    print("Numerical EDA:")
    print(result)


if __name__ == "__main__":
    main()