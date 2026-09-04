import pandas as pd

from ml.eda import analyze_categorical_columns


def main():
    dataframe = pd.DataFrame(
        {
            "city": [
                "Dhaka",
                "Dhaka",
                "Chittagong",
                "Dhaka",
                "Sylhet",
            ],
            "contract": [
                "Monthly",
                "Yearly",
                "Monthly",
                "Monthly",
                "Yearly",
            ],
            "age": [20, 25, 30, 35, 40],
        }
    )

    result = analyze_categorical_columns(dataframe)

    print("Categorical EDA:")
    print(result)


if __name__ == "__main__":
    main()