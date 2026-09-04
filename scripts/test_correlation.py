import pandas as pd

from ml.eda import analyze_correlations


def main():
    dataframe = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "salary": [20000, 25000, 30000, 35000, 40000],
            "experience": [1, 2, 3, 4, 5],
            "city": ["Dhaka", "Dhaka", "Sylhet", "Dhaka", "Chittagong"],
        }
    )

    result = analyze_correlations(dataframe)

    print("Correlation Analysis:")
    print(result)


if __name__ == "__main__":
    main()