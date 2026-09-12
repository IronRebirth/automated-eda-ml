import pandas as pd

from ml.quality.duplicates import analyze_duplicates


def main():
    dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice"],
            "age": [25, 30, 25],
        }
    )

    result = analyze_duplicates(dataframe)

    print("Duplicate Rows:")
    print(result)


if __name__ == "__main__":
    main()