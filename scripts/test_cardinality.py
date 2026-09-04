import pandas as pd

from ml.quality.cardinality import analyze_cardinality


def main():
    dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice", "Charlie"],
            "city": ["Dhaka", "Dhaka", "Chittagong", "Dhaka"],
            "age": [25, 30, 25, 35],
        }
    )

    result = analyze_cardinality(dataframe)

    print("Cardinality:")
    print(result)


if __name__ == "__main__":
    main()