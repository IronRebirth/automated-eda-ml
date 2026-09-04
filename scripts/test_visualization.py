import pandas as pd

from ml.eda import create_numerical_histograms


def main():
    dataframe = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "salary": [30000, 40000, 50000, 60000, 70000],
            "city": [
                "Dhaka",
                "Dhaka",
                "Chittagong",
                "Dhaka",
                "Sylhet",
            ],
        }
    )

    figures = create_numerical_histograms(dataframe)

    print("Generated figures:")
    print(list(figures.keys()))

    for column, figure in figures.items():
        print(
            f"{column}: "
            f"{len(figure.data)} trace(s)"
        )


if __name__ == "__main__":
    main()