import pandas as pd

from ml.eda import create_categorical_bar_charts


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

    figures = create_categorical_bar_charts(dataframe)

    print("Generated categorical figures:")
    print(list(figures.keys()))

    for column, figure in figures.items():
        print(
            f"{column}: "
            f"{len(figure.data)} trace(s)"
        )


if __name__ == "__main__":
    main()