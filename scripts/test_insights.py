import pandas as pd

from ml.eda import (
    analyze_categorical_columns,
    analyze_correlations,
    analyze_numerical_columns,
    analyze_target,
    generate_eda_insights,
)


def main():
    dataframe = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 100],
            "salary": [
                30000,
                31000,
                32000,
                33000,
                34000,
                150000,
            ],
            "city": [
                "Dhaka",
                "Dhaka",
                "Dhaka",
                "Dhaka",
                "Dhaka",
                "Sylhet",
            ],
            "churn": [0, 0, 0, 0, 0, 1],
        }
    )

    numerical_analysis = analyze_numerical_columns(dataframe)
    categorical_analysis = analyze_categorical_columns(dataframe)
    correlation_analysis = analyze_correlations(dataframe)
    target_analysis = analyze_target(dataframe, "churn")

    insights = generate_eda_insights(
        dataframe,
        numerical_analysis,
        categorical_analysis,
        correlation_analysis,
        target_analysis,
    )

    print("EDA Insights:")

    for insight in insights:
        print(f"- {insight}")


if __name__ == "__main__":
    main()