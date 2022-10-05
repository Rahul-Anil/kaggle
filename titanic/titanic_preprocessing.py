import pandas as pd
import numpy as np

# this is used in the initial replacement process
def initials_replacement(initials_list: pd.Series) -> list:
    replacement_list = []
    Initials_short_conv = {
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Don": "M_N",
        "Rev": "Special",
        "Dr": "Dr",
        "Mme": "Mrs",
        "Ms": "Miss",
        "Major": "Special",
        "Lady": "F_N",
        "Sir": "M_N",
        "Mlle": "Miss",
        "Col": "Special",
        "Capt": "Special",
        "Countess": "F_N",
        "Jonkheer": "M_N",
    }

    Initials_unq = list(initials_list.unique())
    for initials in Initials_unq:
        if initials in Initials_short_conv:
            replacement_list.append(Initials_short_conv[initials])
        else:
            replacement_list.append("Special")

    return replacement_list


# this is used to fill in the missing ages using Initials
def filling_missing_ages(df: pd.DataFrame) -> None:
    initials_unq = list(df["Initials"].unique())
    initials_age_mean = (
        df.groupby(["Initials"])["Age"].mean().to_frame(name="Mean_age").reset_index()
    )
    for initials in initials_unq:
        df.loc[(df.Age.isnull()) & (df.Initials == initials), "Age"] = int(
            initials_age_mean[initials_age_mean["Initials"] == initials]["Mean_age"]
        )


def get_Initials(df: pd.DataFrame) -> None:
    df["Initials"] = df["Name"].str.extract("([A-Za-z]+)\.")


def generic_perprocessing(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # get the initials
    get_Initials(train_df)
    get_Initials(test_df)

    # making the initials for the train and test col standard
    train_df["Initials"].replace(
        list(train_df["Initials"].unique()),
        initials_replacement(train_df["Initials"]),
        inplace=True,
    )

    test_df["Initials"].replace(
        list(test_df["Initials"].unique()),
        initials_replacement(test_df["Initials"]),
        inplace=True,
    )

    # filling in the missing ages
    filling_missing_ages(train_df)
    filling_missing_ages(test_df)

    # family size col
    train_df["Family_size"] = train_df["SibSp"] + train_df["Parch"]
    test_df["Family_size"] = test_df["SibSp"] + test_df["Parch"]

    train_df.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1, inplace=True)
    test_df.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1, inplace=True)

    train_df.dropna(axis=0, inplace=True)
    test_df.fillna(test_df["Fare"].mean(), inplace=True)

    return (train_df, test_df)


if __name__ == "__main__":
    train_df = pd.read_csv(
        "/Users/rahulanil/garchomp/projects/kaggle/titanic/data/train.csv"
    )
    test_df = pd.read_csv(
        "/Users/rahulanil/garchomp/projects/kaggle/titanic/data/test.csv"
    )

    train_df, test_df = generic_perprocessing(train_df, test_df)

    print(train_df.head())
