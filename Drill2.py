import pandas as pd
import numpy as np

def get_dataframe(csvFile="titanic_train.csv"):
    return pd.read_csv(csvFile)

def get_age_class_sex():
    """
    I want to find a dataframe that lists just the Age, Pclass, and Sex. Return a DataFrame with these columns
    """
    df = get_dataframe()
    #YOUR CODE HERE
    return df[["Age", "Pclass", "Sex"]]

def get_age_class_sex_over_30():
    """
    Suppose I want the same 3 features, but this time, I only want to see people above the age of 30.
    Ultimately I would likevisual of people over thirty, but let's just focus on getting a dataframe for now.
    """
    df = get_age_class_sex()
    return df[df["Age"] > 30]

def get_100th_person_info():
    """
    I want to access the data of the 100th person, how would I do that?
    """
    df = get_dataframe()
    #YOUR CODE HERE
    return df.loc[99]


def calc_num_unique_ages():
    """
    Time to go back to the main dataframe, I would like to see how many unique ages there were, what function can I use to
    find that?
    """
    df = get_dataframe()
    #YOUR CODE HERE
    return df["Age"].nunique()

def get_cabin_nulls_and_shape():
    """
    I was going through the dataframe and noticed cabin had a lot of null values, how can I see how many null values there
    are? While I'm at it, let me see how big this dataframe is as a whole.
    
    This function should return a tuple of the form (number of null values in Cabin column, how many rows in the dataframe)
    """
    df = get_dataframe()
    #YOUR CODE HERE
    return (df.shape[0] - df["Cabin"].count(), df.shape[0])

def drop_cabin_col():
    """
    The Cabin column seems to have a lot of missing values. Drop this column and return the resulting dataframe
    """
    df = get_dataframe()
    #YOUR CODE HERE
    df.drop(columns=["Cabin"], axis=1, inplace=True)
    return df

def survived_within_class():
    """
    For each Pclass, return the proportion of people who survived. Round answer to nearest 3 decimal places
    Return tuple in the format (Pclass 1 survival proportion, Pclass 2 survival proportion, Pclass 3 survival proportion)
    If there is a ZeroDivisionError, return (-1,-1,-1)
    """
    df = get_dataframe()
    #YOUR CODE HERE
    groups = df.groupby("Pclass")["Survived"]
    try:
        return tuple((groups.sum() / groups.size()).round(3))
    except ZeroDivisionError:
        return (-1, -1, -1)

#print(get_dataframe())
#print(get_age_class_sex())
#print(get_age_class_sex_over_30())
#print(get_100th_person_info())
#print(calc_num_unique_ages())
#print(get_cabin_nulls_and_shape())
#print(drop_cabin_col())
print(survived_within_class())