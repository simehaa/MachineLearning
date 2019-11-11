import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_raw_data(test_size=0.5):
    print("Processing raw data...\n")
    # Raw data
    df = pd.read_csv("./data/ozone_496obs_25vars.txt", sep=" ")
    names = list(df)

    # ----- Split of training and test -----
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    N_train = len(train_df.index)
    N_test = len(test_df.index)

    # These columns only contains binary values (0 or 1)
    categorcal_indices = [1, 2, 3, 4, 5, 6, 7, 9, 13, 14, 17, 19, 20, 22, 23]
    # These columns contains continuous variables
    continuous_indices = [0, 8, 10, 11, 12, 15, 16, 18, 21, 24]
    categorcal_names = []
    continuous_names = []

    for i in range(25):
        if i in categorcal_indices:
            categorcal_names.append(names[i])
        else:
            continuous_names.append(names[i])

    all_names = categorcal_names + continuous_names
    train_continuous_df = train_df[continuous_names]
    test_continuous_df = test_df[continuous_names]
    train_categorical_df = train_df[categorcal_names]
    test_categorical_df = test_df[categorcal_names]

    # ----- Scaling of continuous columns -----
    # Scale columns so that mean=0, std=1
    scl = StandardScaler(with_mean=True, with_std=True)
    # Scale both the continuous columns of the train/test according to training set:
    scl.fit(train_continuous_df)  # only continuous columns
    train_continuous_df = scl.transform(train_continuous_df)
    test_continuous_df = scl.transform(test_continuous_df)

    train_arr = np.concatenate((train_categorical_df, train_continuous_df), axis=1)
    test_arr = np.concatenate((test_categorical_df, test_continuous_df), axis=1)

    train_df = pd.DataFrame(train_arr, columns=all_names)
    test_df = pd.DataFrame(test_arr, columns=all_names)

    # ----- Prints -----
    # Print summary of processed data set to enure that everything is as intended:

    print(f"\tTrain data: {N_train} data points")
    print(f"\tTest data: {N_test} data points\n")

    print("\tContinuous columns.")
    print("\tCovariant  Train mean  Train std  Test mean  Test std")
    print("\t-----------------------------------------------------")
    for name in all_names:
        if name in continuous_names:
            tr_mean = np.mean(train_df[name])
            tr_std = np.std(train_df[name])
            te_mean = np.mean(test_df[name])
            te_std = np.std(test_df[name])
            print(f"\t{name:7}    {tr_mean:4.2e}    {tr_std:4.2e}  ", end="")
            print(f"  {te_mean:4.2e}   {te_std:4.2e}")

    print("\n\tCategorical columns")
    print("\tCovariant  Train ones  Train zeros  Test ones  Test zeros")
    print("\t---------------------------------------------------------")
    for name in all_names:
        if name in categorcal_names:
            tr_ones = np.sum(train_df[name])
            tr_zeros = N_train - tr_ones
            te_ones = np.sum(test_df[name])
            te_zeros = N_test - te_ones
            print(f"\t{name:7}      {tr_ones}      ", end="")
            print(f" {tr_zeros}        {te_ones}       {te_zeros}")

    return train_df, test_df
