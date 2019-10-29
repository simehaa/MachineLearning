import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler


def preprocess_raw_data():
    """
    Read the raw data file in ../data/raw/ and preprocess that data.

    Columns
    -------
        LIMIT_BAL: Amount of given credit in NT dollars
        SEX: Gender (1=male, 2=female)
        EDUCATION: (1=graduate school, 2=university, 3=high school,
            4=others, 5=unknown, 6=unknown)
        MARRIAGE: Marital status (1=married, 2=single, 3=others)
        AGE: Age in years
        PAY_0: Repayment status in September, 2005 (-1=pay duly,
            1=payment delay for one month,
            2=payment delay for two months, ...
            8=payment delay for eight months,
            9=payment delay for nine months and above)
        PAY_2: Repayment status in August, 2005 (scale same as above)
        PAY_3: Repayment status in July, 2005 (scale same as above)
        PAY_4: Repayment status in June, 2005 (scale same as above)
        PAY_5: Repayment status in May, 2005 (scale same as above)
        PAY_6: Repayment status in April, 2005 (scale same as above)
        BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
        BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
        BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
        BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
        BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
        BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
        PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
        PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
        PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
        PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
        PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
        PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
        default.payment.next.month: Default payment (1=yes, 0=no)

    Parameters
    ----------
    new_filename : string
        Filename for the new file with processed data.

    scale : bool
        If True, scale data

    remove_outliers : bool
        If True, remove known outliers in data set.

    Returns
    -------
    None
        Save train and test data in separate files in ../data/processed/
    """
    # Raw data
    dataframe = pd.read_excel("./data/raw/defaulted_cc-clients.xls")
    # 0th row and column are dataframe headers and indices
    data = dataframe.to_numpy()[1:, 1:]

    # ----- Remove Outliers -----
    # Identify indices with correct values, then reassign the data
    # Gender [1, 2]
    correct_gender = np.logical_or(data[:, 1] == 1, data[:, 1] == 2)
    data = data[correct_gender]
    # Eduaction [1, 2, 3, 4]
    correct_education = np.logical_and(data[:, 2] >= 1, data[:, 2] <= 4)
    data = data[correct_education]
    # Marrital status [1, 2, 3]
    correct_marrital_status = np.logical_and(data[:, 3] >= 1, data[:, 3] <= 3)
    data = data[correct_marrital_status]
    # Age, reasonable to assume in range [10, 110]
    correct_age = np.logical_and(data[:, 4] > 10, data[:, 4] < 110)
    data = data[correct_age]
    # Repayment status for 6 previous months [-2, -1, ... , 8 , 9]
    for i in range(5, 11):
        repayment_status = np.logical_and(data[:, i] >= -2, data[:, i] <= 9)
        data = data[repayment_status]

    # ----- Split data set -----
    X_categorical = data[:, 1:4]
    X_continuous = np.concatenate((data[:, 0:1], data[:, 4:-1]), axis=1)
    y = data[:, -1]

    # ----- One Hot Encoding for categorical columns -----
    categories = [[1, 2], [1, 2, 3, 4], [1, 2, 3]]
    enc = OneHotEncoder(handle_unknown="error", categories=categories)
    preprocessor = ColumnTransformer(transformers=[("onehot", enc, [0, 1, 2])])
    X_one_hot_encoded = preprocessor.fit_transform(X_categorical)
    X = np.concatenate((X_one_hot_encoded, X_continuous), axis=1)
    p = X_one_hot_encoded.shape[1]  # index that separates OneHot and continuous colums

    # ----- Split of training and test -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # ----- Upscale data -----
    # fraction (yi=1 / yi=0) = 1.0, this is achieved by
    # randomly resampling the rows with least occurence
    upscale = RandomOverSampler(sampling_strategy=1.0)
    X_train, y_train = upscale.fit_resample(X_train, y_train.astype("int"))
    X_test, y_test = upscale.fit_resample(X_test, y_test.astype("int"))

    # ----- Scaling of continuous columns -----
    scl = StandardScaler(with_mean=True, with_std=True)
    # --- Scale both train/test according to training set:
    # scl.fit(X_train[:, p:].astype("float64")) # only continuous columns
    # X_train[:, p:] = scl.transform(X_train[:, p:].astype("float64"))
    # X_test[:, p:] = scl.transform(X_test[:, p:].astype("float64"))
    # --- Scale train/test independently:
    X_train[:, p:] = scl.fit_transform(X_train[:, p:].astype("float64"))
    X_test[:, p:] = scl.fit_transform(X_test[:, p:].astype("float64"))

    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")
    y_train = y_train.astype("float64")
    y_test = y_test.astype("float64")

    # Save new files
    np.savez("./data/processed/train_data.npz", X=X_train, y=y_train)
    np.savez("./data/processed/test_data.npz", X=X_test, y=y_test)


"""
Columns after preprocessing (30 columns)
-------
        One Hot Encoded Columns
        -----------------------
    0, SEX: Gender (1=male)
    1, SEX: Gender (1=female)
    2, EDUCATION: (1=graduate school)
    3, EDUCATION: (1=university)
    4, EDUCATION: (1=high school)
    5, EDUCATION: (1=others)
    6, MARRIAGE: Marital status (1=married)
    7, MARRIAGE: Marital status (1=single)
    8, MARRIAGE: Marital status (1=others)

        Scaled Columns, with mean=0, std=1
        ----------------------------------
    9, LIMIT_BAL: Amount of given credit in NT dollars
    10, AGE: Age in years
    11, PAY_0: Repayment status in September, 2005 (-1=pay duly,
        1=payment delay for one month,
        2=payment delay for two months, ...
        8=payment delay for eight months,
        9=payment delay for nine months and above)
    12, PAY_2: Repayment status in August, 2005 (scale same as above)
    13, PAY_3: Repayment status in July, 2005 (scale same as above)
    14, PAY_4: Repayment status in June, 2005 (scale same as above)
    15, PAY_5: Repayment status in May, 2005 (scale same as above)
    16, PAY_6: Repayment status in April, 2005 (scale same as above)
    17, BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
    18, BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
    19, BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
    20, BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
    21, BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
    22, BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
    23, PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
    24, PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
    25, PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
    26, PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
    27, PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
    28, PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
    29, default.payment.next.month: Default payment (1=yes, 0=no)
"""
