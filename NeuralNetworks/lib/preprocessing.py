import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler


def preprocess_raw_data(test_size=0.2):
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
    print("\nPreprocessing\n")
    # Raw data
    dataframe = pd.read_excel("./data/raw/defaulted_cc-clients.xls")
    # 0th row and column are dataframe headers and indices
    data = dataframe.to_numpy()[1:, 1:]
    N = data.shape[0]

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

    X_repayment_status_minus_two = np.zeros((data.shape[0], 6))
    X_repayment_status_minus_one = np.zeros((data.shape[0], 6))
    X_repayment_status_minus_two[data[:, 5:11] == -2] = 1.0
    X_repayment_status_minus_one[data[:, 5:11] == -1] = 1.0
    data[:, 5:11][np.logical_or(data[:, 5:11] == -2, data[:, 5:11] == -1)] = 0

    print(f"\t{N -data.shape[0]} outliers removed.")

    # ----- Split data set -----
    X_categorical = data[:, 1:4]
    X_continuous = np.concatenate((data[:, 0:1], data[:, 4:-1]), axis=1)
    y = data[:, -1]

    # ----- One Hot Encoding for categorical columns -----
    # categories = [[1, 2], [1, 2, 3, 4], [1, 2, 3]]
    enc = OneHotEncoder(handle_unknown="error", categories="auto")
    preprocessor = ColumnTransformer(transformers=[("onehot", enc, [0, 1, 2])])
    X_one_hot_encoded = preprocessor.fit_transform(X_categorical)
    X = np.concatenate(
        (
            X_one_hot_encoded,
            X_repayment_status_minus_two,
            X_repayment_status_minus_one,
            X_continuous,
        ),
        axis=1,
    )
    p = X_one_hot_encoded.shape[1]
    p += X_repayment_status_minus_one.shape[1]
    p += X_repayment_status_minus_two.shape[
        1
    ]  # index that separates OneHot and continuous colums

    # ----- Split of training and test -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # ----- Scaling of continuous columns -----
    scl = MinMaxScaler(feature_range=(0.0, 1.0))
    # --- Scale both train/test according to training set:
    scl.fit(X_train[:, p:].astype("float64"))  # only continuous columns
    X_train[:, p:] = scl.transform(X_train[:, p:].astype("float64"))
    X_test[:, p:] = scl.transform(X_test[:, p:].astype("float64"))

    # ----- Upscale training data -----
    # fraction (yi=1 / yi=0) = 1.0, this is achieved by
    # randomly resampling the rows with least occurence
    upscale = RandomOverSampler(sampling_strategy=1.0)
    X_train, y_train = upscale.fit_resample(X_train, y_train.astype("int"))

    # ----- Ensure that the datatype is float -----
    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")
    y_train = y_train.astype("float64")
    y_test = y_test.astype("float64")

    # ----- Save new files-----
    np.savez("./data/processed/train_data.npz", X=X_train, y=y_train)
    np.savez("./data/processed/test_data.npz", X=X_test, y=y_test)
    print("\tNew preprocessed data saved in ./data/preprocessed/.\n")


"""
X-data after preprocessing (29 columns)
---------------------------------------

        One Hot Encoded Columns (1 denoted in parenthesess)
        -----------------------
    0, SEX: Gender (male)
    1, SEX: Gender (female)
    2, EDUCATION: (graduate school)
    3, EDUCATION: (university)
    4, EDUCATION: (high school)
    5, EDUCATION: (others)
    6, MARRIAGE: Marital status (1married)
    7, MARRIAGE: Marital status (1single)
    8, MARRIAGE: Marital status (1others)
    9, PAY_0: Repayment status in September, 2005 (1: status bill paid
    10, PAY_2: Repayment status in August, 2005 (1: status bill paid
    11, PAY_3: Repayment status in July, 2005 (1: status bill paid
    12, PAY_4: Repayment status in June, 2005 (1: status bill paid
    13, PAY_5: Repayment status in May, 2005 (1: status bill paid
    14, PAY_6: Repayment status in April, 2005 (1: status bill paid
    15, PAY_0: Repayment status in September, 2005 (1: status bill NOT paid)
    16, PAY_2: Repayment status in August, 2005 (1: status bill NOT paid)
    17, PAY_3: Repayment status in July, 2005 (1: status bill NOT paid)
    18, PAY_4: Repayment status in June, 2005 (1: status bill NOT paid)
    19, PAY_5: Repayment status in May, 2005 (1: status bill NOT paid)
    20, PAY_6: Repayment status in April, 2005 (1: status bill NOT paid)


        Scaled Columns range [0, 1]
        ----------------------------------
    21, LIMIT_BAL: Amount of given credit in NT dollars
    22, AGE: Scaled age
    23, PAY_0: Repayment delay in September, 2005
    24, PAY_2: Repayment delay in August, 2005
    25, PAY_3: Repayment delay in July, 2005
    26, PAY_4: Repayment delay in June, 2005
    27, PAY_5: Repayment delay in May, 2005
    28, PAY_6: Repayment delay in April, 2005
    29, BILL_AMT1: Amount of bill statement in September, 2005
    30, BILL_AMT2: Amount of bill statement in August, 2005
    31, BILL_AMT3: Amount of bill statement in July, 2005
    32, BILL_AMT4: Amount of bill statement in June, 2005
    33, BILL_AMT5: Amount of bill statement in May, 2005
    34, BILL_AMT6: Amount of bill statement in April, 2005
    35, PAY_AMT1: Amount of previous payment in September, 2005
    36, PAY_AMT2: Amount of previous payment in August, 2005
    37, PAY_AMT3: Amount of previous payment in July, 2005
    38, PAY_AMT4: Amount of previous payment in June, 2005
    39, PAY_AMT5: Amount of previous payment in May, 2005
    40, PAY_AMT6: Amount of previous payment in April, 2005

Y-data after preprocessing (1 column vector)
--------------------------------------------

        Output data
        -----------
    0, default.payment.next.month: Default payment (1=yes, 0=no)
"""
