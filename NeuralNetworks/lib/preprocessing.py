import pandas as pd
import numpy as np


def preprocess_raw_data(new_filename, scale=True, remove_outliers=True):
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
        If True, scale data.

    remove_outliers : bool
        If True, remove known outliers in data set.

    Returns
    -------
    None
        Save new data file in ../data/processed/
    """
    # Raw data
    dataframe = pd.read_excel("./data/raw/defaulted_cc-clients.xls")
    # 0th row and column are dataframe headers and indices
    data = dataframe.to_numpy()[1:, 1:]

    if remove_outliers:
        # Use numpy logical operators to identify all data points
        # with correct values, then reassign the data
        # Gender [1, 2]
        correct_gender = np.logical_or(data[:, 1] == 1, data[:, 1] == 2)
        data = data[correct_gender]
        # Eduaction [1, 2, 3, 4]
        correct_education = np.logical_and(data[:, 2] >= 1, data[:, 2] <= 4)
        data = data[correct_education]
        # Marrital status [1, 2, 3]
        correct_marrital_status = np.logical_or(data[:, 3] >= 1, data[:, 3] <= 3)
        data = data[correct_marrital_status]
        # Age, reasonable to assume [10, 110]
        correct_age = np.logical_and(data[:, 4] > 10, data[:, 4] < 110)
        data = data[correct_age]
        # Repayment status for 6 previous months [-2, -1, ... , 8 , 9]
        for i in range(5, 11):
            repayment_status = np.logical_and(data[:, i] >= -2, data[:, i] <= 9)
            data = data[repayment_status]

    if scale:
        pass
