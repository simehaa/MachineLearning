import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from lib.processing import *


def main():
    train_df, test_df = preprocess_raw_data(test_size=0.5)
    X_train = train_df.loc[:, train_df.columns != "FFVC"]
    y_train = train_df["FFVC"]
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    print(results.summary2())


if __name__ == '__main__':
    main()
