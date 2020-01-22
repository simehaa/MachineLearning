import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from lib.functions import *


def main():
    # Without upsampling
    X, Xt, y, yt, predictors, score_categories, original_y = preprocess_data(
        verbose=True, upsample=False
    )
    plot_histogram(original_y, score_categories)
    correlation_matrix(X, y, predictors)

    # With upsampling
    # Xu, Xtu, yu, ytu, predictors, score_categories, original_y = preprocess_data(
        # verbose=False, upsample=True
    # )
    bagging(Xu, Xtu, yu, ytu)
    random_forest(Xu, Xtu, yu, ytu)
    boosting(Xu, Xtu, yu, ytu)

    plot_feature_importances(
        Xu,
        yu,
        RandomForestClassifier(n_estimators=1000, max_features=9),
        predictors[:-1],
    )
    plot_feature_importances(
        Xu,
        yu,
        GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.15, max_leaf_nodes=6
        ),
        predictors[:-1],
    )


if __name__ == "__main__":
    main()



# Wines with highest quality
"""
      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  ...  density    pH  sulphates  alcohol  quality
267             7.9              0.35         0.46             3.6      0.078  ...  0.99730  3.35       0.86     12.8        8
278            10.3              0.32         0.45             6.4      0.073  ...  0.99760  3.23       0.82     12.6        8
390             5.6              0.85         0.05             1.4      0.045  ...  0.99240  3.56       0.82     12.9        8
440            12.6              0.31         0.72             2.2      0.072  ...  0.99870  2.88       0.82      9.8        8
455            11.3              0.62         0.67             5.2      0.086  ...  0.99880  3.22       0.69     13.4        8
481             9.4              0.30         0.56             2.8      0.080  ...  0.99640  3.15       0.92     11.7        8
495            10.7              0.35         0.53             2.6      0.070  ...  0.99720  3.15       0.65     11.0        8
498            10.7              0.35         0.53             2.6      0.070  ...  0.99720  3.15       0.65     11.0        8
588             5.0              0.42         0.24             2.0      0.060  ...  0.99170  3.72       0.74     14.0        8
828             7.8              0.57         0.09             2.3      0.065  ...  0.99417  3.46       0.74     12.7        8
1061            9.1              0.40         0.50             1.8      0.071  ...  0.99462  3.21       0.69     12.5        8
1090           10.0              0.26         0.54             1.9      0.083  ...  0.99451  2.98       0.63     11.8        8
1120            7.9              0.54         0.34             2.5      0.076  ...  0.99235  3.20       0.72     13.1        8
1202            8.6              0.42         0.39             1.8      0.068  ...  0.99516  3.35       0.69     11.7        8
1269            5.5              0.49         0.03             1.8      0.044  ...  0.99080  3.50       0.82     14.0        8
1403            7.2              0.33         0.33             1.7      0.061  ...  0.99600  3.23       1.10     10.0        8
1449            7.2              0.38         0.31             2.0      0.056  ...  0.99472  3.23       0.76     11.3        8
1549            7.4              0.36         0.30             1.8      0.074  ...  0.99419  3.24       0.70     11.4        8
"""

# Wines with lowest quality
"""
      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  ...  density    pH  sulphates  alcohol  quality
459            11.6             0.580         0.66            2.20      0.074  ...  1.00080  3.25       0.57     9.00        3
517            10.4             0.610         0.49            2.10      0.200  ...  0.99940  3.16       0.63     8.40        3
690             7.4             1.185         0.00            4.25      0.097  ...  0.99660  3.63       0.54    10.70        3
832            10.4             0.440         0.42            1.50      0.145  ...  0.99832  3.38       0.86     9.90        3
899             8.3             1.020         0.02            3.40      0.084  ...  0.99892  3.48       0.49    11.00        3
1299            7.6             1.580         0.00            2.10      0.137  ...  0.99476  3.50       0.40    10.90        3
1374            6.8             0.815         0.00            1.20      0.267  ...  0.99471  3.32       0.51     9.80        3
1469            7.3             0.980         0.05            2.10      0.061  ...  0.99705  3.31       0.55     9.70        3
1478            7.1             0.875         0.05            5.70      0.082  ...  0.99808  3.40       0.52    10.20        3
1505            6.7             0.760         0.02            1.80      0.078  ...  0.99600  3.55       0.63     9.95        3
"""
