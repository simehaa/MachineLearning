# WineQuality
Machine learning project on red wine quality. The methods used were
* Bagging
* Random Forest
* Gradient Boosting

More information about this project can be found in the report.

#### Abstract
A numerical study regarding the classification of red wine preference scores is presented. A large data set (1599 data points) of red wine from the *vinho verde* district in Portugal is analysed by applying three ensemble machine learning techniques. The predictors at hand, were 11 various physicochemical properties of the wines. The response variable was a wine preference score from 0-10, based on the median value of at least three blind testes. The data mining techniques were *random forests*, *bagging* and *gradient boosting*. Due to the imbalanced nature of the outcomes, the most difficult task was to identify the worst and the best wines. Random forest was the method which achieved the best overall accuracy of predicting the exact correct score, with an accuracy of 66.9%. Confusion matrices from all threet models showed that the misclassified wines were more likely to be identified as a more common score (towards 5 and 6). Out of all the predictors: alcohol, sulphates, and volatile acidity was the three most important ones in the random forest and gradient boosting models. The data with the lowest and highest quality wines showed a noticeable difference in the values of these three variables. It was seemingly wines with higher alcohol percentages, lower volatile acidity, and higher levels of sulphates, which predicted a good wine.
