A combination of supervised models, unsupervised models, and NLP methods to help determine ratings for apps on the google play store. 

Text ratings from the google play store were processed using dictvectorizer, then run through dimensionality reduction using sklearn's TruncatedSVD to come up with key factors.

Review data was joined with general properties about the apps, and then were analyzed using a random forest regression model. The predictor managed to have an MSE of roughly 0.07.

Three separate notebooks were used for this project. Two for data processing each of the disparate datasets, one to link them and perform regression analyses.
