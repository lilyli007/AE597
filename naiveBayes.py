from pandas import np

X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

'''Here, we have four data points, with four binary features each. There are two classes, 0 and 1. For class 0 (the 
first and third data points), the first feature is zero two times and nonzero zero times, the second feature is zero 
one time and nonzero one time, and so on. These same counts are then calculated for the data points in the second 
class. Counting the nonzero entries per class in essence looks like this: '''

counts = {}
for label in np.unique(y):
    # iterate over each class
    # count (sum) entries of 1 per feature
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n", counts)

'''The other two naive Bayes models, MultinomialNB and GaussianNB, are slightly different in what kinds of statistics 
they compute. MultinomialNB takes into account the average value of each feature for each class, while GaussianNB 
stores the average value as well as the standard deviation of each feature for each class. 

To make a prediction, a data point is compared to the statistics for each of the classes, and the best matching class 
is predicted. Interestingly, for both MultinomialNB and BernoulliNB, this leads to a prediction formula that is of 
the same form as in the linear models (see “Linear models for classification”). Unfortunately, coef_ for the naive 
Bayes models has a somewhat different meaning than in the linear models, in that coef_ is not the same as w. 


'''