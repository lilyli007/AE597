import mglearn

mglearn.plots.plot_knn_classification(n_neighbors=1)

'''Here, we added three new data points, shown as stars. For each of them, we marked the closest point in the 
training set. The prediction of the one-nearest-neighbor algorithm is the label of that point (shown by the color of 
the cross). 

Instead of considering only the closest neighbor, we can also consider an arbitrary number, k, of neighbors. This is 
where the name of the k-nearest neighbors algorithm comes from. When considering more than one neighbor, 
we use voting to assign a label. This means that for each test point, we count how many neighbors belong to class 0 
and how many neighbors belong to class 1. We then assign the class that is more frequent: in other words, 
the majority class among the k-nearest neighbors. The following example uses the three closest 
neighbors: '''

mglearn.plots.plot_knn_classification(n_neighbors=3)

'''Again, the prediction is shown as the color of the cross. You can see that the prediction for the new data point 
at the top left is not the same as the prediction when we used only one neighbor. 

While this illustration is for a binary classification problem, this method can be applied to datasets with any 
number of classes. For more classes, we count how many neighbors belong to each class and again predict the most 
common class. 

Now let’s look at how we can apply the k-nearest neighbors algorithm using scikit-learn. First, we split our data into 
a training and a test set so we can evaluate generalization performance'''

from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

'''
Next, we import and instantiate the class. This is when we can set parameters, like the number of neighbors to use. 
Here, we set it to 3:
'''

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

'''Now, we fit the classifier using the training set. For KNeighborsClassifier this means storing the dataset, 
so we can compute neighbors during prediction: 
'''

clf.fit(X_train, y_train)

'''To make predictions on the test data, we call the predict method. For each data point in the test set, 
this computes its nearest neighbors in the training set and finds the most common class among these: '''

print("Test set predictions:", clf.predict(X_test))

'''
To evaluate how well our model generalizes, we can call the score method with the test 
data together with the test labels:
'''

print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

'''
We see that our model is about 86% accurate, meaning the model predicted the class correctly for 86% 
of the samples in the test dataset.
'''