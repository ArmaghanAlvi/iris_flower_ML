#Importing libraires and checking versions
import scipy as sp
print(f'scipy: {sp.__version__}')
import numpy as np
print(f'numpy: {np.__version__}')
import matplotlib as plt
print(f'matplotlib: {plt.__version__}')
import pandas as pd
print(f'pandas: {pd.__version__}')
import sklearn
print(f'sci-kit learn: {sklearn.__version__}\n\n')

# Load libraries - modules, functions, and objects
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#Load iris flower dataset
file = "iris.csv" #name of dataset file
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #column names
dataset = read_csv(file, names = names) #reads data and sets column names


"""
-------- Set Summarization --------
"""
#Checking shape - how many instances (rows) and how many attributes (columns) 
# there are in the data file
print("\n\nInstances, attributes per instance:", dataset.shape)

#Peeking at the data - checking the first 20 rows to get a general idea
print('\n\nPeeking rows 0-19:')
print(dataset.head(20))

#Quick statistical summary - checking count, mean, min, max, and percentiles
print('\n\nStats:')
print(dataset.describe())

#Checking number of rows per classification of iris
print('\n\nClass distribution:')
print(dataset.groupby('class').size()) #The groupby function takes in "class" since its the name of a qualitative attribute


"""
-------- Data Vizualization --------
"""
#Univariate plots - plotting each separate/individual variable 
#Box and whisker
dataset.plot(kind = "box", subplots = True, layout = (2, 2), sharex = False, sharey = False) #Create 4 seperate plots that don't share axis in a 2x2 layout
pyplot.show() #lets you display plots visually
#Histogram
dataset.hist() #this one is very simple
pyplot.show()

#Multivariate plots - show interactions and relationships between different variables
#Scatterplots for all variable pairs
scatter_matrix(dataset) #creates a grid of scatter plots to compare different variable relationships
pyplot.show()


"""
-------- Evaluating Algorithms --------
Creating models of the data and estimate
accuracy on unseen data
"""
#First, split off a piece of the dataset - this will be the validation set to test the models
arr = dataset.values
X = arr[:,0:4] #array of first 4 columns - features/predictor variables
Y = arr[:,4] #array of 5th column - target variable: in this case, it is iris classification

#split into training and validation sets
#X_train and Y_train are the training sets and X_validation and Y_validation are the testing/validation sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.20, random_state = 1) 

#This program will use stratified 10-fold cross validation to estimate model accuracy
#What the heck does that mean?
    #k-fold Cross validation: 
        #Cross validation is a statistical method used to estimate the skill of machine learning models
        #Applied machine learning to compare and select a model for a given predictive modeling problem
        #You want tosplit your data into a training and a test set, where the test set is used to evaluate the model trained 
        #  with the training set. However, doing this once isn't reliable, and so you use this method
        #The parameter k represents the number of groups the data set is split into
        #Cross validation is used to estimate the skill of a machine learning model on unseen data. 
        #  AKA: it uses a limited sample to estimate how the model is expected to perform when making predictions 
        #       on data not used during the training of the model (unseen data).
    #Stratified: each fold/split of the dataset will aim to have the same 
    #            distribution by class as the whole training dataset.

#So basically, the program is using a Stratified Cross Validation of 10-folds, where it will split
#the dataset into 10, and train on 9 and test on the remaining 1, and then repeat the process for all 
#combinations of training-testing splits
#It will use random_state to set a random seed to a fixed number so each algorithm is tested on the same splits
#The metric used to evaluate models will be "Accuracy", which is the ratio of correctly predicted instances 
#divided by total number of instances (then multiplied by 100 to get a percentage) 


#We'll test a few algorithms since we don't know what will work best
#These are a few different classification algorithms (since we're trying to evaluate class of iris flower)
models = [] #List of tuples
models.append(('LR', OneVsRestClassifier(LogisticRegression(solver = 'liblinear')))) #Logistic Regression using liblinear algorithm ; compares "one class vs the rest"
models.append(('LDA', LinearDiscriminantAnalysis())) #Linear Discriminant Analysis
models.append(('KNN', KNeighborsClassifier())) #K-Nearest Neighbors Clasifier - based on the idea that similar data exists in proximity
models.append(('CART', DecisionTreeClassifier())) #Decision Tree Classifier - uses a tree algorithm to map observations about features to make conclusions about the target
models.append(('NB', GaussianNB())) #Gaussian Naive Bayes classifier - based on Gaussian distribution and Bayes theorem
models.append(('SVM', SVC(gamma = 'auto'))) #Support Vector Classifier

#Evaluate each model
print('\n\n\nModel Evaluation:')
results = []
names = []
for name, model in models:
    #This is the k-fold thing we talked about - data is randomly shuffled, split into 10 folds, and random_state = 1 ensures
    #consistent shuffling for consistent results
    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True) 
    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy') #cv = kfold means the cross validation strategy is k-fold ; the scoring is based on accuracy, as stated earlier
    results.append(cv_results) #add to results list
    names.append(name) #The first element ("name") of the tuples we made for models
    print('\n%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#Let's compare results with box and whisker plots
pyplot.boxplot(results, labels = names)
pyplot.title('Algorithm Comparison')
pyplot.show()


#Based on evaluation, SVM seems to be the best algorithm
#Lets test it against the validation set we made at the beginning
    #This provides an independent final check on its accuracy
    #It is valuable to keep a separate validation set just in case any mistakes were made in training

#Make predictions with the training dataset
model = SVC(gamma = 'auto')
model.fit(X_train, Y_train) #Applies the training set to teach the model
predictions = model.predict(X_validation) #uses our model to make predictions for the unseen data provided by the X_validation set

#Now that we had our model make its predictions, lets evaluate them
print('\n\n\nEvaluating the model:')
print("Accuracy score:", accuracy_score(Y_validation, predictions)) #calculates the percentage of correct predictions
print("\nConfusion matrix:\n", confusion_matrix(Y_validation, predictions)) #Creates a table that compares actual to predicted (True (correct) and False (incorrect) positives and negatives)
print("\nClassification report:\n", classification_report(Y_validation, predictions)) #generates report on model's performance

