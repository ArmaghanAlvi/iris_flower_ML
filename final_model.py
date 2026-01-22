# Load libraries - modules, functions, and objects
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#Load iris flower dataset
file = "iris.csv" #name of dataset file
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #column names
dataset = read_csv(file, names = names) #reads data and sets column names



"""
-------- ML Model --------
"""
#First, split off a piece of the dataset 
arr = dataset.values
X = arr[:,0:4]
Y = arr[:,4]
#split into training and validation sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.20, random_state = 1) 

#Make predictions with the training dataset
model = SVC(gamma = 'auto')
model.fit(X_train, Y_train) 
predictions = model.predict(X_validation) 

#Now that we had our model make its predictions, lets evaluate them
print('\n\n\nEvaluating the model:')
print("Accuracy score:", accuracy_score(Y_validation, predictions)) #calculates the percentage of correct predictions
print("\nConfusion matrix:\n", confusion_matrix(Y_validation, predictions)) #Creates a table that compares actual to predicted (True (correct) and False (incorrect) positives and negatives)
print("\nClassification report:\n", classification_report(Y_validation, predictions)) #generates report on model's performance