import numpy as np # for making numpy arrays
import pandas as pd # for creating the Data Frame
from sklearn.preprocessing import StandardScaler # for standardizing the data
from sklearn.model_selection import train_test_split # to split our data into training and test data
from sklearn import svm 
from sklearn.metrics import accuracy_score # for evaluating our model - calculating the % accuracy of our prediction

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome',axis = 1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data # data
Y = diabetes_dataset['Outcome'] # model

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 2)
# test_size = 0.2 means test data is 20% and training data is 80%

classifier = svm.SVC(kernel='linear')
# training the Support Vector Machine Classifier
classifier.fit(X_train, Y_train)

# accuracy score on the training data 
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# print('Accuracy score of the training data : ',training_data_accuracy)
# print('Accuracy score of the test data : ',test_data_accuracy)


# For non-diabetic case: input_data = (4,110,92,0,0,37.6,0.191,30)
# For diabetic case:
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
# this part below - is same for most predictor systems (converting input data into numpy array)
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
# this means that we are making the model understand that we are providing only 1 input now, we aren't giving 768 examples
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction) # remember - this prints the value inside a list

if (prediction[0] == 0):
  print('The person is non-diabetic')
else:
  print('The person is diabetic')
