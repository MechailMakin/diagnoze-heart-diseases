#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('heart_2020_cleaned.csv') 
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values


#cleaning the dataset
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
X[:,10] = labelencoder_X.fit_transform(X[:,10]) #omehot
X[:,11] = labelencoder_X.fit_transform(X[:,11])
X[:,14] = labelencoder_X.fit_transform(X[:,14])
X[:,15] = labelencoder_X.fit_transform(X[:,15])
X[:,16] = labelencoder_X.fit_transform(X[:,16])
X[:,7] = labelencoder_X.fit_transform(X[:,7])
X[:,8] = labelencoder_X.fit_transform(X[:,8]) #onehot
X[:,9] = labelencoder_X.fit_transform(X[:,9]) #onehot
X[:,12] = labelencoder_X.fit_transform(X[:,12]) #onehot

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) 

onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [8,9,10,12])],
    remainder='passthrough')
X = onehotencoder.fit_transform(X)


#spliting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#using kernal SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=(0))
classifier.fit(X_train, y_train)

#predict
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)






