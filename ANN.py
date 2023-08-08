#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = 'https://raw.githubusercontent.com/joanby/deeplearning-az/master/datasets/Part%201%20-%20Artificial%20Neural%20Networks%20(ANN)/Churn_Modelling.csv'

#Import DataSet
dataset = pd.read_csv(url)
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#Datos Categoricos
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("Churn_Modelling",
         OneHotEncoder(categories='auto'),
         [1]
         )
    ], remainder='passthrough'
)

X = transformer.fit_transform(X)
X = X[:, 1:]

#Divide the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Scale of Variables
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Build the RNA
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Inicializar la RNA
classifier = Sequential()

#Anadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",
               activation = "relu", input_dim = 11))

#Anadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",
               activation = "relu"))

#Anadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",
                     activation = "sigmoid"))

#Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Ajustamos la RNA al conjunto de Entrenamiento
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)


#Test the Model
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
new_prediction = classifier.predict(sc_x.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print(new_prediction)
print(new_prediction > 0.5)

#Evaluate the Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_Classifier():

    # Inicializar la RNA
    classifier = Sequential()

    # Añadir las capas de entrada y primera capa oculta
    classifier.add(Dense(units = 6, kernel_initializer = "uniform",
                      activation = "relu", input_dim = 11))

    # Añadir la segunda capa oculta
    classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))

    # Añadir la capa de salida
    classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

    # Compilar la RNA
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    #Devolver el clasificator
    return classifier

classifier = KerasClassifier(build_fn = build_Classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator =  classifier, X = x_train, y = y_train, cv = 10, n_jobs= -1, verbose = 1)

mean = accuracies.mean()
variance = accuracies.std()
print(mean)
print(variance)

from sklearn.model_selection import GridSearchCV

def build_Classifier(optimizer):
    # Inicializar la RNA
    classifier = Sequential()

    # Añadir las capas de entrada y primera capa oculta
    classifier.add(Dense(units = 6, kernel_initializer = "uniform",
                      activation = "relu", input_dim = 11))

    # Añadir la segunda capa oculta
    classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))

    # Añadir la capa de salida
    classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

    # Compilar la RNA
    classifier.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics = ["accuracy"])

    return classifier

classifier = KerasClassifier(build_fn = build_Classifier)

parameters = {
    'batch_size' : [25, 32],
    'nb_epoch' : [100, 500],
    'optimizer' : ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score


