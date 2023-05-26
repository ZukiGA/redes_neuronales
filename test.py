import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


# atributosName=setInicial.columns[:-1]
# atributoClase=setInicial.columns[-1]
# clasesName=setInicial[setInicial.columns[-1]].drop_duplicates()
# print(atributosName)
# print(clasesName)

X_train80 = pd.read_csv('./BalanceoRNA3/DiabetesFirst80.csv') #Train 80% de datos
Y_train80 = X_train80['class']

X_test20 = pd.read_csv('./BalanceoRNA3/DiabetesFirst20.csv') #Test 20% de datos
Y_test20 = X_test20['class']

X_trainK1 = pd.read_csv('./BalanceoRNA3/DiabetesK1Train.csv') #Fold 1 train
Y_trainK1 = X_trainK1['class']

X_trainK2 = pd.read_csv('./BalanceoRNA3/DiabetesK1Test.csv') #Fold 1 train
Y_trainK2 = X_trainK2['class']

X_testK2 = pd.read_csv('./BalanceoRNA3/DiabetesK2Train.csv') #Fold 2 test
Y_testK2 = X_testK2['class']

X_trainK3 = pd.read_csv('./BalanceoRNA3/DiabetesK3Train.csv') #Fold 3 train
Y_trainK3 = X_trainK3['class']

X_testK3 = pd.read_csv('./BalanceoRNA3/DiabetesK3Test.csv') #Fold 3 test
Y_testK3 = X_testK3['class']

training_accuracy = [] #lista de aciertos de entrenamiento
test_accuracy = [] #lista de aciertos de test
training_error = [] #lista de errores de entrenamiento
test_error = [] #lista de errores de test

n_epoch = range(10, 200, 10) #granularidad de 10 a 200 de 10 en 10

for epoch in n_epoch:
    # build the model
    clasificador = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(8), 
                    random_state=42,
                    max_iter = epoch)
    clasificador.fit(X_trainK1, Y_testK1)
    # record training set accuracy and error
    training_accuracy.append(clasificador.score(X_trainK1, Y_testK1))
    training_error.append(1.0 - clasificador.score(X_trainK1, Y_testK1))
    # record generalization accuracy and error
    test_accuracy.append(clasificador.score(X_trainK1, Y_testK1))
    test_error.append(1.0 - clasificador.score(X_trainK1, Y_testK1))

