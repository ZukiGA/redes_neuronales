import pandas as pd
from pandas import DataFrame
import numpy as np
#from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import datetime


# atributosName=setInicial.columns[:-1]
# atributoClase=setInicial.columns[-1]
# clasesName=setInicial[setInicial.columns[-1]].drop_duplicates()
# print(atributosName)
# print(clasesName)
def one_hot_encoder(df, orig_column):
  new_columns = df[orig_column].unique()
  for name in new_columns:
    df[f'{orig_column}_{name}'] = (df[orig_column] == name).astype(int)
  df = df.drop(columns=[orig_column])
  return df
'''
#Train 80% de datos
X_train80 = pd.read_csv('./BalanceoRNA3/DiabetesFirst80.csv') 
X_train80 = X_train80.drop_duplicates()
Y_train80 = X_train80[' class']

#Y_train80 = 
#Y_train80 = one_hot_encoder(Y_train80, 'class'))
print(X_train80)
print(Y_train80)
X_train80 = X_train80.drop(columns=' class')

#--------------------------
#Test 20% de datos
X_test20 = pd.read_csv('./BalanceoRNA3/DiabetesFirst20.csv') #Test 20% de datos
X_test20 = X_test20.drop_duplicates()
Y_test20 = X_test20[' class']
X_test20 = X_test20.drop(columns=' class')
'''
#--------------------------
#Fold 1 train
X_trainK1 = pd.read_csv('./BalanceoRNA3/DiabetesK1Train.csv')
X_trainK1 = X_trainK1.drop_duplicates()
Y_trainK1 = X_trainK1[' class'] 
X_trainK1 = X_trainK1.drop(columns=' class')

#--------------------------
#Fold 1 test
X_testK1 = pd.read_csv('./BalanceoRNA3/DiabetesK1Test.csv')
X_testK1 = X_testK1.drop_duplicates()
Y_testK1 = X_testK1[[' class']]
X_testK1 = X_testK1.drop(columns=' class')

#--------------------------
#Fold 2 train
X_trainK2 = pd.read_csv('./BalanceoRNA3/DiabetesK2Train.csv') #Fold 1 train
X_trainK2 = X_trainK2.drop_duplicates()
Y_trainK2 = X_trainK2[' class']
X_trainK2 = X_trainK2.drop(columns=' class')

#--------------------------
#Fold 2 test
X_testK2 = pd.read_csv('./BalanceoRNA3/DiabetesK2Test.csv') #Fold 2 test
X_testK2 = X_testK2.drop_duplicates()
Y_testK2 = X_testK2[' class']
X_testK2 = X_testK2.drop(columns=' class')

#--------------------------
#Fold 3 train
X_trainK3 = pd.read_csv('./BalanceoRNA3/DiabetesK3Train.csv') #Fold 3 train
X_trainK3 = X_trainK3.drop_duplicates()
Y_trainK3 = X_trainK3[' class']
X_trainK3 = X_trainK3.drop(columns=' class')

#--------------------------
#Fold 3 test
X_testK3 = pd.read_csv('./BalanceoRNA3/DiabetesK3Test.csv') #Fold 3 test
X_testK3 = X_testK3.drop_duplicates()
Y_testK3 = X_testK3[' class']
X_testK3 = X_testK3.drop(columns=' class')

# sets = [(X_trainK1, Y_trainK1, X_testK1, Y_testK1),(X_trainK2, Y_trainK2, X_testK2, Y_testK2),(X_trainK3, Y_trainK3, X_testK3, Y_testK3)]
sets = [(X_trainK3, Y_trainK3, X_testK3, Y_testK3)]
training_accuracy = [] #lista de aciertos de entrenamiento
test_accuracy = [] #lista de aciertos de test
training_error = [] #lista de errores de entrenamiento
test_error = [] #lista de errores de test

training_accuracy_record = []
testing_accuracy_record = []
plots_titles = []


output_file = open("output.txt", "w")
n_epoch = range(1, 160, 1) #granularidad de 10 a 200 de 10 en 10
neurons = range(7,10)
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
activations = ['logistic', 'relu', 'identity']


for neuron in neurons:
  for alpha in alphas:
    for activation in activations:
      accuracytrainglobal = 0
      accuracytestglobal = 0
      for X_train, Y_train, X_test, Y_test in sets:
        training_accuracy = []
        training_error = []
        test_accuracy = []
        test_error = []
        for epoch in n_epoch:
            # build the model
            clasificador = MLPClassifier(solver='lbfgs', #metrica de calidad de resutlado
                            alpha=alpha,
                            hidden_layer_sizes=(neuron),
                            random_state=42,
                            activation=activation,
                            max_iter=epoch)
            clasificador.fit(X_train, Y_train)
            # record training set accuracy and error
            training_accuracy.append(clasificador.score(X_train, Y_train))
            training_error.append(1.0 - clasificador.score(X_train, Y_train))
            # record generalization accuracy and error
            test_accuracy.append(clasificador.score(X_test, Y_test))
            test_error.append(1.0 - clasificador.score(X_test, Y_test))
            
            accuracytrain = clasificador.score(X_train, Y_train)
            accuracytest = clasificador.score(X_test, Y_test)
        accuracytrainglobal += accuracytrain
        accuracytestglobal += accuracytest 
        print(f"Neuronas: {neuron} Activation: {activation} alpha: {alpha} Epoch: {epoch} AccTrain: {accuracytrain} AccTest: {accuracytest}", file=output_file)

      accuracytestglobal = accuracytestglobal/len(sets)
      accuracytrainglobal = accuracytrainglobal/len(sets)

      if accuracytestglobal < accuracytrainglobal and accuracytrainglobal > 0.7 and accuracytestglobal > 0.7:
        print(f"AccTrain: {accuracytrainglobal} AccTest: {accuracytestglobal}", file=output_file)
        plt.plot(n_epoch, training_accuracy, label="training accuracy")
        plt.plot(n_epoch, test_accuracy, label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_depth")
        plt.text(5, 8,f"AccTrain: {accuracytrain} AccTest: {accuracytest}")
        plt.savefig(f"./graphs/output-{datetime.datetime.now()}.png")
        plt.clf()   