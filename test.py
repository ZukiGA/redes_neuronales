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
print(Y_trainK1)
#print(Y_trainK1.columns)
'''
#--------------------------
#Fold 2 train
X_trainK2 = pd.read_csv('./BalanceoRNA3/DiabetesK1Train.csv') #Fold 1 train
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
Y_trainK3 = X_trainK3.drop(columns=' class')
    #--------------------------
    #Fold 3 test
X_testK3 = pd.read_csv('./BalanceoRNA3/DiabetesK3Test.csv') #Fold 3 test
X_testK3 = X_testK3.drop_duplicates()
Y_testK3 = X_testK3[' class']
X_trainK3 = X_trainK3.drop(columns=' class')
'''
training_accuracy = [] #lista de aciertos de entrenamiento
test_accuracy = [] #lista de aciertos de test
training_error = [] #lista de errores de entrenamiento
test_error = [] #lista de errores de test



output_file = open("output.txt", "w")
n_epoch = range(5, 200, 5) #granularidad de 10 a 200 de 10 en 10
neurons = range(6,8,1)
lrates = range(21,50,1)
momentums = range(0,50,1)

for neuron in neurons: 
  for learnRate in lrates:
    for momentum in momentums:
      for epoch in n_epoch:
          # build the model
          learningRate = learnRate*0.01
          momentumf = momentum*0.01
          clasificador = MLPClassifier(solver='lbfgs', #metrica de calidad de resutlado
                          alpha=1e-5,
                          hidden_layer_sizes=(neuron),
                          random_state=42,
                          learning_rate_init= learningRate,
                          momentum=momentumf,
                          activation = 'relu',
                          max_iter=epoch)
          clasificador.fit(X_trainK1, Y_trainK1)
          # record training set accuracy and error
          training_accuracy.append(clasificador.score(X_trainK1, Y_trainK1))
          training_error.append(1.0 - clasificador.score(X_trainK1, Y_trainK1))
          # record generalization accuracy and error
          test_accuracy.append(clasificador.score(X_testK1, Y_testK1))
          test_error.append(1.0 - clasificador.score(X_testK1, Y_testK1))
          
          accuracytrain = clasificador.score(X_trainK1, Y_trainK1)
          accuracytest = clasificador.score(X_testK1, Y_testK1)
          if accuracytrain > 0.653 and accuracytest > 0.653:
            print(f"Neuronas: {neuron} LR: {learningRate} MOM: {momentumf} Epoch: {epoch} AccTrain: {accuracytrain} AccTest: {accuracytest}", file=output_file)
          
      if accuracytrain > 0.653 and accuracytest > 0.653:
        # print("Neuronas: {neuron} LR: {learnRate} MOM: {momentum} Epoch: {epoch} AccTrain: {accuracytrain} AccTest: {accuracytest}".format(neuron,learnRate,momentum,epoch,accuracytrain,accuracytest))
        plt.plot(n_epoch, training_accuracy, label="training accuracy")
        plt.plot(n_epoch, test_accuracy, label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_depth")
        plt.legend()
        plt.savefig(f"./graphs/output-{datetime.datetime.now()}.png")
            

          #print("Accuracy on training set: {:.3f}".format(clasificador.score(X_trainK1, Y_trainK1)))
          #print("Accuracy on test set: {:.3f}".format(clasificador.score(X_testK1, Y_testK1)))

'''
plt.plot(n_epoch, training_accuracy, label="training accuracy")
plt.plot(n_epoch, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_depth")
plt.legend()
plt.show()

plt.plot(n_epoch, training_error, label="training error")
plt.plot(n_epoch, test_error, label="test error")
plt.ylabel("Error")
plt.xlabel("n_depth")
plt.legend()
plt.show()
'''