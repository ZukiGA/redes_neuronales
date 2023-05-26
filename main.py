from sklearn.neural_network import MLPClassifier
#import csv
clasificador = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(8), 
                    random_state=42)


#A PARTIR DE AQUÍ DE INICIA CON LA SEPARACIÓN Y CLASIFICACIÓN
valoresTrain = Training[atributosName]
valoresTest = Test[atributosName]
clasesTrain = Training[atributoClase]
clasesTest = Test[atributoClase]

################## Modelo ###########################
modelo = clasificador.fit(valoresTrain, clasesTrain)

################## Clasificar  #################
predict = modelo.predict(valoresTest)

############ Evaluar ###############################
reporte=classification_report(clasesTest, predict, labels=clasesName, output_dict=True)
reporte2=classification_report(clasesTest, predict, labels=clasesName)#, output_dict=True)
print(reporte2)
f1.append(reporte['accuracy'])
    
print(max(f1),sum(f1)/len(f1),min(f1))

# graficas overfitting
# experimento final