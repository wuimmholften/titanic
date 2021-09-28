#Importacion de librerias
from queue import PriorityQueue
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns


#Importar los datos
#Importar los datos de los archivos .csv almacenados
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

#print(df_test.head())
#print(df_train.head())

#ENTENDIMIENTO DE LA DATA

#Verifico la cantidad de datos que hay en el dataset
print('Cantidad de datos:')
print(df_train.shape)
print(df_test.shape)

#Verifico el tipo de datos contenida en ambos dataset
print('Tipos de datos:')
print(df_train.info())
print(df_test.info())
 
#Verifico los datos faltantes de los dataset
print('Datos faltantes:')
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

#Verifico las estadisticas del dataset
print('Estadisticas del dataset:')
print(df_train.describe())
print(df_test.describe())

#PREPROCESAMIENTO DE LA DATA

#Cambio de datos de sexo en números
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)
df_test['Sex'].replace(['female','male'],[0,1],inplace=True)

#Cambio los datos de embarque en números

plt.hist(df_train.loc[df_train["Embarked"].notnull(),"Embarked"],bins=3)   
plt.show()

#Correlacion embarcaciones y supervivientes

sns.barplot(data=df_train,x="Embarked",y="Survived")
plt.title("Correlacion embarcaciones y supervivientes")
plt.show()

#Correlacion entre la clase y superviviente

sns.barplot(data=df_train,x="Pclass",y="Survived")
plt.title("Correlacion entre la clase y superviviente")
plt.show()

df_train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
df_test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)


#Remplazo los datos faltantes en la edad por la media de columna
print(df_train["Age"].mean())
print(df_test["Age"].mean())
promedio = 30
df_train['Age'] = df_train['Age'].replace(np.nan, promedio)
df_test['Age'] = df_test['Age'].replace(np.nan, promedio)

#Creo varios grupos de acuerdo a bandas de las edades
#Bandas: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100
bins = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7']
df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)
df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names) 

#Se elimina la columna de "Cabin" ya que tiene muchos datos perdidos
df_train.drop(['Cabin'], axis = 1, inplace=True)
df_test.drop(['Cabin'], axis = 1, inplace=True)

#Elimino las columnas que considero que son necesarias para el analisis
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)
df_test = df_test.drop(['Name','Ticket'], axis=1)

#Se elimina las filas con los datos perdidos
df_train.dropna(axis=0, how='any', inplace=True)
df_test.dropna(axis=0, how='any', inplace=True)

#Verifico los datos
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

print(df_train.shape)
print(df_test.shape)

print(df_test.head())
print(df_train.head())

#APLICACIONES DE LOS ALGORITMOS DE MACHINE LEARNING 

#Separo la columna con la informacion de los supervivientes
X = np.array(df_train.drop(['Survived'], 1))
y = np.array(df_train['Survived'])

#Separo los datos de "train" en entrenamiento y  prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##Regresión Logistica (1)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred_log = logreg.predict(X_test)
print('Precisión Regresion Logistica:')
print(logreg.score(X_train, y_train))

##Support Vector Machines (2)
scv = SVC()
scv.fit(X_train, y_train)
Y_pred_scv = scv.predict(X_test)
print('Precisión Soporte de Vectores:')
print(scv.score(X_train, y_train))


##K Neighbors (3)
knn = KNeighborsClassifier(n_neighbors= 3)
knn.fit(X_train, y_train)
Y_pred_kn = knn.predict(X_test)
print('Precisión Vecinos mas Cercanos:')
print(knn.score(X_train, y_train))

##Linear Regression (4)
lr = LinearRegression()
lr.fit(X_train, y_train)
Y_pred_lr = lr.predict(X_test)
print('Presición Regresión Lineal:')
print(lr.score(X_train, y_train))


##ESTE ES EL MEJOR DE LOS CINCO
## Decision Tree Classifier (5)
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
Y_pred_tree = tree.predict(X_test)
print('Precisión Decision Tree Classsifier:')
print(tree.score(X_train, y_train))

#PREDICCION UTILIZANDO LOS MODELOS

ids = df_test['PassengerId']

### Regresión Logistica
prediccion_logreg = logreg.predict(df_test.drop('PassengerId', axis=1))
out_logreg = pd.DataFrame({ 'PassengerId' : ids, 'Survived' : prediccion_logreg })
print('Predicción Regresion Logistica:')
print(out_logreg.head())

###Support Vector Machines
prediccion_scv = scv.predict(df_test.drop('PassengerId', axis=1))
out_scv = pd.DataFrame({ 'PassengerId' : ids, 'Survived' : prediccion_scv })
print('Predicción Soporte de Vectores:')
print(out_scv.head())

###K neighbors
prediccion_knn = knn.predict(df_test.drop('PassengerId', axis=1))
out_knn = pd.DataFrame({ 'PassengerId' : ids, 'Survived' : prediccion_knn })
print('Predicción Vecinos mas Cercanos:')
print(out_knn.head())

###Linear Regression
prediccion_lr = lr.predict(df_test.drop('PassengerId', axis=1))
out_lr = pd.DataFrame({ 'PassengerId' : ids, 'Survived' : prediccion_lr })
print('Predicción Regresión Lineal:')
print(out_lr.head())

###Decision Tree Classifier
prediccion_tree = tree.predict(df_test.drop('PassengerId', axis=1))
out_tree = pd.DataFrame({ 'PassengerId' : ids, 'Survived' : prediccion_tree })
print('Predicción Decision Tree Classifier:')
print(out_tree.head())


