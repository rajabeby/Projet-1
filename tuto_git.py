# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:43:08 2024

@author: HP
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import MissingIndicator,KNNImputer,SimpleImputer
from sklearn.impute import IterativeImputer

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedKFold
df=pd.read_spss("EGI-ODD_Depenses.sav")

#                   On commence par modifier la variable "CMAge" pour qu'elle ne contiennent que des nombres

def convertir_en_numerique(valeur):
    try:
        # Essayer de convertir la valeur en numérique
        return int(valeur.replace(' ans', ''))
    except ValueError:
        # Si la conversion échoue, retourner une valeur par défaut
        return 95
df['CMAge'] = df['CMAge'].apply(convertir_en_numerique)
#               GESTION DES VALEURS MANQUANTES
#On calcule la proportion des valeurs manquantes relative à chaque variable,puis on supprime les variables ayant plus de 30% de valeurs manquantes

b=df.isnull()
missing_value=b.sum()
missing_prcent=missing_value/len(df)
missing_table=pd.DataFrame({"Missing value":missing_value, "Proportion":missing_prcent})

iteration=0
suppression=[]
for i in missing_table["Proportion"]:
    if (i>0.3):
        suppression.append(missing_table["Proportion"].index[iteration])
    iteration=iteration+1
A=df.drop(suppression,axis=1)

A["CMAge"]=pd.to_numeric(A["CMAge"])
#On supprime les colonnes contenant les informations personnels
A=A.drop(["Province","mred","Ndur","hhid","CMEducNiv","DecileNat","ODD10_2_1","MenPoids","IndPoids","SCORETOT","PauvSub","HH7","SITE","taille_c","CSP_New2_CM","ED45"],axis=1)
A=A.drop(["FoodPline","Pline","Pline19PPA","PauvMon","DemiDepMed",'PauvNonMon',],axis=1)

#                Imputation
# Pour remplacer les valeurs manquantes on a utilisé différentes techniques d'imputation
# POur les variables catégorielles nous et discrètes nous avons fait une imputation par le mode
# Pour les variables continues, nous avons fait une imputation par la médianne s la distribution de la variable est a un coefficient d'assymétrie de Fisher différent de zéro, et une imputation par la moyenne sinon
# L'objectif étant de minimiser le plus possible le changement de la forme de la distribution de chaque variable




dummy_A=A.select_dtypes(exclude=[np.number])
A=A.select_dtypes(include=[np.number])

x=[]
i=dummy_A.columns
imputer_mode=SimpleImputer(strategy='most_frequent')
dummy_A[i]=imputer_mode.fit_transform(dummy_A[i])

assymetrie=A.select_dtypes(include=[np.number]).skew()

med=[]

iteration=0
for i in assymetrie:
    if abs(i)>0.5:
        
        A[assymetrie.index[iteration]]=A[assymetrie.index[iteration]].fillna(A[assymetrie.index[iteration]].median())
    else:
        A[assymetrie.index[iteration]]=A[assymetrie.index[iteration]].fillna(A[assymetrie.index[iteration]].mean())
    iteration=iteration+1
A=pd.concat([A,dummy_A],axis=1)


##                 Gestion des outliers
# Pour améliorer la qualité d'ajustement de notre modèle, la gestion des valeurs abérrantes est nécéssaire pour éviter que ces dernières n'affectent négativement notre modèle
# Pour cela nous avons choisi d'utiliser l'algorithme de l'Isolation Forest pour détecter ces valeurs
# Cette algorithme repose sur les arbres de décisions. Compte tenu du fait que nos données sont des valeurs multidimensionnelles, les méthodes habituelles de détection des valeurs abérrantes ne sont pas utilisable, d'où le fait de récourir à l'Isolation Forest

from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
# Adapter et transformer les données
A_encoded = enc.fit_transform(dummy_A)
A_encoded = pd.DataFrame(A_encoded.toarray(), columns=enc.get_feature_names_out(dummy_A.columns))



from sklearn.ensemble import IsolationForest


Outliers=pd.concat([A_encoded,A.select_dtypes(include=[np.number])],axis=1)

clf = IsolationForest(max_samples=100, random_state=40)
clf.fit(Outliers)
y_pred_outliers = clf.predict(Outliers)
y_pred_outliers=pd.Series(y_pred_outliers)

iteration=0
suppression=[]
for i in y_pred_outliers:
    if (i==-1):
        suppression.append(iteration)
    iteration=iteration+1   
A=A.drop(A.index[suppression],axis=0)
# Nous avons décidé de supprimer ces valeurs également à cause du fait que le nombre de ces outliers obtenus sont plutôt faibles par rapport à au dataset total
A = A.reset_index(drop=True)
dummy_A=dummy_A.drop(dummy_A.index[suppression],axis=0)
    

    
   

dummy_A=A.select_dtypes(exclude=[np.number])
dummy_A = dummy_A.reset_index(drop=True)

y=dummy_A["Milieu"]
dummy_A=dummy_A.drop(["Milieu"],axis=1)

cluster=A.select_dtypes(include=[np.number])
from sklearn.preprocessing import KBinsDiscretizer
# Nous rappelons que notre objectif est de prédire le milieu des résidence des individus en utilisant un modèle de discrimination qui est la régression logistique.
# Mais compte tenu du grand nombre des variables, cela peut affecter la qualité de notre ajustement.
# Pour éviter cela, nous avons choisi de réduire la dimension de nos données en utilisant des méthodes factorielles
# Nous observons entre autre que notre dataset contient des variables mixtes et une large majorité d'entre-elles sont des variables cartégorielles, ce qui nous empêche d'utiliser des méthodes comme l'ACP ou AFCM.
# cependant il existe également une méthode appelée Analyse Factorielle des Données Mixtes qui permet de reduire la dimension des datasets contenant des variables à la fois numériques et catégorielles.
# Le principe de cette méthode consiste à transformer nos variables numériques en variables catégorielles, pour cela on peut utiliser les algorithmes de partitionnement comme les K-moyennes(K-means)
# Une fois cette étape éffectuée, on fait ensuite une analyse des correspondances multiples sur nos nouvelles données ne contennat que les variables numériques en plus des nouvelles variables obtenuent après avoir transformé nos variables numériques


# Création d'une instance de KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans',subsample=None,random_state=42)

# Transformation du DataFrame
data_binned = discretizer.fit_transform(cluster)
cluster = pd.DataFrame(data_binned, columns=cluster.columns, dtype='int')


dummy_A=pd.concat([cluster,dummy_A],axis=1)

import prince

# Création d'une instance de la classe MCA
mca = prince.MCA(n_components=17)

# Nous ne sélectionnons que les 17 premiers axes factoriels qui représentent plus de 99% l'inertie de notre nuage d'origine

# Réalisation de l'ACM
mca = mca.fit(dummy_A)


Z=mca.row_coordinates(dummy_A)

# Conversion des variables catégorielles en variables factices
y_dummies = pd.get_dummies(y, drop_first=True)
y=y_dummies.astype(float)


# Nous allons éffectuer une validation croisée à trois plis
resultat_train = []

resultat_test = []


cross_validation = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
for train_index, test_index in cross_validation.split(Z, y):
    
    
    X_train = Z.iloc[train_index]
    y_train = y.iloc[train_index]
    
    # Création du modèle de régression logistique
    logit_model = sm.Logit(y_train, sm.add_constant(X_train))


    # Ajustement du modèle
    result = logit_model.fit()  # Vous pouvez remplacer 'lbfgs' par l'optimiseur de votre choix

    
    
    log=LinearDiscriminantAnalysis()
    log.fit(X_train,y_train)
    y_pred_train = result.predict(sm.add_constant(X_train))
    y_pred_train = (y_pred_train > 0.5).astype(int)
    
    
    
    X_test =  Z.iloc[test_index]
    y_test =  y.iloc[test_index]
   
    log=LinearDiscriminantAnalysis()
    log.fit(X_test,y_test)
    y_pred_binary = log.predict(X_test)
    
    # Affichage du résumé du modèle
    #print(result.summary())

    yy = result.predict(sm.add_constant(X_test))
    y_pred_binary = (yy > 0.5).astype(int)
    
    

    report_test = classification_report(y_test, y_pred_binary)
    report_train= classification_report(y_train, y_pred_train)
    resultat_test.append(report_test)
    resultat_train.append(report_train)   

# En ce qui concerne les mesures de performance de notre modèle, nous obtenons à la fois sur les données d'entraînement ainsi que de test un score de 0,89 sur l'Accuracy, la Precision, le Rappel ainsi que le score F1. 
# Seul l'AUC diffère légèrement entre les donnéés, on a un score de 0,885 sur les données de test et un score de 0,882 sur les données d'entraînement.
# Ces scores très proche nous permet d'affirmer sans trop se tromper les bonnes performances de notre modèle sur ces données
