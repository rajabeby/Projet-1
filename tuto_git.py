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
"""
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
# Adapter et transformer les données
A_encoded = enc.fit_transform(dummy_A)
A_encoded = pd.DataFrame(A_encoded.toarray(), columns=enc.get_feature_names_out(dummy_A.columns))
"""


