# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:33:09 2019

@author: OI400241
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

############################################################
#1 - IMPORTAR AS TABELAS PARA TRABALHAR
############################################################

#importar o arquivo de treino

enem_train = pd.read_csv('train.csv')
enem_train.head()

#importar o arquivo de teste

enem_test = pd.read_csv('test.csv')
enem_test.head()

############################################################
#2 - TRATAMENTO DAS TABELAS
############################################################

#Script para normalizar as colunas
#1 - Validações
target = 'NU_NOTA_MT'
print('\nTrain and Test Datasets have the same columns?:',
      enem_train.drop(target,axis=1).columns.tolist()==enem_test.columns.tolist())
print("\nVariables not in test but in train : ", 
      set(enem_train.drop(target,axis=1).columns).difference(set(enem_test.columns)))

#2 - Remover as colunas
dif = list(set(enem_train.drop(target,axis=1).columns).difference(set(enem_test.columns)))

enem_train.drop(dif,axis=1,inplace=True)

#Analisar os dados//
enem_train.columns
enem_train['NU_NOTA_REDACAO'].describe()

#Histograma
sns.distplot(enem_train['NU_NOTA_MT']);

type('NU_NOTA_MT')
type('NU_NOTA_REDACAO')

#Drop NaN linhas
colunas = enem_train.columns[enem_train.columns.slice_indexer('NU_NOTA_LC','NU_NOTA_MT')]
colunas

train = enem_train.dropna(subset=colunas)

#Total de dados faltantes na coluna matematica
train['NU_NOTA_MT'].isnull().sum()
train['NU_NOTA_LC'].isnull().sum()

############################################################
#3 - MODELO DE MACHINE LEARNING 1
############################################################

from sklearn.linear_model import LinearRegression

variaveis = ['NU_NOTA_CN', 'NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_REDACAO']

x = train[variaveis]
y = train['NU_NOTA_MT']

y.head()

modelo = LinearRegression()
x = x.fillna(-1)
modelo.fit(x,y)

#Aplicando no modelo de teste

x_prev = enem_test[variaveis]
x_prev = x_prev.fillna(-1)
x_prev.head()

p = modelo.predict(x_prev)
p

sub = pd.Series(p, index = enem_test['NU_INSCRICAO'], name = 'NU_NOTA_MT')
sub.shape

sub.to_csv("answer.csv", header = True)

############################################################
#4 - MODELO DE MACHINE LEARNING 2 - RANDOM FOREST
############################################################

from sklearn.ensemble import RandomForestRegressor

modelo = RandomForestRegressor(n_estimators = 9000, n_jobs = -1, random_state = 0)

variaveis = ['NU_NOTA_CN', 'NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_REDACAO', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']

x = train[variaveis]
y = train['NU_NOTA_MT']
x = x.fillna(-1)

#treinar o modelo
modelo.fit(x,y)

#Aplicando no modelo de teste
x_prev = enem_test[variaveis]
x_prev = x_prev.fillna(-1)
x_prev.head()

#Aplicar o modelo no teste
p = modelo.predict(x_prev)

sub = pd.Series(p, index = enem_test['NU_INSCRICAO'], name = 'NU_NOTA_MT')
sub.shape

sub.to_csv("answer.csv", header = True)

############################################################
#4 - MODELO DE MACHINE LEARNING 3 - Gradient Boosting
############################################################

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

modelo = GradientBoostingRegressor(n_estimators = 15000, max_depth = 4, 
                                   min_samples_leaf=15, min_samples_split=10, 
                                   learning_rate=0.01, loss='huber', 
                                   random_state = 5)

variaveis = ['NU_NOTA_CN', 'NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_REDACAO', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']

x = train[variaveis]
y = train['NU_NOTA_MT']
x = x.fillna(-1)

#treinar o modelo
modelo.fit(x,y)

#Aplicando no modelo de teste
x_prev = enem_test[variaveis]
x_prev = x_prev.fillna(-1)
x_prev.head()

#Aplicar o modelo no teste
p = modelo.predict(x_prev)

sub = pd.Series(p, index = enem_test['NU_INSCRICAO'], name = 'NU_NOTA_MT')
sub.shape

sub.to_csv("answer.csv", header = True)
