# -*- coding: utf-8 -*-
"""
Analise de Algoritmos e Estruturas de Dados - Profa. Lilian Berton
  Seminario 02: Processamento de Texto e Redução de Dimensionalidade com Feature Hashing
@author: Willian Dihanster Gomes de Oliveira

Referencias:
 - https://kavita-ganesan.com/hashingvectorizer-vs-countvectorizer/#.YLAX7qhKjIV
 - https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py

"""
import time
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

#--------------------------#
# Leitura dos Dataset      #
#--------------------------#
categories = ['sci.electronics', 'sci.space']
train = fetch_20newsgroups(
    subset='train', 
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)
test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)


#---------------------------------#
# Aplicacao do HashingVectorizer  #
#---------------------------------#
tempos = []
for i in range(10):
    inicio = time.time()
    
    # Inicia o vectorizer
    vectorizer = HashingVectorizer(n_features=10000, alternate_sign=False)
    
    # Aplica o vectorizer no dataset de treino e teste
    X_train = vectorizer.fit_transform(train.data)
    y_train = train.target
    
    X_test = vectorizer.transform(test.data)
    y_test = test.target
    
    # Inicia e treina o classificador Multinomial Naive Bayes
    clf = MultinomialNB(alpha=.01)
    clf.fit(X_train, y_train)
    MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
    
    # Faz as predicoes
    y_pred = clf.predict(X_test)
    
    # Salva o tempo gasto nesta iteracao
    fim = time.time()
    tempos.append(fim-inicio)

# Printa o ultim o resultado da Classificacao e o tempo medio gasto
print(classification_report(y_test, y_pred))
print(f'Tempo medio com o HashingVectorizer = {round(sum(tempos)/len(tempos), 15)}s')        


#---------------------------------#
# Aplicacao do CountVectorizer  #
#---------------------------------#
tempos = []
for i in range(10):
    inicio = time.time()

    # Inicia o vectorizer
    vectorizer = CountVectorizer(max_features=None)


    # Aplica o vectorizer no dataset de treino e teste
    X_train = vectorizer.fit_transform(train.data)
    y_train = train.target
    
    X_test = vectorizer.transform(test.data)
    y_test = test.target
    
    # Inicia e treina o classificador Multinomial Naive Bayes
    clf = MultinomialNB(alpha=.01)
    clf.fit(X_train, y_train)
    MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
    
    # Faz as predicoes
    y_pred = clf.predict(X_test)
    
    # Salva o tempo gasto nesta iteracao
    fim = time.time()
    tempos.append(fim-inicio)

# Printa o ultim o resultado da Classificacao e o tempo medio gasto
print(classification_report(y_test, y_pred))
print(f'Tempo medio com o CountVectorizer = {round(sum(tempos)/len(tempos), 15)}s')      



