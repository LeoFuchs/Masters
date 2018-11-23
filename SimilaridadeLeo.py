# encoding: utf-8

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

QGS = pd.read_csv('/home/fuchs/Documentos/MESTRADO/Masters/QGS.csv', sep='\t')
Resultado = pd.read_csv('/home/fuchs/Documentos/MESTRADO/Masters/Resultado.csv', sep='\t')

Saida = open('/home/fuchs/Documentos/MESTRADO/Masters/Saida.csv', 'w')


#with open('/home/fuchs/Documentos/MESTRADO/Masters/QGS.csv') as csvfile:
#    QGS = list(csv.reader(csvfile))

#with open('/home/fuchs/Documentos/MESTRADO/Masters/Resultado.csv') as csvfile:
#    Resultado = list(csv.reader(csvfile))

lenQGS = sum(1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/QGS.csv')) - 1
lenResultado = sum(1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Resultado.csv')) - 1
#QGS = [val for sublist in QGS for val in sublist]

listaQGS = []
listaResultado = []

for i in range (0, lenQGS):
    listaQGS.append(QGS.iloc[i, 0])

print("Lista QGS:", listaQGS)
print("Tamanho Lista QGS:", len(listaQGS))

for i in range (0, lenResultado):
    listaResultado.append(Resultado.iloc[i, 0])

print("Lista Resultado:", listaResultado)
print("Tamanho Lista Resultado:", len(listaResultado))

train_set = [listaQGS, listaResultado]
train_set = [val for sublist in train_set for val in sublist]

print("Lista train_set:", train_set)
print("Elementos train_set", len(train_set))

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)

matSimilaridade = cosine_similarity(tfidf_matrix_train[0:lenQGS], tfidf_matrix_train[lenQGS:lenQGS+lenResultado])
lin, col = matSimilaridade.shape

#print ("Tamanho da matriz de similaridade:", matSimilaridade.shape)

#IMPRIMIR MATRIZ
#for i in range (0, lin):
#    linha = matSimilaridade[i]
#    print(linha)

for i in range (0, lin):
    linha = matSimilaridade[i]
    currentNearest = np.argsort(linha)[-4:] #Pega os x - 1 maiores elementos
    linhaSaida = 'Artigo do QGS:  ' + listaQGS[i] + '\t' + '\n'
    for i in range(1, len(currentNearest)):
        book = currentNearest[-i]
        linhaSaida = linhaSaida + '\t\t\t\t' + listaResultado[book].strip() + '\t' '\n'
    linhaSaida = linhaSaida + "\n"
    Saida.write(linhaSaida)
    Saida.flush()

#print cosine_similarity(tfidf_matrix_train[0], tfidf_matrix_train)

