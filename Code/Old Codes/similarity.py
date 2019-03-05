# encoding: utf-8
import pandas as pd
import numpy as np
import Levenshtein

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

QGS = pd.read_csv('/home/fuchs/Documentos/MESTRADO/Masters/Code/Old Codes/QGS.csv', sep='\t')
Resultado = pd.read_csv('/home/fuchs/Documentos/MESTRADO/Masters/Code/Old Codes/Resultado.csv', sep='\t')

Saida = open('/home/fuchs/Documentos/MESTRADO/Masters/Code/Old Codes/Saida.csv', 'w')


#with open('/home/fuchs/Documentos/MESTRADO/Masters/QGS.csv') as csvfile:
#    QGS = list(csv.reader(csvfile))

#with open('/home/fuchs/Documentos/MESTRADO/Masters/Resultado.csv') as csvfile:
#    Resultado = list(csv.reader(csvfile))

lenQGS = sum(1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Code/Old Codes/QGS.csv')) - 1

lenResultado = sum(1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Code/Old Codes/Resultado.csv')) - 1
#QGS = [val for sublist in QGS for val in sublist]

listaQGS = []
listaResultado = []

for i in range (0, lenQGS):
    listaQGS.append(QGS.iloc[i, 0])

#print("Lista QGS:", listaQGS)
#print("Tamanho Lista QGS:", len(listaQGS))

for i in range (0, lenResultado):
    listaResultado.append(Resultado.iloc[i, 0])

#print("Lista Resultado:", listaResultado)
#print("Tamanho Lista Resultado:", len(listaResultado))

train_set = [listaQGS, listaResultado]
train_set = [val for sublist in train_set for val in sublist]

#print("Lista train_set:", train_set)
#print("Elementos train_set", len(train_set))

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)

matSimilaridade = cosine_similarity(tfidf_matrix_train[0:lenQGS], tfidf_matrix_train[lenQGS:lenQGS+lenResultado])
lin, col = matSimilaridade.shape

counter = 0

for i in range (0, lin):

    linha = matSimilaridade[i]

    currentNearest = np.argsort(linha)[-2:] #Pega os x - 1 maiores elementos

    linhaSaida = 'QGS' + str(i + 1) + ':\t\t\t' + listaQGS[i] + '\t' + '\n'



    for j in range(1, len(currentNearest)):
        book = currentNearest[-j]
        linhaSaida = linhaSaida + '\t\t\t\t' + listaResultado[book].strip() + '\t' '\n'


    print("\n")

    if Levenshtein.distance(listaQGS[i], listaResultado[book]) < 10:
        counter = counter + 1

    linhaSaida = linhaSaida + "\n"

    Saida.write(linhaSaida)
    Saida.flush()

print(counter)

